"""Maps out of band-limited Fourier bins: zero padding, cropping, WDM cells.

The Dirichlet kernels (:func:`pad_maps`, :func:`crop_maps`) and the cell matrices of
:class:`FourierToWDM` (its dense atoms and :meth:`FourierToWDM.after`) are ported from
the galactic-binary toy model, which does not ship; the kernels count the self-conjugate
source bins once.  The atoms are ``wilson.transform.band_atoms``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from domainia import wilson
from domainia.wilson._checks import (index_array, require_bool, require_instance, require_int,
                                     require_str)
from domainia.wilson.backend import pytree_dataclass
from domainia.wilson.grid import carrier_phase
from domainia.wilson.transform import band_atoms
from domainia.wilson.windows import Window, sample_window

from ..domains import FourierDomain, WDMDomain
from .base import _BUILD, LinearMap, _built_for, _last_axis, _require_same_units

__all__ = ["FourierCrop", "FourierToWDM", "FourierZeroPad", "crop_maps", "pad_maps"]


# ---------------------------------------------------------------------------
# Dirichlet kernels (the self-conjugate source bins counted once)
# ---------------------------------------------------------------------------


def _geometric(num, den: int, n: int, n_nu_mod2):
    r"""``sum_{s<n} exp(2 pi i s nu)`` with ``nu = num / den`` (integer ``num``), given
    ``n nu mod 2`` computed exactly by the caller."""
    num = np.asarray(num, np.int64)
    nu = num / den
    exact = np.mod(num, den) == 0
    ratio = np.sin(np.pi * n_nu_mod2) / np.where(exact, 1.0, np.sin(np.pi * nu))
    return np.where(exact, float(n), np.exp(1j * np.pi * (n_nu_mod2 - nu)) * ratio)


def _count_self_conjugate_once(N: int, j_in, A, B):
    """Halve the ``A`` and ``B`` columns of the self-conjugate source bins (absolute bin 0, and
    ``N / 2`` for even ``N``): there ``z`` and ``conj(z)`` are the same DFT coefficient, which
    ``A z + B conj(z)`` would otherwise count twice."""
    j = np.asarray(j_in, np.int64)
    self_conj = (j == 0) | (2 * j == N)
    A[:, self_conj] *= 0.5
    B[:, self_conj] *= 0.5
    return A, B


def pad_maps(N: int, Np: int, k_out, j_in):
    r"""Bins ``k_out`` of the record zero-padded from ``N`` to ``Np`` samples, from the
    record's bins ``j_in``: ``z'_k = A z + B conj(z)``, both ``(len(k_out), len(j_in))``,
    ``A_kj = (1/N) sum_{s<N} e^{2 pi i s (j/N - k/Np)}`` and ``B`` the same at ``-j``, with
    the columns of the self-conjugate bins ``j = 0`` and ``j = N/2`` (even ``N``) halved, so
    ``z'`` is the exact rfft of the padded real record in ``z``'s units."""
    N, Np = require_int("N", N), require_int("Np", Np)
    k = index_array("k_out", k_out)[:, None]
    j_in = index_array("j_in", j_in)
    out = []
    for j in (j_in[None, :], -j_in[None, :]):
        num = j * Np - k * N                                       # nu = num / (N Np)
        # N nu = j - k N / Np, mod 2
        n_nu = np.mod(np.mod(j, 2) * Np - np.mod(k * N, 2 * Np), 2 * Np) / Np
        out.append(_geometric(num, N * Np, N, n_nu) / N)
    return _count_self_conjugate_once(N, j_in, out[0], out[1])


def crop_maps(N: int, start: int, Nc: int, k_out, j_in):
    r"""Bins ``k_out`` of the record cropped to samples ``[start, start + Nc)``, from the
    record's bins ``j_in``: ``z^c_k = A z + B conj(z)``,
    ``A_kj = (1/N) e^{2 pi i j start / N} sum_{s<Nc} e^{2 pi i s (j/N - k/Nc)}``, with the
    columns of the self-conjugate bins ``j = 0`` and ``j = N/2`` (even ``N``) halved, so
    ``z^c`` is the exact rfft of the cropped real record in ``z``'s units."""
    N, start, Nc = require_int("N", N), require_int("start", start), require_int("Nc", Nc)
    k = index_array("k_out", k_out)[:, None]
    j_in = index_array("j_in", j_in)
    out = []
    for j in (j_in[None, :], -j_in[None, :]):
        num = j * Nc - k * N                                       # nu = num / (N Nc)
        # Nc nu = j Nc / N - k, mod 2
        n_nu = np.mod(np.mod(j * Nc, 2 * N) - np.mod(k, 2) * N, 2 * N) / N
        shift = np.exp(2j * np.pi * np.mod(j * start, N) / N)
        out.append(shift * _geometric(num, N * Nc, Nc, n_nu) / N)
    return _count_self_conjugate_once(N, j_in, out[0], out[1])


class _WidelyLinear(LinearMap):
    """``z -> A z + B conj z`` (kind ``"widely"``): there is no complex matrix ``M`` with
    ``M z`` equal to it, so :meth:`rows` (and :meth:`LinearMap.dense`) refuse instead of
    returning ``conj(A^H e + B^T e)``; read ``.A`` and ``.B``.  ``push`` raises in
    :class:`LinearMap`."""

    kind = "widely"

    def rows(self, idx):
        raise TypeError(f"{type(self).__name__} is widely linear (A z + B conj z): it has no "
                        "complex row matrix; use its .A and .B")


@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "built_for"))
@dataclass(frozen=True, eq=False)
class FourierZeroPad(_WidelyLinear):
    """Bins ``dst`` of the record zero-padded from ``src.N`` to ``dst.N`` samples, from the
    bins ``src``: ``z' = A z + B conj z`` (kind ``"widely"``, ``A, B`` from :func:`pad_maps`),
    the rfft of the padded record in ``src``'s units (``dst`` must have the same
    ``(units, sigma0)``)."""

    src: FourierDomain
    dst: FourierDomain
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        require_instance("src", self.src, FourierDomain, "a FourierDomain")
        require_instance("dst", self.dst, FourierDomain, "a FourierDomain")
        _built_for(self, (self.src, self.dst), given=self.A is not _BUILD)
        if self.A is _BUILD:
            s, d = self.src, self.dst
            _require_same_units("FourierZeroPad", s, d)
            if d.N <= s.N or d.dt != s.dt:
                raise ValueError("FourierZeroPad needs a longer record on the same dt")
            A, B = pad_maps(s.N, d.N, d.k0 + np.arange(d.n), s.k0 + np.arange(s.n))
            object.__setattr__(self, "A", A)
            object.__setattr__(self, "B", B)

    def apply(self, z, be=None):
        be = self._be(z, be)
        z = be.asarray(z)
        return (z @ self.A.T) + (be.xp.conj(z) @ self.B.T)

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        return (y @ be.xp.conj(self.A)) + (be.xp.conj(y) @ self.B)


@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "start", "built_for"))
@dataclass(frozen=True, eq=False)
class FourierCrop(_WidelyLinear):
    """Bins ``dst`` of the record cropped to samples ``[start, start + dst.N)``, from the
    bins ``src``: ``z^c = A z + B conj z`` (:func:`crop_maps`), the rfft of the cropped
    record in ``src``'s units (``dst`` must have the same ``(units, sigma0)``).  ``apply`` is
    the ported toy model's template call site ``(h @ A.T) + (conj(h) @ B.T)``;
    :meth:`apply_data` its data call site."""

    src: FourierDomain
    dst: FourierDomain
    start: int = 0
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        require_instance("src", self.src, FourierDomain, "a FourierDomain")
        require_instance("dst", self.dst, FourierDomain, "a FourierDomain")
        object.__setattr__(self, "start", require_int("start", self.start))
        _built_for(self, (self.src, self.dst, self.start), given=self.A is not _BUILD)
        if self.A is _BUILD:
            s, d = self.src, self.dst
            _require_same_units("FourierCrop", s, d)
            if not (0 <= self.start and self.start + d.N <= s.N) or d.dt != s.dt:
                raise ValueError("FourierCrop needs [start, start + dst.N) inside the record")
            A, B = crop_maps(s.N, self.start, d.N, d.k0 + np.arange(d.n), s.k0 + np.arange(s.n))
            object.__setattr__(self, "A", A)
            object.__setattr__(self, "B", B)

    def apply(self, z, be=None):
        be = self._be(z, be)
        z = be.asarray(z)
        return (z @ self.A.T) + (be.xp.conj(z) @ self.B.T)

    def apply_data(self, z, be=None):
        """The ported toy model's DATA call site ``(A @ z.T + B @ conj(z).T).T``: the
        same map as :meth:`apply` in the other matmul orientation (<= 2e-15 apart)."""
        be = self._be(z, be)
        z = be.asarray(z)
        return (self.A @ z.T + self.B @ be.xp.conj(z).T).T

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        return (y @ be.xp.conj(self.A)) + (be.xp.conj(y) @ self.B)


# ---------------------------------------------------------------------------
# The cell geometry of a set of interior bands
# ---------------------------------------------------------------------------


def _band_phase(bands: np.ndarray, Nt: int, phi: np.ndarray) -> np.ndarray:
    """``(nb, Nt)`` carrier sign x quadrature x ``(-1)^n Nt / |phi|``."""
    n = np.arange(Nt)
    out = np.empty((bands.size, Nt), complex)
    for b, m in enumerate(bands):
        quad = np.where((int(m) + n) % 2 == 1, -1j, 1.0)
        out[b] = carrier_phase(int(m), Nt) * quad * (-1.0) ** n * Nt / np.linalg.norm(phi)
    return out


def _analyse(phase: np.ndarray, phi: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``W_b X_b`` for a stack of bands, ``X`` ``(nb, Nt, k)``."""
    Y = np.fft.ifft(np.asarray(phi)[None, :, None] * X, axis=1)
    return np.asarray(phase)[:, :, None] * Y


def _geometry(dom: WDMDomain) -> dict:
    """Bands, window, per-cell positions of a cell set of interior bands."""
    cells = dom.cells
    g = cells.grid
    Nt = g.Nt
    m = np.asarray(cells.m, dtype=np.int64)
    n = np.asarray(cells.n, dtype=np.int64)
    bands = np.unique(m)
    if bands.min() < 1 or bands.max() > g.Nf - 1:
        raise ValueError(f"FourierToWDM needs interior bands 1..{g.Nf - 1}; got {bands.tolist()}")
    h = Nt // 2
    g0 = h * int(bands[0]) - h                                         # absolute bin of window bin 0
    G = h * int(bands[-1] - bands[0]) + Nt
    idx = (h * (bands - bands[0]))[:, None] + np.arange(Nt)[None, :]   # (nb, Nt) window index
    b_of = np.searchsorted(bands, m)
    sel = b_of * Nt + n                                                # row of the full band grid
    return {"bands": bands, "g0": g0, "G": G, "idx": idx, "sel": sel, "Nt": Nt}


def _dense_atoms(geo: dict, Nf: int, dt: float, *, window: Window) -> np.ndarray:
    """``(nb Nt, G)``: row ``b Nt + n`` is the unit-norm atom of cell ``(bands[b], n)`` on the
    window bins, in the Wilson basis of ``window``."""
    bands, Nt, G, idx = geo["bands"], geo["Nt"], geo["G"], geo["idx"]
    atoms = band_atoms(bands, Nf, Nt, dt, window=window)
    U = np.zeros((bands.size * Nt, G), complex)
    for b in range(bands.size):
        U[b * Nt:(b + 1) * Nt, idx[b]] = atoms[b]
    return U


def _kappa(src: FourierDomain, g: wilson.Grid) -> float:
    """``src.rfft_scale * sqrt(2 dt / N)`` (``N``, ``dt`` of the cells' grid): the unit-norm-atom
    analysis of bins in ``src``'s units, times this, is the Wilson coefficients."""
    return src.rfft_scale * float(np.sqrt(2.0 * g.dt / g.N))


@pytree_dataclass(data=("C",), meta=("src", "dst", "impl", "lo", "width", "restrict",
                                     "built_for"))
@dataclass(frozen=True, eq=False)
class FourierToWDM(LinearMap):
    """Wilson coefficients of interior-band cells from band-limited bins: ``cells = Re(C z)``
    (kind ``"cells"``).

    ``src`` is the ``FourierDomain`` the data live on (a wider store of bins, or the band
    window itself); the cells read the window bins ``[g0, g0 + G)``, ``g0 = (m_0 - 1) Nt/2``,
    which ``src`` must cover -- or, with ``restrict=True``, ``src`` may be a sub-window of it
    and the atoms are truncated to ``src``.
    ``C = kappa U``: ``U`` holds the unit-norm atoms in the Wilson basis of ``dst.window``,
    and ``kappa = src.rfft_scale * sqrt(2 dt / N)`` (``N``, ``dt`` of the
    cells' grid) turns bins in ``src.units`` into Wilson coefficients.  ``impl="dense"``
    evaluates ``Re(z[..., lo:lo+width] @ C.T)`` (the golden default);
    ``impl="fft"`` evaluates ``src.rfft_scale * wilson.transform.forward_bands`` (opt-in) on
    the bins converted to complex128 first.  Both impls return float64 cells whatever the
    bins' precision, and they agree to <= 1e-12 relative.  For a record ``x`` and its bins
    ``z`` in ``src.units``, the cells equal ``TimeToWDM.apply(x)`` on the same cells.
    """

    src: FourierDomain
    dst: WDMDomain
    impl: str = "dense"
    restrict: bool = field(default=False, kw_only=True)
    lo: int = field(default=_BUILD, kw_only=True)
    width: int = field(default=_BUILD, kw_only=True)
    C: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        require_instance("src", self.src, FourierDomain, "a FourierDomain")
        require_instance("dst", self.dst, WDMDomain, "a WDMDomain")
        require_str("impl", self.impl)
        object.__setattr__(self, "restrict", require_bool("restrict", self.restrict))
        _built_for(self, (self.src, self.dst, self.impl, self.restrict),
                   given=self.C is not _BUILD)
        if self.C is not _BUILD:
            return
        if self.impl not in ("dense", "fft"):
            raise ValueError(f"impl must be 'dense' or 'fft'; got {self.impl!r}")
        g = self.dst.cells.grid
        if (self.src.N, self.src.dt) != (g.N, g.dt):
            raise ValueError(f"bins of an N={self.src.N} record cannot feed cells of an N={g.N} grid")
        geo = _geometry(self.dst)
        g0, G = geo["g0"], geo["G"]
        k0, n = self.src.k0, self.src.n
        U = _dense_atoms(geo, g.Nf, g.dt, window=self.dst.window)[geo["sel"]]  # (P, G)
        kappa = _kappa(self.src, g)
        if k0 <= g0 and g0 + G <= k0 + n:
            lo, width, C = g0 - k0, G, kappa * U
        elif self.restrict and g0 <= k0 and k0 + n <= g0 + G:
            if self.impl != "dense":
                raise ValueError("restrict=True truncates the atoms: impl='dense' only")
            lo, width, C = 0, n, kappa * U[:, k0 - g0:k0 - g0 + n]
        else:
            raise ValueError(f"the bins [{k0}, {k0 + n}) do not cover the band windows "
                             f"[{g0}, {g0 + G}) (pass restrict=True for a sub-window)")
        object.__setattr__(self, "lo", int(lo))
        object.__setattr__(self, "width", int(width))
        object.__setattr__(self, "C", C)

    kind = "cells"

    @property
    def bands(self) -> np.ndarray:
        """The interior bands of the cells, ascending."""
        return _geometry(self.dst)["bands"]

    @property
    def window(self) -> FourierDomain:
        """The band windows' bins ``[g0, g0 + G)`` of the cells' grid record
        (480 / 876 / 1800 for the galactic-binary toy model).  The record is ``dst``'s grid,
        not ``src``'s: the two agree for a built map (the constructor checks it), while a map
        returned by :meth:`after` reads the bins of the outer record.  It is
        ``dataclasses.replace`` of ``src``, so ``units`` and ``sigma0`` are ``src``'s.  (These
        are band-window *bins*.  The Wilson window of the cells is ``dst.window``.)"""
        geo = _geometry(self.dst)
        g = self.dst.cells.grid
        return replace(self.src, N=g.N, dt=g.dt, k0=geo["g0"], n=geo["G"])

    def apply(self, z, be=None):
        be = self._be(z, be)
        _last_axis(z, self.src.n, "FourierToWDM.apply")
        z = be.asarray(z)
        if self.impl == "dense":
            return be.xp.real(z[..., self.lo:self.lo + self.width] @ self.C.T)
        g = self.dst.cells.grid
        geo = _geometry(self.dst)
        z = be.xp.asarray(z, dtype=np.complex128)        # double, whatever the bins' precision
        w = wilson.transform.forward_bands(z, self.src.k0, geo["bands"], g.Nf, g.Nt, g.dt, be=be,
                                           window=self.dst.window)
        return self.src.rfft_scale * w.reshape(*w.shape[:-2], -1)[..., geo["sel"]]

    def adjoint(self, w, be=None):
        """``C^H w`` on ``src`` (complex), the real-inner-product adjoint."""
        be = self._be(w, be)
        _last_axis(w, self.shape[0], "FourierToWDM.adjoint")
        w = be.asarray(w)
        part = w @ be.xp.conj(self.C)
        out = be.xp.zeros((*w.shape[:-1], self.src.n), dtype=complex)
        return be.set(out, (..., slice(self.lo, self.lo + self.width)), part, copy=False)

    def rows(self, idx):
        idx = np.atleast_1d(index_array("idx", idx))
        C = np.asarray(self.C)
        out = np.zeros((idx.size, self.src.n), complex)
        out[:, self.lo:self.lo + self.width] = C[idx]
        return out

    def push_rows(self, Mp, idx, be=None):
        """``1/2 Re((C[idx] @ M') @ conj(C).T)``;
        ``M'`` is ``(src.n, src.n)``, the bins' covariance in ``src.units``; the ``width`` bins
        ``C`` reads are sliced out of it.  ``C`` carries ``kappa``, so the push carries
        ``kappa^2``."""
        be = be or self._be(Mp, None)
        if Mp.shape != (self.src.n, self.src.n):
            raise ValueError(f"M' must be ({self.src.n}, {self.src.n}) on src's bins; got {Mp.shape}")
        Mp = Mp[self.lo:self.lo + self.width, self.lo:self.lo + self.width]
        C = self.C
        return 0.5 * be.xp.real((C[idx] @ Mp) @ be.xp.conj(C).T)

    def after(self, inner: Any) -> "FourierToWDM":
        """The dense cell matrix of ``self . inner`` for a widely linear ``inner``
        (:class:`FourierZeroPad`, :class:`FourierCrop`):
        ``C = kappa * (analyse(A[idx]) + conj(analyse(B[idx])))``, rows of ``dst``, with
        ``self``'s ``kappa``: ``inner`` keeps the units, so the composite's cells are the
        Wilson coefficients of the padded or cropped record.

        The atoms are rebuilt from ``dst``'s geometry, so ``self.src`` must be exactly the
        band window :attr:`window` of the cells' grid record: a ``restrict=True`` map
        (truncated atoms) or a composite returned by ``after`` (whose ``src`` lies on the
        outer record, of another length ``N``) is refused rather than misread.  The one
        composite that passes, a crop to the whole record (``start = 0``, same ``N``), is
        the map itself (its ``C`` equals the atoms to roundoff)."""
        require_instance("inner", inner, (FourierZeroPad, FourierCrop),
                         "a FourierZeroPad or a FourierCrop")
        if self.impl != "dense" or inner.dst != self.src or self.src != self.window:
            raise ValueError("after() needs a dense map whose src is exactly inner.dst = "
                             "the band window")
        g = self.dst.cells.grid
        geo = _geometry(self.dst)
        phi = sample_window(self.dst.window, g.Nf, g.Nt, g.dt)
        phase = _band_phase(geo["bands"], g.Nt, phi)
        nb, Nt = geo["bands"].size, g.Nt
        A, B = np.asarray(inner.A), np.asarray(inner.B)
        C = _analyse(phase, phi, A[geo["idx"]]).reshape(nb * Nt, -1)
        C = C + np.conj(_analyse(phase, phi, B[geo["idx"]]).reshape(nb * Nt, -1))
        C = _kappa(self.src, g) * C
        return FourierToWDM(inner.src, self.dst, "dense", lo=0, width=inner.src.n, C=C[geo["sel"]],
                            built_for=(inner.src, self.dst, "dense", False))
