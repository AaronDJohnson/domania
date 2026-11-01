r"""WDM II Eq. (45) by the direct grid sum (spec 10.2 / 10.3: the literal tri-diagonal covariance).

.. math::  \Lambda_{pq} = \mathrm{Re}\sum_k w_k\,S(f_k, \bar t_{pq})\,V_p(k)\,V_q^*(k),
           \qquad \bar t_{pq} = t_0 + (n_p + n_q)\,\Delta T/2,

evaluated entry by entry: the spectrum at each pair's arithmetic mid-time, summed over the grid
bins -- no Gram, Bessel or Chebyshev rewrite (user ruling 10.1).  Two discretisations of the
same integral (spec 10.2):

* **record grid** (``LogLinearPSD``, ``GenericPSD``; the sinusoid): ``V_p`` is the rfft of the
  cell's atom on the cells' grid record and ``w_k = eta_k / T`` (Eq. 51 weights, ``T = N dt`` of
  that grid -- ``T'`` on a padded grid).  Block tri-diagonal in ``m`` (the atoms of bands more
  than one apart share no bin), laid out as ``wdm.evolutionary.eq45_gram_plan``'s padded bands:
  its atom spectra, weights and overlap bins are reused as constants, only the sum differs from
  the Gram builder.  ``assembly="blocktri"`` (block Cholesky, jittable) or ``"dense"``.
* **band windows** (``SeparablePSD``; the galactic binary): ``V_p`` is the unit-norm atom on its
  band's ``Nt``-bin window, ``w = 1/2`` and ``S = s_I + g(t) s_G`` of one ``channel``
  (``S / S_ref`` tabulated).  ``assembly="dense"`` on band-ordered cells of contiguous bands (rows
  summed ``row_chunk`` at a time), ``"blocktri"`` on full bands.

``t0`` is the time of column 0 (``-p/2 dT`` on the 8 + 8 padded grid of
:class:`domainia.literal.CenteredPad`): the spectrum's own ``alpha(t)`` / ``g(t)`` is evaluated
at ``t_bar``, so padding columns sit at ``t < 0`` and ``t >= T``.  Every array a factorisation
reads is a host constant passed to ``jit`` as an argument (``GenericPSD`` is NumPy-only); a
matrix that is not positive definite gives ``ok = False`` (``-inf``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from wdm.backend import Backend, get_backend
from wdm.evolutionary import eq45_gram_plan, eq45_plan
from wdm.grid import delta_T
from wdm.transform import band_atoms

from ..covariance.base import Covariance
from ..covariance.blocktri import BlockTridiagonalCovariance, blocks_to_dense, to_layout
from ..covariance.dense import DenseCovariance
from ..covariance.eq45 import _separable_bands, _separable_spectra, keep_mask, window_bins
from ..spectra import GenericPSD, LogLinearPSD, SeparablePSD

__all__ = ["Eq45Direct", "direct_block"]


def direct_block(S: Any, s_idx: Any, Va: Any, Vb: Any, w: Any, be: Backend) -> Any:
    r"""One block of the direct sum: :math:`\mathrm{Re}\sum_k S[s_{ij}, k]\,w_k\,V_a[i, k]\,
    V_b^*[j, k]` -- ``S`` ``(n_s, K)`` the spectrum on the mid-time grid, ``s_idx`` ``(I, J)`` each
    pair's mid-time index, ``Va`` ``(I, K)``, ``Vb`` ``(J, K)``, ``w`` ``(K,)`` (real arithmetic)."""
    xp = be.xp
    Sg = S[s_idx]                                                        # (I, J, K)
    ar, ai = xp.real(Va) * w[None, :], xp.imag(Va) * w[None, :]
    return (xp.einsum("ijk,ik,jk->ij", Sg, ar, xp.real(Vb))
            + xp.einsum("ijk,ik,jk->ij", Sg, ai, xp.imag(Vb)))


def _t_bar(grid: Any, t0: float) -> np.ndarray:
    """The ``2 Nt - 1`` arithmetic mid-times ``t0 + s dT/2``, ``s = n + n'``."""
    return float(t0) + np.arange(2 * grid.Nt - 1) * delta_T(grid.Nf, grid.dt) / 2


def _blocks_map(fn: Any, xs: tuple, empty_shape: tuple[int, ...], be: Backend) -> Any:
    """``be.map(fn, xs)`` over the leading axis; an empty leading axis gives zeros."""
    if xs[0].shape[0] == 0:
        return be.xp.zeros(empty_shape)
    return be.map(lambda a: fn(*a), xs)


class _RecordBlocks:
    """Record-grid direct sum in the padded band layout of ``eq45_gram_plan`` (LogLinear / Generic)."""

    def __init__(self, psd: Any, cells: Any, t0: float) -> None:
        self.psd, self.cells, self.t0 = psd, cells, float(t0)

    def host_constants(self) -> dict[str, Any]:
        g = self.cells.grid
        plan = eq45_plan(g.Nf, g.Nt, g.dt, keep_mask(self.cells), P=2)
        gp = eq45_gram_plan(plan)
        n_kept = plan.n[plan.cells]
        n_slot = np.where(gp.slot >= 0, n_kept[np.maximum(gp.slot, 0)], 0)
        # Padded support slots (gG = gW = 0) repeat the band's last support bin, never bin 0:
        # S there is a value the band already evaluates, so 0 * S stays 0 when S(0) is singular.
        kd = np.zeros((gp.nb, gp.Kb), np.int64)
        for mm, k in enumerate(gp.sup):
            kd[mm, :len(k)] = k
            kd[mm, len(k):] = k[-1] if len(k) else 0
        ku = np.take_along_axis(kd[:-1], gp.uI1.astype(np.int64), axis=1)
        fd, fu = plan.f[kd], plan.f[ku]
        live = np.array([len(c) > 0 for c in plan.band_cells])
        out = {"gG": gp.gG, "gW": gp.gW, "uI1": gp.uI1.astype(np.int64), "uI2": gp.uI2.astype(np.int64),
               "uW": gp.uW, "sd": n_slot[:, :, None] + n_slot[:, None, :],
               "su": n_slot[:-1, :, None] + n_slot[1:, None, :], "tbar": _t_bar(g, self.t0),
               "live_d": live, "live_u": live[:-1] & live[1:]}
        if isinstance(self.psd, LogLinearPSD):
            out.update(Ld=np.asarray(self.psd.L(fd), float), Lu=np.asarray(self.psd.L(fu), float))
        else:
            out.update(fd=fd, fu=fu)
        return out

    def _tables(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        xp = be.xp
        if isinstance(self.psd, LogLinearPSD):
            a = self.psd.alpha(C["tbar"], nu, be)                          # (n_s,)
            return (xp.exp(a[None, :, None] * C["Ld"][:, None, :]),
                    xp.exp(a[None, :, None] * C["Lu"][:, None, :]))
        nu_h = np.asarray(nu)
        t = np.asarray(C["tbar"])[None, :, None]
        return (be.asarray(np.asarray(self.psd(np.asarray(C["fd"])[:, None, :], t, nu_h), float)),
                be.asarray(np.asarray(self.psd(np.asarray(C["fu"])[:, None, :], t, nu_h), float)))

    def evaluate(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        xp = be.xp
        Sd, Su = self._tables(C, nu, be)
        # A band without kept cells has only dummy slots (zero blocks): its S is never read, so
        # a singular S(0) of an absent DC band cannot reach the dummy identity as 0 * inf.
        Sd = xp.where(C["live_d"][:, None, None], Sd, 0.0)
        Su = xp.where(C["live_u"][:, None, None], Su, 0.0)
        gG = C["gG"]
        nb, B = gG.shape[0], gG.shape[1]
        diag = _blocks_map(lambda S, s, V, w: direct_block(S, s, V, V, w, be),
                           (Sd, C["sd"], gG, C["gW"]), (0, B, B), be)
        V1 = xp.take_along_axis(gG[:-1], C["uI1"][:, None, :], axis=2)
        V2 = xp.take_along_axis(gG[1:], C["uI2"][:, None, :], axis=2)
        up = _blocks_map(lambda S, s, Va, Vb, w: direct_block(S, s, Va, Vb, w, be),
                         (Su, C["su"], V1, V2, C["uW"]), (max(nb - 1, 0), B, B), be)
        return diag, up


class _WindowDense:
    """Band-window direct sum, dense, any band-ordered cells of contiguous bands (SeparablePSD)."""

    def __init__(self, psd: Any, cells: Any, channel: int, t0: float, row_chunk: int) -> None:
        self.psd, self.cells, self.channel, self.t0, self.row_chunk = psd, cells, channel, float(t0), row_chunk

    def host_constants(self) -> dict[str, Any]:
        g = self.cells.grid
        m = np.asarray(self.cells.m, int)
        n = np.asarray(self.cells.n, int)
        bands = _separable_bands(m)
        sI, sG, env = _separable_spectra(self.psd, g, bands, self.channel)
        _, G = window_bins(g.Nt, bands)
        atoms = band_atoms(bands, g.Nf, g.Nt, g.dt)
        P, rc = m.size, self.row_chunk
        nch = -(-P // rc)
        U = np.zeros((nch * rc, G), complex)
        for i, (mm, nn) in enumerate(zip(m, n, strict=True)):
            U[i, (g.Nt // 2) * int(mm - bands[0]) + np.arange(g.Nt)] = atoms[int(mm - bands[0]), nn]
        sidx = np.zeros((nch * rc, P), np.int64)
        sidx[:P] = n[:, None] + n[None, :]
        return {"U": U[:P], "Uc": U.reshape(nch, rc, G), "sidx": sidx.reshape(nch, rc, P), "sI": sI, "sG": sG,
                "Bh": np.asarray(env.basis(_t_bar(g, self.t0)), float), "w": np.full(G, 0.5)}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        xp = be.xp
        S = C["sI"][None, :] + xp.exp(nu @ C["Bh"])[:, None] * C["sG"][None, :]          # (n_s, G)
        U, w = C["U"], C["w"]
        rows = be.map(lambda a: direct_block(S, a[0], a[1], U, w, be), (C["sidx"], C["Uc"]))
        P = U.shape[0]
        return rows.reshape(-1, P)[:P]


class _WindowBlocks:
    """Band-window direct sum, block tri-diagonal on full bands (SeparablePSD; a padded grid's
    store bands): per band its ``Nt``-bin window, per band pair the shared half window."""

    def __init__(self, psd: Any, cells: Any, channel: int, t0: float) -> None:
        self.psd, self.cells, self.channel, self.t0 = psd, cells, channel, float(t0)

    def host_constants(self) -> dict[str, Any]:
        g = self.cells.grid
        m = np.asarray(self.cells.m, int)
        n = np.asarray(self.cells.n, int)
        bands = _separable_bands(m)
        Nt, h = g.Nt, g.Nt // 2
        if m.size != bands.size * Nt or not np.array_equal(n, np.tile(np.arange(Nt), bands.size)):
            raise ValueError("block-tridiagonal Eq. (45) on band windows needs every column of each band")
        sI, sG, env = _separable_spectra(self.psd, g, bands, self.channel)
        atoms = band_atoms(bands, g.Nf, Nt, g.dt)
        idx = (h * (bands - bands[0]))[:, None] + np.arange(Nt)[None, :]
        nn = np.arange(Nt)
        s = nn[:, None] + nn[None, :]
        nB = bands.size
        return {"atoms": atoms, "Ua": atoms[:-1, :, h:], "Ub": atoms[1:, :, :h],
                "sId": sI[idx], "sGd": sG[idx], "sIu": sI[idx[:-1, h:]], "sGu": sG[idx[:-1, h:]],
                "sd": np.broadcast_to(s, (nB, Nt, Nt)).copy(), "su": np.broadcast_to(s, (nB - 1, Nt, Nt)).copy(),
                "wd": np.full((nB, Nt), 0.5), "wu": np.full((nB - 1, h), 0.5),
                "Bh": np.asarray(env.basis(_t_bar(g, self.t0)), float)}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        gt = be.xp.exp(nu @ C["Bh"])                                        # g(t_bar), (n_s,)
        Sd = C["sId"][:, None, :] + gt[None, :, None] * C["sGd"][:, None, :]
        Su = C["sIu"][:, None, :] + gt[None, :, None] * C["sGu"][:, None, :]
        A = C["atoms"]
        nB, Nt = A.shape[0], A.shape[1]
        diag = _blocks_map(lambda S, s, V, w: direct_block(S, s, V, V, w, be), (Sd, C["sd"], A, C["wd"]),
                           (0, Nt, Nt), be)
        up = _blocks_map(lambda S, s, Va, Vb, w: direct_block(S, s, Va, Vb, w, be),
                         (Su, C["su"], C["Ua"], C["Ub"], C["wu"]), (max(nB - 1, 0), Nt, Nt), be)
        return diag, up


class _DenseFromPadded:
    """Padded band blocks assembled into the dense matrix (``assembly="dense"``)."""

    def __init__(self, blocks: Any, sizes: tuple[int, ...]) -> None:
        self.blocks, self.sizes = blocks, sizes

    def host_constants(self) -> Any:
        return self.blocks.host_constants()

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = self.blocks.evaluate(C, nu, be)
        d, u = to_layout(diag, up, self.sizes, "ragged", be)
        return blocks_to_dense(d, u, be)


class Eq45Direct(Covariance):
    """Eq. (45) on ``cells`` by the direct grid sum (module docstring).

    Parameters
    ----------
    psd
        ``LogLinearPSD`` / ``GenericPSD`` (record grid; ``GenericPSD`` NumPy only) or
        ``SeparablePSD`` (band windows, ``channel``).
    cells
        A band-ordered ``CellSet``.
    t0
        Time of column 0 of ``cells.grid`` (``0``; ``-p/2 dT`` on the 8 + 8 padded grid).
    assembly
        ``"blocktri"`` (padded band layout; the window path needs full bands) or ``"dense"``
        (pinned dense Cholesky: ``chol_impl``, ``symmetrize`` as :class:`DenseCovariance`).
    row_chunk
        Rows summed at once by the dense window path (memory ``row_chunk x P x G``).
    """

    def __init__(self, psd: Any, cells: Any, *, t0: float = 0.0, assembly: Literal["blocktri", "dense"] = "blocktri",
                 channel: int = 0, row_chunk: int = 32, chol_impl: Literal["numpy", "scipy"] = "numpy",
                 symmetrize: bool = True) -> None:
        if assembly not in ("blocktri", "dense"):
            raise ValueError(f"assembly must be 'blocktri' or 'dense'; got {assembly!r}")
        if not isinstance(row_chunk, int) or row_chunk < 1:
            raise ValueError(f"row_chunk must be a positive int; got {row_chunk!r}")
        self.psd, self.cells, self.assembly, self.t0 = psd, cells, assembly, float(t0)
        self.n = len(cells.m)
        m = np.asarray(cells.m, int)
        if isinstance(psd, SeparablePSD):
            bands = _separable_bands(m)
            self.sizes = tuple(int(np.sum(m == b)) for b in bands)
            if assembly == "dense":
                blocks: Any = _WindowDense(psd, cells, int(channel), self.t0, row_chunk)
            else:
                blocks = _WindowBlocks(psd, cells, int(channel), self.t0)
        elif isinstance(psd, (LogLinearPSD, GenericPSD)):
            self.sizes = tuple(int(np.sum(m == b)) for b in range(cells.grid.Nf + 1))
            blocks = _RecordBlocks(psd, cells, self.t0)
            if assembly == "dense":
                blocks = _DenseFromPadded(blocks, self.sizes)
        else:
            raise TypeError(f"no direct Eq. (45) sum for {type(psd).__name__}")
        self.block_evaluator = blocks
        if assembly == "blocktri":
            self.impl: Covariance = BlockTridiagonalCovariance(blocks, self.sizes, layout="padded")
        else:
            self.impl = DenseCovariance(blocks, self.n, chol_impl=chol_impl, symmetrize=symmetrize)
        self.jittable = self.impl.jittable and not isinstance(psd, GenericPSD)

    @property
    def layout(self) -> str | None:
        """``"padded"`` for ``assembly="blocktri"``, ``None`` for ``"dense"``."""
        return getattr(self.impl, "layout", None)

    def host_constants(self) -> Any:
        return self.impl.host_constants()

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._factor(C, nu, be)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._matrix(C, nu, be)

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        """``(diag, up)`` padded band blocks from the constants ``C`` (``assembly="blocktri"``)."""
        if not isinstance(self.impl, BlockTridiagonalCovariance):
            raise ValueError("band_blocks needs assembly='blocktri'")
        return self.impl.band_blocks(C, nu, be)

    def blocks(self, nu: Any, be: Backend | None = None) -> Any:
        """The raw direct sum at ``nu``: padded ``(diag, up)`` (blocktri) or the matrix (dense)."""
        be = be or get_backend(nu)
        C = self.constants(be)
        C = C["blocks"] if isinstance(self.impl, BlockTridiagonalCovariance) else C
        return self.block_evaluator.evaluate(C, be.asarray(nu), be)
