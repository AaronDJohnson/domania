r"""The text's padded covariance ``(M Lambda M)_KK`` on the kept cells, exactly (WDM II Eq. 94).

``Lambda`` lives on every cell ``R`` of the extended grid (``base``: diagonal, block tri-diagonal
or dense) and ``M`` is the data mask in cell space, given either by a
:class:`~domainia.literal.CenteredPad` (``M = I - B B^T`` from the time-domain mask: the sinusoid
pads, the only production caller -- ``wdm_models.sinusoid.literal``, QR route) or by explicit rows
(``rows``: the kept cells' rows of ``M`` over ``R``; a tested generalisation that no production
code reaches -- the GB literal pads use ``domainia.covariance.ZeroPadded``'s dense
``"cell_congruence"`` on their Fourier-domain ``PadMask`` instead).  Split the kept cells ``K``
into the edge set ``E`` (rows of ``M`` that are not ``e_p``) and ``I = K \ E``, and let ``M~`` be
``M`` with the rows ``I`` replaced by ``e_p`` (``M~ = M`` when ``E`` holds every row that is not
``e_p``).  Then

.. math::  (\tilde M\Lambda\tilde M^{\mathsf T})_{KK} = \begin{pmatrix} R_{EE} & R_{EI} \\
           R_{EI}^{\mathsf T} & \Lambda_{II}\end{pmatrix},\qquad
           R_{EK} = (M_E\Lambda)_{.K},\quad R_{EE} = M_E \Lambda M_E^{\mathsf T},

with ``M_E Lambda = Lambda_{E.} - B_E (Lambda B)^T`` (pad) or ``A_E Lambda`` (rows) -- exact
algebra, ``M`` truncated nowhere but outside ``E`` (the GB release's 1e-8 rows are such an ``E``).
The default ``E`` is ``CenteredPad.edge_set()`` (round-off): every kept cell on the paper's grids,
so the literal matrix is the full ``(M Lambda M)_KK``.

Factorisations (``factorization``):

* ``"qr"`` (default): ``(M~ Lambda M~^T)_KK = G G^T`` with ``G = (M~ L)_{K.}``,
  ``Lambda = L L^T`` the base's own Cholesky (diagonal ``sqrt(v)``, block bidiagonal, dense), and
  ``G^T = Q R``: the factor is ``R^T`` (signs fixed).  Exact algebra on the literal entries and
  well conditioned: ``cond(G) = cond(M Lambda M)^{1/2}``.  With ``p = 16`` the mask ``M_KK`` has
  ``Nf/2`` eigenvalues near ``5e-9`` (64 x 256: 32, the smallest ``4.9e-9``), so ``M Lambda M``
  reaches ``cond ~ 1e12``; on the 16 x 32 test grid the assembled Cholesky then loses up to
  ``3.7e-2`` nat against the exact time-sample identity while QR stays within ``3e-9``;
* ``"cholesky"``: the assembled matrix, factorised by the Schur complement on ``E`` over the
  interior's own factor (``domainia.covariance.edges.schur_factor``) or, for ``E = K``, a dense
  Cholesky -- the arrow; cheaper, accurate only while ``M Lambda M`` is well conditioned.

``assembly="dense"`` is the small-grid oracle: ``M~`` from the dense ``W'`` (pad) or the given
rows, ``M~_K Lambda M~_K^T`` by two products, a dense Cholesky.  A factor failure gives
``ok = False`` (``-inf``).  What ``-inf`` means differs between the routes: the QR route
factorises ``Lambda`` itself first (``base_ok``), so it gives ``-inf`` whenever ``Lambda`` on the
extended cells is not positive definite -- possible for a non-diagonal base (the sinusoid
``tri_pad``: Eq. 45 need not be positive definite) even where ``(M Lambda M)_KK`` is; the
assembled routes (``factorization="cholesky"``; ``assembly="dense"``, i.e. ``dense=True`` in
:func:`masked_covariance`) give ``-inf`` only when ``(M Lambda M)_KK`` is not positive definite.
The two ``-inf`` masks can therefore differ for a non-diagonal base (a diagonal base with positive
variances never does); the QR default is the chains' (their time-marginal pad also factorised
``Lambda_ext``).  The same data's time-sample density (``C_dd = [W'^T Lambda W']_dd / dt^2``) is
this cell density plus ``ln|det A|`` (``CenteredPad.log_abs_det_lift``;
:func:`~domainia.literal.time_marginal` computes it exactly in ``O(p Nf)^3``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from domainia.wilson.backend import Backend

from ..covariance.base import CholFactor, Covariance, DiagFactor
from ..covariance.blocktri import blocktri_cov_factor, dummy_slots, to_layout
from ..covariance.dense import dense_factor
from ..covariance.diagonal import DiagonalCovariance
from ..covariance.edges import assemble_edge_rows, schur_factor
from ..linalg import BandLayout, blocktri_factor
from .pad import CenteredPad

__all__ = ["MaskedCovariance", "base_kind", "dense_lower", "masked_covariance", "raw_padded_blocks"]

Assembly = Literal["arrow", "dense"]
Factorization = Literal["qr", "cholesky"]


def base_kind(base: Covariance) -> str:
    """``"diagonal"`` (a ``DiagonalCovariance``), ``"blocks"`` (``band_blocks`` + a block
    layout) or ``"dense"`` (anything with ``_matrix``)."""
    if isinstance(base, DiagonalCovariance):
        return "diagonal"
    if hasattr(base, "band_blocks") and getattr(base, "layout", None) is not None:
        return "blocks"
    return "dense"


def raw_padded_blocks(base: Covariance, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
    """The base's band blocks as zero-padded ``(nb, B, B)`` / ``(nb - 1, B, B)`` arrays (dummy
    slots zero, diagonal blocks as the builder returned them)."""
    xp = be.xp
    d, u = base.band_blocks(C, nu, be)
    if not isinstance(d, (list, tuple)):
        return d, u
    sizes = tuple(int(s) for s in base.sizes)
    B = max(sizes)
    D = xp.stack([xp.pad(x, ((0, B - s), (0, B - s))) for x, s in zip(d, sizes, strict=True)])
    U = (xp.stack([xp.pad(x, ((0, B - sizes[m]), (0, B - sizes[m + 1]))) for m, x in enumerate(u)])
         if len(u) else xp.zeros((0, B, B)))
    return D, U


def dense_lower(Ls: Any, Xs: Any, layout: BandLayout, be: Backend) -> Any:
    """The dense lower factor ``L`` (``Lambda = L L^T``) of a padded block Cholesky
    (``domainia.linalg.blocktri_factor``: ``L_bb = Ls[b]``, ``L_{b+1, b} = Xs[b]^T``) in the
    natural (band-major) cell order."""
    xp = be.xp
    nb, B = Ls.shape[0], Ls.shape[1]
    L = xp.zeros((nb * B, nb * B))
    for b in range(nb):
        L = be.set(L, (slice(b * B, (b + 1) * B), slice(b * B, (b + 1) * B)), Ls[b], copy=False)
        if b:
            L = be.set(L, (slice(b * B, (b + 1) * B), slice((b - 1) * B, b * B)), Xs[b - 1].T, copy=False)
    back = be.asarray(layout.back)
    return L[back][:, back]


class MaskedCovariance(Covariance):
    """``(M Lambda M)_KK`` (module docstring).

    Parameters
    ----------
    base
        ``Lambda`` on the extended cells ``R`` (``pad.cells``, or the cells the ``rows`` are on).
    pad
        A :class:`CenteredPad` (``M = I - B B^T``); or ``None`` with ``kpos`` and ``rows``.
    kpos, rows
        Without ``pad``: the kept cells' positions among ``R`` and the rows of ``M`` for the kept
        positions ``edge`` (``(|E|, |R|)``; every other kept row is ``e_p``).
    edge
        Kept positions whose ``M`` rows are applied (pad default: ``pad.edge_set()``; rows: required).
    assembly
        ``"arrow"`` or ``"dense"`` (the oracle; pad grids ``N' <= 4096``).
    factorization
        ``"qr"`` or ``"cholesky"`` (``assembly="arrow"``; the dense oracle is a Cholesky).
    chol_impl
        The dense / Schur Cholesky (``"numpy"`` or ``"scipy"``).
    """

    def __init__(self, base: Covariance, pad: CenteredPad | None = None, *, kpos: Any = None, rows: Any = None,
                 edge: Any = None, assembly: Assembly = "arrow", factorization: Factorization = "qr",
                 chol_impl: Literal["numpy", "scipy"] = "numpy") -> None:
        if assembly not in ("arrow", "dense"):
            raise ValueError(f"assembly must be 'arrow' or 'dense'; got {assembly!r}")
        if factorization not in ("qr", "cholesky"):
            raise ValueError(f"factorization must be 'qr' or 'cholesky'; got {factorization!r}")
        if chol_impl not in ("numpy", "scipy"):
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        if (pad is None) == (rows is None):
            raise ValueError("give either pad (a CenteredPad) or kpos + rows (explicit mask rows)")
        if pad is not None:
            n_R, kp = pad.cells.size, pad.kpos
            e = pad.edge_set() if edge is None else np.unique(np.asarray(edge, dtype=np.int64))
        else:
            if kpos is None or edge is None:
                raise ValueError("explicit mask rows need kpos and edge")
            kp = np.asarray(kpos, dtype=np.int64)
            n_R = int(np.asarray(rows).shape[1])
            e = np.asarray(edge, dtype=np.int64)
            if np.any(np.diff(e) <= 0) or np.asarray(rows).shape[0] != e.size:
                raise ValueError("edge must be strictly increasing with one row of M each")
        if base.n != n_R:
            raise ValueError(f"the base covers {base.n} cells; the extended grid has {n_R}")
        self.n = int(kp.size)
        if e.size and (e.min() < 0 or e.max() >= self.n):
            raise ValueError(f"edge positions must lie in [0, {self.n}) (the kept cells)")
        self.base, self.pad, self.kpos, self.rows = base, pad, kp, None if rows is None else np.asarray(rows, float)
        self.assembly, self.factorization, self.chol_impl = assembly, factorization, chol_impl
        self.e, self.i = e, np.setdiff1d(np.arange(self.n), e)
        self.kind = base_kind(base)
        self.cols: list[np.ndarray] = []
        if self.kind == "blocks":                     # interior slots per band (static: np.ix_ under jit)
            sizes = tuple(int(s) for s in base.sizes)
            off = np.concatenate([[0], np.cumsum(sizes)])
            band = np.searchsorted(off, np.arange(int(off[-1])), side="right") - 1
            slot = np.arange(int(off[-1])) - off[band]
            kI = kp[self.i]
            self.cols = [slot[kI][band[kI] == b] for b in range(len(sizes))]
        self.jittable = base.jittable

    def children(self) -> dict[str, Covariance]:
        return {"base": self.base}

    # -- constants ----------------------------------------------------------------------
    def host_constants(self) -> dict[str, Any]:
        kpos = self.kpos
        C: dict[str, Any] = {"kpos": kpos, "e": self.e, "i": self.i, "interior": {}, "kE": kpos[self.e],
                             "kI": kpos[self.i]}
        if self.pad is not None:
            C["B"] = np.asarray(self.pad.padding_impulses())
            if self.assembly == "dense":
                C["M"] = self.pad.dense_mask()
        else:
            C["AE"] = self.rows
        if self.kind == "blocks":
            C["layout_R"] = BandLayout.from_sizes(tuple(int(s) for s in self.base.sizes))
            sizes_I = tuple(int(c.size) for c in self.cols)
            if self.i.size:
                C["interior"] = {"layout": BandLayout.from_sizes(sizes_I), "dummy": dummy_slots(sizes_I)}
        return C

    # -- Lambda on the extended cells -----------------------------------------------------
    def _blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        """Padded, symmetrised blocks with identity dummies (``to_layout``)."""
        d, u = raw_padded_blocks(self.base, C["base"], nu, be)
        return to_layout(d, u, tuple(self.base.sizes), "padded", be, dummy=be.asarray(dummy_slots(self.base.sizes)))

    def _apply(self, C: Any, nu: Any, X: Any, be: Backend, blocks: Any) -> Any:
        """``Lambda X`` for ``X`` ``(|R|, k)``."""
        xp = be.xp
        if self.kind == "diagonal":
            return self.base.variances(C["base"], nu, be)[:, None] * X
        if self.kind == "dense":
            return self.base._matrix(C["base"], nu, be) @ X
        d, u = blocks
        lay = C["layout_R"]
        Xp = lay.pack(X, be=be)                                             # (nb, B, k), dummies 0
        Yp = xp.einsum("bij,bjk->bik", d, Xp)
        if u.shape[0]:
            zero = xp.zeros_like(Xp[:1])
            Yp = Yp + xp.concatenate([xp.einsum("bij,bjk->bik", u, Xp[1:]), zero], axis=0)
            Yp = Yp + xp.concatenate([zero, xp.einsum("bji,bjk->bik", u, Xp[:-1])], axis=0)
        return lay.unpack(Yp, be=be)

    def _rows_lambda(self, C: Any, nu: Any, be: Backend, blocks: Any) -> Any:
        """``M_E Lambda`` (``(|E|, |R|)``)."""
        xp = be.xp
        if self.pad is None:
            return self._apply(C, nu, C["AE"].T, be, blocks).T
        B, kE = C["B"], C["kE"]
        unit = be.set(xp.zeros((B.shape[0], kE.size)), (kE, np.arange(kE.size)), 1.0, copy=False)
        Y = self._apply(C, nu, B, be, blocks)                               # Lambda B
        return self._apply(C, nu, unit, be, blocks).T - B[kE] @ Y.T          # Lambda_{E.} - B_E Y^T

    def edge_rows(self, C: Any, nu: Any, be: Backend, blocks: Any = None) -> Any:
        """``R_EK = (M~ Lambda M~^T)_{E, K}`` (``(|E|, |K|)``, columns in kept order)."""
        WE = self._rows_lambda(C, nu, be, blocks)
        if self.pad is None:
            WEE = WE @ C["AE"].T
        else:
            B, kE = C["B"], C["kE"]
            WEE = WE[:, kE] - (WE @ B) @ B[kE].T
        return be.set(WE[:, C["kpos"]], (slice(None), C["e"]), WEE, copy=False)

    def _interior(self, C: Any, nu: Any, be: Backend, blocks: Any) -> Any:
        """``Lambda_II``'s own factor (diagonal, block tri-diagonal or dense Cholesky)."""
        kI = C["kI"]
        if self.kind == "diagonal":
            return DiagFactor.build(self.base.variances(C["base"], nu, be)[kI], be)
        if self.kind == "dense":
            L = self.base._matrix(C["base"], nu, be)
            return CholFactor.build(L[kI][:, kI], impl=self.chol_impl, be=be)
        d, u = blocks
        cols = self.cols
        diag = [d[b][np.ix_(c, c)] for b, c in enumerate(cols)]
        up = [u[b][np.ix_(cols[b], cols[b + 1])] for b in range(len(cols) - 1)]
        Ci = C["interior"]
        return blocktri_cov_factor(diag, up, tuple(int(c.size) for c in cols), "padded", Ci["layout"], be,
                                   Ci["dummy"])

    def _mask_rows_dense(self, C: Any, be: Backend) -> Any:
        """``M~_{K.}`` (``(|K|, |R|)``): ``e_p`` rows on ``I``, ``M`` rows on ``E``."""
        xp = be.xp
        kpos = C["kpos"]
        Mt = be.set(xp.zeros((kpos.size, self.base.n)), (np.arange(kpos.size), kpos), 1.0, copy=False)
        if self.pad is None:
            ME = C["AE"]
        else:
            B, kE = C["B"], C["kE"]
            ME = be.set(-(B[kE] @ B.T), (np.arange(kE.size), kE), 1.0 - xp.sum(B[kE] * B[kE], axis=1), copy=False)
        return be.set(Mt, C["e"], ME, copy=False)

    def _mask_times(self, C: Any, X: Any, be: Backend) -> Any:
        """``(M~ X)_{K.}`` for ``X`` ``(|R|, k)``."""
        kpos, e = C["kpos"], C["e"]
        if self.pad is None:
            ME_X = C["AE"] @ X
        else:
            B, kE = C["B"], C["kE"]
            ME_X = X[kE] - B[kE] @ (B.T @ X)
        return be.set(X[kpos], e, ME_X, copy=False)

    def _qr_factor(self, C: Any, nu: Any, be: Backend, blocks: Any) -> CholFactor:
        """``G = (M~ L)_{K.}``, ``G^T = Q R``; the factor ``R^T`` with a positive diagonal."""
        xp = be.xp
        if self.kind == "diagonal":
            v = self.base.variances(C["base"], nu, be)
            base_ok = xp.all(v > 0) & xp.all(xp.isfinite(v))
            sv = xp.sqrt(xp.where(v > 0, v, xp.nan))
            G = self._mask_rows_dense(C, be) * sv[None, :]
        else:
            if self.kind == "blocks":
                d, u = blocks
                bt = blocktri_factor(d, u, layout="padded", band_layout=C["layout_R"], be=be)
                L, base_ok = dense_lower(bt.Ls, bt.Xs, C["layout_R"], be), bt.ok
            else:
                S = self.base._matrix(C["base"], nu, be)
                F = CholFactor.build(0.5 * (S + S.T), impl="numpy", be=be)
                L, base_ok = F.L, F.ok
            G = self._mask_times(C, L, be)
        R = xp.linalg.qr(G.T, mode="r")
        dg = xp.diagonal(R)
        sgn = xp.where(dg < 0, -1.0, 1.0)
        logdet = 2.0 * xp.sum(xp.log(xp.abs(dg)))
        ok = base_ok & xp.all(xp.isfinite(dg)) & xp.all(dg != 0)
        return CholFactor(R.T * sgn[None, :], logdet, ok, "numpy")

    # -- factor / matrix ------------------------------------------------------------------
    def _dense_oracle(self, C: Any, nu: Any, be: Backend) -> Any:
        """``M~_K Lambda M~_K^T`` with the dense ``M`` (pad: ``W' chi W'^T / dt``; rows: as given)."""
        lam = self.base._matrix(C["base"], nu, be)
        if self.pad is None:
            Mt = self._mask_rows_dense(C, be)
        else:
            Mt = be.set(C["M"][C["kpos"]], C["i"], 0.0, copy=False)
            Mt = be.set(Mt, (C["i"], C["kI"]), 1.0, copy=False)
        return Mt @ lam @ Mt.T

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.assembly == "dense":
            return dense_factor(self._dense_oracle(C, nu, be), chol_impl=self.chol_impl, symmetrize=True, be=be)
        blocks = self._blocks(C, nu, be) if self.kind == "blocks" else None
        if self.factorization == "qr":
            return self._qr_factor(C, nu, be, blocks)
        R = self.edge_rows(C, nu, be, blocks)
        if not self.i.size:
            return dense_factor(R, chol_impl=self.chol_impl, symmetrize=True, be=be)
        T = self._interior(C, nu, be, blocks)
        return schur_factor(T, R[:, C["e"]], R[:, C["i"]], chol_impl=self.chol_impl, be=be, form="solve",
                            C_T=C["interior"])

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.assembly == "dense":
            return self._dense_oracle(C, nu, be)
        blocks = self._blocks(C, nu, be) if self.kind == "blocks" else None
        R = self.edge_rows(C, nu, be, blocks)
        if not self.i.size:
            return R
        lam = self.base._matrix(C["base"], nu, be)
        kI = C["kI"]
        return assemble_edge_rows(lam[kI][:, kI], R, C["e"], C["i"], be)


def masked_covariance(base: Covariance, pad: CenteredPad, *, edge: Any = None, dense: bool = False,
                      factorization: Factorization = "qr",
                      chol_impl: Literal["numpy", "scipy"] = "numpy") -> MaskedCovariance:
    """``(M Lambda M)_KK`` of ``base`` on ``pad``'s kept cells: the arrow (``dense=False``) or the
    dense oracle (``dense=True``, ``ModelOptions(dense=True)``)."""
    return MaskedCovariance(base, pad, edge=edge, assembly="dense" if dense else "arrow",
                            factorization=factorization, chol_impl=chol_impl)
