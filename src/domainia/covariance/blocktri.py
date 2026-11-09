"""``BlockTridiagonalCovariance``: band blocks factorised by the block Cholesky.

Two layouts of the same factorisation:

* ``"ragged"`` -- Python lists of per-band blocks, a host loop; eager on both
  backends (the NumPy golden default);
* ``"padded"`` -- ``(nb, B, B)`` arrays with identity dummy slots (``BandLayout``), a
  ``be.scan`` over the bands; jit-able (the JAX golden default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import dataclasses

import numpy as np
from domainia.wilson._checks import require_int, require_str
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass

from ..linalg import BandLayout, _failed, blocktri_factor, blocktri_solve, blocktri_whiten
from .base import Covariance, QuadForm, _check_leading, as_evaluator, factor_constants, quad_form

__all__ = ["BlockTriCovFactor", "BlockTridiagonalCovariance", "blocks_to_dense", "blocktri_cov_factor",
           "dummy_slots", "to_layout"]

Layout = Literal["ragged", "padded"]


def dummy_slots(sizes: tuple[int, ...]) -> np.ndarray:
    """``(nb, B)``: 1.0 on the dummy slots of the padded layout, 0.0 on the cells."""
    B = max(sizes)
    return np.where(np.arange(B)[None, :] < np.asarray(sizes)[:, None], 0.0, 1.0)


def to_layout(diag: Any, up: Any, sizes: tuple[int, ...], layout: Layout, be: Backend,
              dummy: Any = None) -> tuple[Any, Any]:
    """``(diag, up)`` in ``layout``.

    Ragged input: lists (``diag[m]`` ``(s_m, s_m)``, ``up[m]`` ``(s_m, s_{m+1})``); padded
    input: ``(nb, B, B)`` / ``(nb - 1, B, B)`` arrays whose slot ``i < s_m`` of band ``m``
    is the band's ``i``-th cell and whose dummy slots are zero (the identity is ADDED
    there: a padded input already carrying it would get 2.0).  Padded output symmetrises
    the diagonal blocks and puts the identity on dummy slots; ``dummy`` is
    :func:`dummy_slots` (from the constants under ``jit``).  A single band (``nb = 1``) has
    no upper blocks: ``up`` is ``(0, B, B)``.
    """
    xp = be.xp
    ragged_in = isinstance(diag, (list, tuple))
    if layout == "ragged":
        if ragged_in:
            return list(diag), list(up)
        return ([diag[m][:s, :s] for m, s in enumerate(sizes)],
                [up[m][:sizes[m], :sizes[m + 1]] for m in range(len(sizes) - 1)])
    B = max(sizes)
    if ragged_in:
        diag = xp.stack([xp.pad(d, ((0, B - s), (0, B - s))) for d, s in zip(diag, sizes, strict=True)])
        up = (xp.stack([xp.pad(u, ((0, B - sizes[m]), (0, B - sizes[m + 1])))
                        for m, u in enumerate(up)]) if len(up) else xp.zeros((0, B, B)))
    dummy = be.asarray(dummy_slots(sizes)) if dummy is None else dummy
    diag = 0.5 * (diag + xp.swapaxes(diag, 1, 2))
    diag = diag + dummy[:, :, None] * xp.eye(B)[None]
    return diag, up


def blocks_to_dense(diag: list[Any], up: list[Any], be: Backend) -> Any:
    """The dense matrix of ragged band blocks (tests and dense assembly; ``O(n^2)`` memory)."""
    xp = be.xp
    nb = len(diag)
    sizes = [d.shape[0] for d in diag]

    def blk(i: int, j: int) -> Any:
        if i == j:
            return diag[i]
        if j == i + 1:
            return up[i]
        if i == j + 1:
            return up[j].T
        return xp.zeros((sizes[i], sizes[j]))

    return xp.block([[blk(i, j) for j in range(nb)] for i in range(nb)])


@pytree_dataclass(data=("bt", "logdet", "ok"), meta=("layout",))
@dataclass(frozen=True, eq=False)
class BlockTriCovFactor:
    """A :class:`domainia.linalg.BlockTriFactor` with vectors in cell order (band-major).

    The padded layout's :class:`~domainia.linalg.BandLayout` is a constant: it is not stored
    (``bt.band_layout`` is ``None``) and comes from the covariance's constants,
    ``C["layout"]``."""

    bt: Any
    logdet: Any
    ok: Any
    layout: str

    n_u = 0

    def _b_shape(self, C: Any = None) -> tuple[int, ...]:
        """``(n,)``: the sum of the ragged blocks' sizes, or the padded layout's ``n`` (static
        meta of the ``BandLayout`` in the constants ``C``; its ``Ls`` are ``(nb, B, B)``)."""
        if self.layout == "ragged":
            return (sum(L.shape[0] for L in self.bt.Ls),)
        return (factor_constants(self, C, "_b_shape")["layout"].n,)

    def _apply(self, fn: Any, b: Any, C: Any, be: Backend, what: str) -> Any:
        # domainia.linalg takes natural, band-major vectors on both layouts: the padded
        # layout packs, runs the ``*_packed`` kernels and unpacks inside (the dummy entries
        # are exactly 0).
        b = be.asarray(b)
        if self.layout == "ragged":
            # A failed ragged factor stops at its first bad block, so its blocks do not add
            # up to n: skip the check and let ``fn`` raise the failure.
            if not _failed(self):
                _check_leading(b, self._b_shape(), f"BlockTriCovFactor.{what}")
            return fn(self.bt, b, be=be)
        C = factor_constants(self, C, what)
        _check_leading(b, self._b_shape(C), f"BlockTriCovFactor.{what}")
        return fn(self.bt, b, band_layout=C["layout"], be=be)

    def _be(self, b: Any, be: Backend | None) -> Backend:
        # The factor's arrays count too (as CholFactor): a host b against a device factor
        # runs on the device, not silently on the host.
        Ls = self.bt.Ls
        return be or get_backend(b, *(Ls if isinstance(Ls, (list, tuple)) else (Ls,)))

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        return self._apply(blocktri_solve, b, C, self._be(b, be), "solve")

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        return self._apply(blocktri_whiten, b, C, self._be(b, be), "whiten")

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        return quad_form(self, b, form, self._be(b, be), C)


def blocktri_cov_factor(diag: Any, up: Any, sizes: tuple[int, ...], layout: Layout, lay: Any,
                        be: Backend, dummy: Any = None) -> BlockTriCovFactor:
    """Factor band blocks in ``layout`` (``lay``: the :class:`BandLayout`, ``dummy``:
    :func:`dummy_slots`, both for ``"padded"``).  The returned factor does not keep ``lay``."""
    d, u = to_layout(diag, up, sizes, layout, be, dummy)
    bt = blocktri_factor(d, u, layout=layout, band_layout=lay, be=be)
    if layout == "padded":
        bt = dataclasses.replace(bt, band_layout=None)
    return BlockTriCovFactor(bt, bt.logdet, bt.ok, layout)


class BlockTridiagonalCovariance(Covariance):
    """``C`` block tri-diagonal in the band index.

    ``blocks`` is an :class:`~domainia.covariance.base.Evaluator` (or callable
    ``fn(nu, be)``) returning ``(diag, up)`` ragged or padded (:func:`to_layout`);
    ``sizes`` the cells per band, a sequence of integers (an integer ndarray included).
    ``layout="ragged"`` runs eagerly on both backends.
    """

    def __init__(self, blocks: Any, sizes: tuple[int, ...], *, layout: Layout = "ragged") -> None:
        if require_str("layout", layout) not in ("ragged", "padded"):
            raise ValueError(f"layout must be 'ragged' or 'padded'; got {layout!r}")
        self.blocks = as_evaluator(blocks)
        self.sizes = tuple(require_int(f"sizes[{m}]", s) for m, s in enumerate(sizes))
        self.n = int(sum(self.sizes))
        self.layout = layout
        self.jittable = layout == "padded"

    def host_constants(self) -> Any:
        padded = self.layout == "padded"
        return {"blocks": self.blocks.host_constants(),
                "layout": BandLayout.from_sizes(self.sizes) if padded else None,
                "dummy": dummy_slots(self.sizes) if padded else None}

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        return self.blocks.evaluate(C["blocks"], nu, be)

    def _factor(self, C: Any, nu: Any, be: Backend) -> BlockTriCovFactor:
        diag, up = self.band_blocks(C, nu, be)
        return blocktri_cov_factor(diag, up, self.sizes, self.layout, C["layout"], be, C["dummy"])

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = to_layout(*self.band_blocks(C, nu, be), self.sizes, "ragged", be)
        return blocks_to_dense(diag, up, be)

    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        """Band by band from the ragged blocks, ``D_m x_m + U_m x_{m+1} + U_{m-1}^T x_{m-1}``
        (``x_m`` the band-``m`` slice of ``b``): never the dense matrix."""
        diag, up = to_layout(*self.band_blocks(C, nu, be), self.sizes, "ragged", be)
        off = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        x = [b[off[m]:off[m + 1]] for m in range(len(self.sizes))]
        out = []
        for m, D in enumerate(diag):
            y = D @ x[m]
            if m + 1 < len(diag):
                y = y + up[m] @ x[m + 1]
            if m > 0:
                y = y + up[m - 1].T @ x[m - 1]
            out.append(y)
        return be.xp.concatenate(out)
