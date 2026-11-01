r"""Evolutionary-spectrum covariance of WDM cells (WDM II Eq. 45) for any callable PSD.

.. math::  \Lambda_{pq} = \frac1T\,\mathrm{Re}\sum_k \eta_k\,S(f_k, \bar t_{pq})\,
           \hat g_p(k)\,\hat g_q^*(k),\qquad \bar t_{pq} = (t_{n_p} + t_{n_q})/2,

with :math:`\bar t` used literally, no lag truncation, and only :math:`|m - m'| \le 1`
(an admissible window's compact support makes every other block zero).

==========  =====================================================================
builder     evaluator
==========  =====================================================================
``interp``  ``eq45_coeffs_interp`` + ``eq45_blocks_cheb``: Chebyshev in time
``direct``  ``eq45_direct``: the dense direct sum (small grids and tests)
==========  =====================================================================

``assembly="blocktri"`` factorises the band blocks (:class:`BlockTridiagonalCovariance`,
``layout`` ``"ragged"`` | ``"padded"``); ``assembly="dense"`` assembles the matrix
(:class:`DenseCovariance`).  Both builders call the PSD on the host, so the covariance
runs eagerly on every backend (``jittable`` is always False).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from domainia.wilson.backend import Backend, get_backend
from domainia.wilson.evolutionary import eq45_blocks_cheb, eq45_coeffs_interp, eq45_direct, eq45_plan
from domainia.wilson.grid import packed_cells
from domainia.wilson.windows import MEYER, Window, require_window

from .base import Covariance
from .blocktri import BlockTridiagonalCovariance, blocks_to_dense, to_layout
from .dense import DenseCovariance

__all__ = ["BUILDERS", "EvolutionaryCovariance", "keep_mask"]

Builder = Literal["interp", "direct"]
BUILDERS: tuple[str, ...] = ("interp", "direct")


def keep_mask(cells: Any) -> np.ndarray:
    """Boolean mask over ``wilson.grid.packed_cells`` of ``cells.grid`` selecting ``cells``.

    Raises unless ``cells`` lists active cells in band order (increasing packed position),
    the order of every Eq. (45) block and of ``eq45_plan``'s kept cells.
    """
    g = cells.grid
    m_all, n_all = packed_cells(g.Nf, g.Nt)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(m_all, n_all, strict=True))}
    try:
        pos = np.array([index[int(a), int(b)] for a, b in zip(cells.m, cells.n, strict=True)], int)
    except KeyError as err:
        raise ValueError(f"cell {err.args[0]} is not an active cell of {g}") from None
    if pos.size and np.any(np.diff(pos) <= 0):
        raise ValueError("cells must be listed in band order (increasing packed position)")
    keep = np.zeros(m_all.size, bool)
    keep[pos] = True
    return keep


class _ChebBlocks:
    """``interp``: Chebyshev-in-time blocks on the plan's kept cells."""

    def __init__(self, psd: Any, grid: Any, keep: np.ndarray, P: int, t_max: float | None,
                 window: Window) -> None:
        self.psd, self.grid, self.keep, self.P, self.t_max = psd, grid, keep, P, t_max
        self.window = window

    def host_constants(self) -> dict[str, Any]:
        g = self.grid
        return {"plan": eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=self.P, t_max=self.t_max,
                                  window=self.window)}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> tuple[list[Any], list[Any]]:
        plan = C["plan"]
        nu_h = np.asarray(nu)

        def Sn(f: Any, t: Any) -> Any:
            return self.psd(f, t, nu_h)

        coeffs = eq45_coeffs_interp(Sn, plan.f, plan.t_max, plan.P, k=plan.k_used)
        return eq45_blocks_cheb(plan, coeffs, be=be)


class _DirectMatrix:
    """``direct``: the dense direct sum."""

    def __init__(self, psd: Any, grid: Any, keep: np.ndarray, t_max: float | None,
                 window: Window) -> None:
        self.psd, self.grid, self.keep, self.t_max, self.window = psd, grid, keep, t_max, window

    def host_constants(self) -> dict[str, Any]:
        g = self.grid
        return {"plan": eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=2, t_max=self.t_max,
                                  window=self.window)}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        nu_h = np.asarray(nu)
        return be.asarray(eq45_direct(C["plan"], lambda f, t: self.psd(f, t, nu_h)))


class _DenseFromBlocks:
    """Band blocks assembled into the dense matrix (``assembly="dense"``)."""

    def __init__(self, blocks: Any, sizes: tuple[int, ...]) -> None:
        self.blocks, self.sizes = blocks, sizes

    def host_constants(self) -> Any:
        return self.blocks.host_constants()

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = to_layout(*self.blocks.evaluate(C, nu, be), self.sizes, "ragged", be)
        return blocks_to_dense(diag, up, be)


class EvolutionaryCovariance(Covariance):
    """Eq. (45) on ``cells`` for a callable ``psd(f, t, nu)`` (``GenericPSD``).

    Parameters
    ----------
    psd
        Any callable, elementwise, broadcasting ``psd(f, t, nu)``.
    cells
        A ``CellSet`` in band order.
    builder
        ``"interp"`` (default) or ``"direct"`` (needs ``assembly="dense"``).
    assembly, layout
        ``"blocktri"`` (``layout`` ``"ragged"``, default, or ``"padded"``) or ``"dense"``
        (``chol_impl``, ``symmetrize`` as :class:`DenseCovariance`).
    t_max
        End of the Chebyshev time domain (default: the cells' grid ``T``).
    P
        Chebyshev order of ``interp`` (24).
    window
        The Wilson window of the cells' basis (default :data:`MEYER`).
    """

    def __init__(self, psd: Any, cells: Any, *, builder: Builder = "interp", assembly: str = "blocktri",
                 layout: str | None = None, t_max: float | None = None, P: int = 24,
                 chol_impl: Literal["numpy", "scipy"] = "numpy", symmetrize: bool = True,
                 window: Window = MEYER) -> None:
        if builder not in BUILDERS:
            raise ValueError(f"unknown builder {builder!r}; expected one of {BUILDERS}")
        if not callable(psd):
            raise ValueError(f"builder {builder!r} needs a callable PSD psd(f, t, nu); "
                             f"{type(psd).__name__} is not callable")
        if assembly not in ("blocktri", "dense"):
            raise ValueError(f"assembly must be 'blocktri' or 'dense'; got {assembly!r}")
        if builder == "direct" and assembly != "dense":
            raise ValueError("builder 'direct' needs assembly='dense'")
        if layout is not None and assembly == "dense":
            raise ValueError(f"layout applies to assembly='blocktri' only; got layout={layout!r} "
                             "with assembly='dense'")
        if chol_impl not in ("numpy", "scipy"):
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        require_window(window)
        self.psd, self.cells, self.builder, self.assembly = psd, cells, builder, assembly
        self.window = window
        self.n = len(cells.m)
        keep = keep_mask(cells)
        m_all, _ = packed_cells(cells.grid.Nf, cells.grid.Nt)
        sizes = tuple(int(np.sum(keep & (m_all == mm))) for mm in range(cells.grid.Nf + 1))
        blocks: Any = (_ChebBlocks(psd, cells.grid, keep, P, t_max, window) if builder == "interp"
                       else _DirectMatrix(psd, cells.grid, keep, t_max, window))
        self.sizes = sizes
        self.block_evaluator = blocks
        if assembly == "blocktri":
            self.impl: Covariance = BlockTridiagonalCovariance(blocks, sizes, layout=layout or "ragged")
        else:
            matrix = blocks if builder == "direct" else _DenseFromBlocks(blocks, sizes)
            self.impl = DenseCovariance(matrix, self.n, chol_impl=chol_impl, symmetrize=symmetrize)
        self.jittable = False

    @property
    def layout(self) -> str | None:
        """The block layout (``None`` for ``assembly="dense"``)."""
        return getattr(self.impl, "layout", None)

    def host_constants(self) -> Any:
        return self.impl.host_constants()

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._factor(C, nu, be)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._matrix(C, nu, be)

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        """``(diag, up)`` from the constants ``C`` (``assembly="blocktri"`` only)."""
        if not isinstance(self.impl, BlockTridiagonalCovariance):
            raise ValueError("band_blocks needs assembly='blocktri'")
        return self.impl.band_blocks(C, nu, be)

    def blocks(self, nu: Any, be: Backend | None = None) -> Any:
        """The builder's raw output at ``nu``: ragged band blocks (``interp``) or the
        matrix (``direct``)."""
        be = be or get_backend(nu)
        C = self.constants(be)
        C = C["blocks"] if isinstance(self.impl, BlockTridiagonalCovariance) else C
        return self.block_evaluator.evaluate(C, be.asarray(nu), be)
