r"""``Lambda = W C W^T`` by the fast WDM transform on both sides (WDM II Eq. 9; spec 10.3).

The text builds the exact sinusoid covariance in time (Eqs. 95-97) and then "transform[s] the
time-domain covariance to the WDM domain": :math:`\Lambda = W C W^{\mathsf T}`.  ``W`` is never
formed: each row of ``C`` goes through ``wdm.forward_time_many`` (``x -> W x`` on the packed
cells, ``W W^T = dt I``), then each row of ``C W^T`` again.  ``rows`` gives the edge rows
:math:`W_E C W^{\mathsf T}` (the exact-edge targets), with ``W_E`` the atoms of the cells ``E``
(:func:`wdm_rows`: :math:`W^{\mathsf T} e_p = dt\,W^{-1} e_p`).  Written once over the backend
shim; under ``jax.jit`` the grid, ``rows`` and ``chunk`` are static.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import wdm
from wdm.backend import NUMPY, Backend, get_backend
from wdm.grid import packed_cells
from wdm.transform import forward_time_many, inverse_time_many

__all__ = ["wdm_congruence", "wdm_rows"]


def wdm_rows(grid: wdm.Grid, rows: Any, *, be: Backend | None = None) -> Any:
    """``(len(rows), N)``: the rows ``rows`` (positions among ``wdm.grid.packed_cells``) of the
    packed WDM matrix, i.e. the cells' atoms sampled in time (``dt W^{-1} e_p``)."""
    be = be or NUMPY
    rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
    m, n = packed_cells(grid.Nf, grid.Nt)
    if rows.size and (rows.min() < 0 or rows.max() >= m.size):
        raise ValueError(f"rows must lie in [0, {m.size}) (packed cells of {grid})")
    E = be.xp.zeros((rows.size, grid.Nf + 1, grid.Nt))
    E = be.set(E, (np.arange(rows.size), m[rows], n[rows]), 1.0, copy=False)
    return grid.dt * inverse_time_many(E, grid.Nf, grid.Nt, grid.dt, be=be)


def _forward_rows(X: Any, grid: wdm.Grid, m: np.ndarray, n: np.ndarray, be: Backend, chunk: int | None) -> Any:
    """``X W^T``: every row of ``X`` (a record) to its packed cells, ``chunk`` rows at a time."""
    if chunk is None or X.shape[0] <= chunk:
        return forward_time_many(X, grid.Nf, grid.Nt, grid.dt, be=be)[:, m, n]
    parts = [forward_time_many(X[i:i + chunk], grid.Nf, grid.Nt, grid.dt, be=be)[:, m, n]
             for i in range(0, X.shape[0], chunk)]
    return be.xp.concatenate(parts, axis=0)


def wdm_congruence(C: Any, grid: wdm.Grid, *, rows: Any = None, chunk: int | None = None,
                   be: Backend | None = None) -> Any:
    """``W C W^T`` on every packed cell of ``grid`` (``(N, N)``), or ``W_rows C W^T``
    (``(len(rows), N)``) -- ``C`` any ``(N, N)`` time-domain matrix (not assumed symmetric).

    ``chunk`` bounds the rows transformed at once (memory: one ``(chunk, N/2 + 1)`` complex
    spectrum per pass); the result does not depend on it."""
    be = be or get_backend(C)
    C = be.asarray(C)
    if C.shape != (grid.N, grid.N):
        raise ValueError(f"C must be ({grid.N}, {grid.N}) for {grid}; got {C.shape}")
    if chunk is not None and (not isinstance(chunk, int) or chunk < 1):
        raise ValueError(f"chunk must be a positive int or None; got {chunk!r}")
    m, n = packed_cells(grid.Nf, grid.Nt)
    Y = _forward_rows(C, grid, m, n, be, chunk)                         # C W^T
    if rows is None:
        return _forward_rows(Y.T, grid, m, n, be, chunk).T              # (Y^T W^T)^T = W C W^T
    return wdm_rows(grid, rows, be=be) @ Y                              # W_rows C W^T
