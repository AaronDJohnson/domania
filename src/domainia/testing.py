"""Dense oracles for tests and small grids (NumPy only, never on a
likelihood path).  Every oracle here is ``O(N^2)`` memory or worse: keep ``N`` <~ 2048.

* :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a cell set
  (``wilson.matrix.active_W``) and ``W_K Sigma W_K^T``.
* :func:`eq45_dense`: Eq. (45) on a cell set by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).

:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.

The oracles follow the package's rule: a wrong kind (``cells`` that is not a ``CellSet``)
raises ``TypeError``, a wrong value ``ValueError``.

Test windows (``domainia.wilson.windows``):

* :func:`smoothstep`: a non-Meyer ramp for :class:`~domainia.wilson.RampWindow`.
* :func:`wide_profile`, :data:`ALT_WINDOW`: a non-Meyer admissible profile and the
  :class:`~domainia.wilson.SampledWindow` built from it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from domainia import wilson
from domainia.wilson._checks import require_callable, require_instance
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
from domainia.wilson.windows import MEYER, SampledWindow, Window

from .domains import CellSet

__all__ = ["ALT_WINDOW", "dense_W", "eq45_dense", "smoothstep", "wdm_congruence", "wide_profile"]

Array = NDArray[np.float64]


def dense_W(cells: CellSet, *, window: Window = MEYER) -> Array:
    """``(cells.size, N)``: row ``p`` is cell ``p``'s atom sampled in time, so ``W x`` are
    the cells of ``x`` (``wilson.matrix.active_W`` rows, ``W_a W_a^T = dt I``) in the
    Wilson basis of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    return wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=window)[cells.packed_positions()]


def wdm_congruence(Sigma: Array, cells: CellSet, *, window: Window = MEYER) -> Array:
    """``W_K Sigma W_K^T`` for a time-domain covariance ``Sigma (N, N)`` (``W_K`` is
    :func:`dense_W` with ``window``)."""
    W = dense_W(cells, window=window)
    return W @ np.asarray(Sigma, dtype=float) @ W.T


def eq45_dense(cells: CellSet, Sn: Callable[[Any, Any], Any], *, P: int = 24,
               t_max: float | None = None, window: Window = MEYER) -> Array:
    """Eq. (45) on ``cells`` by the direct sum, ``Sn(f, t)`` elementwise (``A_n`` inside),
    with the atoms of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    require_callable("Sn", Sn)
    plan = eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max, window=window)
    return eq45_direct(plan, Sn)


def smoothstep(x: Any) -> Array:
    """``3 x^2 - 2 x^3`` on [0, 1] (clipped): a ramp with ``nu(x) + nu(1 - x) = 1``."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def wide_profile(u: Any) -> Array:
    """A non-Meyer admissible profile ``psi(u)``, ``u = f / dF``: 1 for ``|u| < 1/8``, a
    smoothstep cosine rolloff to 0 at ``|u| = 7/8``, 0 beyond.  On [0, 1],
    ``psi(u)^2 + psi(u - 1)^2 = 1`` because ``smoothstep(1 - x) = 1 - smoothstep(x)``."""
    a = np.abs(np.asarray(u, dtype=float))
    x = np.clip((a - 0.125) / 0.75, 0.0, 1.0)
    return np.where(a < 0.125, 1.0, np.where(a < 0.875, np.cos(np.pi / 2.0 * smoothstep(x)), 0.0))


#: A non-default admissible window for tests (other flat width, other ramp than Meyer).
ALT_WINDOW = SampledWindow(wide_profile, "wide-smoothstep")
