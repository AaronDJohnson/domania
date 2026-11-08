"""Diagonal WDM variances of a :class:`~domainia.spectra.GenericPSD` ``S(f, t, nu)``.

* :class:`BandAveragedVariance` (WDM II Eq. 36) -- the window-weighted band average,
  ``domainia.wilson.noise.evolutionary_band_variance`` of the bound PSD at the cells;
* :class:`MidpointVariance` (Eq. 39) -- ``0.5 S(f_m, t_n)`` at the band centre
  ``f_m = m dF`` and the cell time ``t_n = n dT``.

Both take a ``CellSet`` or a ``WDMDomain``.  ``BandAveragedVariance`` averages over the
window of the cells' basis: the ``WDMDomain``'s window, or ``window=`` (default Meyer) for a
``CellSet``.  The midpoint variance is window-free.

Both are pytrees (``domainia.wilson.backend.pytree_dataclass``; the window is static
metadata), but evaluating them calls the PSD on the host, so neither is jittable
(``jittable = False``): a ``DiagonalCovariance`` built on one factorises eagerly on every
backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from domainia import wilson
from domainia.wilson.backend import Backend, pytree_dataclass
from domainia.wilson.windows import Window

from .domains import CellSet, _cells_and_window
from .maps.base import _BUILD, on_backend
from .spectra import GenericPSD

__all__ = ["BandAveragedVariance", "MidpointVariance"]


def _cell_arrays(cells: CellSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = cells.grid
    m = np.asarray(cells.m, dtype=np.int64)
    n = np.asarray(cells.n, dtype=np.int64)
    return m, n, n * wilson.delta_T(g.Nf, g.dt)                     # t_n = n dT


def _require_generic(psd: Any, what: str) -> None:
    if not isinstance(psd, GenericPSD):
        raise TypeError(f"no {what} variance for {type(psd).__name__}: wrap the spectrum in "
                        "domainia.spectra.GenericPSD")


@pytree_dataclass(data=("t",), meta=("psd", "cells", "window"))
@dataclass(frozen=True, eq=False)
class BandAveragedVariance:
    """Eq. 36: ``0.5 <S(f, t_n)>`` over the window squared around each cell's band.

    ``cells`` is a ``CellSet`` or a ``WDMDomain`` (stored as its cells).  ``window`` is the
    Wilson window of the cells' basis: ``None`` (default) means the ``WDMDomain``'s window,
    or :data:`~domainia.wilson.MEYER` for a ``CellSet``; a ``WDMDomain`` with another
    ``window`` raises ``ValueError``.
    """

    jittable: ClassVar[bool] = False

    psd: Any
    cells: CellSet
    t: Any = field(default=_BUILD, kw_only=True)
    window: Window | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.t is not _BUILD:
            return
        _require_generic(self.psd, "band-averaged")
        cells, window = _cells_and_window(self.cells, self.window)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "t", _cell_arrays(cells)[2])

    def on(self, be: Backend) -> "BandAveragedVariance":
        """The constants moved to ``be``."""
        return on_backend(self, be, ("t",))

    def __call__(self, nu: Any, *, channel: int = 0, be: Backend | None = None) -> Any:
        g = self.cells.grid
        m, n, _ = _cell_arrays(self.cells)
        full = wilson.noise.evolutionary_band_variance(g.Nf, g.Nt, g.dt, self.psd.bind(nu),
                                                       window=self.window)
        return full[m, n]


@pytree_dataclass(data=("t",), meta=("psd", "cells"))
@dataclass(frozen=True, eq=False)
class MidpointVariance:
    """Eq. 39: ``0.5 S(f_m, t_n)``.  ``cells`` is a ``CellSet`` or a ``WDMDomain`` (stored as
    its cells; the midpoint variance does not depend on the window)."""

    jittable: ClassVar[bool] = False

    psd: Any
    cells: CellSet
    t: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.t is not _BUILD:
            return
        _require_generic(self.psd, "midpoint")
        cells, _ = _cells_and_window(self.cells)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "t", _cell_arrays(cells)[2])

    def on(self, be: Backend) -> "MidpointVariance":
        """The constants moved to ``be``."""
        return on_backend(self, be, ("t",))

    def __call__(self, nu: Any, *, channel: int = 0, be: Backend | None = None) -> Any:
        g = self.cells.grid
        m, _, t = _cell_arrays(self.cells)
        f = wilson.band_frequencies(g.Nf, g.dt)[m]
        return 0.5 * np.asarray(self.psd(f, t, nu))
