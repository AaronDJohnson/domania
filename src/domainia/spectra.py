"""Noise spectra.

* :class:`GenericPSD` -- any elementwise, broadcasting ``Sn(f, t, nu)``: what the
  evolutionary covariance and the WDM variances evaluate.
* :class:`TabulatedSpectrum` -- ``(C, n)`` values on absolute rfft bins of a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import numpy as np
from domainia.wilson.backend import Backend

from .domains import FourierDomain

__all__ = ["GenericPSD", "TabulatedSpectrum"]


@dataclass(frozen=True, eq=False)
class TabulatedSpectrum:
    """``values[c, j]`` at absolute rfft bin ``domain.k0 + j`` of ``domain``'s record.

    ``values`` is a read-only float64 copy (the caller's array is never aliased), as
    D-A's ``CellSet`` keeps its indices.
    """

    domain: FourierDomain
    values: np.ndarray

    def __post_init__(self) -> None:
        v = np.array(self.values, dtype=float)                 # a copy, values unchanged
        if v.ndim != 2 or v.shape[1] != self.domain.n:
            raise ValueError(f"values must be (channels, {self.domain.n}); got {v.shape}")
        v.setflags(write=False)
        object.__setattr__(self, "values", v)

    @property
    def channels(self) -> int:
        return self.values.shape[0]

    def _check(self, N: int, dt: float, lo: int, hi: int) -> None:
        d = self.domain
        if (N, dt) != (d.N, d.dt):
            raise ValueError(f"bins of an N={N}, dt={dt} record; this spectrum is on N={d.N}, dt={d.dt}")
        if lo < d.k0 or hi > d.k0 + d.n:
            raise ValueError(f"bins [{lo}, {hi}) are outside the tabulated [{d.k0}, {d.k0 + d.n})")

    def on(self, domain: FourierDomain) -> np.ndarray:
        """``(C, domain.n)``: the values on ``domain``'s bins; raises outside the table."""
        self._check(domain.N, domain.dt, domain.k0, domain.k0 + domain.n)
        lo = domain.k0 - self.domain.k0
        return self.values[:, lo:lo + domain.n]

    def at(self, bins: Any) -> np.ndarray:
        """``(C, *bins.shape)``: the values at absolute bins; raises outside the table.

        Integer bins only: a non-integral float (``99.9999``) raises instead of being
        truncated by the int64 cast.
        """
        b = np.asarray(bins)
        if b.dtype.kind not in "iu" and not (b.dtype.kind == "f" and np.all(np.isfinite(b))
                                             and np.array_equal(b, np.rint(b))):
            raise ValueError(f"bins must be integers; got {b.dtype} values")
        bins = b.astype(np.int64)
        if bins.size:
            self._check(self.domain.N, self.domain.dt, int(bins.min()), int(bins.max()) + 1)
        return self.values[:, bins - self.domain.k0]


@dataclass(frozen=True, eq=False)
class GenericPSD:
    """Any elementwise, broadcasting ``Sn(f, t, nu)`` (Eq. 45 ``interp``/``direct``, dense
    exact covariances, tests)."""

    fn: Callable[[Any, Any, Any], Any]

    capabilities: ClassVar[frozenset[str]] = frozenset({"interp", "direct"})

    def __call__(self, f: Any, t: Any, nu: Any, be: Backend | None = None) -> Any:
        return self.fn(f, t, nu)

    def bind(self, nu: Any) -> Callable[[Any, Any], Any]:
        return lambda f, t: self.fn(f, t, nu)
