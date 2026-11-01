"""Admissible Wilson windows.

A Wilson basis on the half-shifted lattice is orthonormal when its window
:math:`\\hat\\phi` is *admissible* on the slab :math:`f \\in [-\\Delta F, \\Delta F)`,
:math:`\\Delta F = 1/(2 N_f \\delta t)`:

1. real and even;
2. zero for :math:`|f| \\ge \\Delta F` (compact support inside one slab: the transform
   stays exact and every band covariance block-tridiagonal);
3. a partition of unity on the lattice,
   :math:`\\hat\\phi(f)^2 + \\hat\\phi(f - \\Delta F)^2 = 1/\\Delta F` on :math:`[0, \\Delta F]`
   (the Hz normalisation of :func:`domainia.wilson.filter.phi_centered`).

The WDM transform is the Wilson basis with the Meyer window, :data:`MEYER`.  Every
kernel takes ``window=`` and samples it through :func:`sample_window`.  Built-in
windows validate each ``(window, Nf, Nt, dt)`` once, cache the read-only samples in a
bounded LRU, and raise :class:`WindowError` (naming the condition, the window and the
worst bin) when a condition fails.

Examples
--------
>>> import numpy as np; from domainia import wilson
>>> phi = wilson.MEYER.sample(8, 16, 1.0)
>>> bool(np.array_equal(phi, wilson.filter.phi_centered(8, 16, 1.0)))
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .filter import phi_centered
from .grid import delta_F, validate_dt, validate_grid

__all__ = ["MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError", "sample_window"]

type Array = NDArray[np.float64]

#: Relative tolerance of the partition-of-unity check (relative to 1/dF).
PARTITION_RTOL = 1e-12
#: Absolute tolerance of the ramp symmetry check nu(x) + nu(1 - x) = 1.
RAMP_ATOL = 1e-12
_RAMP_POINTS = 257
_SUPPORT_POINTS = 32            # per side: 64 points of |u| in [1, 2]
_CACHE_SIZE = 256


class WindowError(ValueError):
    """An inadmissible window.  The message names the condition, the window and the bin."""


@runtime_checkable
class Window(Protocol):
    """A Wilson window: frozen and hashable (static under ``jax.jit``)."""

    @property
    def name(self) -> str: ...

    def sample(self, Nf: int, Nt: int, dt: float) -> Array:
        """Centred ``(Nt,)`` samples in ``phi_centered``'s ordering and Hz normalisation."""
        ...


def _fail(name: str, condition: str, Nf: int, Nt: int, dt: float, detail: str) -> None:
    raise WindowError(f"window {name!r} fails the {condition} condition on the "
                      f"(Nf={Nf}, Nt={Nt}, dt={dt!r}) slab: {detail}")


def _as_samples(name: str, raw: Any, Nf: int, Nt: int, dt: float) -> np.ndarray:
    a = np.asarray(raw)
    if a.dtype.kind not in "fiu":
        _fail(name, "real", Nf, Nt, dt, f"samples have dtype {a.dtype}")
    return np.array(a, dtype=np.float64)                         # a fresh float64 copy


def _check_samples(name: str, phi: np.ndarray, Nf: int, Nt: int, dt: float) -> None:
    """The admissibility conditions on the sampled slab (module docstring)."""
    if phi.shape != (Nt,):
        _fail(name, "shape", Nf, Nt, dt, f"expected ({Nt},) samples, got {phi.shape}")
    if not np.all(np.isfinite(phi)):
        k = int(np.flatnonzero(~np.isfinite(phi))[0])
        _fail(name, "finiteness", Nf, Nt, dt, f"bin {k} is {phi[k]!r}")
    h = Nt // 2
    if phi[0] != 0.0:
        _fail(name, "support", Nf, Nt, dt, f"bin 0 (f = -dF) is {phi[0]!r}, not 0")
    diff = np.abs(phi[h + 1:] - phi[h - 1:0:-1])
    if diff.size and np.max(diff) > 0.0:
        j = int(np.argmax(diff)) + 1
        _fail(name, "evenness", Nf, Nt, dt,
              f"worst at bin {h + j}: {phi[h + j]!r} but its mirror bin {h - j} is {phi[h - j]!r}")
    inv_dF = 1.0 / delta_F(Nf, dt)
    err = np.abs(phi[h:] ** 2 + phi[:h] ** 2 - inv_dF)
    j = int(np.argmax(err))
    if err[j] > PARTITION_RTOL * inv_dF:
        _fail(name, "partition of unity", Nf, Nt, dt,
              f"worst at bin {h + j}: phi(f)^2 + phi(f - dF)^2 = {phi[h + j] ** 2 + phi[j] ** 2!r}, "
              f"1/dF = {inv_dF!r} (relative error {err[j] / inv_dF:.3e})")


@lru_cache(maxsize=_CACHE_SIZE)
def _checked(window: _Admissible, Nf: int, Nt: int, dt: float) -> Array:
    phi = _as_samples(window.name, window._samples(Nf, Nt, dt), Nf, Nt, dt)
    _check_samples(window.name, phi, Nf, Nt, dt)
    phi.setflags(write=False)
    return phi


class _Admissible:
    """The built-in windows' ``sample``: validate the grid, then the cached checked samples."""

    __slots__ = ()
    name: str

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        raise NotImplementedError

    def sample(self, Nf: int, Nt: int, dt: float) -> Array:
        """Validated, read-only, cached samples (centred, Hz normalisation)."""
        validate_grid(Nf, Nt)
        validate_dt(dt)
        return _checked(self, int(Nf), int(Nt), float(dt))


def _check_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError(f"window name must be a non-empty string; got {name!r}")


def _check_callable(what: str, fn: Any) -> None:
    if not callable(fn):
        raise TypeError(f"{what} must be callable; got {type(fn).__name__}")


def _meyer_geometry(Nf: int, Nt: int, dt: float) -> tuple[np.ndarray, float, float, float]:
    """``(|f|, A, B, amp)`` exactly as ``phi_centered`` computes them."""
    dF = delta_F(Nf, dt)
    T = Nf * Nt * dt
    f = (np.arange(Nt) - Nt // 2) / T
    return np.abs(f), dF / 4.0, dF / 2.0, 1.0 / np.sqrt(dF)


@dataclass(frozen=True)
class Meyer(_Admissible):
    """The Meyer window: flat to dF/4, ``cos(pi/2 nu_d)`` rolloff to 3 dF/4, ``nu_d = I_x(d, d)``.

    ``Meyer(d).sample`` is :func:`domainia.wilson.filter.phi_centered` ``(Nf, Nt, dt, d)``
    bit for bit; ``Meyer()`` (d = 4) is the WDM transform's window.
    """

    d: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.d, bool) or not isinstance(self.d, (int, np.integer)):
            raise TypeError(f"Meyer d must be an int; got {type(self.d).__name__}")
        if self.d < 1:
            raise ValueError(f"Meyer d must be >= 1; got {self.d}")
        object.__setattr__(self, "d", int(self.d))

    @property
    def name(self) -> str:
        return f"meyer(d={self.d})"

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        return phi_centered(Nf, Nt, dt, self.d)


@dataclass(frozen=True)
class RampWindow(_Admissible):
    """Meyer's geometry with a user ramp ``nu`` on [0, 1]: ``nu(0) = 0``, ``nu(1) = 1`` and
    ``nu(x) + nu(1 - x) = 1`` (checked on 257 points at construction).  Two RampWindows are
    equal when they hold the same function object and name."""

    ramp: Callable[[Array], Array]
    name: str

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_callable("ramp", self.ramp)
        x = np.linspace(0.0, 1.0, _RAMP_POINTS)                    # dyadic: 1 - x[k] == x[-1-k]
        nu = np.asarray(self.ramp(x))
        if nu.shape != x.shape or nu.dtype.kind not in "fiu" or not np.all(np.isfinite(nu)):
            raise WindowError(f"window {self.name!r} fails the ramp condition: the ramp must map "
                              f"{x.size} points to {x.size} finite real values")
        nu = nu.astype(np.float64)
        if nu[0] != 0.0 or nu[-1] != 1.0:
            raise WindowError(f"window {self.name!r} fails the ramp condition: nu(0) = {nu[0]!r} "
                              f"and nu(1) = {nu[-1]!r}; need exactly 0 and 1")
        err = np.abs(nu + nu[::-1] - 1.0)
        k = int(np.argmax(err))
        if err[k] > RAMP_ATOL:
            raise WindowError(f"window {self.name!r} fails the ramp condition: nu(x) + nu(1 - x) - 1 "
                              f"= {err[k]:.3e} at x = {x[k]!r}")

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        f_abs, A, B, amp = _meyer_geometry(Nf, Nt, dt)
        nu = np.asarray(self.ramp(np.clip((f_abs - A) / B, 0.0, 1.0)), dtype=np.float64)
        out = np.where(f_abs < A, amp, 0.0)
        rolloff = (f_abs >= A) & (f_abs < A + B)
        return np.where(rolloff, amp * np.cos(np.pi / 2.0 * nu), out)


@dataclass(frozen=True)
class SampledWindow(_Admissible):
    """Any dimensionless profile ``psi(u)``, ``u = f / dF``: flat value 1,
    ``psi(u)^2 + psi(u - 1)^2 = 1`` on [0, 1] and ``psi = 0`` for ``|u| >= 1``.

    ``sample`` returns ``psi(f / dF) / sqrt(dF)`` (the Hz normalisation).  ``psi`` is also
    evaluated on 64 points of ``|u|`` in [1, 2] at construction and must be exactly 0
    there: mass outside the slab would be truncated silently.  Two SampledWindows are equal
    when they hold the same function object and name.
    """

    fn: Callable[[Array], Array]
    name: str

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_callable("fn", self.fn)
        u = np.linspace(1.0, 2.0, _SUPPORT_POINTS)
        u = np.concatenate([-u[::-1], u])
        psi = np.asarray(self.fn(u))
        if psi.shape != u.shape:
            raise WindowError(f"window {self.name!r} fails the support condition: fn maps {u.size} "
                              f"points to shape {psi.shape}")
        bad = np.flatnonzero(psi != 0.0)
        if bad.size:
            k = int(bad[0])
            raise WindowError(f"window {self.name!r} fails the support condition: psi({u[k]!r}) = "
                              f"{psi[k]!r}, but a window must vanish for |f| >= dF")

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        dF = delta_F(Nf, dt)
        T = Nf * Nt * dt
        u = (np.arange(Nt) - Nt // 2) / T / dF
        return np.asarray(self.fn(u), dtype=np.float64) / np.sqrt(dF)


def sample_window(window: Any, Nf: int, Nt: int, dt: float) -> Array:
    """The kernels' single entry point: the validated, read-only samples of ``window``.

    Built-in windows return their cached samples.  Any other object implementing
    :class:`Window` is validated on every call; anything else raises ``TypeError``.
    """
    if isinstance(window, _Admissible):
        return window.sample(Nf, Nt, dt)
    if not isinstance(window, Window):
        raise TypeError(f"window must be a domainia.wilson Window (Meyer, RampWindow, "
                        f"SampledWindow); got {type(window).__name__}")
    validate_grid(Nf, Nt)
    validate_dt(dt)
    name = str(window.name)
    phi = _as_samples(name, window.sample(Nf, Nt, dt), int(Nf), int(Nt), float(dt))
    _check_samples(name, phi, int(Nf), int(Nt), float(dt))
    phi.setflags(write=False)
    return phi


#: The WDM transform's window: Meyer with d = 4.
MEYER = Meyer()
