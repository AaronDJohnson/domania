"""Data arrays between time, frequency and WDM.

The layouts, on the trailing axes of an array with any leading axes (typically ``(K, ...)``):

* ``"time"``: real ``(N,)``;
* ``"frequency"``: complex ``(N // 2 + 1,)``, ``dt * numpy.fft.rfft(x)``;
* ``"wdm"``: real ``(Nf, Nt)``, the Wilson coefficients of ``window`` in the folded layout
  (:func:`domainia.wilson.grid.cells` gives each slot's band and time).

Every route goes through the rfft ``numpy.fft.rfft(x)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import wilson
from .wilson import MEYER, Window
from .wilson._checks import require_array, require_int
from .wilson.grid import RESIDUE, _fold_fix

__all__ = ["transform"]

KINDS = ("time", "frequency", "wdm")


def _grid(role: str, kind: Any, wdm: Any, N: int) -> tuple[int, int] | None:
    """The validated ``(Nf, Nt)`` of a ``"wdm"`` side, ``None`` on the others."""
    if not isinstance(kind, str):
        raise TypeError(f"{role} must be a string, got {type(kind).__name__}")
    if kind not in KINDS:
        raise ValueError(f"{role} must be one of {KINDS}, got {kind!r}")
    if kind != "wdm":
        if wdm is not None:
            raise ValueError(f"{role}_wdm is only for a 'wdm' {role}; {role} is {kind!r}")
        return None
    if wdm is None:
        raise ValueError(f"a 'wdm' {role} needs {role}_wdm=(Nf, Nt)")
    try:
        n = len(wdm)
    except TypeError:
        raise TypeError(f"{role}_wdm must be an (Nf, Nt) pair, got {type(wdm).__name__}") from None
    if n != 2:
        raise ValueError(f"{role}_wdm must be an (Nf, Nt) pair, got {wdm!r}")
    Nf, Nt = wdm
    wilson.grid.validate_grid(Nf, Nt)
    if Nf * Nt != N:
        raise ValueError(f"{role}_wdm=({Nf}, {Nt}) holds {Nf * Nt} samples, not N={N}")
    return int(Nf), int(Nt)


def _checked(x: Any, kind: str, N: int, grid: tuple[int, int] | None) -> np.ndarray:
    """``x`` as a float64 (complex128 on frequency) copy, after the input checks."""
    a = require_array("x", x, "iufc", "a numeric array")
    if a.dtype.kind == "c" and kind != "frequency":
        raise ValueError(f"x is complex, but {kind} data are real")
    shape = {"time": (N,), "frequency": (N // 2 + 1,)}.get(kind) or grid
    if a.shape[a.ndim - len(shape):] != shape:
        raise ValueError(f"x has shape {a.shape}, but {kind} data end in shape {shape}"
                         + (_fold_fix(a.shape, *grid) if grid else ""))
    wilson.grid.validate_finite(a, "x")
    a = a.astype(np.complex128 if kind == "frequency" else np.float64)
    if kind != "frequency":
        return a
    ends = np.zeros(shape, bool)
    ends[-1] = N % 2 == 0
    ends[0] = True
    off = np.where(ends, np.abs(a.imag), 0.0)
    peak = np.max(np.abs(a), axis=-1, keepdims=True)
    rel = np.divide(off, peak, out=np.zeros_like(off), where=peak > 0)
    bad = np.argwhere(rel > RESIDUE)
    if bad.size:
        *lead, k = (int(i) for i in bad[0])
        raise ValueError(f"x{lead if lead else ''}: the imaginary part of rfft bin {k} "
                         f"({'Nyquist' if k else 'DC'}) is {rel[tuple(bad[0])]:.2g} of the "
                         f"slice's peak; at most {RESIDUE:g} is accepted")
    return a


def transform(x: Any, source: str, target: str, *, N: int, dt: float,
              source_wdm: tuple[int, int] | None = None, target_wdm: tuple[int, int] | None = None,
              window: Window = MEYER) -> np.ndarray:
    """``x`` on ``source`` re-expressed on ``target`` (module docstring for the layouts).

    ``source`` and ``target`` are each ``"time"``, ``"frequency"`` or ``"wdm"``.  ``N`` is the
    record's sample count, always required.  ``source_wdm`` and ``target_wdm`` are the
    ``(Nf, Nt)`` of a ``"wdm"`` side, required there and refused elsewhere, with Nf and Nt
    positive and even and ``Nf * Nt == N``.  The two WDM grids may differ: WDM -> WDM then
    regrids through the rfft.  When the source equals the target, grid included, the result
    is a validated canonical copy.

    The result is float64 (complex128 on frequency); on frequency it is exactly canonical, with
    the imaginary parts at DC and even-``N`` Nyquist 0.  A frequency input may carry a residue
    there up to :data:`~domainia.wilson.grid.RESIDUE` of its slice's peak (``max |x|`` over the
    last axis, for each leading index); a larger one is refused, naming the bin.  A WDM output
    is the folded output of :func:`domainia.wilson.forward` as it is.

    Raises ``TypeError`` for a masked or non-numeric ``x`` (bool, str, object), a non-string
    ``source``/``target``, a WDM grid that is not a sequence, a non-integer ``N``, ``Nf`` or
    ``Nt``, a non-real ``dt`` or a non-window ``window``; ``ValueError`` for an unknown domain,
    a missing or misplaced WDM grid or one that is not a valid ``(Nf, Nt)`` pair, ``N < 1``, a
    non-positive or non-finite ``dt``, complex ``x`` on time or WDM, a wrong trailing shape (for
    a padded ``(..., Nf+1, Nt)`` or a flat ``(..., N)`` WDM ``x`` the message names the fix,
    ``wilson.grid.fold`` or ``.reshape(..., Nf, Nt)``), a non-finite value, or a residue above
    :data:`~domainia.wilson.grid.RESIDUE`; and ``WindowError``, a
    ``ValueError``, for a window inadmissible on a WDM grid.  A real ``x`` on frequency is
    taken as complex.
    """
    N = require_int("N", N)
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    wilson.grid.validate_dt(dt)
    wilson.windows.require_window(window)
    dt = float(dt)
    sg, tg = _grid("source", source, source_wdm, N), _grid("target", target, target_wdm, N)
    y = _checked(x, source, N, sg)
    if (source, sg) != (target, tg):
        if source == "time":
            R = np.fft.rfft(y, axis=-1)
        elif source == "frequency":
            R = y / dt
        else:
            R = wilson.inverse(y, *sg, dt, window=window)
        if target == "time":
            y = np.fft.irfft(R, n=N, axis=-1)
        elif target == "frequency":
            y = dt * R
        else:
            y = wilson.forward(R, *tg, dt, window=window)
    if target == "frequency":
        y[..., 0] = y[..., 0].real
        if N % 2 == 0:
            y[..., -1] = y[..., -1].real
    return y
