"""Data arrays between time, frequency and WDM (domainia 3.0 spec, section 2).

The layouts, on the trailing axes of an array with any leading axes (typically ``(K, ...)``):

* ``"time"``: real ``(N,)``;
* ``"frequency"``: complex ``(N // 2 + 1,)``, ``dt * numpy.fft.rfft(x)``;
* ``"wdm"``: real ``(Nf + 1, Nt)``, the Wilson coefficients of ``window``, edge bands
  included.

Every route goes through the rfft ``numpy.fft.rfft(x)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import wilson
from .wilson import MEYER, Window
from .wilson._checks import require_dtype, require_int

__all__ = ["transform"]

KINDS = ("time", "frequency", "wdm")
#: The largest canonical-form residue an input may carry, as a fraction of its slice's peak.
RESIDUE = 1e-10


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
    Nf, Nt = wdm
    wilson.validate_grid(Nf, Nt)
    if Nf * Nt != N:
        raise ValueError(f"{role}_wdm=({Nf}, {Nt}) holds {Nf * Nt} samples, not N={N}")
    return int(Nf), int(Nt)


def _checked(x: Any, kind: str, N: int, grid: tuple[int, int] | None) -> np.ndarray:
    """``x`` as a float64 (complex128 on frequency) copy, after the input checks."""
    if isinstance(x, np.ma.MaskedArray):
        raise TypeError("x is a masked array; numpy.asarray would drop the mask, so fill it first")
    a = np.asarray(x)
    require_dtype("x", a, "iufc", "a numeric array")
    if a.dtype.kind == "c" and kind != "frequency":
        raise ValueError(f"x is complex, but {kind} data are real")
    shape = {"time": (N,), "frequency": (N // 2 + 1,)}.get(kind) or (grid[0] + 1, grid[1])
    nd = len(shape)
    if a.shape[a.ndim - nd:] != shape:
        raise ValueError(f"x has shape {a.shape}, but {kind} data end in shape {shape}")
    wilson.validate_finite(a, "x")
    a = a.astype(np.complex128 if kind == "frequency" else np.float64)
    if kind == "time":
        return a
    if kind == "frequency":
        ends = np.zeros(shape, bool)
        ends[-1] = N % 2 == 0
        ends[0] = True
        off = np.where(ends, np.abs(a.imag), 0.0)
    else:
        off = np.where(wilson.active_mask(*grid), 0.0, np.abs(a))
    peak = np.max(np.abs(a), axis=tuple(range(-nd, 0)), keepdims=True)
    rel = np.divide(off, peak, out=np.zeros_like(off), where=peak > 0)
    bad = np.argwhere(rel > RESIDUE)
    if bad.size:
        at = [int(i) for i in bad[0]]
        lead, k = at[:-nd], at[-nd:]
        where = (f"the imaginary part of rfft bin {k[0]} ({'Nyquist' if k[0] else 'DC'})"
                 if kind == "frequency" else f"inactive WDM cell (m={k[0]}, n={k[1]})")
        raise ValueError(f"x{lead if lead else ''}: {where} is {rel[tuple(at)]:.2g} of the "
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

    The result is float64 (complex128 on frequency) and exactly canonical: inactive WDM cells
    are 0, and so are the imaginary parts at DC and even-``N`` Nyquist.  An input may carry a
    residue there up to :data:`RESIDUE` of its slice's peak (``max |x|`` over the domain axes,
    for each leading index); a larger one is refused, naming the cell or bin.

    Raises ``TypeError`` for a masked or non-numeric ``x`` (bool, str, object), a non-string
    ``source``/``target``, or a non-integer ``N``, ``Nf`` or ``Nt``; ``ValueError`` for an
    unknown domain, a missing or misplaced or bad WDM grid, a bad ``N`` or ``dt``, complex
    ``x`` on time or WDM, a wrong trailing shape, a non-finite value, or a residue above
    :data:`RESIDUE`.  A real ``x`` on frequency is taken as complex.
    """
    N = require_int("N", N)
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    wilson.validate_dt(dt)
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
    elif target == "wdm":
        y[..., ~wilson.active_mask(*tg)] = 0.0
    return y
