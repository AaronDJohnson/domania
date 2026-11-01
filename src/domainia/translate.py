"""Move full-record data between the time, Fourier and WDM domains (spec section 3).

NumPy only.  Data are ``{channel: ndarray}`` dicts (insertion order kept) on one of the
full-record domains:

* ``TimeDomain(N, dt, t0)``: real ``(N,)`` per channel;
* ``FourierDomain.full(N, dt)``: complex ``(N // 2 + 1,)`` per channel, ``dt * numpy.fft.rfft(x)``;
* ``WDMDomain.full(Grid(Nf, Nt, dt), window)``: real dense ``(Nf + 1, Nt)`` per channel, the
  Wilson coefficients of ``window``, inactive cells exactly 0.

Every pair goes through the rfft hub ``R = numpy.fft.rfft(x)``: time ``rfft``/``irfft``,
Fourier ``X / dt``/``dt R``, WDM ``wilson.inverse_many``/``wilson.forward_many``.  Partial
or packed cell sets stay with :mod:`domainia.maps`.

In the real coordinates (time samples; WDM cells; Fourier ``Re X_0``, ``sqrt(2) Re/Im X_k``,
``Re X_{N/2}``) every domain is the time record scaled by ``exp(log_coordinate_scale(domain))``:
1 for time, ``sqrt(dt)`` for WDM (any admissible window), ``dt sqrt(N)`` for Fourier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from domainia import wilson
from domainia.domains import FourierDomain, TimeDomain, WDMDomain

__all__ = ["log_coordinate_scale", "transform"]

type Domain = TimeDomain | FourierDomain | WDMDomain


def _record(domain: Any, role: str) -> tuple[int, float]:
    """``(N, dt)`` of a full-record domain; refuses every other domain."""
    if isinstance(domain, TimeDomain):
        return domain.N, domain.dt
    if isinstance(domain, FourierDomain):
        if domain.units != "dt_rfft" or domain.k0 != 0 or domain.n != domain.N // 2 + 1:
            raise ValueError(f"{role}: translate handles only FourierDomain.full(N, dt) (every rfft "
                             f"bin, dt_rfft units); got {domain!r}")
        return domain.N, domain.dt
    if isinstance(domain, WDMDomain):
        g = domain.cells.grid
        if domain.cells.size != g.N:
            raise ValueError(f"{role}: translate handles only WDMDomain.full(grid, window) (every "
                             f"active cell, scale 1); got {domain.cells.size} of {g.N} cells")
        if domain.scale != 1.0:
            raise ValueError(f"{role}: translate handles only WDMDomain.full(grid, window) (every "
                             f"active cell, scale 1); got scale={domain.scale!r}")
        return g.N, g.dt
    raise TypeError(f"{role} must be a full-record domain (TimeDomain, FourierDomain.full or "
                    f"WDMDomain.full); got {type(domain).__name__}")


def _check_pair(source: Any, target: Any) -> None:
    Ns, dts = _record(source, "source")
    Nt, dtt = _record(target, "target")
    if Ns != Nt or dts != dtt:
        raise ValueError(f"source and target describe different records: N={Ns}, dt={dts!r} vs "
                         f"N={Nt}, dt={dtt!r}")
    if isinstance(source, TimeDomain) and isinstance(target, TimeDomain) and source.t0 != target.t0:
        raise ValueError(f"source and target time domains start at different t0 ({source.t0!r} vs "
                         f"{target.t0!r}): transform does not shift records")


def _shape(domain: Domain) -> tuple[int, ...]:
    if isinstance(domain, TimeDomain):
        return (domain.N,)
    if isinstance(domain, FourierDomain):
        return (domain.N // 2 + 1,)
    g = domain.cells.grid
    return (g.Nf + 1, g.Nt)


def _active(domain: WDMDomain) -> np.ndarray:
    g = domain.cells.grid
    return wilson.active_mask(g.Nf, g.Nt)


def _stack(channel_data: Any, domain: Domain) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate ``channel_data`` on ``domain``; ``(names, (C, *shape) float64/complex128)``."""
    if not isinstance(channel_data, Mapping):
        raise TypeError(f"channel_data must be a mapping {{channel: array}}; got "
                        f"{type(channel_data).__name__}")
    if not channel_data:
        raise ValueError("channel_data is empty: pass at least one channel")
    shape = _shape(domain)
    kind = type(domain).__name__
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for name, value in channel_data.items():
        if not isinstance(name, str):
            raise TypeError(f"channel names must be strings; got {name!r}")
        if not name:
            raise ValueError("channel names must be non-empty")
        a = np.asarray(value)
        if a.dtype.kind not in "iufc":
            raise TypeError(f"channel {name!r}: values must be numeric; got dtype {a.dtype}")
        if a.dtype.kind == "c" and not isinstance(domain, FourierDomain):
            raise ValueError(f"channel {name!r}: complex values on a real {kind}")
        if a.shape != shape:
            raise ValueError(f"channel {name!r}: shape {a.shape}, but this {kind} needs shape {shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"channel {name!r}: values must be finite")
        if isinstance(domain, FourierDomain):
            a = a.astype(np.complex128)
            ends = ((0, "DC"), (shape[0] - 1, "Nyquist")) if domain.N % 2 == 0 else ((0, "DC"),)
            for k, label in ends:
                if a[k].imag != 0.0:
                    raise ValueError(f"channel {name!r}: rfft bin {k} ({label}) must be real; its "
                                     f"imaginary part is {a[k].imag!r}")
        else:
            a = a.astype(np.float64)
            if isinstance(domain, WDMDomain):
                bad = (a != 0.0) & ~_active(domain)
                if np.any(bad):
                    m, n = (int(i) for i in np.argwhere(bad)[0])
                    raise ValueError(f"channel {name!r}: inactive WDM cell (m={m}, n={n}) is "
                                     f"{a[m, n]!r}; inactive cells must be exactly 0")
        names.append(name)
        arrays.append(a)
    return tuple(names), np.stack(arrays)


def _to_rfft(X: np.ndarray, domain: Domain) -> np.ndarray:
    if isinstance(domain, TimeDomain):
        return np.fft.rfft(X, axis=-1)
    if isinstance(domain, FourierDomain):
        return X / domain.dt
    g = domain.cells.grid
    return wilson.inverse_many(X, g.Nf, g.Nt, g.dt, window=domain.window)


def _from_rfft(R: np.ndarray, domain: Domain) -> np.ndarray:
    if isinstance(domain, TimeDomain):
        return np.fft.irfft(R, n=domain.N, axis=-1)
    if isinstance(domain, FourierDomain):
        return domain.dt * R
    g = domain.cells.grid
    return wilson.forward_many(R, g.Nf, g.Nt, g.dt, window=domain.window)


def _finish(names: tuple[str, ...], Y: np.ndarray, domain: Domain) -> dict[str, np.ndarray]:
    if isinstance(domain, FourierDomain):
        out = np.array(Y, dtype=np.complex128)
        out[..., 0] = out[..., 0].real
        if domain.N % 2 == 0:
            out[..., -1] = out[..., -1].real
    else:
        out = np.array(Y, dtype=np.float64)
        if isinstance(domain, WDMDomain):
            out[..., ~_active(domain)] = 0.0
    if not np.all(np.isfinite(out)):
        raise ValueError(f"the result on the {type(domain).__name__} is not finite (overflow?)")
    return {name: out[i].copy() for i, name in enumerate(names)}


def transform(channel_data: Mapping[str, Any], source: Domain, target: Domain) -> dict[str, np.ndarray]:
    """``channel_data`` on ``source`` re-expressed on ``target`` (module docstring).

    The same domain returns validated copies.  WDM to WDM across another ``(Nf, Nt)`` or
    window goes through the spectrum.  Raises ``TypeError`` for a non-mapping, a non-domain or
    non-numeric values, and ``ValueError`` for a partial domain, another record (``N``,
    ``dt``, or ``t0`` between time domains), an empty dict, a wrong shape, non-finite
    values, complex values on a real domain, a non-real DC (or even-``N`` Nyquist) bin, a
    non-zero inactive WDM cell, or a non-finite result.
    """
    _check_pair(source, target)
    names, X = _stack(channel_data, source)
    Y = X if source == target else _from_rfft(_to_rfft(X, source), target)
    return _finish(names, Y, target)


def log_coordinate_scale(domain: Domain) -> float:
    """``log`` of the factor between ``domain``'s real coordinates and the time samples:
    0 for time, ``log(dt) / 2`` for WDM, ``log(dt) + log(N) / 2`` for Fourier."""
    N, dt = _record(domain, "domain")
    if isinstance(domain, TimeDomain):
        return 0.0
    if isinstance(domain, WDMDomain):
        return 0.5 * math.log(dt)
    return math.log(dt) + 0.5 * math.log(N)
