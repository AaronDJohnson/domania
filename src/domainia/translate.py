"""Move full-record data between the time, Fourier and WDM domains (spec section 3).

NumPy only: a JAX array is copied to host NumPy (results have float64 dtype; a float32
input keeps its float32 precision), and ``transform`` cannot be traced under ``jax.jit`` or
``jax.grad``, where it raises JAX's ``TracerArrayConversionError`` (a ``TypeError``).  Data
are ``{channel: ndarray}`` dicts (insertion order kept) on one of the full-record domains:

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

A covariance :class:`Operator` lives on one full-record domain.  :func:`translate_operator`
gives a :class:`TranslatedOperator` that sees it from another domain lazily: every call
moves the dicts with :func:`transform` and applies the coordinate Jacobian
``Delta = log_coordinate_scale(target) - log_coordinate_scale(native)``.  The
real-coordinate convention (sqrt(2) Re/Im interiors, real endpoints, masks) stays inside
the native operator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from domainia import wilson
from domainia.domains import FourierDomain, TimeDomain, WDMDomain

__all__ = ["Domain", "Operator", "TranslatedOperator", "log_coordinate_scale", "transform",
           "translate_operator"]

#: The domains :func:`transform` and :func:`log_coordinate_scale` take (full-record ones only).
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
                             f"active cell, scale 1); got scale={float(domain.scale)!r}")
        return g.N, g.dt
    raise TypeError(f"{role} must be a full-record domain (TimeDomain, FourierDomain.full or "
                    f"WDMDomain.full); got {type(domain).__name__}")


def _check_pair(source: Any, target: Any, roles: tuple[str, str] = ("source", "target")) -> None:
    """Both full-record domains of one record; ``roles`` name the two in the messages."""
    a, b = roles
    N_src, dt_src = _record(source, a)
    N_dst, dt_dst = _record(target, b)
    if N_src != N_dst or dt_src != dt_dst:
        raise ValueError(f"{a} and {b} describe different records: N={N_src}, "
                         f"dt={float(dt_src)!r} vs N={N_dst}, dt={float(dt_dst)!r}")
    if isinstance(source, TimeDomain) and isinstance(target, TimeDomain) and source.t0 != target.t0:
        raise ValueError(f"{a} and {b} time domains start at different t0 "
                         f"({float(source.t0)!r} vs {float(target.t0)!r}): transform does not "
                         f"shift records")


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


def _relative(value: float, a: np.ndarray) -> str:
    """``|value|`` as a fraction of the channel's largest ``|a|``: the canonical-form refusals
    tell round-off (~1e-16) from another convention (near 1) by it."""
    peak = max(float(np.max(np.abs(a))), float(np.finfo(np.float64).tiny))
    return f"{abs(value) / peak:.2g} of the channel's largest |value|"


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
        if isinstance(value, np.ma.MaskedArray):
            raise TypeError(f"channel {name!r}: masked arrays are refused (numpy.asarray drops the "
                            f"mask, so the masked values would leak into the result); fill or "
                            f"excise the masked samples first")
        a = np.asarray(value)
        if a.dtype.kind not in "iufc":
            raise TypeError(f"channel {name!r}: values must be numeric; got dtype {a.dtype}")
        if a.dtype.kind == "c" and not isinstance(domain, FourierDomain):
            raise ValueError(f"channel {name!r}: complex values on a real {kind}")
        if a.shape != shape:
            raise ValueError(f"channel {name!r}: shape {a.shape}, but this {kind} needs shape {shape}")
        try:
            wilson.validate_finite(a, f"channel {name!r}")
        except ValueError as err:
            if not isinstance(domain, WDMDomain):
                raise
            first = int(np.flatnonzero(~np.isfinite(a))[0])
            m, n = (int(i) for i in np.unravel_index(first, shape))
            raise ValueError(f"{err} (Index {first} is WDM cell (m={m}, n={n}) of the dense "
                             f"{shape} layout.)") from None
        if isinstance(domain, FourierDomain):
            a = a.astype(np.complex128)
            even = domain.N % 2 == 0
            ends = ((0, "DC"), (shape[0] - 1, "Nyquist")) if even else ((0, "DC"),)
            fix = (f"take the real part of the DC/Nyquist bins first, e.g. "
                   f"X[..., [0, {shape[0] - 1}]] = X[..., [0, {shape[0] - 1}]].real" if even else
                   "take the real part of the DC bin first, e.g. X[..., 0] = X[..., 0].real")
            for k, label in ends:
                if a[k].imag != 0.0:
                    im = float(a[k].imag)
                    raise ValueError(f"channel {name!r}: rfft bin {k} ({label}) must be real; its "
                                     f"imaginary part is {im!r} ({_relative(im, a)}); if that is "
                                     f"round-off, {fix}; a value near the peak means another "
                                     f"layout or FFT convention")
        else:
            a = a.astype(np.float64)
            if isinstance(domain, WDMDomain):
                bad = (a != 0.0) & ~_active(domain)
                if np.any(bad):
                    m, n = (int(i) for i in np.argwhere(bad)[0])
                    g = domain.cells.grid
                    v = float(a[m, n])
                    raise ValueError(f"channel {name!r}: inactive WDM cell (m={m}, n={n}) is "
                                     f"{v!r} ({_relative(v, a)}); inactive cells must be exactly "
                                     f"0; if that is round-off, zero them first, e.g. "
                                     f"w[..., ~domainia.wilson.active_mask({g.Nf}, {g.Nt})] = 0; "
                                     f"a value near the peak means another layout or active-cell "
                                     f"convention")
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
    window goes through the spectrum.  ``t0`` is compared only between two time domains;
    Fourier and WDM domains carry no time origin, so the caller owns the epoch.

    Inputs must be canonical, exactly: inactive WDM cells exactly 0, and the Fourier DC bin
    (and the even-``N`` Nyquist bin) exactly real.  Raw ``wilson.forward*``/``wilson.inverse*``
    outputs and slices of a full FFT (``numpy.fft.fft(x)[:N // 2 + 1]``) carry ~1e-16
    residue there on many grids, so they are refused; the refusal names the fix (zero the
    inactive cells, or take the real part of the end bins).  The canonical routes are
    ``transform`` itself and ``numpy.fft.rfft``.

    Raises ``TypeError`` for a non-mapping, a non-domain, a masked array (any
    ``numpy.ma.MaskedArray``, even one with no masked entries) or non-numeric values, and
    ``ValueError`` for a partial domain, another record (``N``, ``dt``, or ``t0`` between time
    domains), an empty dict, a wrong shape, non-finite values, complex values on a real
    domain, a non-real DC (or even-``N`` Nyquist) bin, a non-zero inactive WDM cell, or a
    non-finite result.
    """
    _check_pair(source, target)
    names, X = _stack(channel_data, source)
    Y = X if source == target else _from_rfft(_to_rfft(X, source), target)
    return _finish(names, Y, target)


def log_coordinate_scale(domain: Domain) -> float:
    """``s``, the log of the scale per real coordinate: ``domain``'s real coordinates (module
    docstring) are ``exp(s) Q x`` for the time samples ``x`` and an orthogonal ``Q``, so every
    energy is ``exp(2 s) |x|^2``.

    0 for time, ``log(dt) / 2`` for WDM (any admissible window), ``log(dt) + log(N) / 2`` for
    Fourier (odd ``N`` too).  Raises like :func:`transform` for a domain it does not handle:
    ``ValueError`` for a partial Fourier or WDM domain (or a scaled WDM one), ``TypeError``
    for anything else.
    """
    N, dt = _record(domain, "domain")
    if isinstance(domain, TimeDomain):
        return 0.0
    if isinstance(domain, WDMDomain):
        return 0.5 * math.log(dt)
    return math.log(dt) + 0.5 * math.log(N)


@runtime_checkable
class Operator(Protocol):
    """A covariance on a full-record domain, acting on ``{channel: array}`` dicts in that
    domain's layout.

    The methods act on the stacked real coordinates of all channels: ``r`` stacks every
    channel's real coordinates in the domain (module docstring), and ``C`` is a symmetric
    positive semi-definite covariance on that stacked vector, whose range is the retained
    subspace.  Channels may be correlated at each point (``C`` need not be block-diagonal
    over channels); scalar results are totals over the stacked vector.

    * ``apply``: ``C r``.
    * ``project``: the orthogonal projector onto the retained subspace, applied to ``r``.
    * ``solve``: ``C^+ r`` with ``C^+`` the Moore-Penrose inverse: the inverse on the retained
      subspace, zero on the excluded directions.
    * ``quadratic_form``: ``r^T C^+ r``.
    * ``log_determinant``: the log pseudo-determinant (the sum of the logs of the non-zero
      eigenvalues of ``C``).
    * ``degrees_of_freedom``: the rank of ``C`` (the retained dimension).

    A full-rank operator has the identity as ``project`` and the inverse as ``solve``.  The
    retained subspace need not be a set of coordinates: a time gap seen in WDM is spread over
    many cells.

    The dicts that ``apply``, ``solve`` and ``project`` return must meet :func:`transform`'s
    input rules exactly, because :class:`TranslatedOperator` sends them through it unchanged
    (no tolerance, no repair): inactive WDM cells exactly 0, and the Fourier DC bin (and the
    even-``N`` Nyquist bin) exactly real.  An output computed with the ``domainia.wilson``
    kernels or a full FFT can carry ~1e-16 there, and ``TranslatedOperator`` then raises a
    ``ValueError`` that names the native method and quotes ``transform``'s refusal.  Meet the
    rules before returning: zero the inactive cells
    (``w[..., ~domainia.wilson.active_mask(Nf, Nt)] = 0``) and take the real part of the end
    bins (``X[..., 0] = X[..., 0].real``, likewise ``X[..., -1]`` for even ``N``).
    """

    @property
    def domain(self) -> Domain: ...

    @property
    def channel_names(self) -> tuple[str, ...]: ...

    @property
    def degrees_of_freedom(self) -> int: ...

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]: ...

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]: ...

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]: ...

    def quadratic_form(self, values: Mapping[str, Any]) -> float: ...

    def log_determinant(self) -> float: ...


@dataclass(frozen=True, eq=False)
class TranslatedOperator:
    """``native`` seen from ``target`` (module docstring).  It stores only these two: no
    dense matrix is ever built.  It holds a reference to ``native`` (a view, not a copy).

    * ``apply(x)`` = to_target(native.apply(to_native(x))) * exp(2 Delta)
    * ``solve(x)`` = to_target(native.solve(to_native(x))) * exp(-2 Delta)
    * ``project(x)`` = to_target(native.project(to_native(x)))
    * ``quadratic_form(x)`` = native.quadratic_form(to_native(x))
    * ``log_determinant()`` = native.log_determinant() + 2 * degrees_of_freedom * Delta

    ``to_target`` is :func:`transform`, so a native output that breaks the output contract
    (:class:`Operator`) is refused with a plain ``ValueError`` or ``TypeError`` (as
    ``transform`` refused it), naming the native method, quoting ``transform``'s refusal and
    chaining it as the cause: ``ValueError`` off the canonical form, for a wrong shape or
    non-finite values, ``TypeError`` for a masked array, non-numeric values or JAX tracers.  A
    native output with other channels than the operator's raises ``ValueError``, and one that
    is not a mapping ``TypeError``, likewise naming the native method.
    """

    native: Operator
    target: Domain

    def __post_init__(self) -> None:
        if isinstance(self.native, TranslatedOperator) or not isinstance(self.native, Operator):
            raise TypeError(f"native must be a native domainia.translate.Operator (translate_operator "
                            f"re-targets a TranslatedOperator); got {type(self.native).__name__}")
        _check_pair(self.native.domain, self.target, roles=("native.domain", "target"))

    @property
    def domain(self) -> Domain:
        return self.target

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.native.channel_names)

    @property
    def degrees_of_freedom(self) -> int:
        return int(self.native.degrees_of_freedom)

    def _delta(self) -> float:
        return log_coordinate_scale(self.target) - log_coordinate_scale(self.native.domain)

    def _to_native(self, values: Any) -> dict[str, np.ndarray]:
        if not isinstance(values, Mapping):
            raise TypeError(f"values must be a mapping {{channel: array}}; got {type(values).__name__}")
        if set(values) != set(self.channel_names):
            raise ValueError(f"values have channels {tuple(values)}; the operator has channels "
                             f"{self.channel_names}")
        return transform(values, self.target, self.native.domain)

    def _to_target(self, method: str, values: Any, factor: float = 1.0) -> dict[str, np.ndarray]:
        """``native.<method>``'s output ``values`` on the target, times ``factor``; a bad
        output is blamed on the native method, not on the caller."""
        who = f"{type(self.native).__name__}.{method}"
        if not isinstance(values, Mapping):
            raise TypeError(f"{who} returned a {type(values).__name__}, not a {{channel: array}} "
                            f"mapping")
        if set(values) != set(self.channel_names):
            raise ValueError(f"{who} returned channels {tuple(values)}; the operator has channels "
                             f"{self.channel_names}")
        try:
            out = transform(values, self.native.domain, self.target)
        except (TypeError, ValueError) as err:
            # The plain type: a subclass may not take a message (JAX's TracerArrayConversionError
            # takes the tracer).
            kind = TypeError if isinstance(err, TypeError) else ValueError
            raise kind(f"{who} returned a dict that breaks the Operator output contract "
                       f"(see domainia.translate.Operator): {err}") from err
        return out if factor == 1.0 else {k: v * factor for k, v in out.items()}

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._to_target("apply", self.native.apply(self._to_native(values)),
                               np.exp(2.0 * self._delta()))

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._to_target("solve", self.native.solve(self._to_native(values)),
                               np.exp(-2.0 * self._delta()))

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._to_target("project", self.native.project(self._to_native(values)))

    def quadratic_form(self, values: Mapping[str, Any]) -> float:
        return float(self.native.quadratic_form(self._to_native(values)))

    def log_determinant(self) -> float:
        return float(self.native.log_determinant()) + 2.0 * self.degrees_of_freedom * self._delta()


def translate_operator(op: Operator, target: Domain) -> Operator:
    """``op`` seen from ``target``.  A :class:`TranslatedOperator` is re-targeted (its native
    operator is reused, never nested); a target equal to the native domain returns the native
    operator itself, and a ``TranslatedOperator`` already on ``target`` returns itself."""
    if not isinstance(op, Operator):
        raise TypeError(f"op must implement domainia.translate.Operator; got {type(op).__name__}")
    native = op.native if isinstance(op, TranslatedOperator) else op
    _check_pair(native.domain, target, roles=("native.domain", "target"))
    if target == native.domain:
        return native
    if isinstance(op, TranslatedOperator) and target == op.target:
        return op
    return TranslatedOperator(native, target)
