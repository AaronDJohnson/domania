"""Per-point K x K noise covariance from a model (domainia 3.0 spec, section 3).

A model returns the one-sided K x K cross-spectral density [data units^2/Hz], oriented so
that ``E[X_a conj(X_b)] = (T_obs / 2) S_ab(f_k)`` at an interior rfft bin, with
``S[b, a] = conj(S[a, b])``.  Each output is linear in the model, and exactly Hermitian with a
real diagonal at every point.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .wilson._checks import require_callable, require_dtype, require_int, validate_dt

__all__ = ["frequency_covariance"]

#: The model check's tolerance at a point, as a fraction of that point's largest |S_cd|.
TOL = 1e-12


def _hermitian(out: Any, points: tuple[int, ...], name: str) -> np.ndarray:
    """``H = (S + S^H) / 2`` at each point of a model's output ``S = out``, after the check.

    ``S`` must be numeric (else ``TypeError``), of shape ``(K, K, *p)`` with K >= 1 and ``p``
    equal to ``points`` up to axes of size 1, and finite (else ``ValueError``).  At each point,
    with ``tol = TOL * max_cd |S_cd|`` there, ``|S_ab - conj(S_ba)|`` (a != b), ``|Im S_aa|``
    and ``-Re S_aa`` must not exceed ``tol`` (else ``ValueError``, naming the first entry), so
    a point whose entries are all 0 passes.  ``H`` has ``S``'s shape, so a size-1 axis is never
    expanded, and is float64 for a real ``S``, complex128 otherwise.
    """
    S = np.asarray(out)
    require_dtype(name, S, "iufc", "a numeric array")
    if (S.ndim != 2 + len(points) or not 0 < S.shape[0] == S.shape[1]
            or any(s not in (1, p) for s, p in zip(S.shape[2:], points))):
        shape = ", ".join(map(str, points))
        raise ValueError(f"{name} must have shape (K, K, {shape}) with K >= 1 (an axis after "
                         f"the first two may be 1); got {S.shape}")
    if not np.isfinite(S).all():
        at = tuple(np.argwhere(~np.isfinite(S))[0].tolist())
        raise ValueError(f"{name} is not finite at {at}")
    S = S.astype(np.complex128 if S.dtype.kind == "c" else np.float64)
    peak = np.max(np.abs(S), axis=(0, 1))
    asym = np.abs(S - np.conj(np.swapaxes(S, 0, 1)))
    asym[np.diag_indices(S.shape[0])] = 0.0
    diag = np.moveaxis(np.diagonal(S), -1, 0)
    for what, value in (("|S[{0}, {1}] - conj(S[{1}, {0}])|", asym),
                        ("|Im S[{0}, {0}]|", np.abs(diag.imag)), ("-Re S[{0}, {0}]", -diag.real)):
        bad = np.argwhere(value > TOL * peak)
        if bad.size:
            at = tuple(bad[0].tolist())
            k = value.ndim - peak.ndim
            raise ValueError(f"{name} at point {at[k:]}: {what.format(*at[:k])} is "
                             f"{value[at] / peak[at[k:]]:.2g} of max |S_cd| there; at most "
                             f"{TOL:g} is accepted")
    return 0.5 * (S + np.conj(np.swapaxes(S, 0, 1)))


def frequency_covariance(S: Callable[[np.ndarray], Any], *, N: int, dt: float) -> np.ndarray:
    """The covariance per rfft bin of ``X = dt * numpy.fft.rfft(x)`` under a stationary model.

    ``S(f)`` is called once, with ``f = numpy.fft.rfftfreq(N, dt)`` (``f_k = k / (N dt)``), and
    returns shape ``(K, K, N // 2 + 1)``, or ``(K, K, 1)`` for a constant model; K is read from
    it.  It is checked and projected to ``H = (S + S^H) / 2`` per bin (:func:`_hermitian`).

    Returns ``C[a, b, k] = (N dt / 2) H_ab(f_k)``, complex128 of shape ``(K, K, N // 2 + 1)``:
    ``E[X_a[k] conj(X_b[k])]`` for a periodic record, the Whittle approximation otherwise.  It is
    real at DC and at even-``N`` Nyquist, where a real record has no imaginary part.

    Raises ``TypeError`` for a non-callable ``S``, a non-integer ``N`` or a non-numeric output,
    and ``ValueError`` for ``N < 1``, a bad ``dt`` or an output that fails the check.
    """
    require_callable("S", S)
    N = require_int("N", N)
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    validate_dt(dt)
    f = np.fft.rfftfreq(N, float(dt))
    H = _hermitian(S(f), f.shape, "S(f)")
    C = np.empty(H.shape[:2] + f.shape, np.complex128)
    C[...] = (0.5 * N * float(dt)) * H
    C[..., 0] = C[..., 0].real
    if N % 2 == 0:
        C[..., -1] = C[..., -1].real
    return C
