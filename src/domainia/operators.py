"""Explicit covariance matrices as :class:`domainia.translate.Operator` s, and the matrix of
any operator (spec section 2).

* :class:`DenseOperator`: an explicit covariance matrix on the stacked real coordinates of
  a full-record domain.
* :func:`operator_matrix`: the matrix of any ``Operator``, native or translated, in the
  stacked real coordinates of its domain (``K N`` applies: for small ``N``).

The stacked real coordinates of ``K`` channels are the rows of
:func:`domainia.translate.to_coordinates` in ``channel_names`` order, concatenated
(channel-major): ``r = to_coordinates(values, domain, channel_names).reshape(-1)``, of
length ``K N``.  Every operator here passes its ``channel_names`` to ``to_coordinates``, so
a dict built in another channel order still meets the right blocks of the matrix.  NumPy
only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from domainia import wilson
from domainia.translate import (
    Domain,
    Operator,
    _channel_names,
    _record,
    from_coordinates,
    to_coordinates,
)

__all__ = ["DenseOperator", "operator_matrix"]


@dataclass(frozen=True, eq=False)
class DenseOperator:
    """An explicit covariance matrix ``C`` as an :class:`~domainia.translate.Operator`.

    ``matrix`` is ``(K N, K N)`` on the stacked real coordinates of ``domain`` (module
    docstring), ``K = len(channel_names)``, so channels may be correlated.  It must be
    finite and symmetric to 1e-12 relative (``max |C - C^T| <= 1e-12 max |C|``); the
    operator then stores ``(C + C^T) / 2`` as ``matrix``, a read-only float64 copy.

    ``numpy.linalg.eigh`` decomposes it once.  An eigenvalue below ``-tol`` raises
    ``ValueError`` (``C`` is not positive semi-definite); the eigenpairs with
    ``lambda > tol`` are kept (``V``, ``Lambda``).  ``tol = max |lambda| K N eps`` (the
    default of ``numpy.linalg.matrix_rank``), or ``rtol max |lambda|`` when ``rtol`` is
    given.

    * ``apply``: ``C r``; ``solve``: ``V Lambda^-1 V^T r``; ``project``: ``V V^T r``;
    * ``quadratic_form``: ``r^T solve(r)``; ``log_determinant``: ``sum log lambda``;
    * ``degrees_of_freedom``: the number of kept eigenpairs.

    Inputs go through :func:`~domainia.translate.to_coordinates` (``transform``'s input
    rules; channels other than ``channel_names`` raise ``ValueError``), and outputs come from
    :func:`~domainia.translate.from_coordinates`, so they meet the ``Operator`` output
    contract exactly.  Raises ``TypeError`` for a wrong kind (a masked, non-numeric or
    complex ``matrix``, a non-domain, bad ``channel_names``, a non-real ``rtol``) and
    ``ValueError`` for a wrong value.
    """

    matrix: Any
    domain: Domain
    channel_names: tuple[str, ...]
    rtol: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        N, _ = _record(self.domain, "domain")
        names = _channel_names(self.channel_names)
        n = len(names) * N
        if isinstance(self.matrix, np.ma.MaskedArray):
            raise TypeError("matrix: masked arrays are refused (numpy.asarray drops the mask)")
        C = np.asarray(self.matrix)
        eps = float(np.finfo(C.dtype).eps) if C.dtype.kind == "f" and C.dtype.itemsize < 8 \
            else float(np.finfo(np.float64).eps)
        if C.dtype.kind not in "iuf":
            raise TypeError(f"matrix must be a real numeric array; got dtype {C.dtype}")
        if C.shape != (n, n):
            raise ValueError(f"matrix has shape {C.shape}; {len(names)} channel(s) of {N} "
                             f"coordinates need shape ({n}, {n})")
        wilson.validate_finite(C, "matrix")
        C = C.astype(np.float64)
        peak = float(np.max(np.abs(C)))
        asym = float(np.max(np.abs(C - C.T)))
        if asym > 1e-12 * peak:
            raise ValueError(f"matrix is not symmetric: max |C - C^T| is {asym / peak:.2g} of "
                             f"max |C| (more than 1e-12)")
        rtol = self.rtol
        if rtol is not None:
            if isinstance(rtol, bool) or not isinstance(rtol, (int, float, np.integer, np.floating)):
                raise TypeError(f"rtol must be a real number or None; got {type(rtol).__name__}")
            rtol = float(rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError(f"rtol must be finite and >= 0; got {rtol!r}")
        S = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(S)
        top = float(np.max(np.abs(w)))
        tol = top * n * eps if rtol is None else rtol * top
        if w[0] < -tol:
            raise ValueError(f"matrix is not positive semi-definite: its smallest eigenvalue "
                             f"{float(w[0])!r} is below -tol = {-tol!r}")
        keep = w > tol
        V, w = V[:, keep], w[keep]
        for a in (S, V, w):
            a.setflags(write=False)
        object.__setattr__(self, "matrix", S)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "_V", V)
        object.__setattr__(self, "_w", w)

    @property
    def degrees_of_freedom(self) -> int:
        return int(self._w.size)  # type: ignore[attr-defined]

    def _r(self, values: Any) -> np.ndarray:
        return to_coordinates(values, self.domain, self.channel_names).reshape(-1)

    def _values(self, r: np.ndarray) -> dict[str, np.ndarray]:
        return from_coordinates(r.reshape(len(self.channel_names), -1), self.domain, self.channel_names)

    def _solve(self, r: np.ndarray) -> np.ndarray:
        V, w = self._V, self._w  # type: ignore[attr-defined]
        return V @ ((V.T @ r) / w)

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self.matrix @ self._r(values))

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self._solve(self._r(values)))

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        V = self._V  # type: ignore[attr-defined]
        return self._values(V @ (V.T @ self._r(values)))

    def quadratic_form(self, values: Mapping[str, Any]) -> float:
        r = self._r(values)
        return float(r @ self._solve(r))

    def log_determinant(self) -> float:
        return float(np.sum(np.log(self._w)))  # type: ignore[attr-defined]


def operator_matrix(op: Operator) -> np.ndarray:
    """``(K N, K N)``: the matrix of ``op.apply`` on the stacked real coordinates of
    ``op.domain`` (module docstring), for any :class:`~domainia.translate.Operator`, native
    or translated.

    Column ``j`` is the stacked ``to_coordinates(op.apply(from_coordinates(e_j)))``, with
    ``op.channel_names`` as the channel order.  It costs ``K N`` applies, so it is for small
    ``N`` (tests and dense references).  Translating a matrix:
    ``operator_matrix(translate_operator(DenseOperator(C, time, ("A",)), wdm))`` is
    ``W C W^T``.
    """
    if not isinstance(op, Operator):
        raise TypeError(f"op must implement domainia.translate.Operator; got {type(op).__name__}")
    names = _channel_names(op.channel_names)
    N, _ = _record(op.domain, "op.domain")
    K = len(names)
    out = np.empty((K * N, K * N))
    e = np.zeros(K * N)
    for j in range(K * N):
        e[j] = 1.0
        y = op.apply(from_coordinates(e.reshape(K, N), op.domain, names))
        out[:, j] = to_coordinates(y, op.domain, names).reshape(-1)
        e[j] = 0.0
    return out
