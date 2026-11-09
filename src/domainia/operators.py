"""Covariances as :class:`domainia.translate.Operator` s, and the matrix of any operator.

* :class:`DenseOperator`: an explicit covariance matrix on the stacked real coordinates of
  a full-record domain.
* :class:`CovarianceOperator`: a :mod:`domainia.covariance` covariance fixed at a noise
  state ``nu`` on a (possibly partial) WDM support, so that
  :func:`~domainia.translate.translate_operator` can see it from time or Fourier.
* :func:`operator_matrix`: the matrix of any ``Operator``, native or translated, in the
  stacked real coordinates of its domain (``K N`` applies: for small ``N``).

The stacked real coordinates of ``K`` channels are the rows of
:func:`domainia.translate.to_coordinates` in ``channel_names`` order, concatenated
(channel-major): ``r = to_coordinates(values, domain, channel_names).reshape(-1)``, of
length ``K N``.  Every operator here passes its ``channel_names`` to ``to_coordinates``, so
a dict built in another channel order still meets the right blocks of the matrix.  NumPy
only.  Importing this module does not import :mod:`domainia.covariance`:
:class:`CovarianceOperator` imports it when it is constructed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from domainia import wilson
from domainia.domains import CellSet, WDMDomain
from domainia.translate import (
    Domain,
    Operator,
    _channel_names,
    _record,
    from_coordinates,
    to_coordinates,
)
from domainia.wilson._checks import REAL_KINDS, require_dtype, require_instance, require_real
from domainia.wilson.backend import NUMPY

if TYPE_CHECKING:
    from domainia.covariance import Covariance

__all__ = ["CovarianceOperator", "DenseOperator", "operator_matrix"]


@dataclass(frozen=True, eq=False)
class DenseOperator:
    """An explicit covariance matrix ``C`` as an :class:`~domainia.translate.Operator`.

    ``matrix`` is ``(K N, K N)`` on the stacked real coordinates of ``domain`` (module
    docstring), ``K = len(channel_names)``, so channels may be correlated.  It must be
    finite and symmetric to ``f`` relative (``max |C - C^T| <= f max |C|``), with
    ``f = 1e-12``, or ``1e-12 eps32 / eps64`` (about 5.4e-4) for a float32 ``matrix``, so
    that a float32 product ``B D B^T`` nobody symmetrised passes.  The operator then stores
    ``C / 2 + C^T / 2`` (which cannot overflow) as ``matrix``, a read-only float64 copy.

    ``numpy.linalg.eigh`` decomposes it once; a non-finite eigenvalue (the matrix's
    spectrum is beyond float64) and the zero matrix (no direction to keep) raise
    ``ValueError``.  One relative tolerance ``rtol`` does both jobs, with
    ``tol = max |lambda| rtol``: an eigenvalue below ``-tol`` raises ``ValueError`` (``C``
    is not positive semi-definite), and the eigenpairs with ``lambda > tol`` are kept
    (``V``, ``Lambda``).  So ``rtol`` drops the directions with an eigenvalue at or below
    ``rtol max |lambda|`` and forgives negative eigenvalues down to ``-rtol max |lambda|``.
    The default ``rtol=None`` is ``K N eps``, the default of ``numpy.linalg.matrix_rank``
    (``eps`` is float32's for a float32 ``matrix`` and float64's otherwise).  A given
    ``rtol`` must be ``>= 0`` and ``< 1`` (from 1 on, no eigenvalue is below ``-tol``); when
    one refuses an eigenvalue within ``K N eps max |lambda|`` of zero, the message says that
    it is round-off and that the default accepts it.  ``op.rtol`` reads back the relative
    tolerance used, never ``None``, so ``dataclasses.replace(op)`` rebuilds the same
    operator.

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
    #: The relative tolerance (class docstring).  ``None``, the default, becomes ``K N eps``
    #: at construction, so after it ``rtol`` is the float used, never ``None``.
    rtol: float = field(default=None, kw_only=True)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        N, _ = _record(self.domain, "domain")
        names = _channel_names(self.channel_names)
        n = len(names) * N
        if isinstance(self.matrix, np.ma.MaskedArray):
            raise TypeError("matrix: masked arrays are refused (numpy.asarray drops the mask)")
        C = np.asarray(self.matrix)
        require_dtype("matrix", C, REAL_KINDS, "a real numeric array")
        eps64 = float(np.finfo(np.float64).eps)
        eps = float(np.finfo(np.float32).eps) if C.dtype == np.float32 else eps64
        f = 1e-12 * (eps / eps64)                       # 1e-12, or about 5.4e-4 for float32
        if C.shape != (n, n):
            raise ValueError(f"matrix has shape {C.shape}; {len(names)} channel(s) of {N} "
                             f"coordinates need shape ({n}, {n})")
        wilson.validate_finite(C, "matrix")
        C = C.astype(np.float64)
        peak = float(np.max(np.abs(C)))
        asym = float(np.max(np.abs(C - C.T)))
        if asym > f * peak:
            raise ValueError(f"matrix is not symmetric: max |C - C^T| is {asym / peak:.2g} of "
                             f"max |C| (more than {f:.2g})")
        if self.rtol is None:
            rtol = n * eps
        else:
            rtol = require_real("rtol", self.rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError(f"rtol must be finite and >= 0; got {rtol!r}")
            if rtol >= 1.0:
                raise ValueError(f"rtol must be < 1; got {rtol!r}: no eigenvalue is below "
                                 f"-rtol max |lambda| when rtol >= 1, so every matrix would pass")
        S = 0.5 * C + 0.5 * C.T                         # 0.5 * (C + C.T) overflows above ~9e307
        w, V = np.linalg.eigh(S)
        if not np.isfinite(w).all():
            raise ValueError("matrix: numpy.linalg.eigh returned a non-finite eigenvalue (the "
                             "spectrum of the matrix is beyond float64); scale it down")
        top = float(np.max(np.abs(w)))
        if top == 0.0:
            raise ValueError("matrix is zero: an operator needs a direction to keep")
        tol = top * rtol                                # the default: top * (n * eps)
        if w[0] < -tol:
            floor = top * (n * eps)
            if self.rtol is not None and w[0] >= -floor:
                raise ValueError(f"matrix is not positive semi-definite at rtol={rtol!r}: its "
                                 f"smallest eigenvalue {float(w[0])!r} is below -tol = {-tol!r}, "
                                 f"but it is within round-off of zero (at or above -K N eps "
                                 f"max |lambda| = {-floor!r}), and the default rtol=None accepts it")
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


@dataclass(frozen=True, eq=False)
class CovarianceOperator:
    """A :mod:`domainia.covariance` covariance ``cov`` fixed at the noise state ``nu`` on the
    WDM ``support``, as an :class:`~domainia.translate.Operator`.

    ``support`` is the :class:`~domainia.domains.WDMDomain` (cells and window) the
    covariance lives on; it may be partial.  The operator's ``domain`` is
    ``WDMDomain.full(support.cells.grid, window=support.window)``.  Let ``K`` be the number
    of channels, ``s = support.size`` and ``P`` the ``(K s, K N)`` selection of the
    support's cells in every channel (channel-major).  The operator is ``C = P^T Sigma P``
    with ``Sigma = cov`` at ``nu``, and its methods are the Moore-Penrose structure of that
    ``C`` (``F = cov.factor(nu)``):

    * ``apply``: ``cov.multiply``; ``solve``: ``F.solve``; ``project``: zeros off the
      support (the identity for a full support);
    * ``quadratic_form``: ``F.quad``; ``log_determinant``: ``F.logdet``;
      ``degrees_of_freedom``: ``K s``.

    Layout: the values go through :func:`~domainia.translate.to_coordinates` (``(K, N)``,
    ``channel_names`` order) and are cut to the support, ``R[:, support.cells.packed_positions()]``
    (``(K, s)``).  That array is flattened channel-major for a single covariance
    (``cov.n == K s``, so the covariance may correlate the channels) and kept as rows for a
    :class:`~domainia.covariance.ChannelStack` (one child per channel, each with
    ``n == s``).  Results are put back with exact zeros off the support and returned
    through :func:`~domainia.translate.from_coordinates`.

    Validation: ``TypeError`` for a wrong kind (``cov`` not a ``Covariance``, ``support``
    not a ``WDMDomain``, ``channel_names`` against the ``ChannelAxis`` rules, a non-numeric
    ``nu``); ``ValueError`` for a size or channel-count mismatch, a covariance ``window``
    (when not ``None``) or ``cells`` (when it has them) other than the support's, a
    non-zero ``n_u``, or ``F.ok`` false (the factor is computed once, eagerly on NumPy, at
    construction).  NumPy only.
    """

    cov: Covariance
    nu: Any
    support: WDMDomain
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        # Imported here, not at module level: ``import domainia`` imports this module and
        # must not import domainia.covariance (the package docstring).
        from domainia.covariance import ChannelStack, Covariance

        cov, support = self.cov, self.support
        require_instance("cov", cov, Covariance, "a domainia.covariance.Covariance")
        require_instance("support", support, WDMDomain, "a WDMDomain")
        names = _channel_names(self.channel_names)
        K, s = len(names), support.size
        stack = isinstance(cov, ChannelStack)
        parts = cov.covs if stack else (cov,)
        if stack:
            if len(parts) != K:
                raise ValueError(f"the ChannelStack has {len(parts)} channels; channel_names has {K}")
            if any(c.n != s for c in parts):
                raise ValueError(f"every ChannelStack channel needs n = support.size = {s}; got "
                                 f"n = {tuple(c.n for c in parts)}")
        elif cov.n != K * s:
            raise ValueError(f"cov.n = {cov.n}, but {K} channel(s) on a support of {s} cells need "
                             f"cov.n = {K * s}")
        for c in parts:
            window = getattr(c, "window", None)
            if window is not None and window != support.window:
                raise ValueError(f"the covariance is built for the window {window.name} but the "
                                 f"support uses {support.window.name}")
            cells = getattr(c, "cells", None)
            if isinstance(cells, CellSet) and cells != support.cells:
                raise ValueError(f"the covariance's cells ({cells!r}) are not support.cells "
                                 f"({support.cells!r})")
        if cov.n_u != 0:
            raise ValueError(f"cov.n_u = {cov.n_u}: an Operator has no nuisance directions")
        nu = np.asarray(self.nu)
        require_dtype("nu", nu, REAL_KINDS, "a real numeric array")
        nu = np.array(nu, dtype=np.float64)
        nu.setflags(write=False)
        F = cov.factor(nu, NUMPY)
        if not bool(F.ok):
            raise ValueError(f"{type(cov).__name__} is not positive definite at nu")
        pos = support.cells.packed_positions()
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "_domain", WDMDomain.full(support.cells.grid, window=support.window))
        object.__setattr__(self, "_factor", F)
        object.__setattr__(self, "_pos", pos)
        object.__setattr__(self, "_stack", stack)

    @property
    def domain(self) -> WDMDomain:
        return self._domain  # type: ignore[attr-defined]

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.channel_names) * self.support.size

    def _b(self, values: Any) -> np.ndarray:
        """``values`` on the support: ``(K, s)`` rows for a ChannelStack, else ``(K s,)``."""
        pos, stack = self._pos, self._stack  # type: ignore[attr-defined]
        R = to_coordinates(values, self.domain, self.channel_names)[:, pos]
        return R if stack else R.reshape(-1)

    def _values(self, y: Any) -> dict[str, np.ndarray]:
        """The support vector ``y`` back on the full domain: exact zeros off the support."""
        K = len(self.channel_names)
        R = np.zeros((K, self.domain.cells.grid.N))
        R[:, self._pos] = np.asarray(y).reshape(K, -1)  # type: ignore[attr-defined]
        return from_coordinates(R, self.domain, self.channel_names)

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self.cov.multiply(self.nu, self._b(values), be=NUMPY))

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self._factor.solve(self._b(values)))  # type: ignore[attr-defined]

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self._b(values))

    def quadratic_form(self, values: Mapping[str, Any]) -> float:
        return float(self._factor.quad(self._b(values)))  # type: ignore[attr-defined]

    def log_determinant(self) -> float:
        return float(self._factor.logdet)  # type: ignore[attr-defined]


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
        raise TypeError(f"op must implement domainia.translate.Operator, got {type(op).__name__}")
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
