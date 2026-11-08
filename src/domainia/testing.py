"""Dense oracles for tests and small grids (NumPy only, never on a
likelihood path).  Every oracle here is ``O(N^2)`` memory or worse: keep ``N`` <~ 2048.

* :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a cell set
  (``wilson.matrix.active_W``) and ``W_K Sigma W_K^T``.
* :func:`eq45_dense`: Eq. (45) on a cell set by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).
* :func:`blocktri_dense`: a ragged block-tridiagonal matrix assembled densely.
* :func:`hhat`, :func:`eq54_covariance`, :func:`dense_lambda_generic`: the generic
  Eq. (54) covariance of a modulated stationary process on Fourier bins and its WDM
  form (ported from the galactic-binary toy model, which does not ship).

:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.

Test windows (``domainia.wilson.windows``):

* :func:`smoothstep`: a non-Meyer ramp for :class:`~domainia.wilson.RampWindow`.
* :func:`wide_profile`, :data:`ALT_WINDOW`: a non-Meyer admissible profile and the
  :class:`~domainia.wilson.SampledWindow` built from it.

Test operators (``domainia.translate``):

* :class:`PointwiseOperator`: a minimal native :class:`~domainia.translate.Operator`
  (independent time samples or WDM cells with per-channel variances).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from domainia import wilson
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
from domainia.wilson.windows import MEYER, SampledWindow, Window

from .domains import CellSet, TimeDomain, WDMDomain

__all__ = [
    "ALT_WINDOW",
    "PointwiseOperator",
    "blocktri_dense",
    "dense_W",
    "dense_lambda_generic",
    "eq45_dense",
    "eq54_covariance",
    "hhat",
    "smoothstep",
    "wdm_congruence",
    "wide_profile",
]

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


def dense_W(cells: CellSet, *, window: Window = MEYER) -> Array:
    """``(cells.size, N)``: row ``p`` is cell ``p``'s atom sampled in time, so ``W x`` are
    the cells of ``x`` (``wilson.matrix.active_W`` rows, ``W_a W_a^T = dt I``) in the
    Wilson basis of ``window``."""
    g = cells.grid
    return wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=window)[cells.packed_positions()]


def wdm_congruence(Sigma: Array, cells: CellSet, *, window: Window = MEYER) -> Array:
    """``W_K Sigma W_K^T`` for a time-domain covariance ``Sigma (N, N)`` (``W_K`` is
    :func:`dense_W` with ``window``)."""
    W = dense_W(cells, window=window)
    return W @ np.asarray(Sigma, dtype=float) @ W.T


def eq45_dense(cells: CellSet, Sn: Callable[[Any, Any], Any], *, P: int = 24,
               t_max: float | None = None, window: Window = MEYER) -> Array:
    """Eq. (45) on ``cells`` by the direct sum, ``Sn(f, t)`` elementwise (``A_n`` inside),
    with the atoms of ``window``."""
    g = cells.grid
    plan = eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max, window=window)
    return eq45_direct(plan, Sn)


def blocktri_dense(diag: Sequence[Array], up: Sequence[Array | None]) -> Array:
    """The symmetric matrix with diagonal blocks ``diag`` and upper blocks ``up``."""
    sizes = [np.asarray(d).shape[0] for d in diag]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    C = np.zeros((off[-1], off[-1]))
    for i, D in enumerate(diag):
        C[off[i]:off[i + 1], off[i]:off[i + 1]] = D
    for i, U in enumerate(up):
        if U is not None:
            C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = U
            C[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = np.asarray(U).T
    return C


def hhat(mu: Array, basis_L: Array, lags: NDArray[np.int64]) -> CArray:
    r""":math:`\hat h[q] = \frac1L \sum_t \sqrt{g(t)}\,e^{-2\pi i q t/L}` at integer ``lags``,
    ``g = exp(mu @ basis_L)`` on the ``L``-point decimation grid."""
    L = basis_L.shape[1]
    R = np.fft.rfft(np.exp(0.5 * (np.asarray(mu) @ basis_L))) / L
    lags = np.asarray(lags)
    out = R[np.abs(lags)]
    return np.where(lags < 0, np.conj(out), out)


def eq54_covariance(s_I: Array, s_G_src: Array, h: CArray, *, margin: int) -> CArray:
    r"""Eq. (54) on ``G`` grid bins: :math:`M = \mathrm{diag}(s_I) + A A^\dagger`,
    :math:`A_{j\kappa} = \sqrt{s_G(\kappa)}\,\hat h[j - \kappa]` over the source columns
    :math:`\kappa = -\mathrm{margin} \ldots G + \mathrm{margin} - 1` (grid units).

    ``s_G_src`` has ``G + 2 margin`` entries; ``h`` is :math:`\hat h` at the lags
    ``-(G + margin) .. G + margin``.
    """
    s_I = np.asarray(s_I, dtype=float)
    G = s_I.size
    s_G_src = np.asarray(s_G_src, dtype=float)
    if s_G_src.size != G + 2 * margin:
        raise ValueError(f"s_G_src needs G + 2 margin = {G + 2 * margin} entries; got {s_G_src.size}")
    h = np.asarray(h)
    if h.size != 2 * (G + margin) + 1:
        raise ValueError(f"h needs the {2 * (G + margin) + 1} lags -(G+margin)..G+margin; got {h.size}")
    j = np.arange(G)[:, None]
    k = np.arange(-margin, G + margin)[None, :]
    A = np.sqrt(s_G_src)[None, :] * h[(j - k) + (G + margin)]
    return np.diag(s_I).astype(complex) + A @ A.conj().T


def dense_lambda_generic(U: CArray, s_I: Array, s_G_src: Array, h: CArray, *,
                         margin: int) -> Array:
    r""":math:`\Lambda = \tfrac12\mathrm{Re}(U M U^\dagger)` with ``M`` from
    :func:`eq54_covariance` and ``U`` the unit-norm atoms on the ``G`` grid bins."""
    U = np.asarray(U)
    M = eq54_covariance(s_I, s_G_src, h, margin=margin)
    return 0.5 * np.real(U @ M @ U.conj().T)


def smoothstep(x: Any) -> Array:
    """``3 x^2 - 2 x^3`` on [0, 1] (clipped): a ramp with ``nu(x) + nu(1 - x) = 1``."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def wide_profile(u: Any) -> Array:
    """A non-Meyer admissible profile ``psi(u)``, ``u = f / dF``: 1 for ``|u| < 1/8``, a
    smoothstep cosine rolloff to 0 at ``|u| = 7/8``, 0 beyond.  On [0, 1],
    ``psi(u)^2 + psi(u - 1)^2 = 1`` because ``smoothstep(1 - x) = 1 - smoothstep(x)``."""
    a = np.abs(np.asarray(u, dtype=float))
    x = np.clip((a - 0.125) / 0.75, 0.0, 1.0)
    return np.where(a < 0.125, 1.0, np.where(a < 0.875, np.cos(np.pi / 2.0 * smoothstep(x)), 0.0))


#: A non-default admissible window for tests (other flat width, other ramp than Meyer).
ALT_WINDOW = SampledWindow(wide_profile, "wide-smoothstep")


@dataclass(frozen=True, eq=False)
class PointwiseOperator:
    """A minimal native :class:`domainia.translate.Operator` for tests: independent samples
    (``TimeDomain``) or independent active cells (``WDMDomain.full``, dense layout,
    inactive cells masked out) with per-channel variances."""

    domain: TimeDomain | WDMDomain
    variances: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.domain, TimeDomain):
            mask = np.ones(self.domain.N, dtype=bool)
        elif isinstance(self.domain, WDMDomain):
            g = self.domain.cells.grid
            if self.domain.cells.size != g.N:
                raise ValueError("PointwiseOperator needs every active cell: use WDMDomain.full(grid)")
            mask = wilson.active_mask(g.Nf, g.Nt)
        else:
            raise TypeError(f"PointwiseOperator needs a TimeDomain or WDMDomain; got "
                            f"{type(self.domain).__name__}")
        if not isinstance(self.variances, Mapping):
            raise TypeError(f"variances must be a {{channel: array}} mapping; got "
                            f"{type(self.variances).__name__}")
        if not self.variances:
            raise ValueError("variances must be a non-empty {channel: array} mapping")
        var = {}
        for name, v in self.variances.items():
            if not isinstance(name, str):
                raise TypeError(f"channel names must be strings; got {name!r}")
            a = np.array(v, dtype=np.float64)
            if a.shape != mask.shape:
                raise ValueError(f"channel {name!r}: variances have shape {a.shape}; need {mask.shape}")
            if not np.all(np.isfinite(a[mask])) or np.any(a[mask] <= 0.0):
                raise ValueError(f"channel {name!r}: variances must be finite and positive")
            # 1 at the inactive cells: np.where evaluates both branches, so solve and
            # quadratic_form would otherwise divide by a 0 or NaN there and warn.
            a = np.where(mask, a, 1.0)
            a.setflags(write=False)
            var[name] = a
        object.__setattr__(self, "variances", var)
        object.__setattr__(self, "_mask", mask)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.variances)

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.variances) * int(self._mask.sum())

    def _values(self, values: Any) -> dict[str, Array]:
        if not isinstance(values, Mapping):
            raise TypeError(f"values must be a mapping; got {type(values).__name__}")
        if set(values) != set(self.variances):
            raise ValueError(f"values have channels {tuple(values)}; need {self.channel_names}")
        out = {}
        for name in self.variances:
            a = np.asarray(values[name], dtype=np.float64)
            if a.shape != self._mask.shape:
                raise ValueError(f"channel {name!r}: shape {a.shape}; need {self._mask.shape}")
            out[name] = a
        return out

    def apply(self, values: Any) -> dict[str, Array]:
        return {k: np.where(self._mask, self.variances[k] * v, 0.0) for k, v in self._values(values).items()}

    def solve(self, values: Any) -> dict[str, Array]:
        return {k: np.where(self._mask, v / self.variances[k], 0.0) for k, v in self._values(values).items()}

    def project(self, values: Any) -> dict[str, Array]:
        return {k: np.where(self._mask, v, 0.0) for k, v in self._values(values).items()}

    def quadratic_form(self, values: Any) -> float:
        return float(sum(np.sum(np.where(self._mask, v * v / self.variances[k], 0.0))
                         for k, v in self._values(values).items()))

    def log_determinant(self) -> float:
        return float(sum(np.sum(np.log(a[self._mask])) for a in self.variances.values()))
