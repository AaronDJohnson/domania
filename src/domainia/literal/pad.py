r"""The text's zero padding (WDM II Sec. VII D, Eqs. 92-94; spec 10.1 ruling 2: 8 + 8).

"We extend the grid by ``p`` WDM time columns by adding ``p N_f / 2`` zero samples to each end of
the time-domain data.  The values of dT and dF along with the times of the columns holding the
data are unchanged."  :class:`CenteredPad` is that extended grid: ``Nt' = Nt + p`` columns, the
data at columns ``p/2 .. p/2 + Nt - 1`` (``layout="centered"``; ``"end"`` puts the ``p`` columns
after the data, the chains' layout), column ``j`` at time ``t_j = (j - p/2) dT``, so the padding
columns sit at ``t < 0`` and ``t >= T`` (:func:`time_shifted` moves a spectrum's clock there).

With ``chi`` the data mask on the extended record and ``W'`` the extended grid's packed WDM
matrix (``W' W'^T = dt I``):

* the data are ``w'_K = (W' chi x)_K`` on the kept cells ``K`` (:meth:`CenteredPad.lift_map`);
* ``M = W' chi W'^T / dt = I - B B^T`` with ``B = W'_{., P} / sqrt(dt)`` the padding samples'
  impulse responses (:meth:`CenteredPad.padding_impulses`; exact algebra, ``B^T B = I_P``);
* the edge set ``E`` (:meth:`CenteredPad.edge_set`): the kept cells whose atom puts more than
  ``energy_tol`` of its energy on the padding.  Because ``B B^T`` is a projector, row ``p`` of
  ``M`` differs from ``e_p`` by exactly ``sqrt(psi_p)`` in 2-norm, so the default
  ``energy_tol = 2**-106`` (unit round-off squared) keeps every row that is not ``e_p`` to double
  precision.  The Meyer (``nu_4``) atoms have power-law time tails: on every paper grid every
  kept cell leaks above that (sinusoid 64 x 256: ``min psi = 4.7e-18``; 3/6/12-month GB
  windows ``9.8e-18 / 4.6e-19 / 7.5e-22``), so ``E = K`` and ``(M Lambda M)_KK`` is dense;
* ``ln det M_KK = 2 ln|det B_{K^c}|`` (``B^T B = I`` and Sylvester: ``det(I - B_K B_K^T) =
  det(B_{K^c}^T B_{K^c})``; ``K^c`` the ``p Nf`` padding cells, a square block), so the lift
  ``A = W'_{K, data}`` has ``ln|det A| = N/2 ln dt + 1/2 ln det M_KK``
  (:meth:`CenteredPad.log_abs_det_lift`): the cell density and the time-sample density of the
  same data differ by exactly ``-ln|det A|``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from domainia import wilson
from domainia.wilson.backend import NUMPY, Backend, pytree_dataclass
from domainia.wilson.grid import active_mask, delta_T
from domainia.wilson.transform import forward_time_many, inverse_time_many

from ..domains import CellSet, TimeDomain, WDMDomain
from ..maps.base import Compose, LinearMap, _last_axis
from ..maps.time import TimeToWDM
from ..spectra import GenericPSD, LogLinearPSD, SeparablePSD

__all__ = ["CenterPadTime", "CenteredPad", "EDGE_ENERGY_TOL", "PAD_LAYOUTS", "time_shifted"]

PAD_LAYOUTS = ("centered", "end")
#: Unit round-off squared: a kept cell leaking at most this much energy into the padding has a
#: mask row equal to ``e_p`` to double precision (``|M_p. - e_p|_2 = sqrt(psi_p) <= 2**-53``).
EDGE_ENERGY_TOL = 2.0 ** -106
#: Largest extended record :meth:`CenteredPad.dense_mask` builds (the dense oracle, tests only).
DENSE_MASK_MAX = 4096


@pytree_dataclass(data=(), meta=("src", "lead", "trail"))
@dataclass(frozen=True, eq=False)
class CenterPadTime(LinearMap):
    """``lead`` zero samples, the record, ``trail`` zero samples (the 8 + 8 pad: ``lead = trail =
    8 Nf``).  ``dst`` starts ``lead`` samples before ``src`` (``t0 - lead dt``)."""

    src: TimeDomain
    lead: int
    trail: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "lead", operator.index(self.lead))
        object.__setattr__(self, "trail", operator.index(self.trail))
        if self.lead < 0 or self.trail < 0:
            raise ValueError(f"lead and trail must be >= 0; got {self.lead}, {self.trail}")

    @property
    def dst(self) -> TimeDomain:
        return TimeDomain(self.src.N + self.lead + self.trail, self.src.dt, self.src.t0 - self.lead * self.src.dt)

    def apply(self, x, be=None):
        _last_axis(x, self.src.N, "CenterPadTime.apply")
        be = self._be(x, be)
        x = be.asarray(x)
        out = be.xp.zeros((*x.shape[:-1], self.dst.N), dtype=x.dtype)
        return be.set(out, (..., slice(self.lead, self.lead + self.src.N)), x, copy=False)

    def adjoint(self, y, be=None):
        _last_axis(y, self.dst.N, "CenterPadTime.adjoint")
        return y[..., self.lead:self.lead + self.src.N]


@dataclass(frozen=True)
class _ShiftedEnvelope:
    """``g(t + t0)`` of an ``Envelope``: ``basis`` / ``log_g`` shifted; ``ghat`` refused (a shift
    multiplies it by a phase the pads never read)."""

    inner: Any
    t0: float

    @property
    def n_params(self) -> int:
        return self.inner.n_params

    def basis(self, t: Any) -> np.ndarray:
        return self.inner.basis(np.asarray(t, dtype=float) + self.t0)

    def log_g(self, t: Any, mu: Any, be: Backend | None = None) -> Any:
        return mu @ self.basis(t)

    def ghat(self, lags: Any, mu: Any, be: Backend | None = None) -> Any:
        raise NotImplementedError("a time-shifted envelope has no ghat (the padded grids never read it)")


@dataclass(frozen=True)
class _ShiftedLogLinearPSD(LogLinearPSD):
    """``LogLinearPSD`` evaluated at ``t + t0``.  Its capabilities are empty: the Gram / Bessel /
    operator builders compute ``alpha`` from ``nu`` themselves and would ignore the shift."""

    t0: float = 0.0
    capabilities = frozenset()

    def alpha(self, t: Any, nu: Any, be: Backend | None = None) -> Any:
        from domainia.wilson.backend import get_backend

        be = be or get_backend(t, *(nu if isinstance(nu, (tuple, list)) else (nu,)))
        return super().alpha(be.asarray(t, dtype=float) + self.t0, nu, be)


def time_shifted(psd: Any, t0: float) -> Any:
    """``psd`` with its clock moved: ``S'(f, t) = S(f, t + t0)``.  On the padded grid column ``j``
    is at ``j dT`` for the WDM code and at ``(j - p/2) dT`` for the text, so the extended-grid
    variances use ``time_shifted(psd, CenteredPad.t0)``.  ``LogLinearPSD`` (``alpha`` shifted;
    the result serves ``BandAveragedVariance`` / ``MidpointVariance`` / :class:`Eq45Direct`),
    ``SeparablePSD`` (the envelope's ``basis`` shifted) and ``GenericPSD``."""
    t0 = float(t0)
    if isinstance(psd, LogLinearPSD):
        return _ShiftedLogLinearPSD(psd.f_floor, psd.f_ref, psd.T, t0=getattr(psd, "t0", 0.0) + t0)
    if isinstance(psd, SeparablePSD):
        comps = tuple((s, None if e is None else _ShiftedEnvelope(e, t0)) for s, e in psd.components)
        return SeparablePSD(comps, psd.channels)
    if isinstance(psd, GenericPSD):
        fn = psd.fn
        return GenericPSD(lambda f, t, nu: fn(f, np.asarray(t, dtype=float) + t0, nu))
    raise TypeError(f"no time shift for {type(psd).__name__}")


@dataclass(frozen=True)
class CenteredPad:
    """The text's padded grid around the data grid ``grid`` (module docstring).

    ``p`` total padding columns (16 in the paper), ``layout`` ``"centered"`` (``p/2`` each side;
    ``p/2`` must be even so the kept cells keep their DC / Nyquist parity) or ``"end"``."""

    grid: wilson.Grid
    p: int = 16
    layout: Literal["centered", "end"] = "centered"

    def __post_init__(self) -> None:
        if not isinstance(self.grid, wilson.Grid):
            raise TypeError(f"grid must be a wilson.Grid; got {type(self.grid).__name__}")
        if self.layout not in PAD_LAYOUTS:
            raise ValueError(f"layout must be one of {PAD_LAYOUTS}; got {self.layout!r}")
        p = operator.index(self.p)
        if p < 2 or p % 2:
            raise ValueError(f"p must be a positive even number of columns; got {self.p!r}")
        if self.layout == "centered" and (p // 2) % 2:
            raise ValueError(f"the centered layout needs p/2 even (DC / Nyquist parity); got p = {p}")

    # -- geometry -------------------------------------------------------------------------
    @property
    def ext(self) -> wilson.Grid:
        """The extended grid ``(Nf, Nt + p, dt)``."""
        return wilson.Grid(self.grid.Nf, self.grid.Nt + self.p, self.grid.dt)

    @property
    def lead(self) -> int:
        """Padding columns before the data (``p/2`` centered, ``0`` end)."""
        return self.p // 2 if self.layout == "centered" else 0

    @property
    def t0(self) -> float:
        """The text time of extended column 0: ``-lead dT``."""
        return -self.lead * delta_T(self.grid.Nf, self.grid.dt)

    @property
    def samples(self) -> tuple[int, int]:
        """The data samples ``[start, stop)`` of the extended record."""
        start = self.lead * self.grid.Nf
        return start, start + self.grid.N

    @property
    def cells(self) -> CellSet:
        """Every active cell of the extended grid (``R``, packed order): where ``Lambda`` lives."""
        return CellSet.all_active(self.ext)

    @property
    def kept(self) -> CellSet:
        """The kept cells ``K``: the data columns ``lead .. lead + Nt - 1`` of the extended grid."""
        return self.cells.drop_columns(self.lead, self.p - self.lead)

    @property
    def kpos(self) -> np.ndarray:
        """Positions of ``K`` among ``R``."""
        return self.cells.index_of(self.kept)

    @property
    def padding(self) -> CellSet:
        """The ``p Nf`` padding cells ``K^c``."""
        return self.cells.complement(self.kept)

    def column_times(self) -> np.ndarray:
        """``(Nt + p,)``: the text time of every extended column, ``(j - lead) dT``."""
        return (np.arange(self.ext.Nt) - self.lead) * delta_T(self.grid.Nf, self.grid.dt)

    # -- data -----------------------------------------------------------------------------
    def time_map(self) -> CenterPadTime:
        """The record (``N`` samples from ``t = 0``) onto the extended record (``chi x``)."""
        return CenterPadTime(TimeDomain(self.grid.N, self.grid.dt), self.lead * self.grid.Nf,
                             (self.p - self.lead) * self.grid.Nf)

    def lift_map(self) -> Compose:
        """``x -> (W' chi x)_K`` (Eq. 92 on the kept cells): the padded likelihood's data map."""
        pad = self.time_map()
        return Compose(TimeToWDM(pad.dst, WDMDomain(self.kept)), pad)

    def lift(self, x: Any, be: Backend | None = None) -> Any:
        """:meth:`lift_map` applied to ``x`` (``(N,)`` or ``(..., N)``)."""
        return self.lift_map().apply(x, be)

    # -- the mask -------------------------------------------------------------------------
    def padding_samples(self) -> np.ndarray:
        """The ``p Nf`` padding sample indices of the extended record."""
        start, stop = self.samples
        return np.r_[np.arange(0, start), np.arange(stop, self.ext.N)]

    def padding_impulses(self, be: Backend | None = None) -> Any:
        """``B = W'_{R, P} / sqrt(dt)``, ``(N', p Nf)``: ``M = I - B B^T`` on ``R`` (host build)."""
        e = self.ext
        P = self.padding_samples()
        X = np.zeros((P.size, e.N))
        X[np.arange(P.size), P] = 1.0
        B = forward_time_many(X, e.Nf, e.Nt, e.dt)[:, active_mask(e.Nf, e.Nt)].T / np.sqrt(e.dt)
        return (be or NUMPY).asarray(B)

    def leakage(self) -> np.ndarray:
        """``(|K|,)``: each kept atom's energy fraction on the padding samples, ``psi_p = (B B^T)_pp``,
        by direct summation of the squared atom samples (no FFT correlation: its round-off floor
        ~1e-16 would hide the tails).  Resolution ~1e-30 (the atoms' own round-off)."""
        e = self.ext
        mask = active_mask(e.Nf, e.Nt)
        P = self.padding_samples()
        psi = np.full((e.Nf + 1, e.Nt), np.nan)
        for m in range(e.Nf + 1):
            for a in (0, 1):
                if not mask[m, a]:
                    continue
                E = np.zeros((1, e.Nf + 1, e.Nt))
                E[0, m, a] = 1.0
                g2 = inverse_time_many(E, e.Nf, e.Nt, e.dt)[0] ** 2
                cols = np.arange(a, e.Nt, 2)
                cols = cols[mask[m, cols]]
                idx = (P[None, :] - ((cols - a) * e.Nf)[:, None]) % e.N
                psi[m, cols] = g2[idx].sum(axis=1) / g2.sum()
        K = self.kept
        return psi[K.m_array, K.n_array]

    def edge_set(self, energy_tol: float = EDGE_ENERGY_TOL) -> np.ndarray:
        """Positions among ``K`` of the cells whose mask row is not ``e_p`` at ``energy_tol``
        (default: unit round-off squared -- every kept cell on the paper's grids)."""
        if not energy_tol >= 0.0:
            raise ValueError(f"energy_tol must be >= 0; got {energy_tol!r}")
        return np.flatnonzero(self.leakage() > energy_tol)

    def mask_rows(self, rows: Any = None) -> np.ndarray:
        """``(len(rows), N')``: rows of ``M`` for the kept positions ``rows`` (all of ``K`` by
        default), ``e_p - B_p B^T`` (host NumPy)."""
        rows = np.arange(self.kept.size) if rows is None else np.atleast_1d(np.asarray(rows, dtype=np.int64))
        B = np.asarray(self.padding_impulses())
        kp = self.kpos[rows]
        out = -(B[kp] @ B.T)
        out[np.arange(rows.size), kp] += 1.0
        return out

    def dense_mask(self) -> np.ndarray:
        """``M = W' chi W'^T / dt`` from the dense ``W'`` (``wilson.matrix.active_W``; the oracle)."""
        e = self.ext
        if e.N > DENSE_MASK_MAX:
            raise ValueError(f"dense_mask is the small-grid oracle (N' <= {DENSE_MASK_MAX}); N' = {e.N}")
        W = wilson.matrix.active_W(e.Nf, e.Nt, e.dt)
        start, stop = self.samples
        return W[:, start:stop] @ W[:, start:stop].T / e.dt

    def log_det_mask(self) -> float:
        """``ln det M_KK = 2 ln|det B_{K^c}|`` (module docstring; one ``p Nf`` square slogdet)."""
        B = np.asarray(self.padding_impulses())
        Kc = self.cells.index_of(self.padding)
        sign, logabs = np.linalg.slogdet(B[Kc])
        if sign == 0:
            raise ValueError("B_{K^c} is singular: the padding cells do not span the padding samples")
        return 2.0 * float(logabs)

    def log_abs_det_lift(self) -> float:
        """``ln|det A|``, ``A = W'_{K, data}`` the lift on the data samples:
        ``N/2 ln dt + 1/2 ln det M_KK`` (``A A^T = dt M_KK``)."""
        return 0.5 * self.grid.N * float(np.log(self.grid.dt)) + 0.5 * self.log_det_mask()
