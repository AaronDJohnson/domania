"""Per-pixel noise variance grids in the WDM domain.

For stationary Gaussian noise with one-sided PSD $S_n(f)$, the
expected variance of a normalized WDM coefficient $w_{mn}$ is
approximately

$$\\sigma_{mn}^2 \\approx S_n(f_m) / 2,$$

uniformly across all pixels.  Both variance grids here are the filter-weighted
band average of WDM Paper II, Eq. 36:

* :func:`band_averaged_variance` — filter-weighted mean of $S_n$ over
  each band; exact, and the right choice for steep PSDs.
* :func:`evolutionary_band_variance` — the same mean of a time-varying PSD
  $S_n(f, t)$ at each time pixel, for locally-stationary noise.

A Monte-Carlo estimator, :func:`domainia.testing.empirical_variance`, checks them.

Both grids have shape ``(Nf+1, Nt)``.  They fill the inactive edge pixels
($m = 0$ and $m = N_f$ at odd $n$) by the same formula as the active ones, so they hold
about $S_n/2$ there, not 0; :func:`domainia.wdm_covariance` sets those cells to 0.

Every kernel here that reaches the window takes ``window=`` (a
:class:`domainia.wilson.windows.Window`; default :data:`domainia.wilson.MEYER`) and
computes in the Wilson basis of that window.

Argument order follows the package convention: the grid first
(``Nf, Nt, dt``), then the PSD. It matches :mod:`domainia.wilson.matrix`, so the PSD is
always the trailing argument.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .grid import (
    band_frequencies,
    delta_F,
    delta_T,
    n_samples,
    sanitize_psd,
    validate_dt,
    validate_grid,
)
from .windows import MEYER, Window, sample_window

type Array = NDArray[np.floating]
type PsdFunc = Callable[[Array], Array]
type PsdFtFunc = Callable[[Array, Array], Array]

__all__ = ["band_averaged_variance", "evolutionary_band_variance"]


def band_averaged_variance(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
    *,
    window: Window = MEYER,
) -> Array:
    """Filter-weighted mean of $S_n$ over each band, divided by 2.

    For each band $m$, computes the effective PSD

    $$S_n^{\\text{eff}}(m)
    = \\frac{\\sum_j \\hat\\phi^2(f_j - f_m)\\,S_n(f_j)}
            {\\sum_j \\hat\\phi^2(f_j - f_m)}$$

    and returns $\\sigma_{mn}^2 = S_n^{\\text{eff}}(m)/2$ broadcast
    over $n$.  This is exact: it reproduces the diagonal of the WDM
    covariance $W\\,C\\,W^T$ to machine precision, because the pixel
    variance is the $\\hat\\phi^2$-weighted average of the PSD across
    the band.

    .. note::
       Before 2026-07-27 this used the filter-weighted **harmonic**
       mean $\\bigl(\\sum \\hat\\phi^2/S_n\\bigr)^{-1}\\sum\\hat\\phi^2$,
       which is not the pixel variance and was measurably *less*
       accurate than $S_n(f_m)/2$ at the band centre in every interior band.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_func : callable
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, shape (Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> steep = lambda f: 1e-3 * (1.0 + 1e2 / (f + 1 / (2 * dt * Nf)) ** 2)
    >>> var = wilson.band_averaged_variance(Nf, Nt, dt, steep)
    >>> var.shape
    (9, 16)
    >>> # exact: reproduces the diagonal of the WDM covariance
    >>> truth = np.diag(wilson.matrix.build_C_wdm_active(Nf, Nt, dt, steep))
    >>> bool(abs(var[1, 0] - truth[Nt // 2]) / truth[Nt // 2] < 1e-12)
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    dt = float(dt)
    phi = sample_window(window, Nf, Nt, dt)
    phi2 = phi**2
    f_offsets = (np.arange(Nt) - Nt // 2) / (n_samples(Nf, Nt) * dt)

    dF = delta_F(Nf, dt)
    f_nyq = 0.5 / dt

    # Fold every band's slab onto the physical band [0, f_Nyq] in one shot.
    # The DC band's slab runs below zero and folds about 0; the Nyquist band's
    # runs above f_Nyq and folds about f_Nyq -- that is where the discrete
    # spectrum actually aliases. Folding the top band about 0 (plain ``abs``)
    # samples the PSD at the wrong frequencies and biases band m=Nf by several
    # percent. One vectorised call also means one user-PSD invocation instead
    # of Nf+1, which matters for an expensive sensitivity curve.
    f_bins = np.arange(Nf + 1)[:, None] * dF + f_offsets[None, :]
    f_folded = np.abs(f_bins)
    f_folded = np.where(f_folded > f_nyq, 2.0 * f_nyq - f_folded, f_folded)
    Sn_vals = sanitize_psd(Sn_func(f_folded), "band_averaged_variance")
    if Sn_vals.shape != f_folded.shape:
        raise ValueError(
            f"Sn_func must be elementwise: expected shape {f_folded.shape}, "
            f"got {Sn_vals.shape}"
        )

    # Weight by every bin the window touches. A bin where S_n vanishes still
    # belongs in the denominator -- it contributes zero power, which is exactly
    # what pulls the band average down. Dropping such bins from both sums (as
    # an earlier ``Sn_vals > 0`` mask did) rescales the average by the surviving
    # weight and overstates a band with a spectral cutoff in it by up to 3.2x.
    # The Meyer window always has positive bins -- measured 3/4 of them at
    # Nf=2, Nt=4 and 1535/2048 at Nf=1024, Nt=2048, across dt from 1e-6 to 1e6 --
    # and so does every admissible window (its partition of unity makes phi(f) or
    # phi(f - dF) nonzero at every bin), so ``in_band`` is never empty. An earlier
    # revision carried an ``if not np.any(in_band): Sn_eff = inf`` fallback for
    # that case. It was unreachable, and a silent infinite variance is the last
    # thing this package should produce, so it is gone rather than merely untested.
    in_band = phi2 > 0
    weights = phi2[in_band]
    Sn_eff = Sn_vals[:, in_band] @ weights / np.sum(weights)

    return np.broadcast_to(0.5 * Sn_eff[:, None], (Nf + 1, Nt)).copy()


# ---------------------------------------------------------------------------
# Evolutionary PSDs S(f, t) -- moved from wdm_dev ``evolutionary.py`` (verbatim)
# ---------------------------------------------------------------------------


def _psd_at(Sn: PsdFtFunc, f: Array, t: Array, context: str) -> Array:
    """Evaluate an elementwise, broadcasting ``Sn(f, t)`` and sanitize it."""
    shape = np.broadcast_shapes(np.shape(f), np.shape(t))
    out = sanitize_psd(Sn(f, t), context)
    if out.shape != shape:
        raise ValueError(f"{context}: Sn(f, t) must be elementwise and broadcast to "
                         f"{shape}; got {out.shape}")
    return out


def evolutionary_band_variance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc, *,
                               bands: Array | None = None,
                               window: Window = MEYER) -> Array:
    r"""The filter-weighted band average of :math:`S(f, t_n)` at every cell centre,

    .. math::  \sigma^2_{mn} = \frac12\,
               \frac{\sum_j \tilde\phi^2(f_j - f_m)\,S(f_j, t_n)}{\sum_j \tilde\phi^2(f_j - f_m)},

    WDM Paper II, Eq. 36 (eq:wdm_variance_band_average; the edge-band factor is not
    applied -- inactive edge cells carry the same value, as in
    :func:`band_averaged_variance`).  ``Sn(f, t)`` is elementwise and broadcasting
    (argument order ``(f, t)``).  In the stationary limit this is
    :func:`band_averaged_variance`.  Returns ``(Nf + 1, Nt)``, or
    ``(len(bands), Nt)`` for selected bands.  Was wdm_dev
    ``evolutionary.band_averaged_evolutionary_variance``.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn : callable
        ``(f, t) -> S(f, t)``, elementwise and broadcasting.
    bands : array of int, optional
        The bands to return (all by default).
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.
    """
    validate_grid(Nf, Nt)
    N = Nf * Nt
    phi2 = sample_window(window, Nf, Nt, dt) ** 2
    dt = float(dt)
    # An admissible window has compact support: a slab weight of exactly 0 cannot
    # contribute, so dropping those bins is an exact restriction.
    sup = np.flatnonzero(phi2)
    phi2 = phi2[sup]
    f_off = (np.arange(Nt) - Nt // 2)[sup] / (N * dt)
    m = np.arange(Nf + 1) if bands is None else np.asarray(bands, int)
    fb = band_frequencies(Nf, dt)[m][:, None] + f_off[None, :]
    f_nyq = 0.5 / dt
    ff = np.abs(fb)
    ff = np.where(ff > f_nyq, 2.0 * f_nyq - ff, ff)                       # (B, J)
    t = np.arange(Nt) * delta_T(Nf, dt)
    S = _psd_at(Sn, ff[:, None, :], t[None, :, None],
                "evolutionary_band_variance")                              # (B, Nt, J)
    return 0.5 * (S * phi2).sum(axis=-1) / phi2.sum()
