"""``domainia.wilson``: Wilson-basis kernels on the half-shifted lattice.  With the
default window :data:`MEYER` they are the WDM transform.  Every kernel that samples
the window takes ``window=`` (:mod:`domainia.wilson.windows`).

The WDM transform is a Wilson-basis time-frequency tiling with a Meyer
scaling-function window.  Despite the historical name, it is **not** a wavelet
transform: the frequency grid is uniform (not dyadic) and the basis
functions are Wilson-basis trigonometric atoms on a half-shifted
lattice.  The acronym stands for *Wilson--Daubechies--Meyer*, the
mathematicians whose work underlies the construction.

Public API
----------
Forward / inverse data transforms:

* :func:`forward`, :func:`inverse` (any leading axes: a stack of series sharing one
  grid goes through one FFT)
* :func:`forward_time`, :func:`inverse_time` (time-domain wrappers)

A validated ``(Nf, Nt, dt)`` container, which removes the one misuse the
package cannot otherwise detect (swapping the two same-typed ints):

* :class:`Grid` -- unpacks as ``(Nf, Nt, dt)``, so ``f(x, *grid)`` works

Input validation (in :mod:`domainia.wilson.grid`; all applied automatically by the entry
points):

* :func:`grid.validate_grid`, :func:`grid.validate_dt`, :func:`grid.validate_finite`,
  :func:`grid.sanitize_psd`

Grid helpers (in :mod:`domainia.wilson.grid`):

* :func:`grid.n_samples`, :func:`grid.band_frequencies`, :func:`grid.time_pixel_centers`
* :func:`grid.delta_F`, :func:`grid.delta_T`
* :func:`grid.active_mask`

Noise variance grids (WDM Paper II, Eq. 36):

* :func:`band_averaged_variance`, :func:`evolutionary_band_variance`

Explicit matrix construction (in :mod:`domainia.wilson.matrix`):

* :func:`matrix.build_W`, :func:`matrix.active_W`
* :func:`matrix.build_C_time`, :func:`matrix.build_C_wdm`,
  :func:`matrix.build_C_wdm_active`
* :func:`matrix.rfft_to_real`, :func:`matrix.real_to_rfft`

Fast stationary noise covariance (in :mod:`domainia.wilson.stationary`):

* :func:`stationary.build_lambda_blocks_full_active` (interior + edges,
  matches dense $W_a\\,C\\,W_a^T$ to machine precision)
* :func:`stationary.assemble_lambda_full_active`
* :func:`stationary.is_diagonal`
* :func:`stationary.psd_from_lambda_blocks` (fast inverse: $\\Lambda \\to S_n(f)$)
* :func:`stationary.apply_lambda_from_psd` ($\\Lambda x$ without assembling
  $\\Lambda$; $O(N \\log N_t)$ per call)
* :func:`stationary.build_K_full_active`, :func:`stationary.psd_from_K`
  (compact $\\Lambda$: one length-$N_t$ vector per band pair, and the inverse
  taken straight from it)

Filter (in :mod:`domainia.wilson.filter`):

* :func:`filter.phi_centered`

Windows (in :mod:`domainia.wilson.windows`):

* :class:`Window` (the protocol), :class:`Meyer`, :class:`RampWindow`,
  :class:`SampledWindow`, :data:`MEYER` (the WDM transform's window, ``Meyer(4)``),
  :class:`windows.WindowError`
* :func:`windows.sample_window` (the kernels' single entry point: validated, read-only
  samples of any :class:`Window`, a window of your own included),
  :func:`windows.require_window`
"""

from __future__ import annotations

from . import evolutionary, filter, grid, matrix, noise, stationary, windows
from .grid import Grid
from .noise import band_averaged_variance, evolutionary_band_variance
from .transform import forward, forward_time, inverse, inverse_time
from .windows import MEYER, Meyer, RampWindow, SampledWindow, Window

__all__ = [
    "MEYER",
    "Grid",
    "Meyer",
    "RampWindow",
    "SampledWindow",
    "Window",
    "band_averaged_variance",
    "evolutionary",
    "evolutionary_band_variance",
    "filter",
    "forward",
    "forward_time",
    "grid",
    "inverse",
    "inverse_time",
    "matrix",
    "noise",
    "stationary",
    "windows",
]
