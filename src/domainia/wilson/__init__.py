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

Input validation (all applied automatically by the entry points):

* :func:`validate_grid`, :func:`validate_dt`, :func:`validate_finite`,
  :func:`grid.sanitize_psd`

Grid helpers:

* :func:`n_samples`, :func:`band_frequencies`, :func:`time_pixel_centers`
* :func:`delta_F`, :func:`delta_T`
* :func:`active_mask`, :func:`count_active`
* :func:`pack`, :func:`unpack` -- dense ``(Nf+1, Nt)`` <-> packed ``(N,)`` active cells
  in band order

Noise variance grids:

* :func:`band_averaged_variance`
* :func:`empirical_variance`

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

* :func:`filter.phi_centered`, :func:`filter.phi_fft_order`

Windows (in :mod:`domainia.wilson.windows`):

* :class:`Window` (the protocol), :class:`Meyer`, :class:`RampWindow`,
  :class:`SampledWindow`, :data:`MEYER` (the WDM transform's window, ``Meyer(4)``),
  :class:`WindowError`
* :func:`sample_window` (the kernels' single entry point: validated, read-only samples of
  any :class:`Window`, a window of your own included), :func:`require_window`
"""

from __future__ import annotations

from . import (
    evolutionary,
    filter,
    matrix,
    noise,
    stationary,
    windows,
)
from .filter import phi_centered, phi_fft_order
from .grid import (
    Grid,
    active_mask,
    band_frequencies,
    count_active,
    delta_F,
    delta_T,
    n_samples,
    pack,
    time_pixel_centers,
    unpack,
    validate_dt,
    validate_finite,
    validate_grid,
)
from .noise import band_averaged_variance, empirical_variance
from .transform import forward, forward_time, inverse, inverse_time
from .windows import (
    MEYER,
    Meyer,
    RampWindow,
    SampledWindow,
    Window,
    WindowError,
    require_window,
    sample_window,
)

__all__ = [
    "MEYER",
    "Grid",
    "Meyer",
    "RampWindow",
    "SampledWindow",
    "Window",
    "WindowError",
    "active_mask",
    "band_averaged_variance",
    "band_frequencies",
    "count_active",
    "delta_F",
    "delta_T",
    "empirical_variance",
    # Submodules
    "evolutionary",
    "filter",
    # Transform
    "forward",
    "forward_time",
    "inverse",
    "inverse_time",
    "matrix",
    # Grid
    "n_samples",
    "noise",
    "pack",
    # Filter
    "phi_centered",
    "phi_fft_order",
    # Windows
    "require_window",
    "sample_window",
    "stationary",
    "time_pixel_centers",
    "unpack",
    "validate_dt",
    "validate_finite",
    "validate_grid",
    "windows",
]
