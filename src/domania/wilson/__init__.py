"""``domania.wilson``: Wilson-basis kernels on the half-shifted lattice.  With the
default window :data:`MEYER` they are the WDM transform.  Every kernel that samples
the window takes ``window=`` (:mod:`domania.wilson.windows`).

The WDM transform is a Wilson-basis time-frequency tiling with a Meyer
scaling-function window.  Despite the name, it is **not** a wavelet
transform: the frequency grid is uniform (not dyadic) and the basis
functions are Wilson-basis trigonometric atoms on a half-shifted
lattice.  The acronym stands for *Wilson--Daubechies--Meyer*.

Public API
----------
* :func:`forward`, :func:`inverse` (rfft <-> WDM) and :func:`forward_time`,
  :func:`inverse_time` (time <-> WDM), with any leading axes.
* :class:`Grid`: a validated ``(Nf, Nt, dt)`` that unpacks as ``(Nf, Nt, dt)``.
* :func:`band_averaged_variance`, :func:`evolutionary_band_variance`: noise variance
  grids (WDM Paper II, Eq. 36).
* :class:`Window` (the protocol), :class:`Meyer`, :class:`RampWindow`,
  :class:`SampledWindow`, :data:`MEYER` (``Meyer(4)``, the WDM transform's window).
* The submodules :mod:`domania.wilson.grid` (grid helpers and the input checks the entry
  points apply), :mod:`domania.wilson.filter` (the Meyer filter),
  :mod:`domania.wilson.windows` (:func:`~domania.wilson.windows.sample_window` and the
  :class:`~domania.wilson.windows.WindowError` of a failed window check),
  :mod:`domania.wilson.noise`, :mod:`domania.wilson.matrix` (explicit matrices, small grids),
  :mod:`domania.wilson.stationary` (fast stationary noise covariance) and
  :mod:`domania.wilson.evolutionary` (kernels for evolutionary noise).
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
