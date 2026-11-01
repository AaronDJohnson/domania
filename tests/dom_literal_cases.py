"""Shared small-grid cases of the literal-primitive tests (L-A6, L-A7): a padded grid and its
extended-grid Lambda (band / mid variances with the shifted clock, or the direct Eq. 45)."""

from __future__ import annotations

import numpy as np

from domainia.covariance import DiagonalCovariance
from domainia.literal import CenteredPad, Eq45Direct, time_shifted
from domainia.spectra import LogLinearPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

NUS = [np.array([-1.7, 0.9]), np.array([-2.3, -1.4])]


def setup_case(g, p, base, layout="centered", tri_assembly="blocktri"):
    """``(pad, Lambda_ext)``: band / mid (time-shifted variances) or tri (Eq45Direct, t0) on
    every active cell of the padded grid; alpha(t) on the DATA record's T."""
    pad = CenteredPad(g, p=p, layout=layout)
    psd = LogLinearPSD(0.5, 1.0, g.N * g.dt)
    sh = time_shifted(psd, pad.t0)
    if base == "band":
        return pad, DiagonalCovariance(BandAveragedVariance(sh, pad.cells), pad.cells.size)
    if base == "mid":
        return pad, DiagonalCovariance(MidpointVariance(sh, pad.cells), pad.cells.size)
    return pad, Eq45Direct(psd, pad.cells, t0=pad.t0, assembly=tri_assembly)


def rel(a, b):
    """Max abs difference relative to ``max|b|``."""
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.abs(np.asarray(b)).max()
