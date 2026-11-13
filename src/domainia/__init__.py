"""domainia: translate data arrays between time, frequency and WDM, and evaluate a noise
model's per-point K x K covariance on a frequency or WDM grid.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`domainia.wilson.MEYER`); every kernel that samples the window
  takes ``window=`` (any admissible :class:`domainia.wilson.Window`: ``Meyer(d)``,
  ``RampWindow``, ``SampledWindow``).
* :func:`transform`: data arrays, with any leading axes, between time, frequency and WDM
  (:mod:`domainia._transform`).
* :func:`frequency_covariance`: a stationary K x K noise model as its covariance per rfft bin
  (:mod:`domainia._covariance`).
* :func:`wdm_covariance`: an evolutionary (or stationary) K x K noise model as its covariance
  per WDM cell (:mod:`domainia._covariance`).
* Not imported here: :mod:`domainia.testing` (dense oracles, test windows and the
  Monte-Carlo noise helpers).
"""

from __future__ import annotations

from . import wilson
from ._covariance import frequency_covariance, wdm_covariance
from ._transform import transform

__version__ = "3.0.0"

__all__ = ["__version__", "frequency_covariance", "transform", "wdm_covariance", "wilson"]
