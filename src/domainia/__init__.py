"""domainia: data, linear maps and covariance across the time, Fourier and time-frequency
(Wilson / WDM) domains.

Importing ``domainia`` never imports JAX: the JAX backend is reached only through
:mod:`domainia.wilson.backend`.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`MEYER`); every kernel takes ``window=`` (any admissible
  :class:`Window`: ``Meyer(d)``, ``RampWindow``, ``SampledWindow``).
* :mod:`domainia.translate`: ``transform`` (``{channel: array}`` dicts) and
  ``translate_operator`` (lazy covariance change of basis) between the full-record
  ``TimeDomain``, ``FourierDomain.full`` and ``WDMDomain.full``.
* :mod:`domainia.domains`: ``TimeDomain``, ``FourierDomain``, ``CellSet``, ``WDMDomain``,
  ``ChannelAxis`` (frozen, hashable: static under ``jax.jit``).
* :mod:`domainia.data`: ``TimeSeries``, ``FourierBins``, ``WDMCells`` (pytrees).
* :mod:`domainia.linalg`: array kernels (block-tridiagonal, Hermitian block, PCG).
* Not imported here: :mod:`domainia.maps` (linear maps between partial domains),
  :mod:`domainia.covariance`, :mod:`domainia.variance`, :mod:`domainia.spectra`, and
  :mod:`domainia.testing` (dense oracles, test windows and operators).
"""

from __future__ import annotations

from . import data, domains, linalg, translate, wilson
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from .translate import Operator, TranslatedOperator, log_coordinate_scale, transform, translate_operator
from .wilson import MEYER, Grid, Meyer, RampWindow, SampledWindow, Window, WindowError

__version__ = "2.0.0"

__all__ = [
    "MEYER",
    "CellSet",
    "ChannelAxis",
    "FourierBins",
    "FourierDomain",
    "Grid",
    "Meyer",
    "Operator",
    "RampWindow",
    "SampledWindow",
    "TimeDomain",
    "TimeSeries",
    "TranslatedOperator",
    "WDMCells",
    "WDMDomain",
    "Window",
    "WindowError",
    "__version__",
    "data",
    "domains",
    "linalg",
    "log_coordinate_scale",
    "transform",
    "translate",
    "translate_operator",
    "wilson",
]
