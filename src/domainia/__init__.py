"""domainia: data and covariance across the time, Fourier and time-frequency (Wilson / WDM)
domains.

Importing ``domainia`` (any of its modules) never imports JAX.  JAX is imported only on use:
when the JAX backend is built (:func:`domainia.wilson.backend.jax_backend` or
``domainia.wilson.backend.get_backend(backend="jax")``) to pass as the ``be=`` of a kernel,
map or covariance that takes one, when float64 is turned on
(:func:`domainia.wilson.backend.enable_x64`), or when JAX device constants are built.  The
JAX backend needs float64: call ``enable_x64()`` (or set ``JAX_ENABLE_X64=true``) first, or
it raises ``RuntimeError``.  A kernel that takes ``be=``, given JAX arrays and no ``be=``,
infers the JAX backend, with the same requirement.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`MEYER`); every kernel that samples the window takes
  ``window=`` (any admissible :class:`Window`: ``Meyer(d)``, ``RampWindow``,
  ``SampledWindow``).
* :func:`transform`: data arrays, with any leading axes, between time, frequency and WDM
  (:mod:`domainia._transform`).
* Not imported here: :mod:`domainia.testing` (dense oracles and test windows).
"""

from __future__ import annotations

from . import wilson
from ._transform import transform
from .wilson import MEYER, Grid, Meyer, RampWindow, SampledWindow, Window, WindowError

__version__ = "2.0.0"

__all__ = [
    "MEYER",
    "Grid",
    "Meyer",
    "RampWindow",
    "SampledWindow",
    "Window",
    "WindowError",
    "__version__",
    "transform",
    "wilson",
]
