"""domainia: data on domains, the maps between them, covariance operators and Gaussian
likelihoods over the WDM transform (WDM Paper II release, spec section 4).

Importing ``domainia`` never imports JAX: the JAX backend is reached only through :mod:`domainia.wilson.backend`.

Modules
-------
* :mod:`domainia.domains`: ``TimeDomain``, ``FourierDomain``, ``CellSet``, ``WDMDomain``,
  ``ChannelAxis`` (frozen, hashable: static under ``jax.jit``)
* :mod:`domainia.data`: ``TimeSeries``, ``FourierBins``, ``WDMCells`` (pytrees; the
  domain and channels are static metadata)
* :mod:`domainia.linalg`: array kernels -- block-tridiagonal (ragged / padded), Hermitian
  block, preconditioned CG (the factor objects -- diagonal, dense, Schur, padded marginal --
  are :mod:`domainia.covariance`'s, one family since S4)
* :mod:`domainia.assets`: ``Artifact`` (identity + sha256, fail-closed, never writes
  unless asked)
* :mod:`domainia.testing`: dense oracles for tests (not imported here)
"""

from __future__ import annotations

from . import assets, data, domains, linalg
from .assets import Artifact, ArtifactError
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain

__version__ = "2.0.0"

__all__ = [
    "Artifact",
    "ArtifactError",
    "CellSet",
    "ChannelAxis",
    "FourierBins",
    "FourierDomain",
    "TimeDomain",
    "TimeSeries",
    "WDMCells",
    "WDMDomain",
    "__version__",
    "assets",
    "data",
    "domains",
    "linalg",
]
