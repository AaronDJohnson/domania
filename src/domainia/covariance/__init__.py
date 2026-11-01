"""Covariance operators on WDM cells: one protocol, one factor per noise state.

Modules: ``base`` (protocol, factors, ``Covariance``), ``diagonal``, ``dense``,
``blocktri``, ``evolutionary`` (Eq. 45 for any callable PSD), ``stack`` (``ChannelStack``).
"""

from .base import CholFactor, Covariance, DiagFactor, Evaluator, Factor, QuadForm, as_evaluator, quad_form
from .blocktri import BlockTriCovFactor, BlockTridiagonalCovariance
from .dense import DenseCovariance, dense_factor
from .diagonal import DiagonalCovariance
from .evolutionary import EvolutionaryCovariance, keep_mask
from .stack import ChannelStack, StackFactor

__all__ = [
    "BlockTriCovFactor",
    "BlockTridiagonalCovariance",
    "ChannelStack",
    "CholFactor",
    "Covariance",
    "DenseCovariance",
    "DiagFactor",
    "DiagonalCovariance",
    "Evaluator",
    "EvolutionaryCovariance",
    "Factor",
    "QuadForm",
    "StackFactor",
    "as_evaluator",
    "dense_factor",
    "keep_mask",
    "quad_form",
]
