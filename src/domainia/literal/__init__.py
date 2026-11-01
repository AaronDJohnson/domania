"""Text-literal primitives (part 5, spec section 10): every matrix entry from the paper's
formula, factorised by exact algebra only.

* :mod:`.congruence` -- ``Lambda = W C W^T`` (Eq. 9) by the fast transform on both sides.
* :mod:`.eq45_direct` -- Eq. (45) by the direct grid sum at each pair's arithmetic mid-time.
* :mod:`.pad` -- the 8 + 8 zero-padded grid (Eqs. 92-94): kept cells, ``M = I - B B^T``, the
  edge set at round-off, ``ln|det A|`` of the lift; :func:`time_shifted` spectra.
* :mod:`.masked` -- ``(M Lambda M)_KK`` exactly: QR of ``(M L)_K`` or the arrow, the dense oracle.
* :mod:`.time` -- the same likelihood through the time samples (cell density + ``ln|det A|``).
"""

from .congruence import wdm_congruence, wdm_rows
from .eq45_direct import Eq45Direct, direct_block
from .masked import MaskedCovariance, masked_covariance
from .pad import EDGE_ENERGY_TOL, CenteredPad, CenterPadTime, time_shifted
from .time import ColumnRotated, time_marginal

__all__ = ["CenterPadTime", "CenteredPad", "ColumnRotated", "EDGE_ENERGY_TOL", "Eq45Direct", "MaskedCovariance",
           "direct_block", "masked_covariance", "time_marginal", "time_shifted", "wdm_congruence", "wdm_rows"]
