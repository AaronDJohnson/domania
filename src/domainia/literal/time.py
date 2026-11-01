r"""The text's padded likelihood through the time samples: exact algebra, ``O((p N_f)^3)`` per state.

``(M Lambda M)_KK = A C_dd A^T`` with ``A = W'_{K, data}`` the lift and
``C_dd = [W'^T Lambda W']_dd / dt^2``, so the cell density of ``w'_K = A x`` (the text) is the
time-sample density of ``x`` minus ``ln|det A|`` (:meth:`CenteredPad.log_abs_det_lift`) at every
noise state.  The time-sample density is the release's ``ZeroPadded("time_marginal")`` (WT
``PaddedMarginal``: the precision Schur complement over the ``p N_f`` padding samples, no
truncation), written for padding appended at the end of the record.  The WDM transform is
covariant under shifts by an even number of columns, so the 8 + 8 layout is the end layout with
``Lambda``'s columns rotated: :class:`ColumnRotated` is ``P^T Lambda P`` (cell ``(m, n)`` of the
end-layout grid is cell ``(m, (n + p/2) mod Nt')`` of the 8 + 8 grid) and :func:`time_marginal`
wraps it in ``ZeroPadded``.  The data are then the record's ``N`` samples (``Identity`` map), and
the value is the stored chains' convention for the pads (a density of ``x``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from domainia.wilson.backend import Backend

from ..covariance.base import Covariance, DiagFactor
from ..covariance.blocktri import blocktri_cov_factor, dummy_slots
from ..covariance.dense import dense_factor
from ..covariance.edges import ZeroPadded
from ..domains import _packed_index
from ..linalg import BandLayout
from .masked import base_kind, raw_padded_blocks
from .pad import CenteredPad

__all__ = ["ColumnRotated", "time_marginal"]


class ColumnRotated(Covariance):
    """``base`` (on every active cell of ``pad.ext``, packed order) with its columns rotated by
    ``pad.lead``: entry ``(r, s)`` is ``base``'s entry of the cells ``(m_r, (n_r + lead) mod Nt')``
    and ``(m_s, (n_s + lead) mod Nt')`` -- the 8 + 8 grid seen from the end layout."""

    def __init__(self, base: Covariance, pad: CenteredPad) -> None:
        R = pad.cells
        if base.n != R.size:
            raise ValueError(f"the base covers {base.n} cells; the extended grid has {R.size}")
        self.base, self.pad, self.n = base, pad, base.n
        e = pad.ext
        self.perm = _packed_index(R.m_array, (R.n_array + pad.lead) % e.Nt, e.Nf, e.Nt)
        self.kind = base_kind(base)
        self.jittable = base.jittable
        if self.kind == "blocks":
            self.sizes = tuple(int(s) for s in base.sizes)
            self.layout = "padded"

    def children(self) -> dict[str, Covariance]:
        return {"base": self.base}

    def host_constants(self) -> dict[str, Any]:
        C: dict[str, Any] = {"perm": self.perm}
        if self.kind == "blocks":
            sizes = self.sizes
            off = np.concatenate([[0], np.cumsum(sizes)])
            B = max(sizes)
            sp = np.tile(np.arange(B), (len(sizes), 1))
            for b, s in enumerate(sizes):
                sp[b, :s] = self.perm[off[b]:off[b] + s] - off[b]
            C.update(sp=sp, layout=BandLayout.from_sizes(sizes), dummy=dummy_slots(sizes))
        return C

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        """The base's raw padded blocks with every band's columns rotated."""
        d, u = raw_padded_blocks(self.base, C["base"], nu, be)
        sp = C["sp"]
        nb = sp.shape[0]
        b = np.arange(nb)[:, None, None]
        dr = d[b, sp[:, :, None], sp[:, None, :]]
        ur = u[b[:-1], sp[:-1, :, None], sp[1:, None, :]]
        return dr, ur

    def variances(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.base.variances(C["base"], nu, be)[C["perm"]]

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        p = C["perm"]
        return self.base._matrix(C["base"], nu, be)[p][:, p]

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        if self.kind == "diagonal":
            return DiagFactor.build(self.variances(C, nu, be), be)
        if self.kind == "blocks":
            dr, ur = self.band_blocks(C, nu, be)
            return blocktri_cov_factor(dr, ur, self.sizes, "padded", C["layout"], be, C["dummy"])
        return dense_factor(self._matrix(C, nu, be), chol_impl="numpy", symmetrize=True, be=be)


def time_marginal(base: Covariance, pad: CenteredPad, *, form: Literal["solve", "reduce"] = "solve",
                  chol_impl: Literal["numpy", "scipy"] = "scipy") -> ZeroPadded:
    """The time-sample density of the padded model (``ZeroPadded("time_marginal")`` over
    ``ColumnRotated(base, pad)``; ``base`` itself for the end layout).  Cell density = this
    minus ``pad.log_abs_det_lift()``."""
    rotated = ColumnRotated(base, pad) if pad.lead else base
    return ZeroPadded(rotated, strategy="time_marginal", grid=pad.ext, pad_columns=pad.p, form=form,
                      chol_impl=chol_impl)
