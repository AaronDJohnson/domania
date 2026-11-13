"""Meyer scaling function used as the WDM filter.

The Meyer scaling function $\\hat\\phi(f)$ has compact frequency support
on $[-3\\Delta F/4,\\,+3\\Delta F/4]$ and a smooth ($C^{d-1}$)
rolloff controlled by the regularized incomplete beta function $\\nu_d$.
Frequencies are in Hz throughout, with the Paper I (eq:Meyer-def)
normalisation: the flat value is $1/\\sqrt{\\Delta F}$.

It is the window that, paired with the half-shifted Wilson lattice and the
$C_{mn}$ phase factor, gives an orthonormal time-frequency basis at
critical density (the partition-of-unity property).

The shape parameter ``d`` (default ``4``) sets the smoothness of the rolloff; the basis
functions decay as $|t|^{-(d+1)}$ in time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import betainc

from ._checks import require_int
from .grid import delta_F, validate_dt, validate_grid

type Array = NDArray[np.floating]

__all__ = [
    "nu_d",
    "phi_centered",
]


def nu_d(x: Array, d: int = 4) -> Array:
    """Regularized incomplete beta function $I_x(d, d)$, element-wise, with ``x`` clipped to
    ``[0, 1]``: the Meyer rolloff's ramp, $\\nu_d(0) = 0$, $\\nu_d(1) = 1$, $C^{d-1}$ smooth.

    Raises
    ------
    TypeError
        If ``d`` is not an integer (``bool`` included).
    ValueError
        If ``d < 1``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.filter.nu_d(np.array([0.0, 0.5, 1.0]))
    array([0. , 0.5, 1. ])
    """
    require_int("d", d)
    if d < 1:
        # betainc(0, 0, x) is NaN, so a bad d silently yields a NaN filter.
        raise ValueError(f"d must be >= 1, got {d}")
    return betainc(d, d, np.clip(x, 0.0, 1.0))


def phi_centered(Nf: int, Nt: int, dt: float, d: int = 4) -> Array:
    """Meyer filter $\\hat\\phi(f)$ (Paper I eq:Meyer-def), shape ``(Nt,)``, centered at
    index $N_t/2$, with rolloff smoothness ``d``.

    The grid spans frequencies $f \\in [-N_t/2,\\,N_t/2-1]/T$ with
    $T = N_f N_t \\delta t$.  In the flat region ($|f| < \\Delta F/4$)
    the value is $1/\\sqrt{\\Delta F}$; in the rolloff region
    ($\\Delta F/4 \\le |f| < 3\\Delta F/4$) it follows the Meyer
    cosine taper; otherwise it is zero.

    Raises
    ------
    TypeError
        If ``Nf``, ``Nt`` or ``d`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid or ``dt`` is invalid, or if $\\Delta F = 1/(2\\,\\delta t N_f)$
        is not a positive finite double (see :func:`domainia.wilson.grid.delta_F`).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> phi = wilson.filter.phi_centered(8, 16, 1.0)
    >>> phi.shape, bool(phi[8] == phi.max())   # flat top includes index Nt/2
    ((16,), True)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    dF = delta_F(Nf, dt)
    T = Nf * Nt * float(dt)
    f = (np.arange(Nt) - Nt // 2) / T
    A = dF / 4.0
    B = dF / 2.0
    amp = 1.0 / np.sqrt(dF)

    f_abs = np.abs(f)
    out = np.where(f_abs < A, amp, 0.0)
    rolloff = (f_abs >= A) & (f_abs < A + B)
    out = np.where(
        rolloff,
        amp * np.cos(np.pi / 2.0 * nu_d((f_abs - A) / B, d)),
        out,
    )
    return out
