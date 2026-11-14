"""Explicit WDM transform matrix and noise covariance.

The WDM forward transform is a linear map $w = W\\,x$ where $W$ is a
real $N \\times N$ matrix whose rows are the folded slots in flat order (slot $(i, j)$ is
row $i\\,N_t + j$; :func:`~domania.wilson.grid.cells` gives its band and time).  $W$ is
orthogonal **up to the sampling scale**:

$$W W^T = W^T W = \\delta t\\,I,$$

i.e. $W = \\sqrt{\\delta t}\\,Q$ with $Q$ orthogonal.  Invert with
$x = W^T w / \\delta t$.  The $\\delta t$ is the same factor that
appears in Parseval ($\\sum w^2 = \\delta t \\sum x^2$); $W$ is
exactly orthogonal only when $\\delta t = 1$.

Every builder takes ``window=`` (default :data:`domania.wilson.MEYER`) and returns the
matrix of that window's Wilson basis; the identities above hold for every admissible
window (:mod:`domania.wilson.windows`).

This module exposes:

* :func:`build_W` — the $N \\times N$ matrix ($W W^T = \\delta t\\,I$).
* :func:`build_C_time` — Toeplitz time-domain noise covariance from
  a PSD (vectorized via ``irfft`` of the power spectrum).
* :func:`build_C_wdm` — WDM-domain noise covariance via the similarity
  transform $C^{(\\text{WDM})} = W\\,C^{(\\text{time})}\\,W^T$.
* :func:`rfft_to_real`, :func:`real_to_rfft` — the complex rfft as a real vector of
  length $N$ (the real-DFT representation).

These functions are $O(N^2)$ in memory and $O(N^3)$ in compute: references for
small $N$ (roughly $N \\lesssim 2048$); the FFT-based transforms are in
:mod:`domania.wilson.transform`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .grid import (
    _require_even_positive_int,
    _require_int,
    cells,
    n_samples,
    sanitize_psd,
    validate_dt,
    validate_grid,
)
from .transform import _inverse_time
from .windows import MEYER, Window

type Array = NDArray[np.floating]
type CArray = NDArray[np.complexfloating]
type PsdFunc = Callable[[Array], Array]

__all__ = [
    "build_C_time",
    "build_C_wdm",
    "build_W",
    "real_to_rfft",
    "rfft_to_real",
]


def build_W(
    Nf: int,
    Nt: int,
    dt: float,
    *,
    window: Window = MEYER,
) -> Array:
    """Build the $N \\times N$ WDM transform matrix for ``window``.

    Row $p = i\\,N_t + j$ is the basis function $g_{mn}$ of folded slot $(i, j)$ sampled in
    time ($(m, n)$ from :func:`~domania.wilson.grid.cells`), so $W$ is *defined* by "column
    $k_0$ is the flattened WDM transform of a delta at sample $k_0$".  It is *built* from the
    lattice's shift-covariance instead, which agrees with the definition to rounding.

    Returns
    -------
    ndarray, shape (N, N)
        Real matrix with $W W^T = \\delta t\\,I$.  Rows are the folded slots in flat order
        (``W @ x`` is ``forward_time(x).reshape(N)``); columns are the time samples
        $k_0 = 0, \\ldots, N - 1$.

    Notes
    -----
    Cost: $2 N_f$ inverse transforms plus $O(N^2)$ of copying, and
    $O(N^2)$ memory.

    Examples
    --------
    >>> import numpy as np; from domania import wilson
    >>> W = wilson.matrix.build_W(4, 8, 1.0)
    >>> W.shape, bool(np.allclose(W @ W.T, 1.0 * np.eye(32)))
    ((32, 32), True)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    m, n = cells(Nf, Nt)
    W = np.empty((Nf, Nt, N))

    # Since W^T W = dt*I, row p is exactly dt * inverse(e_p).  The lattice is
    # shift-covariant in steps of TWO time pixels -- one step flips the carrier
    # parity (-1)^{(m+1)n} -- so slots j and j+2 of a row (the same band, in row 0
    # too) differ only by a circular shift of 2*Nf samples.  Probing j = 0 and
    # j = 1 per row and rolling the rest turns N transforms into 2*Nf.
    seed = np.zeros((Nf + 1, Nt))
    for i in range(Nf):
        for j0 in (0, 1):
            seed[:] = 0.0
            seed[m[i, j0], n[i, j0]] = 1.0
            row = dt * _inverse_time(seed, Nf, Nt, dt, window=window)
            for j in range(j0, Nt, 2):
                W[i, j] = np.roll(row, (j - j0) * Nf)

    return W.reshape(N, N)


def build_C_time(N: int, dt: float, Sn_func: PsdFunc) -> Array:
    """Toeplitz time-domain noise covariance of ``N`` samples from a one-sided PSD.

    For stationary noise with one-sided PSD ``Sn_func(f)`` $= S_n(f)$, the autocovariance
    is $R(\\tau) = E[n_k n_{k+\\tau}]$, and the covariance matrix is
    Toeplitz: $C[k, k'] = R(|k - k'|)$.  Built via the inverse rfft
    of the discrete two-sided power spectrum.

    Returns
    -------
    ndarray, shape (N, N)
        Real symmetric Toeplitz covariance matrix.

    Raises
    ------
    TypeError
        If ``N`` is not an integer (``bool`` included), or ``dt`` is not a real number.
    ValueError
        If ``N`` is not positive and even, or ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domania import wilson
    >>> C = wilson.matrix.build_C_time(32, 1.0, lambda f: 1e-3 * np.ones_like(f))
    >>> C.shape, bool(np.allclose(C, C.T))
    ((32, 32), True)
    """
    validate_dt(dt)
    N = _require_int("N", N)
    if N <= 0 or N % 2 != 0:
        raise ValueError(f"N must be a positive even integer, got {N}")
    freqs = np.fft.rfftfreq(N, d=float(dt))
    Sn = sanitize_psd(Sn_func(freqs), "build_C_time")
    sigma2 = N * Sn / (2.0 * dt)  # E[|X_j|^2]

    # Autocovariance R[tau] = (1/N^2) * sum_j (full sigma2_j) e^{2 pi i j tau / N}
    # Two-sided: sigma2_j for j=0,...,N/2 then mirror j=N-j' for j' = 1,...,N/2-1.
    # For the rfft-based inverse, build a "spectrum" g[j] s.t. irfft(g, N)
    # gives us R[tau] up to a normalization.
    # irfft(g, N)[tau] = (1/N) [g[0] + g[N/2] (-1)^tau
    #                          + 2 Re sum_{j=1}^{N/2-1} g[j] e^{2pi i j tau/N}]
    # We want R[tau] = (1/N^2) [sigma2[0] + sigma2[N/2] (-1)^tau
    #                            + 2 sum_{j=1}^{N/2-1} sigma2[j] cos(2pi j tau/N)].
    # So g[j] = sigma2[j] / N (real-valued) yields R = irfft(g, N).
    g = sigma2 / N
    R = np.fft.irfft(g, n=N)

    # Build the Toeplitz matrix; use circular distance to match the discrete
    # autocovariance produced by the irfft (periodic).
    idx = np.arange(N)
    tau = np.abs(idx[:, None] - idx[None, :])
    tau = np.minimum(tau, N - tau)
    return R[tau]


def build_C_wdm(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: PsdFunc,
    *,
    window: Window = MEYER,
) -> Array:
    """WDM-domain covariance via the similarity transform $W\\,C^{(\\text{time})}\\,W^T$.

    Returns the $N \\times N$ covariance matrix of the WDM coefficients, rows and columns in
    :func:`build_W`'s folded flat order, under stationary noise with PSD $S_n(f)$.
    Block-tridiagonal in the band index $m$ for every admissible ``window``
    (compact support inside one slab): only $|m - m'| \\le 1$ blocks are
    nonzero (up to numerical noise).  Row 0 holds both edge bands, so in slot rows this is
    block-tridiagonal plus the $(0, N_f - 1)$ corner pair.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric real covariance matrix.

    Examples
    --------
    >>> import numpy as np; from domania import wilson
    >>> C = wilson.matrix.build_C_wdm(4, 8, 1.0, lambda f: 1e-3 * np.ones_like(f))
    >>> C.shape, float(np.round(np.diag(C)[0], 6))   # ~ S_n / 2
    ((32, 32), 0.0005)
    """
    N = n_samples(Nf, Nt)
    W = build_W(Nf, Nt, dt, window=window)
    C_time = build_C_time(N, dt, Sn_func)
    return W @ C_time @ W.T


def rfft_to_real(x_rfft: CArray, N: int) -> Array:
    """Pack a complex rfft into a real vector of length $N$.

    The packing is

    $$\\bm{x}_R = (\\text{Re}\\,X_0,\\,\\text{Re}\\,X_1,\\,\\text{Im}\\,X_1,
    \\,\\ldots,\\,\\text{Re}\\,X_{N/2-1},\\,\\text{Im}\\,X_{N/2-1},
    \\,\\text{Re}\\,X_{N/2})^T,$$

    yielding $1 + 2(N/2 - 1) + 1 = N$ entries, from ``x_rfft`` of shape ``(N//2+1,)``.

    Examples
    --------
    >>> import numpy as np; from domania import wilson
    >>> x = np.random.default_rng(0).standard_normal(32)
    >>> packed = wilson.matrix.rfft_to_real(np.fft.rfft(x), 32)
    >>> packed.shape
    (32,)
    >>> wilson.matrix.rfft_to_real(np.fft.rfft(x[:5]), 5)
    Traceback (most recent call last):
    ValueError: N must be even, got 5
    """
    # Interleave Re/Im of bins j = 1..N/2-1 into slots (2j-1, 2j), i.e. the
    # strided ranges 1:N-2:2 and 2:N-1:2.  Even N only: for odd N the last stored bin
    # is not Nyquist and its imaginary part would be silently dropped, yet N//2+1 is
    # still the right bin count.
    N = _require_even_positive_int("N", N)
    x_rfft = np.asarray(x_rfft)
    if x_rfft.shape != (N // 2 + 1,):
        raise ValueError(
            f"x_rfft must have shape ({N // 2 + 1},) for N={N}; got {x_rfft.shape}"
        )
    xf = np.zeros(N)
    interior = x_rfft[1 : N // 2]
    xf[0] = x_rfft[0].real
    xf[1 : N - 2 : 2] = interior.real
    xf[2 : N - 1 : 2] = interior.imag
    xf[N - 1] = x_rfft[N // 2].real
    return xf


def real_to_rfft(x_real: Array, N: int) -> CArray:
    """Inverse of :func:`rfft_to_real`: ``(N,)`` real to ``(N//2+1,)`` complex.

    Examples
    --------
    >>> import numpy as np; from domania import wilson
    >>> x = np.random.default_rng(0).standard_normal(32)
    >>> packed = wilson.matrix.rfft_to_real(np.fft.rfft(x), 32)
    >>> back = wilson.matrix.real_to_rfft(packed, 32)
    >>> bool(np.allclose(back, np.fft.rfft(x)))
    True
    >>> wilson.matrix.real_to_rfft(np.zeros(5), 5)
    Traceback (most recent call last):
    ValueError: N must be even, got 5
    """
    N = _require_even_positive_int("N", N)
    x_real = np.asarray(x_real)
    if x_real.shape != (N,):
        raise ValueError(f"x_real must have shape ({N},); got {x_real.shape}")
    n_rfft = N // 2 + 1
    out = np.zeros(n_rfft, dtype=complex)
    out[0] = x_real[0]
    out[1 : N // 2] = x_real[1 : N - 2 : 2] + 1j * x_real[2 : N - 1 : 2]
    out[N // 2] = x_real[N - 1]
    return out
