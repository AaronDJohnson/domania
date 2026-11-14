"""Forward and inverse WDM transforms (RFFT-only, linear-centred slab).

These implementations follow Algorithms 1 and 2 of
``notes/Time_Frequency_Analysis/wdm_transform_algorithms.tex``: the
data lives only in the one-sided ``rfft`` $X[k]$, $k = 0, \\ldots, N/2$;
the two-sided spectrum is never materialised.  Out-of-range slab
indices are served by Hermitian symmetry on read (forward) or
discarded (inverse, where they are redundant by Hermitian symmetry of
the real WDM wavelets).

Every kernel here takes ``window=`` (a :class:`domainia.wilson.windows.Window`;
default :data:`domainia.wilson.MEYER`, the Meyer window of the WDM transform) and
computes in the Wilson basis of that window: $\\hat\\phi_c$ below is the window's
centred samples, from :func:`domainia.wilson.windows.sample_window`.  The formulas
and the edge normalisation are the same for every admissible window.

The public kernels take and return the folded layout ``(..., Nf, Nt)``
(:func:`domainia.wilson.grid.cells`).  Each computes through a private twin (``_forward``, ...)
on the padded ``(..., Nf+1, Nt)`` grid, where row ``m`` is band ``m``; the other kernels call
the twins.

The slab is linear-centred:

$$\\ell_{m,k} \\equiv k + (m - 1)\\,N_t/2,\\qquad k = 0, \\ldots, N_t - 1.$$

Interior bands ($1 \\le m \\le N_f - 1$) read directly from the rfft;
only the DC band ($m = 0$) and the Nyquist band ($m = N_f$) ever take
the Hermitian-symmetry branch ($N_t$ slab fetches each).

Forward:

$$w_{mn} = \\frac{\\sqrt{2}}{N_f}\\,(-1)^{(m+1)n}\\,
\\Re\\!\\left[C_{mn}^{*}\\,\\mathrm{IFFT}\\!\\bigl(u_m\\bigr)[n]\\right],$$

with $u_m[k] = X_{\\ell_{m,k}}\\,\\hat\\phi_c[k]$ (centred filter,
peaked at $k = N_t/2$) and $C_{mn} = 1$ if $m+n$ even, $C_{mn} = i$
otherwise.

Inverse:

$$X[\\ell_{m,k}] \\mathrel{+}= \\frac{1}{\\sqrt{2}\\,\\delta t}\\,
\\hat\\phi_c[k]\\,\\mathrm{FFT}\\!\\bigl(w_{m\\cdot}\\,C_{mn}\\,
(-1)^{(m+1)n}\\bigr)[k],$$

with writes outside $[0, N/2]$ discarded.

The DC and Nyquist edge-band coefficients are divided by $\\sqrt{2}$ so
that all basis functions have unit norm.  This makes the WDM transform
matrix orthogonal up to the sampling scale ($W_a W_a^T = \\delta t\\,I$,
so invert with $x = W_a^T w/\\delta t$) and the Parseval identity exact:
$\\sum_{m,n} w_{mn}^2 = \\delta t \\sum_k x_k^2$.  This is the package's only
coefficient convention.

Precision
---------
Every entry point computes in double and returns float64 (complex128 for a
spectrum).  Input of another accepted dtype (a lower-precision float, an
integer) is promoted first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._checks import REAL_KINDS, require_dtype, require_int
from .grid import (
    _fold_fix,
    active_mask,
    cells,
    n_samples,
    unfold,
    validate_dt,
    validate_finite,
    validate_grid,
)
from .windows import MEYER, Window, require_window, sample_window

type Array = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type CArray = NDArray[np.complexfloating]

__all__ = [
    "atom_spectrum",
    "base_atom_spectra",
    "forward",
    "forward_time",
    "inverse",
    "inverse_time",
]

# ---------------------------------------------------------------------------
# FFT backend: numpy, deliberately
# ---------------------------------------------------------------------------
# ``scipy.fft`` is faster on a bare FFT but not end to end here (the parity assembly and
# the scatter dominate), and its rounding differs from numpy's.


def _parity(m: int | np.integer, n_arr: NDArray[np.integer]) -> tuple[BoolArray, Array]:
    """``((m+n) even, (-1)^{(m+1)n})`` for one band.

    :func:`domainia.wilson.stationary._carriers` holds the 2-D form of the same rule.
    """
    even = (m + n_arr) % 2 == 0
    sign = 1.0 - 2.0 * (((m + 1) * n_arr) % 2)
    return even, sign


def _build_slab(X_rfft: CArray, m: int, Nf: int, Nt: int) -> CArray:
    """Linear-centred slab $u_m[k] = X[\\ell_{m,k}]$ for $k \\in [0, N_t-1]$:
    ``(..., N//2+1)`` complex to ``(..., Nt)`` complex.

    Hermitian extension is applied inline: negative indices use the
    conjugate of the positive bin (only fires at $m = 0$, $k < N_t/2$);
    over-Nyquist indices use the conjugate of the mirror bin (only
    fires at $m = N_f$, $k > N_t/2$).  For interior bands every lookup
    is a direct rfft read.
    """
    N = n_samples(Nf, Nt)
    ell = np.arange(Nt) + (m - 1) * (Nt // 2)
    # Every slot is written below.
    slab = np.zeros((*X_rfft.shape[:-1], Nt), dtype=complex)
    neg = np.flatnonzero(ell < 0)
    over = np.flatnonzero(ell > N // 2)
    direct = np.flatnonzero((ell >= 0) & (ell <= N // 2))
    slab[..., direct] = X_rfft[..., ell[direct]]
    if neg.size:
        slab[..., neg] = np.conj(X_rfft[..., -ell[neg]])
    if over.size:
        slab[..., over] = np.conj(X_rfft[..., N - ell[over]])
    return slab


def _scatter_interior_bands(
    X: CArray,
    wmn_work: Array,
    Nf: int,
    Nt: int,
    phi_c: Array,
    lam: float,
) -> CArray:
    """Accumulate the interior bands into the rfft accumulator ``X``; returns it.

    One batched length-$N_t$ FFT, then a contiguous slice-add per band. Within
    a band the slab indices are consecutive and duplicate-free, so ``X[a:b] +=``
    is exact and much cheaper than ``np.add.at``; consecutive bands overlap by
    $N_t/2$, which the sequential loop handles correctly.

    ``X`` is ``(..., N//2+1)`` and ``wmn_work`` ``(..., Nf+1, Nt)``.  There is always an
    interior band: :func:`domainia.wilson.grid.validate_grid` requires ``Nf`` even, so
    ``Nf >= 2``.
    """
    n_arr = np.arange(Nt)
    m_int = np.arange(1, Nf)
    m_par = m_int & 1
    n_par = n_arr & 1
    even_int = m_par[:, None] == n_par[None, :]
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
    scaled_int = wmn_work[..., 1:Nf, :] * sign_int
    # g = w' * C_mn * (-1)^{(m+1)n}: real where (m+n) even, imaginary otherwise.
    G = np.where(even_int, scaled_int, 0.0) + 1j * np.where(even_int, 0.0, scaled_int)
    W_int = np.fft.fft(G, axis=-1) * (lam * phi_c)
    # Sequential slice-adds in band order.
    # Every rfft bin receives at most two addends (adjacent interior bands
    # overlap by Nt/2; the edge bands only touch bins also covered by band 1 or
    # band Nf-1, plus bin N/2 alone), and a + b == b + a in IEEE arithmetic, so
    # the order is immaterial.  Each slice is duplicate-free.
    for i, m_i in enumerate(m_int):
        start = int(m_i - 1) * (Nt // 2)
        X[..., start:start + Nt] += W_int[..., i, :]
    return X


def _scatter_edge_bands(
    X: CArray, wmn_work: Array, Nf: int, Nt: int, N: int, phi_c: Array, lam: float,
) -> CArray:
    """Accumulate the DC and Nyquist bands into ``X``; returns it.

    These scatter partly outside $[0, N/2]$; those writes are redundant by
    Hermitian symmetry of the real WDM wavelets and are discarded.  ``ell`` is strictly
    increasing, so the surviving indices are duplicate-free and a plain ``+=`` is exact.
    """
    n_arr = np.arange(Nt)
    for m in (0, Nf):
        even, sign = _parity(m, n_arr)
        scaled = wmn_work[..., m, :] * sign
        g = np.where(even, scaled, 0.0) + 1j * np.where(even, 0.0, scaled)
        Wm = np.fft.fft(g, axis=-1)
        ell = n_arr + (m - 1) * (Nt // 2)
        valid = np.flatnonzero((ell >= 0) & (ell <= N // 2))
        X[..., ell[valid]] += lam * phi_c[valid] * Wm[..., valid]
    return X


def _gather_bands(x_rfft: CArray, Nf: int, Nt: int, dt: float, *,
                  window: Window = MEYER) -> Array:
    """Coefficient grid(s) from spectra: ``(..., N//2+1) -> (..., Nf+1, Nt)``.

    The whole of the forward algorithm; :func:`_forward` validates and calls it.

    Interior bands $m = 1 \\ldots N_f-1$: every slab index
    $\\ell_{m,k} = k + (m-1)N_t/2$ lies in $[0, N/2]$
    (max $= (N_f-1)N_t/2 - 1 < N/2$), so the slab is a pure gather from the
    rfft with no Hermitian branch, and all of them go through one batched
    length-$N_t$ IFFT.  Edge bands $m \\in \\{0, N_f\\}$ take the
    Hermitian-symmetry slab path.
    """
    phi_c = sample_window(window, Nf, Nt, dt)
    kappa = np.sqrt(2.0) / Nf
    n_arr = np.arange(Nt)
    out: Array = np.zeros((*x_rfft.shape[:-1], Nf + 1, Nt))

    m_int = np.arange(1, Nf)
    ell = (m_int - 1)[:, None] * (Nt // 2) + n_arr[None, :]  # (Nf-1, Nt)
    Y = np.fft.ifft(x_rfft[..., ell] * phi_c, axis=-1)

    # The parity patterns depend only on the parities of m and n, so build them
    # from 1D parity vectors instead of forming (m+1)*n over the whole grid.
    m_par = m_int & 1
    n_par = n_arr & 1
    even_int = m_par[:, None] == n_par[None, :]  # (m+n) even <=> parities equal
    # Re(C_mn^* y_m[n]): even -> Re(y_m); odd -> Re(-i y_m) = Im(y_m).
    re_or_im_int = np.where(even_int, Y.real, Y.imag)
    # (-1)^{(m+1)n}; exponent parity = ((m+1) & 1) * (n & 1).
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
    out[..., 1:Nf, :] = kappa * sign_int * re_or_im_int

    for m in (0, Nf):
        y_m = np.fft.ifft(_build_slab(x_rfft, m, Nf, Nt) * phi_c, axis=-1)
        even, sign = _parity(m, n_arr)
        out[..., m, :] = kappa * sign * np.where(even, y_m.real, y_m.imag)

    # Unit-norm edge basis functions (module docstring).
    root2 = np.sqrt(2.0)
    out[..., 0, :] /= root2
    out[..., Nf, :] /= root2
    return out


def _forward(x_rfft: CArray, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
    """:func:`forward` on the padded grid: ``(..., N//2+1)`` to ``(..., Nf+1, Nt)``."""
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    x_rfft = np.asarray(x_rfft)
    if x_rfft.shape[-1:] != (N // 2 + 1,):
        raise ValueError(
            f"x_rfft must have shape (..., {N // 2 + 1}) for Nf={Nf}, Nt={Nt}; "
            f"got {x_rfft.shape}"
        )
    validate_finite(x_rfft, "forward: x_rfft")
    return _gather_bands(x_rfft, Nf, Nt, dt, window=window)


def forward(
    x_rfft: CArray,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    window: Window = MEYER,
) -> Array:
    """Forward WDM transform (Algorithm 1): $\\text{rfft}(x) \\to w_{mn}$.

    Parameters
    ----------
    x_rfft : ndarray, complex, shape (..., N//2+1)
        One-sided DFT of the real time series, as returned by
        ``np.fft.rfft(x)`` (no $\\delta t$ scaling).  Any leading axes: each
        slice is transformed on its own, and a stack goes through one batched FFT.

    Returns
    -------
    ndarray, float, shape (..., Nf, Nt)
        Real WDM coefficients in the folded layout: slot ``(i, j)`` holds band ``m[i, j]`` at
        time ``n[i, j]``, with ``m, n = wilson.grid.cells(Nf, Nt)``.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` or ``Nt`` is not positive and even, if the last axis of
        ``x_rfft`` does not have length ``N//2 + 1``, or if it holds a ``nan`` or
        ``inf`` (:func:`domainia.wilson.grid.validate_finite`).

    Notes
    -----
    Cost: $O(N \\log N_t)$.  The $N_f-1$ interior bands go through one batched
    length-$N_t$ IFFT, so the working set is $O(N_f N_t)$, not the $O(N_t)$ of a
    per-band loop.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> w.shape
    (8, 16)
    >>> bool(np.isclose(np.sum(w**2), dt * np.sum(x**2)))  # Parseval
    True
    >>> xs = np.random.default_rng(1).standard_normal((2, 3, Nf * Nt))   # leading axes
    >>> wilson.forward(np.fft.rfft(xs), Nf, Nt, dt).shape
    (2, 3, 8, 16)
    """
    m, n = cells(Nf, Nt)
    return _forward(x_rfft, Nf, Nt, dt, window=window)[..., m, n]


def _inverse(wmn: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> CArray:
    """:func:`inverse` on the padded grid ``(..., Nf+1, Nt)``, whose inactive cells must be 0."""
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    wmn = np.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.shape[-2:] != (Nf + 1, Nt):
        raise ValueError(f"wmn must have shape (..., {Nf + 1}, {Nt}); got {wmn.shape}")
    validate_finite(wmn, "inverse: wmn")

    phi_c = sample_window(window, Nf, Nt, dt)
    wmn_work = wmn.astype(np.float64)              # a fresh copy, so scaling it in place is safe
    wmn_work[..., [0, Nf], :] *= np.sqrt(2.0)      # undo forward's unit-norm edge scaling

    lam = 1.0 / (np.sqrt(2.0) * dt)
    X = np.zeros((*wmn.shape[:-2], N // 2 + 1), dtype=complex)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
    return X


def inverse(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    window: Window = MEYER,
) -> CArray:
    """Inverse WDM transform (Algorithm 2): $w_{mn} \\to \\text{rfft}(x)$.

    Each band scatters into the one-sided rfft accumulator at its linear-centred slab
    indices.

    Parameters
    ----------
    wmn : ndarray, float, shape (..., Nf, Nt)
        WDM coefficients in the folded layout of :func:`forward`, with any leading axes.

    Returns
    -------
    ndarray, complex, shape (..., N//2+1)
        Reconstructed one-sided DFT (compatible with ``np.fft.irfft``).

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, ``dt`` is not a real number, or ``wmn``
        is not a real numeric array: a complex grid would have its imaginary part
        silently discarded, and a bool, str or object grid is refused too.
    ValueError
        If the grid is invalid, if ``wmn`` does not end in ``(Nf, Nt)`` (for a padded
        ``(..., Nf+1, Nt)`` or a flat ``(..., N)`` array the message names the fix,
        :func:`domainia.wilson.grid.fold` or ``.reshape(..., Nf, Nt)``), or if it holds a
        ``nan`` or ``inf`` (see :func:`domainia.wilson.grid.validate_finite`).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> bool(np.allclose(wilson.inverse(w, Nf, Nt, dt), np.fft.rfft(x)))
    True
    >>> wilson.inverse(w + 0j, Nf, Nt, dt)
    Traceback (most recent call last):
    TypeError: wmn must be a real numeric array, got dtype complex128
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    wmn = np.asarray(wmn)
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.shape[-2:] != (Nf, Nt):
        raise ValueError(f"wmn must have shape (..., {Nf}, {Nt}); got {wmn.shape}"
                         + _fold_fix(wmn.shape, Nf, Nt))
    validate_finite(wmn, "inverse: wmn")
    return _inverse(unfold(wmn, Nf, Nt), Nf, Nt, dt, window=window)


def forward_time(
    x: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    window: Window = MEYER,
) -> Array:
    """Time-domain forward transform, ``forward(np.fft.rfft(x), ...)``: a real ``x`` of shape
    ``(..., N)``, $N = N_f N_t$, to the folded ``(..., Nf, Nt)``.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if the last axis of ``x`` does not have ``Nf*Nt``
        samples, or if it holds a ``nan`` or ``inf`` (gapped data often arrives
        ``nan``-filled: fill the gaps first).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> wilson.forward_time(x, Nf, Nt, dt).shape
    (8, 16)
    >>> x[5] = np.nan                      # one gap-filled sample
    >>> wilson.forward_time(x, Nf, Nt, dt)    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: forward_time: x: 1 of 128 entries are not finite...
    """
    validate_grid(Nf, Nt)
    x = np.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.shape[-1:] != (expected,):
        raise ValueError(
            f"x must have shape (..., {expected}) = (..., Nf*Nt) for Nf={Nf}, Nt={Nt}; "
            f"got {x.shape}"
        )
    # Check here rather than only on the rfft: reporting "index 5 of x" beats
    # reporting that every bin of the transformed spectrum went non-finite.
    validate_finite(x, "forward_time: x")
    # At least double before the rfft, which keeps a lower-precision float's precision.
    # A complex x stays complex, and the rfft refuses it.
    x_rfft = np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False))
    return forward(x_rfft, Nf, Nt, dt, window=window)


def _inverse_time(wmn: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
    """:func:`inverse_time` on the padded grid, whose inactive cells must be 0."""
    return np.fft.irfft(_inverse(wmn, Nf, Nt, dt, window=window), n=n_samples(Nf, Nt))


def inverse_time(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    window: Window = MEYER,
) -> Array:
    """Time-domain inverse transform, ``np.fft.irfft(inverse(wmn, ...), n=N)``:
    the folded ``(..., Nf, Nt)`` to ``(..., N)``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward_time(x, Nf, Nt, dt)
    >>> bool(np.allclose(wilson.inverse_time(w, Nf, Nt, dt), x, atol=1e-12))
    True
    >>> w8 = wilson.forward_time(x, Nf, Nt, dt, window=wilson.Meyer(8))       # another window
    >>> bool(np.allclose(wilson.inverse_time(w8, Nf, Nt, dt, window=wilson.Meyer(8)), x,
    ...                  atol=1e-12))
    True
    """
    return np.fft.irfft(inverse(wmn, Nf, Nt, dt, window=window), n=n_samples(Nf, Nt))


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def base_atom_spectra(Nf: int, Nt: int, dt: float, *,
                      window: Window = MEYER) -> dict[tuple[int, int], CArray]:
    """rfft of the atoms of the cells ``(m, a)``, ``a in {0, 1}`` (active only).

    Every other atom of band ``m`` is a circular shift of one of these by an
    even number of columns (:func:`atom_spectrum`).  ``dt * inverse(e)``:
    :func:`inverse` returns ``rfft(W_a^{-1} e) = rfft(atom) / dt``.
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    mask = active_mask(Nf, Nt)
    out: dict[tuple[int, int], CArray] = {}
    for m in range(Nf + 1):
        for a in (0, 1):
            if not mask[m, a]:
                continue
            e = np.zeros((Nf + 1, Nt))
            e[m, a] = 1.0
            out[m, a] = dt * _inverse(e, Nf, Nt, dt, window=window)
    return out


def _check_base_spectrum(base: dict[tuple[int, int], CArray], G: CArray, Nf: int, Nt: int,
                         dt: float) -> None:
    """Reject a ``base`` that :func:`base_atom_spectra` did not build for ``(Nf, Nt, dt)``.

    An ``O(N)`` fingerprint: ``2 Nf`` entries pin ``Nf``; the length ``N//2+1`` of ``G``
    then pins ``Nt``; the energy of the atom pins ``dt``, because every active atom has
    energy ``dt`` to rounding (``W_a W_a^T = dt I``).  ``N = Nf Nt`` is even, so the last
    bin is the Nyquist bin.  The window is not pinned: the atoms of every admissible window
    have energy ``dt``.
    """
    N = n_samples(Nf, Nt)
    if len(base) != 2 * Nf or np.shape(G) != (N // 2 + 1,):
        raise ValueError(
            f"base does not hold the base atoms of the ({Nf}, {Nt}) grid: "
            f"{len(base)} entries of shape {np.shape(G)}, expected {2 * Nf} of "
            f"shape ({N // 2 + 1},); build it with base_atom_spectra({Nf}, {Nt}, dt, window=...)"
        )
    G = np.asarray(G)
    energy = (2.0 * np.vdot(G, G).real - abs(G[0]) ** 2 - abs(G[-1]) ** 2) / N
    if not abs(energy - dt) <= 1e-9 * dt:
        raise ValueError(
            f"base was not built for dt={dt}: its atoms have energy {energy:.17g}, "
            f"and every atom of base_atom_spectra({Nf}, {Nt}, dt, window=...) has energy dt"
        )


def atom_spectrum(m: int, n: int, Nf: int, Nt: int, dt: float, *,
                  base: dict[tuple[int, int], CArray] | None = None,
                  window: Window = MEYER) -> CArray:
    """rfft of the atom of the active cell ``(m, n)`` (length ``N//2+1``); ``n`` must lie in
    ``0..Nt-1`` (no wrap-around).

    Pass ``base`` (:func:`base_atom_spectra` of the grid) to avoid recomputing it.  A
    ``base`` built for another ``(Nf, Nt, dt)`` is rejected; one built with another window
    is not detected.  With ``base`` given, ``window`` is unused, but it must still be a
    window (``TypeError`` otherwise).
    """
    require_window(window)
    validate_grid(Nf, Nt)
    validate_dt(dt)
    require_int("m", m)
    require_int("n", n)
    if not 0 <= n < Nt:
        raise ValueError(f"n must lie in 0..{Nt - 1} on the ({Nf}, {Nt}) grid; got {n}")
    base = base_atom_spectra(Nf, Nt, dt, window=window) if base is None else base
    a = n % 2
    if (m, a) not in base:
        raise ValueError(f"({m}, {n}) is not an active cell of the ({Nf}, {Nt}) grid")
    _check_base_spectrum(base, base[m, a], Nf, Nt, dt)
    k = np.arange(n_samples(Nf, Nt) // 2 + 1)
    return base[m, a] * np.exp(-2j * np.pi * k * (n - a) / Nt)
