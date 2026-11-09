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
centred samples, from :func:`domainia.wilson.windows.sample_window`.  The formulas,
the edge normalisation and the precision rules are the same for every admissible
window.

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

Both algorithms produce $w_{mn}$ identical to the FFT-natural
implementation in earlier versions of this module to machine precision
(the slab/carrier reordering is algebraically equivalent, see the note
``wdm_transform_algorithms.tex`` \\S Differences).

The DC and Nyquist edge-band coefficients are divided by $\\sqrt{2}$ so
that all basis functions have unit norm.  This makes the WDM transform
matrix orthogonal up to the sampling scale ($W_a W_a^T = \\delta t\\,I$,
so invert with $x = W_a^T w/\\delta t$) and the Parseval identity exact:
$\\sum_{m,n} w_{mn}^2 = \\delta t \\sum_k x_k^2$.

This is the package's **only** coefficient convention -- there is no flag.
The raw "code" alternative, in which the edge basis functions have
$\\|g_{0n}\\|^2 = 2$, used to be selectable via ``normalize_edges=False``
throughout the package.  It bought nothing but a way to get the edge bands
silently wrong: the two conventions differ by a factor of two on the edge
bands and nowhere else, no downstream function could detect a mismatch, and
every non-trivial consumer (``matrix``, ``stationary``, ``inner_product``)
either assumed or hard-coded the normalized one anyway.

Precision
---------
Single precision propagates: a ``float32`` series (or ``complex64`` spectrum)
is transformed in single precision and returns ``float32``.  Every constant
that multiplies a working array (the sign patterns, ``kappa``, ``lam``, the
window samples, the edge-band ``sqrt(2)``, the band phases) is cast to the
working dtype first, so every FFT receives a single-precision array, on NumPy
and on JAX.  Anything else, integers included, promotes to double.  It is
opt-in by dtype rather than by flag, so the precision of a result is readable
off its own ``.dtype``.

A single-precision run that overflows raises ``ValueError`` instead of
returning ``inf`` or ``nan``.  float32 tops out at 3.4e38 and the window
samples grow with ``sqrt(dt)``, so finite input can overflow inside a product
or an FFT (complex64 bins of order 1e36 at ``dt=1e4`` do, in
``forward_bands``), and NumPy only warns on the way.  Each of the ten entry
points (``forward``/``inverse``, their ``_time`` and ``_many`` forms, and
``forward_bands``/``inverse_bands``) checks the single-precision arrays it
computes, and the message names the function whose array overflowed and
suggests float64 input.  The check is host work, like the input check: it is
skipped for double results and under ``jax.jit``, where a tracer has no values
to read and an overflow still comes back as ``inf``/``nan``.

float64 runs are bit for bit what they were before the casts (a cast to
float64 is a no-op, and no float64 expression is regrouped).  The Meyer
baseline, ``tests/wilson/test_meyer_baseline.py``, pins them bit for bit
except the rescaled families (``forward_bands``, ``inverse_bands``,
``FourierToWDM``), which it compares by tolerance.  float32 runs differ
from the frozen oracle at single-precision rounding, because the oracle formed
some products in double before storing them.

**This is a memory feature, not a speed one**, and the docstrings say so
because the obvious guess is wrong.  Measured end to end it is 0.87-1.19x --
i.e. nothing -- for the same reason ``scipy.fft`` was reverted above: the FFT
is not the bottleneck.  What it buys is half the memory.  Round-trip accuracy
drops to ~2e-7 relative, from ~4e-16.

Where it reaches, and where it stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **The transform** -- ``forward``/``inverse`` and their ``_time`` and
  ``_many`` forms, and the band kernels ``forward_bands``/``inverse_bands`` --
  follows the input dtype.
* **The variance builders** ``whittle_variance``, ``band_averaged_variance``
  and ``evolutionary_variance`` follow the *PSD's* dtype, so a ``float32`` PSD
  paired with a ``float32`` series keeps the whole coefficient/variance
  working set single-precision.
* **``empirical_variance`` stays double** on purpose. It is a Monte-Carlo
  reference; computing it in single precision would add numerical noise to a
  noise estimate, which defeats the point of having it.
* **``matrix`` and ``stationary`` stay double** on purpose. They are $O(N^2)$
  verification and covariance machinery, where precision *is* the deliverable
  and the arrays are too large to want in the first place.
* **The maps of ``domainia.maps`` compute in double.** ``TimeToWDM``,
  ``WDMToTime``, ``TimeToFourier`` and ``FourierToWDM(impl="fft")`` convert their
  input to float64 (bins to complex128) before they call these kernels, so a
  ``float32`` record or ``complex64`` bins give float64 results there.  Single
  precision is reached only by calling the kernels here directly.

An earlier revision claimed "half the memory" while only the coefficient grid
was single -- the variance grid it is invariably paired with stayed double, so
the real working set was ~33% smaller, not 50%. Both are single now, and the
claim is measured rather than assumed.

Before any of this, ``float32`` input was silently upcast and returned
``float64`` -- twice the memory the caller asked for, with no indication.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._checks import REAL_KINDS, index_array, require_dtype, require_int
from .backend import NUMPY, Backend, get_backend, is_tracer
from .grid import (
    active_mask,
    carrier_phase,
    n_samples,
    validate_dt,
    validate_finite,
    validate_grid,
)
from .windows import MEYER, Window, require_window, sample_window

_F64 = np.dtype(np.float64)
_C128 = np.dtype(np.complex128)

type Array = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type CArray = NDArray[np.complexfloating]

__all__ = [
    "atom_spectrum",
    "band_atoms",
    "base_atom_spectra",
    "forward",
    "forward_bands",
    "forward_many",
    "forward_time",
    "forward_time_many",
    "inverse",
    "inverse_bands",
    "inverse_many",
    "inverse_time",
    "inverse_time_many",
]

# ---------------------------------------------------------------------------
# FFT backend: numpy, deliberately
# ---------------------------------------------------------------------------
# ``scipy.fft`` was tried and reverted. Recorded here so it is not retried on
# the strength of a microbenchmark:
#
#   * Isolated, scipy IS faster: 1.64x single-threaded and 2.78x at
#     ``workers=8`` on a bare (255, 512) complex ifft.
#   * In this transform it is worth **0.93-1.14x** end to end, measured across
#     four grids from Nf=64 to Nf=1024. The FFT is not the bottleneck -- the
#     parity assembly (``np.where`` plus complex construction) and the scatter
#     dominate -- so Amdahl eats the gain.
#   * ``workers=8`` bought 1.00-1.23x for the same reason.
#
# The cost was real: scipy is not bit-identical to numpy (~3.7e-16 relative,
# same algorithm family, different rounding), and the coefficients are kept
# byte-for-byte stable across every change (the Meyer outputs are pinned bit for
# bit by ``tests/wilson/test_meyer_baseline.py``). Trading a verifiable property
# for measurement noise is a bad deal.
#
# Worth revisiting only if the parity assembly is ever vectorised further, at
# which point the FFT share of the runtime rises and the arithmetic changes.


def _working_dtypes(a: NDArray[np.generic]) -> tuple[np.dtype, np.dtype]:
    """``(real, complex)`` working dtypes implied by an input array.

    Single precision propagates: a ``float32`` series or ``complex64`` spectrum
    is transformed in single precision (every constant is cast to the working
    dtype before it multiplies a working array) and comes back ``float32``.
    Anything else -- including integer input -- promotes to double.

    This is opt-in by dtype rather than by flag, which keeps the default exact
    and makes the precision of a result readable off its own ``.dtype`` instead
    of requiring the caller to remember what they passed.
    """
    if a.dtype in (np.float32, np.complex64):
        return np.dtype(np.float32), np.dtype(np.complex64)
    return np.dtype(np.float64), np.dtype(np.complex128)


def _check_finite(values: NDArray[np.generic], context: str) -> None:
    """:func:`domainia.wilson.validate_finite` on concrete input; skipped for JAX tracers.

    Validation is host work, not arithmetic: a tracer has no values to check,
    and a concrete JAX array is checked on the host like a NumPy one.
    """
    if not is_tracer(values):
        validate_finite(values, context)


def _check_single_output(out: NDArray[np.generic], entry: str) -> None:
    """Raise ``ValueError`` if a single-precision result of ``entry`` is not finite.

    The entry points check their input first (:func:`_check_finite`), so a
    ``nan`` or ``inf`` here was made by the transform itself: a float32
    product, FFT or scaling overflowed.  NumPy only warns on the way and JAX
    says nothing, so the caller would get ``nan`` cells.  Each entry point
    checks the arrays it computes itself, so the message names the function
    whose array overflowed.

    Host work, as :func:`_check_finite` is: skipped for a double-precision
    result (float64 runs gain no pass) and for a JAX tracer.  It does not
    change ``np.errstate``.
    """
    if _working_dtypes(out)[0] != np.float32 or is_tracer(out):
        return
    arr = np.asarray(out)
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    n_bad = int(np.count_nonzero(bad))
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    raise ValueError(
        f"{entry}: {n_bad} of {arr.size} entries of the single-precision result are not "
        f"finite ({n_nan} nan, {n_bad - n_nan} inf), from finite input: the computation "
        "overflowed float32, whose largest value is 3.4e38. Pass float64 (or complex128) "
        "input to compute in double."
    )


def _set_rounded(a: Array, idx: object, v: Array, be: Backend) -> Array:
    """``a[idx] = v`` with ``v`` cast to ``a.dtype`` (``copy=False``).

    The transforms form ``v`` in the working dtype (their constants are cast
    first), so the cast is a no-op for them.  It stays as the guard that keeps
    the scatter dtype-safe on JAX: storing float64 values into a float32 array
    warns in jax 0.9 (``FutureWarning``: "scatter inputs have incompatible
    types"), and a later JAX will reject it.
    """
    return be.set(a, idx, be.asarray(v, dtype=a.dtype), copy=False)


def _add_rounded(a: Array, idx: object, v: Array, be: Backend) -> Array:
    """``a[idx] += v`` for a duplicate-free ``idx``, stored in ``a.dtype``.

    Gather, add, cast, set: the functional form of NumPy's in-place ``+=``,
    so float64 bits are the in-place statement's.  The inverse transforms hand
    it a ``v`` already in ``a``'s dtype, so a float32 run adds in single
    precision and the cast is a no-op; the cast keeps the scatter dtype-safe
    on JAX (see :func:`_set_rounded`).
    """
    return be.set(a, idx, be.asarray(a[idx] + v, dtype=a.dtype), copy=False)


def _parity(m: int | np.integer, n_arr: NDArray[np.integer]) -> tuple[BoolArray, Array]:
    """``((m+n) even, (-1)^{(m+1)n})`` for one band.

    The Wilson parity/carrier convention is the most error-prone thing in the
    package -- :func:`domainia.wilson.stationary._carriers` carries the 2-D form of the
    same rule.  Both live in exactly one place each so that a sign fix cannot
    land in one copy and miss another.
    """
    even = (m + n_arr) % 2 == 0
    sign = 1.0 - 2.0 * (((m + 1) * n_arr) % 2)
    return even, sign


def _build_slab(
    X_rfft: CArray, m: int, Nf: int, Nt: int, cdtype: np.dtype = _C128,
    be: Backend = NUMPY,
) -> CArray:
    """Linear-centred slab $u_m[k] = X[\\ell_{m,k}]$ for $k \\in [0, N_t-1]$.

    Hermitian extension is applied inline: negative indices use the
    conjugate of the positive bin (only fires at $m = 0$, $k < N_t/2$);
    over-Nyquist indices use the conjugate of the mirror bin (only
    fires at $m = N_f$, $k > N_t/2$).  For interior bands every lookup
    is a direct rfft read.

    **Shape-generic in the leading axes.** The slab indices depend only on
    ``(m, Nf, Nt)``, never on the data, so a stack of spectra gathers with the
    same index arrays under ``...`` indexing.  :func:`forward_many` used to
    call this once per row inside a list comprehension and ``np.stack`` the
    results -- which left the two edge bands unbatched and cost **85% of
    ``forward_many``'s runtime** at ``Nf=8, Nt=16, B=512``, the small-grid
    many-series case the batched path exists for.

    Parameters
    ----------
    X_rfft : ndarray, complex, shape (..., N//2+1)
    m : int
    Nf, Nt : int
    cdtype : dtype, optional
        Complex working dtype.

    Returns
    -------
    ndarray, complex, shape (..., Nt)
    """
    N = n_samples(Nf, Nt)
    ell = np.arange(Nt) + (m - 1) * (Nt // 2)
    # Every slot is written below, so zeros (not np.empty) costs nothing and is
    # the only allocation JAX offers.  Integer positions replace the boolean
    # masks: the same assignments, and concrete indices for JAX.
    slab = be.xp.zeros((*X_rfft.shape[:-1], Nt), dtype=cdtype)
    neg = np.flatnonzero(ell < 0)
    over = np.flatnonzero(ell > N // 2)
    direct = np.flatnonzero((ell >= 0) & (ell <= N // 2))
    slab = be.set(slab, (..., direct), X_rfft[..., ell[direct]], copy=False)
    if neg.size:
        slab = be.set(slab, (..., neg), be.xp.conj(X_rfft[..., -ell[neg]]), copy=False)
    if over.size:
        slab = be.set(slab, (..., over), be.xp.conj(X_rfft[..., N - ell[over]]), copy=False)
    return slab


def _prepare_edges_for_inverse(
    wmn: Array, Nf: int, Nt: int, rdtype: np.dtype = _F64, be: Backend = NUMPY
) -> Array:
    """Working copy of ``wmn`` with the edge bands put into ``inverse``'s form.

    Two things, both edge-only:

    1. **Zero the parity-killed entries.** They correspond to zero rows of $W$,
       so the adjoint must ignore them. Without this, junk there leaks into the
       reconstruction -- measured: a unit perturbation of one inactive DC pixel
       moved the *active* pixels of ``forward(inverse(w))`` by 2.5 absolute.
       Anything produced by :func:`forward` already carries ~1e-16 there, so
       this is a no-op on well-formed input.
    2. **Undo the unit-norm $\\sqrt2$** that :func:`forward` applied.

    ``active_mask`` is built once. It used to be called twice, once per edge
    band, allocating the whole $(N_f+1)\\times N_t$ boolean array each time to
    read a single row -- 5.1% of ``inverse`` at $N_f=64$, $N_t=128$.

    Shape-generic in the leading axes, so :func:`inverse` and
    :func:`inverse_many` share it.
    """
    work = wmn.astype(rdtype)          # NumPy: always a fresh copy, so copy=False below is safe
    mask = active_mask(Nf, Nt)
    root2 = rdtype.type(np.sqrt(2))   # mask[m] * root2 is then in the working dtype too
    # ``work[..., 0, :] *= mask[0] * root2`` as a functional update.
    work = _set_rounded(work, (..., 0, slice(None)), work[..., 0, :] * (mask[0] * root2), be)
    work = _set_rounded(work, (..., Nf, slice(None)), work[..., Nf, :] * (mask[Nf] * root2), be)
    return work


def _scatter_interior_bands(
    X: CArray,
    wmn_work: Array,
    Nf: int,
    Nt: int,
    phi_c: Array,
    lam: float,
    be: Backend = NUMPY,
) -> CArray:
    """Accumulate the interior bands into the rfft accumulator ``X``; returns it.

    One batched length-$N_t$ FFT, then a contiguous slice-add per band. Within
    a band the slab indices are consecutive and duplicate-free, so ``X[a:b] +=``
    is exact and much cheaper than ``np.add.at``; consecutive bands overlap by
    $N_t/2$, which the sequential loop handles correctly.

    Shape-generic in the leading axes: ``X`` may be ``(N//2+1,)`` or
    ``(B, N//2+1)`` and ``wmn_work`` correspondingly, so :func:`inverse` and
    :func:`inverse_many` run the same code and cannot drift apart.

    There is no ``Nf < 2`` guard because there is no such grid:
    :func:`domainia.wilson.validate_grid` requires ``Nf`` positive **and** even, so the
    smallest legal value is 2 and every caller has at least one interior band.
    """
    n_arr = np.arange(Nt)
    m_int = np.arange(1, Nf)
    m_par = m_int & 1
    n_par = n_arr & 1
    even_int = m_par[:, None] == n_par[None, :]
    # In the working dtype, so a float32 G (and so the FFT) stays complex64;
    # the callers pass ``phi_c`` and ``lam`` in that dtype too.
    sign_int = (1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])).astype(wmn_work.dtype)
    xp = be.xp
    scaled_int = wmn_work[..., 1:Nf, :] * sign_int
    # g = w' * C_mn * (-1)^{(m+1)n}: real where (m+n) even, imaginary otherwise.
    G = xp.where(even_int, scaled_int, 0.0) + 1j * xp.where(even_int, 0.0, scaled_int)
    W_int = be.fft.fft(G, axis=-1) * (lam * phi_c)
    # Sequential slice-adds in band order, exactly the oracle's ``+=`` loop.
    # Every rfft bin receives at most two addends (adjacent interior bands
    # overlap by Nt/2; the edge bands only touch bins also covered by band 1 or
    # band Nf-1, plus bin N/2 alone), and a + b == b + a in IEEE arithmetic, so
    # the order is immaterial on any backend.  Each slice is duplicate-free.
    for i, m_i in enumerate(m_int):
        start = int(m_i - 1) * (Nt // 2)
        X = _add_rounded(X, (..., slice(start, start + Nt)), W_int[..., i, :], be)
    return X


def _scatter_edge_bands(
    X: CArray, wmn_work: Array, Nf: int, Nt: int, N: int, phi_c: Array, lam: float,
    be: Backend = NUMPY,
) -> CArray:
    """Accumulate the DC and Nyquist bands into ``X``; returns it.

    These scatter partly outside $[0, N/2]$; those writes are redundant by
    Hermitian symmetry of the real WDM wavelets and are discarded.

    Shape-generic in the leading axes, as :func:`_scatter_interior_bands`.
    ``ell`` is strictly increasing, so the surviving indices are
    duplicate-free and a plain ``+=`` is exact -- ``np.add.at`` was only ever
    guarding against duplicates that cannot occur here.
    """
    xp = be.xp
    n_arr = np.arange(Nt)
    for m in (0, Nf):
        even, sign = _parity(m, n_arr)
        scaled = wmn_work[..., m, :] * sign.astype(wmn_work.dtype)
        g = xp.where(even, scaled, 0.0) + 1j * xp.where(even, 0.0, scaled)
        Wm = be.fft.fft(g, axis=-1)
        ell = n_arr + (m - 1) * (Nt // 2)
        valid = np.flatnonzero((ell >= 0) & (ell <= N // 2))
        # ``X[..., ell[valid]] += v``: the indices are strictly increasing, so the
        # fancy-index add is one addition per bin either way.
        X = _add_rounded(X, (..., ell[valid]), lam * phi_c[valid] * Wm[..., valid], be)
    return X


def _gather_bands(x_rfft: CArray, Nf: int, Nt: int, dt: float, be: Backend = NUMPY, *,
                  window: Window = MEYER) -> Array:
    """Coefficient grid(s) from spectra: ``(..., N//2+1) -> (..., Nf+1, Nt)``.

    The whole of the forward algorithm, shape-generic in the leading axes, so
    :func:`forward` and :func:`forward_many` are thin validation wrappers over
    one implementation rather than two copies of it.

    Interior bands $m = 1 \\ldots N_f-1$: every slab index
    $\\ell_{m,k} = k + (m-1)N_t/2$ lies in $[0, N/2]$
    (max $= (N_f-1)N_t/2 - 1 < N/2$), so the slab is a pure gather from the
    rfft with no Hermitian branch, and all of them go through one batched
    length-$N_t$ IFFT.  Unconditional: :func:`domainia.wilson.validate_grid` requires $N_f$
    positive **and** even, so $N_f \\ge 2$ and there is always at least one
    interior band.

    Edge bands $m \\in \\{0, N_f\\}$ take the Hermitian-symmetry slab path.
    Two bands, so the Python loop over them is irrelevant to the cost.
    """
    xp = be.xp
    rdtype, cdtype = _working_dtypes(x_rfft)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)          # host constant
    # Every constant below is in the working dtype, so a float32 run stays float32
    # after the FFT (for float64 the casts are no-ops).
    kappa = rdtype.type(np.sqrt(2.0) / Nf)
    n_arr = np.arange(Nt)
    out: Array = xp.zeros((*x_rfft.shape[:-1], Nf + 1, Nt), dtype=rdtype)

    m_int = np.arange(1, Nf)
    ell = (m_int - 1)[:, None] * (Nt // 2) + n_arr[None, :]  # (Nf-1, Nt)
    Y = be.fft.ifft(x_rfft[..., ell] * phi_c, axis=-1)

    # The parity patterns depend only on the parities of m and n, so build them
    # from 1D parity vectors instead of forming (m+1)*n over the whole grid.
    m_par = m_int & 1
    n_par = n_arr & 1
    even_int = m_par[:, None] == n_par[None, :]  # (m+n) even <=> parities equal
    # Re(C_mn^* y_m[n]): even -> Re(y_m); odd -> Re(-i y_m) = Im(y_m).
    re_or_im_int = xp.where(even_int, Y.real, Y.imag)
    # (-1)^{(m+1)n}; exponent parity = ((m+1) & 1) * (n & 1).
    sign_int = (1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])).astype(rdtype)
    # ``out[..., 1:Nf, :] = (kappa * sign_int) * re_or_im_int``, in the working dtype.
    out = _set_rounded(out, (..., slice(1, Nf), slice(None)), kappa * sign_int * re_or_im_int, be)

    for m in (0, Nf):
        y_m = be.fft.ifft(_build_slab(x_rfft, m, Nf, Nt, cdtype, be) * phi_c, axis=-1)
        even, sign = _parity(m, n_arr)
        out = _set_rounded(out, (..., m, slice(None)),
                           kappa * sign.astype(rdtype) * xp.where(even, y_m.real, y_m.imag), be)

    # Unit-norm edge basis functions. Not optional: see the module docstring.
    # ``out[..., 0, :] /= sqrt(2)``, with sqrt(2) in the working dtype.
    root2 = rdtype.type(np.sqrt(2))
    out = _set_rounded(out, (..., 0, slice(None)), out[..., 0, :] / root2, be)
    out = _set_rounded(out, (..., Nf, slice(None)), out[..., Nf, :] / root2, be)
    return out


def forward(
    x_rfft: CArray,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Forward WDM transform: $\\text{rfft}(x) \\to w_{mn}$.

    RFFT-only implementation following Algorithm 1 of
    ``wdm_transform_algorithms.tex``: the full-length two-sided spectrum is
    never built, only the one-sided rfft is read.  Batching the interior bands
    trades memory for speed -- the working set is $O(N_f N_t)$, several times
    the input rfft, not the $O(N_t)$ of a per-band loop.  See the Notes below.

    Parameters
    ----------
    x_rfft : ndarray, complex, shape (N//2+1,)
        One-sided DFT of the real time series, as returned by
        ``np.fft.rfft(x)`` (no $\\delta t$ scaling).
    Nf : int
        Number of frequency bands; $N = N_f N_t$.
    Nt : int
        Number of time bins.
    dt : float
        Sampling interval.
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Notes
    -----
    Cost: $O(N \\log N_t)$.  The $N_f-1$ interior bands are done in one
    batched length-$N_t$ IFFT; only the two edge bands take the
    Hermitian-symmetry path.  Working set is $O(N_f N_t)$ -- the batched
    slab is materialised -- not the $O(N_t)$ of a per-band loop.
    :func:`inverse` costs modestly more than :func:`forward` (it scatters
    into an accumulator rather than gathering).

    Returns
    -------
    ndarray, float, shape (Nf+1, Nt)
        Real WDM coefficients.  Edge bands have $N_t/2$ active values;
        the inactive entries vanish by Hermitian symmetry of the slab,
        realized to floating-point rounding (~1e-16 relative to the
        peak) rather than bit-exactly.  Compare them with a tolerance.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` or ``Nt`` is not positive and even, if ``x_rfft``
        does not have length ``N//2 + 1``, or if it holds a ``nan`` or
        ``inf`` (see :func:`domainia.wilson.validate_finite` -- one bad bin would
        otherwise silently make every output coefficient ``nan``).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> w.shape
    (9, 16)
    >>> bool(np.isclose(np.sum(w**2), dt * np.sum(x**2)))  # Parseval
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    be = be or get_backend(x_rfft)
    x_rfft = be.asarray(x_rfft)
    if x_rfft.shape != (N // 2 + 1,):
        raise ValueError(
            f"x_rfft must have shape ({N // 2 + 1},) for Nf={Nf}, Nt={Nt}; "
            f"got {x_rfft.shape}"
        )
    _check_finite(x_rfft, "forward: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
    _check_single_output(out, "forward")
    return out


def forward_many(
    x_rfft: CArray,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Forward-transform a stack of spectra: $(B, N/2+1) \\to (B, N_f+1, N_t)$.

    Same result as looping :func:`forward` over the first axis, but the whole
    stack goes through **one** batched IFFT and one parity assembly instead of
    $B$ of each.  The saving is the per-call Python and NumPy-dispatch
    overhead, which is what dominates for small grids -- exactly the regime of
    a global fit evaluating many sources on the same grid.

    Bit-identical to the loop, not merely close; :func:`forward` and this
    function run the same arithmetic in the same order.

    Parameters
    ----------
    x_rfft : ndarray, complex, shape (B, N//2+1)
        Stack of one-sided spectra sharing one grid.
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, float, shape (B, Nf+1, Nt)

    Raises
    ------
    ValueError
        If ``x_rfft`` is not 2-D with second axis ``N//2+1``, or holds a
        ``nan``/``inf``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((5, Nf * Nt))
    >>> W = wilson.forward_many(np.fft.rfft(xs, axis=-1), Nf, Nt, dt)
    >>> W.shape
    (5, 9, 16)
    >>> bool(np.allclose(W[2], wilson.forward(np.fft.rfft(xs[2]), Nf, Nt, dt)))
    True
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    be = be or get_backend(x_rfft)
    x_rfft = be.asarray(x_rfft)
    if x_rfft.ndim != 2 or x_rfft.shape[1] != N // 2 + 1:
        raise ValueError(
            f"x_rfft must have shape (B, {N // 2 + 1}) for Nf={Nf}, Nt={Nt}; "
            f"got {x_rfft.shape}"
        )
    _check_finite(x_rfft, "forward_many: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
    _check_single_output(out, "forward_many")
    return out


def forward_time_many(
    x: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: a stack of time series $(B, N) \\to (B, N_f+1, N_t)$.

    Equivalent to ``forward_many(np.fft.rfft(x, axis=-1), ...)``.

    Parameters
    ----------
    x : ndarray, float, shape (B, N)
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, float, shape (B, Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((4, Nf * Nt))
    >>> wilson.forward_time_many(xs, Nf, Nt, dt).shape
    (4, 9, 16)
    """
    validate_grid(Nf, Nt)
    be = be or get_backend(x)
    x = be.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(
            f"x must have shape (B, {expected}) = (B, Nf*Nt) for Nf={Nf}, "
            f"Nt={Nt}; got {x.shape}"
        )
    _check_finite(x, "forward_time_many: x")
    x_rfft = be.fft.rfft(x, axis=-1)
    # Before forward_many's input check, which would report an overflow here as a gap.
    _check_single_output(x_rfft, "forward_time_many")
    return forward_many(x_rfft, Nf, Nt, dt, be=be, window=window)


def inverse(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> CArray:
    """Inverse WDM transform: $w_{mn} \\to \\text{rfft}(x)$.

    RFFT-only implementation following Algorithm 2 of
    ``wdm_transform_algorithms.tex``.  Each band scatters into the
    one-sided rfft accumulator at its linear-centred slab indices;
    writes outside $[0, N/2]$ (only at the edge bands) are discarded
    as redundant by Hermitian symmetry of the real WDM wavelets.

    Parameters
    ----------
    wmn : ndarray, float, shape (Nf+1, Nt)
        WDM coefficient grid.
    Nf, Nt : int
        Frequency and time bin counts.
    dt : float
        Sampling interval.
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, complex, shape (N//2+1,)
        Reconstructed one-sided DFT (compatible with ``np.fft.irfft``).

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, ``dt`` is not a real number, or ``wmn``
        is not a real numeric array: a complex grid would have its imaginary part
        silently discarded, and a bool, str or object grid is refused too.
    ValueError
        If the grid is invalid, if ``wmn`` is wrongly shaped, or
        if it holds a ``nan`` or ``inf`` (see :func:`domainia.wilson.validate_finite`).

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
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    wmn = be.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.shape != (Nf + 1, Nt):
        raise ValueError(f"wmn must have shape ({Nf + 1}, {Nt}); got {wmn.shape}")
    _check_finite(wmn, "inverse: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype, be)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = be.xp.zeros(N // 2 + 1, dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
    _check_single_output(X, "inverse")
    return X


def inverse_many(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> CArray:
    """Inverse-transform a stack: $(B, N_f+1, N_t) \\to (B, N/2+1)$.

    The batched counterpart of :func:`inverse`, and the return leg of
    :func:`forward_many`.  Shares every helper with the unbatched form -- the
    scatter routines are generic in the leading axes -- so the two cannot
    drift apart.  Bit-identical to looping :func:`inverse`.

    Parameters
    ----------
    wmn : ndarray, float, shape (B, Nf+1, Nt)
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, complex, shape (B, N//2+1)

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((4, Nf * Nt))
    >>> W = wilson.forward_time_many(xs, Nf, Nt, dt)
    >>> wilson.inverse_many(W, Nf, Nt, dt).shape
    (4, 65)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    wmn = be.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.ndim != 3 or wmn.shape[1:] != (Nf + 1, Nt):
        raise ValueError(
            f"wmn must have shape (B, {Nf + 1}, {Nt}) for Nf={Nf}, Nt={Nt}; "
            f"got {wmn.shape}"
        )
    _check_finite(wmn, "inverse_many: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype, be)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = be.xp.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
    _check_single_output(X, "inverse_many")
    return X


def inverse_time_many(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: $(B, N_f+1, N_t) \\to$ a stack of series $(B, N)$.

    Equivalent to ``np.fft.irfft(inverse_many(wmn, ...), n=N, axis=-1)``.

    Parameters
    ----------
    wmn : ndarray, float, shape (B, Nf+1, Nt)
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, float, shape (B, N)

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> xs = np.random.default_rng(0).standard_normal((4, Nf * Nt))
    >>> W = wilson.forward_time_many(xs, Nf, Nt, dt)
    >>> bool(np.allclose(wilson.inverse_time_many(W, Nf, Nt, dt), xs, atol=1e-12))
    True
    """
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    x = be.fft.irfft(inverse_many(wmn, Nf, Nt, dt, be=be, window=window), n=N, axis=-1)
    _check_single_output(x, "inverse_time_many")
    return x


def forward_time(
    x: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: time-domain $x_k \\to w_{mn}$.

    Equivalent to ``forward(np.fft.rfft(x), ...)``.

    Parameters
    ----------
    x : ndarray, float, shape (N,)
        Real time series with $N = N_f N_t$ samples.
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, float, shape (Nf+1, Nt)

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if ``x`` does not have ``Nf*Nt`` samples, or
        if it holds a ``nan`` or ``inf``.  Gapped detector data often arrives
        ``nan``-filled; see :func:`domainia.wilson.validate_finite` for why that has to
        be an error rather than something to propagate.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> wilson.forward_time(x, Nf, Nt, dt).shape
    (9, 16)
    >>> x[5] = np.nan                      # one gap-filled sample
    >>> wilson.forward_time(x, Nf, Nt, dt)    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: forward_time: x: 1 of 128 entries are not finite...
    """
    validate_grid(Nf, Nt)
    be = be or get_backend(x)
    x = be.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.shape != (expected,):
        raise ValueError(
            f"x must have shape ({expected},) = (Nf*Nt,) for Nf={Nf}, Nt={Nt}; "
            f"got {x.shape}"
        )
    # Check here rather than only on the rfft: reporting "index 5 of x" beats
    # reporting that every bin of the transformed spectrum went non-finite.
    _check_finite(x, "forward_time: x")
    x_rfft = be.fft.rfft(x)
    # Before forward's input check, which would report an overflow here as a gap.
    _check_single_output(x_rfft, "forward_time")
    return forward(x_rfft, Nf, Nt, dt, be=be, window=window)


def inverse_time(
    wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: $w_{mn} \\to$ time-domain $x_k$.

    Equivalent to ``np.fft.irfft(inverse(wmn, ...), n=N)``.

    Parameters
    ----------
    wmn : ndarray, float, shape (Nf+1, Nt)
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, float, shape (N,)

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
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    x_rfft = inverse(wmn, Nf, Nt, dt, be=be, window=window)
    x = be.fft.irfft(x_rfft, n=N)
    _check_single_output(x, "inverse_time")
    return x


# ---------------------------------------------------------------------------
# Atoms and band-limited analysis
# ---------------------------------------------------------------------------


def base_atom_spectra(Nf: int, Nt: int, dt: float, *,
                      window: Window = MEYER) -> dict[tuple[int, int], CArray]:
    """rfft of the atoms of the cells ``(m, a)``, ``a in {0, 1}`` (active only).

    Every other atom of band ``m`` is a circular shift of one of these by an
    even number of columns (:func:`atom_spectrum`).  ``dt * inverse_many(e)``:
    :func:`inverse` returns ``rfft(W_a^{-1} e) = rfft(atom) / dt``.  With the default
    window (:data:`domainia.wilson.MEYER`) these spectra and the Eq. 45 constants built from
    them are bit-identical to the Meyer baseline (pinned by
    ``tests/wilson/test_meyer_baseline.py``).

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.
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
            out[m, a] = dt * inverse_many(e[None], Nf, Nt, dt, window=window)[0]
    return out


def _check_base_spectrum(base: dict[tuple[int, int], CArray], G: CArray, Nf: int, Nt: int,
                         dt: float) -> None:
    """Reject a ``base`` that :func:`base_atom_spectra` did not build for ``(Nf, Nt, dt)``.

    Host-side fingerprint, ``O(N)`` like the phase ramp it guards: ``2 Nf``
    entries pins ``Nf``; the length ``N//2+1`` of ``G`` then pins ``Nt``; the
    energy of the atom pins ``dt``, because every active atom has energy
    exactly ``dt`` (``W_a W_a^T = dt I``; measured to <= 5e-16 relative).
    ``N = Nf Nt`` is even, so the last bin is the Nyquist bin.  The window is
    not pinned: the atoms of every admissible window have energy ``dt``.
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
    """rfft of the atom of the active cell ``(m, n)`` (length ``N//2+1``).

    Pass ``base`` (:func:`base_atom_spectra`) to avoid recomputing it.  ``n``
    must lie in ``0..Nt-1`` (no wrap-around), and a ``base`` built for another
    ``(Nf, Nt, dt)`` is rejected.

    Parameters
    ----------
    m, n : int
        The active cell: band ``m``, time bin ``n``.
    Nf, Nt : int
    dt : float
    base : dict, optional
        :func:`base_atom_spectra` of the grid; built here (with ``window``) when omitted.
        When ``base`` is given, ``window`` is not used: ``base`` carries the window it was
        built with (from ``base_atom_spectra(..., window=)``), and a base built with
        another window is not detected.
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.  It must be a window even when ``base`` is
        given (``TypeError`` otherwise).
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


def _check_interior_bands(bands: NDArray[np.integer], Nf: int) -> NDArray[np.integer]:
    bands = index_array("bands", bands).ravel()
    if bands.size == 0 or bands.min() < 1 or bands.max() > Nf - 1:
        raise ValueError(f"bands must be interior bands 1..{Nf - 1}; got {bands.tolist()}")
    return bands


def band_atoms(bands: NDArray[np.integer], Nf: int, Nt: int, dt: float, *,
               window: Window = MEYER) -> CArray:
    """``(nb, Nt, Nt)`` complex unit-norm atoms of ``bands`` on their ``Nt``-bin windows.

    Atom ``[b, n, s]`` multiplies the rfft bin ``(m_b - 1) Nt/2 + s`` of a
    record's spectrum.  The atoms are unit-norm (the spectrum of a basis function, not a
    coefficient): for rfft bins ``z = numpy.fft.rfft(x)`` of an ``N = Nf Nt``-sample record,
    ``sqrt(2 dt / N) * Re(atoms[b, n] . z[window])`` is the Wilson coefficient
    ``forward(z)[m_b, n]``, which :func:`forward_bands` returns.  Ported verbatim from the
    galactic-binary toy model (not shipped), with ``window``'s samples (Hz normalisation).

    Parameters
    ----------
    bands : array_like of int
        Interior bands, each in ``1..Nf-1``.
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    bands = _check_interior_bands(bands, Nf)
    phi = sample_window(window, Nf, Nt, dt)
    s = np.arange(Nt)
    carrier = np.exp(2j * np.pi * np.outer(np.arange(Nt), s - Nt // 2) / Nt)  # (n, s)
    atoms = np.empty((len(bands), Nt, Nt), complex)
    for b, m in enumerate(bands):
        ph = carrier_phase(int(m), Nt)
        quad = (int(m) + np.arange(Nt)) % 2
        u = ph[:, None] * phi[None, :] * carrier
        u = np.where(quad[:, None] == 1, -1j * u, u)
        atoms[b] = u / np.linalg.norm(phi)
    return atoms


def _band_phase(bands: NDArray[np.integer], Nt: int, phi: Array) -> CArray:
    """``(nb, Nt)``: ``carrier_phase * quad * (-1)^n * Nt / |phi|``."""
    n = np.arange(Nt)
    out = np.empty((bands.size, Nt), complex)
    for b, m in enumerate(bands):
        quad = np.where((int(m) + n) % 2 == 1, -1j, 1.0)
        out[b] = carrier_phase(int(m), Nt) * quad * (-1.0) ** n * Nt / np.linalg.norm(phi)
    return out


def _band_windows(bands: NDArray[np.integer], k0: int, n_bins: int, Nt: int) -> NDArray[np.integer]:
    """``(nb, Nt)`` positions, inside a bin array starting at absolute bin ``k0``,
    of every band window bin ``(m - 1) Nt/2 + s``."""
    ell = (bands[:, None] - 1) * (Nt // 2) + np.arange(Nt)[None, :] - int(k0)
    if ell.min() < 0 or ell.max() >= n_bins:
        raise ValueError(
            f"the bins [{k0}, {k0 + n_bins}) do not cover the band windows "
            f"[{ell.min() + k0}, {ell.max() + k0 + 1})"
        )
    return ell


def forward_bands(z: CArray, k0: int, bands: NDArray[np.integer], Nf: int, Nt: int,
                  dt: float, *, be: Backend | None = None, window: Window = MEYER) -> Array:
    """Wilson coefficients of ``bands`` from band-limited rfft bins: ``(..., n) -> (..., nb, Nt)``.

    ``z[..., j]`` is the rfft bin ``k0 + j`` in ``numpy.fft.rfft`` units (no ``dt``).  On a
    whole record, ``z = rfft(x)``, the result is ``forward(z)[bands]`` (<= 1e-12 relative).
    Cell ``(b, n)`` is ``sqrt(2 dt / N) Re(band_atoms[b, n] . z[window_b])``, ``N = Nf Nt``,
    computed with one length-``Nt`` IFFT per band:
    ``sqrt(2 dt / N) Re(phase * ifft(phi * z[window]))``.  It agrees with the dense
    ``sqrt(2 dt / N) Re(band_atoms @ z)`` to ~1e-15 (not bitwise).  Interior bands only.
    Cost ``O(nb Nt log Nt)`` against ``O(nb Nt^2)`` dense.
    A non-finite ``z`` is rejected (as in :func:`forward`): one ``nan`` bin
    would otherwise make every cell of each band whose window covers it
    ``nan``, without a warning.

    Parameters
    ----------
    z : ndarray, complex, shape (..., n)
        rfft bins ``k0 .. k0+n-1`` of a record (``numpy.fft.rfft`` units); they must cover
        every band window.
    k0 : int
        Absolute rfft bin of ``z[..., 0]``.
    bands : array_like of int
        Interior bands, each in ``1..Nf-1``.
    Nf, Nt : int
    dt : float
    be : Backend, optional
        Array backend (default: the backend of ``z``).
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    require_int("k0", k0)
    be = be or get_backend(z)
    bands = _check_interior_bands(bands, Nf)
    z = be.asarray(z)
    ell = _band_windows(bands, k0, z.shape[-1], Nt)
    _check_finite(z, "forward_bands: z")
    rdtype, cdtype = _working_dtypes(z)
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi).astype(cdtype)
    Y = be.fft.ifft(phi.astype(rdtype) * z[..., ell], axis=-1)
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    out = be.xp.real(phase * Y) * rdtype.type(np.sqrt(2.0 * dt / n_samples(Nf, Nt)))
    _check_single_output(out, "forward_bands")
    return out


def inverse_bands(w: Array, k0: int, n_bins: int, bands: NDArray[np.integer], Nf: int,
                  Nt: int, dt: float, *, be: Backend | None = None,
                  window: Window = MEYER) -> CArray:
    """rfft bins from the Wilson coefficients of ``bands``: ``(..., nb, Nt) -> (..., n_bins)``.

    It is ``inverse(w_full)[k0:k0 + n_bins]`` (<= 1e-12 relative), ``w_full`` the
    ``(Nf + 1, Nt)`` grid holding ``w`` on ``bands`` and zeros elsewhere: bin ``k0 + j``
    receives ``sqrt(N / (2 dt)) sum_n w[b, n] conj(atom[b, n, s])`` from every band window
    covering it (``sqrt(N / (2 dt)) phi * fft(conj(phase) * w) / Nt``, scattered band by
    band).  It is ``N / (2 dt)`` times the adjoint of :func:`forward_bands`: for the real
    inner products ``<w, w'> = sum w w'`` and ``<z, z'> = Re sum conj(z) z'``,
    ``<forward_bands(z), w> == (2 dt / N) <z, inverse_bands(w)>`` in exact arithmetic.
    ``inverse_bands(forward_bands(z))`` is ``z`` on the covered bins only when ``z``'s
    Wilson coefficients vanish outside ``bands``.  ``w`` must be real and finite (as in
    :func:`inverse`).

    Parameters
    ----------
    w : ndarray, float, shape (..., nb, Nt)
        Cells of ``bands``.
    k0 : int
        Absolute rfft bin of the first output bin.
    n_bins : int
        Number of output bins; ``[k0, k0+n_bins)`` must cover every band window.
    bands : array_like of int
        Interior bands, each in ``1..Nf-1``.
    Nf, Nt : int
    dt : float
    be : Backend, optional
        Array backend (default: the backend of ``w``).
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    require_int("k0", k0)
    require_int("n_bins", n_bins)
    be = be or get_backend(w)
    bands = _check_interior_bands(bands, Nf)
    w = be.asarray(w)
    # The adjoint identity holds only for real cells: a complex w would give a complex
    # result that is not the adjoint of any cells.
    require_dtype("w", w, REAL_KINDS, "a real numeric array")
    if w.shape[-2:] != (bands.size, Nt):
        raise ValueError(f"w must end in ({bands.size}, {Nt}); got {w.shape}")
    _check_finite(w, "inverse_bands: w")
    ell = _band_windows(bands, k0, n_bins, Nt)
    rdtype, cdtype = _working_dtypes(w)
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi)
    V = phi.astype(rdtype) * be.fft.fft(np.conj(phase).astype(cdtype) * w, axis=-1) / Nt
    out = be.xp.zeros((*w.shape[:-2], n_bins), dtype=cdtype)
    for b in range(bands.size):
        out = be.add_at(out, (..., ell[b]), V[..., b, :], copy=False)
    # rfft units: N / (2 dt) times the adjoint of forward_bands' sqrt(2 dt / N).
    out = out * rdtype.type(np.sqrt(n_samples(Nf, Nt) / (2.0 * dt)))
    _check_single_output(out, "inverse_bands")
    return out
