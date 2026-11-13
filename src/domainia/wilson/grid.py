"""Time-frequency grid: pixel layout, indexing, and active-pixel masks.

The WDM transform partitions the time-frequency plane into a grid of
$N_f+1$ frequency bands (indexed by $m = 0,\\ldots,N_f$) and $N_t$
time bins (indexed by $n = 0,\\ldots,N_t-1$).  The total time-domain
sample count is $N = N_f N_t$.

The pixel widths are
$$\\Delta T = N_f\\,\\delta t,\\qquad \\Delta F = 1/(2\\,\\delta t\\,N_f).$$

Edge bands ($m=0$ DC, $m=N_f$ Nyquist) carry only $N_t/2$ active
coefficients each (parity-selected); interior bands carry all $N_t$.
The total active count is therefore
$(N_f - 1)\\,N_t + N_t/2 + N_t/2 = N$.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._checks import REAL_KINDS, require_dtype
from ._checks import require_int as _require_int, validate_dt, validate_finite

type Array = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.integer]

__all__ = [
    "Grid",
    "active_mask",
    "band_frequencies",
    "band_indices",
    "carrier_phase",
    "count_active",
    "delta_F",
    "delta_T",
    "edge_columns",
    "endpoint_weights",
    "n_samples",
    "pack",
    "packed_cells",
    "sanitize_psd",
    "unpack",
    "time_pixel_centers",
    "validate_dt",
    "validate_finite",
    "validate_grid",
]


@dataclass(frozen=True)
class Grid:
    """A validated ``(Nf, Nt, dt)`` triple, with the derived quantities.

    Every function in the package takes ``Nf, Nt, dt`` positionally, and the
    one misuse that cannot be detected downstream is swapping ``Nf`` and
    ``Nt`` -- both are positive even ints, so ``Nf*Nt`` is unchanged and no
    validation can see it. Building a ``Grid`` once and unpacking it at each
    call site removes the opportunity.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> g = wilson.Grid(Nf=8, Nt=16, dt=1.0)
    >>> g.N, g.dF, g.dT
    (128, 0.0625, 8.0)
    >>> x = np.random.default_rng(0).standard_normal(g.N)
    >>> wilson.forward_time(x, *g).shape          # unpacks as (Nf, Nt, dt)
    (9, 16)
    >>> wilson.Grid(Nf=9, Nt=16, dt=1.0)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    """

    Nf: int
    Nt: int
    dt: float

    def __post_init__(self) -> None:
        validate_grid(self.Nf, self.Nt)
        validate_dt(self.dt)

    def __iter__(self) -> Iterator[int | float]:
        """Unpack as ``(Nf, Nt, dt)`` so ``f(x, *grid)`` just works."""
        yield self.Nf
        yield self.Nt
        yield self.dt

    @property
    def N(self) -> int:
        """Total time-domain sample count $N = N_f N_t$."""
        return n_samples(self.Nf, self.Nt)

    @property
    def dF(self) -> float:
        """Frequency-pixel width $\\Delta F$."""
        return delta_F(self.Nf, self.dt)

    @property
    def dT(self) -> float:
        """Time-pixel width $\\Delta T$."""
        return delta_T(self.Nf, self.dt)


def _require_even_positive_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``, requiring it to be positive and even.
    Same messages and exceptions as :func:`validate_grid`."""
    val = _require_int(name, value)
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")
    if val % 2 != 0:
        raise ValueError(f"{name} must be even, got {val}")
    return val


def validate_grid(Nf: int, Nt: int) -> None:
    """Raise unless ``Nf`` and ``Nt`` are positive even integers.

    Parameters
    ----------
    Nf : int
        Number of frequency bands.
    Nt : int
        Number of time bins.

    Raises
    ------
    TypeError
        If either argument is not an integer (``bool`` included).
    ValueError
        If either argument is not positive and even.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.validate_grid(8, 16)          # even and positive: fine
    >>> wilson.validate_grid(9, 16)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    >>> wilson.validate_grid(8.0, 16)
    Traceback (most recent call last):
    TypeError: Nf must be an integer, got float
    """
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        _require_int(name, val)
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")
        if val % 2 != 0:
            raise ValueError(f"{name} must be even, got {val}")


def sanitize_psd(values: object, context: str) -> Array:
    """Apply the package-wide policy for non-finite PSD samples.

    Sensitivity curves routinely return ``inf`` at ``f = 0`` (a LISA
    acceleration term goes as $f^{-4}$), so an infinite sample is a normal
    input meaning "this bin carries no measurable power". Every module maps it
    to ``0.0``, which is what keeps :func:`domainia.wilson.matrix.build_C_time` finite and
    positive semi-definite.

    ``nan`` is *not* treated that way: it has no physical reading, so it raises.
    Complex input raises ``TypeError`` rather than being silently truncated to its
    real part, and so does a bool, str or object array.

    The point of routing every module through here is **uniformity**. These
    paths previously disagreed: ``build_C_time`` and ``stationary`` mapped
    non-finite to ``0.0``, ``noise.band_averaged_variance`` dropped the bin
    from a weighted average (a different answer), and the Gaussian log-likelihood
    propagated it to ``+inf``. The same PSD gave three answers depending on which
    function you called.

    Negative samples are rejected outright: a negative PSD is not a convention,
    it is a bug in the caller's model.

    Parameters
    ----------
    values : array-like
        PSD samples.
    context : str
        Name used in the error message.

    Returns
    -------
    ndarray, float64
        Finite, non-negative PSD values with non-finite entries set to ``0.0``.

    Raises
    ------
    TypeError
        If ``values`` is not a real numeric array (a complex, bool, str or object dtype).
    ValueError
        If any entry is negative or ``nan``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.sanitize_psd([1e-3, np.inf], 'demo')   # inf -> no power
    array([0.001, 0.   ])
    >>> wilson.grid.sanitize_psd([-1.0], 'demo')
    Traceback (most recent call last):
    ValueError: demo: PSD has negative value(s); a power spectral density must be non-negative
    >>> wilson.grid.sanitize_psd([1e-3 + 0j], 'demo')
    Traceback (most recent call last):
    TypeError: demo: PSD must be a real numeric array, got dtype complex128
    """
    arr = np.asarray(values)
    require_dtype(f"{context}: PSD", arr, REAL_KINDS, "a real numeric array")
    arr = arr.astype(np.float64)
    if np.any(np.isnan(arr)):
        # inf has a physical reading (no measurable power in that bin); nan does
        # not -- it means the caller's model produced something undefined.
        raise ValueError(
            f"{context}: PSD has {int(np.count_nonzero(np.isnan(arr)))} NaN "
            "value(s). Unlike inf, a NaN has no physical reading here; fix the "
            "PSD model rather than letting it become zero power."
        )
    if np.any(arr < 0.0):
        raise ValueError(
            f"{context}: PSD has negative value(s); a power spectral density "
            "must be non-negative"
        )
    return np.where(np.isfinite(arr), arr, 0.0)


def n_samples(Nf: int, Nt: int) -> int:
    """Total time-domain sample count $N = N_f N_t$.

    Parameters
    ----------
    Nf, Nt : int
        Frequency and time bin counts.

    Returns
    -------
    int
        $N = N_f \\cdot N_t$.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer (``bool`` included).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.n_samples(8, 16)
    128
    """
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        # int() would silently truncate: n_samples(8.7, 16.2) == 128, and
        # bool is an int subclass, so n_samples(True, 16) would give 16.
        _require_int(name, val)
    return int(Nf) * int(Nt)


def delta_F(Nf: int, dt: float) -> float:
    """Frequency-pixel width $\\Delta F = 1/(2\\,\\delta t\\,N_f)$.

    Parameters
    ----------
    Nf : int
        Number of frequency bands.
    dt : float
        Sampling interval.

    Returns
    -------
    float
        Pixel width in Hz.

    Raises
    ------
    TypeError
        If ``Nf`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` is not positive and even, if ``dt`` is not positive
        and finite, or if $2\\,\\delta t N_f$ is so small or so large that the
        width is not a positive finite double (lisa/wdm b5973b7: an ``inf`` or
        ``0.0`` width made :func:`domainia.wilson.forward_time` return exact zeros).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.delta_F(8, 1.0)
    0.0625
    >>> wilson.delta_F(0, 1.0)
    Traceback (most recent call last):
    ValueError: Nf must be positive, got 0
    >>> wilson.delta_F(8, 5e-324)          # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: delta_F is not representable for Nf=8, dt=5e-324...
    """
    _require_even_positive_int("Nf", Nf)
    validate_dt(dt)
    width = 1.0 / (2.0 * float(dt) * Nf)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(
            f"delta_F is not representable for Nf={Nf}, dt={dt!r}: "
            f"1/(2*dt*Nf) is {width!r}. Both arguments are individually legal, "
            "but their product has left the normal double range, and the "
            "transform would return a grid of exact zeros without raising."
        )
    return width


def delta_T(Nf: int, dt: float) -> float:
    """Time-pixel width $\\Delta T = N_f\\,\\delta t$.

    Parameters
    ----------
    Nf : int
    dt : float

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.delta_T(8, 1.0)
    8.0
    """
    validate_dt(dt)
    return float(Nf) * float(dt)


def band_frequencies(Nf: int, dt: float) -> Array:
    """Band-center frequencies $f_m = m\\,\\Delta F$ for $m = 0,\\ldots,N_f$.

    Parameters
    ----------
    Nf : int
    dt : float

    Returns
    -------
    ndarray, shape (Nf+1,)
        Endpoints satisfy ``freqs[0] == 0`` (DC) and
        ``freqs[Nf] == 1/(2*dt)`` (Nyquist).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> f = wilson.band_frequencies(8, 1.0)
    >>> f.shape, float(f[0]), float(f[-1])   # DC .. Nyquist
    ((9,), 0.0, 0.5)
    """
    validate_grid(Nf, 2)  # only Nf is constrained here; Nt is a placeholder
    return np.arange(Nf + 1) * delta_F(Nf, dt)


def time_pixel_centers(Nf: int, Nt: int, dt: float) -> Array:
    """Pixel-center times $t_n = n\\,\\Delta T$ for $n = 0,\\ldots,N_t-1$.

    These are the time coordinates of the WDM lattice: the energy centroid of
    the basis function $g_{mn}$ sits at exactly $n\\,\\Delta T$.

    Parameters
    ----------
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, shape (Nt,)

    Notes
    -----
    .. note::
       Before 2026-07-27 this returned $(n + 1/2)\\,\\Delta T$, a half-pixel
       too late. Measured against the energy centroid of a single-pixel
       inverse transform, the offset was exactly $-0.5\\,\\Delta T$ at every
       grid tested.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.time_pixel_centers(8, 4, 1.0)
    array([ 0.,  8., 16., 24.])
    """
    validate_grid(Nf, Nt)
    return np.arange(Nt) * delta_T(Nf, dt)


def band_indices(m: int, Nf: int, Nt: int) -> IntArray:
    """Circular DFT bin indices for band $m$ in FFT-natural ordering.

    Returns the $N_t$ indices $j$ that the forward transform extracts
    when computing band $m$ via ``np.fft.ifft``.  The offsets within
    the band run as $[0, 1, \\ldots, N_t/2-1, -N_t/2, \\ldots, -1]$
    relative to the band center $j_0 = m N_t/2$, with circular wrap
    by $N = N_f N_t$.

    Parameters
    ----------
    m : int
        Band index, $0 \\le m \\le N_f$.
    Nf, Nt : int

    Returns
    -------
    ndarray of int, shape (Nt,)

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.band_indices(1, 8, 4)
    array([2, 3, 0, 1])
    """
    N = n_samples(Nf, Nt)
    center = m * (Nt // 2)
    offsets = np.fft.ifftshift(np.arange(Nt) - Nt // 2)
    return (center + offsets) % N


def carrier_phase(m: int, Nt: int) -> Array:
    """Carrier phase $(-1)^{mn}$ as a length-$N_t$ array.

    Equals an array of ones when $m$ is even, and the alternating
    sequence $[1, -1, 1, -1, \\ldots]$ when $m$ is odd.

    Parameters
    ----------
    m : int
    Nt : int

    Returns
    -------
    ndarray, shape (Nt,)
        Float array of $\\pm 1$ values.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.carrier_phase(1, 4)
    array([ 1., -1.,  1., -1.])
    >>> wilson.grid.carrier_phase(1.0, 4)
    Traceback (most recent call last):
    TypeError: m must be an integer, got float
    """
    _require_int("m", m)
    if m % 2 == 0:
        return np.ones(Nt)
    return 1.0 - 2.0 * (np.arange(Nt) % 2)


def active_mask(Nf: int, Nt: int) -> BoolArray:
    """Boolean mask of shape ``(Nf+1, Nt)`` marking active (nonzero) pixels.

    - Interior bands ($1 \\le m \\le N_f - 1$): all $N_t$ time bins active.
    - DC band ($m=0$): only even $n$ are active ($N_t/2$ pixels).
    - Nyquist band ($m=N_f$): only $n$ with $(N_f + n)$ even are active
      ($N_t/2$ pixels).

    Total active count equals $N = N_f N_t$.

    Parameters
    ----------
    Nf, Nt : int

    Returns
    -------
    ndarray of bool, shape (Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> mask = wilson.active_mask(8, 16)
    >>> mask.shape, int(mask.sum())   # total active == N
    ((9, 16), 128)
    """
    validate_grid(Nf, Nt)
    mask = np.ones((Nf + 1, Nt), dtype=bool)
    n = np.arange(Nt)
    mask[0] = n % 2 == 0
    mask[Nf] = (Nf + n) % 2 == 0
    return mask


def count_active(Nf: int, Nt: int) -> int:
    """Total active WDM pixels.

    Always equals $N = N_f N_t$ for valid grids.

    Parameters
    ----------
    Nf, Nt : int

    Returns
    -------
    int

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.count_active(8, 16)
    128
    """
    return int(np.sum(active_mask(Nf, Nt)))


# ---------------------------------------------------------------------------
# Packed (band-order) cell vectors
# ---------------------------------------------------------------------------
#
# The N active cells of an (Nf+1, Nt) grid, listed in BAND order: DC with even
# n, then bands 1..Nf-1 with every n, then Nyquist with (Nf+n) even -- i.e.
# ``np.nonzero(active_mask(Nf, Nt))``.  This is the order of
# ``matrix.active_W`` and of every WDM Paper II covariance.  The Nyquist->DC fold of
# lisa/wdm 0.2.0 (and the manuscript's L1373 interleave) is not offered.


def packed_cells(Nf: int, Nt: int) -> tuple[IntArray, IntArray]:
    """``(m, n)`` of every active cell in band order (length ``N`` each).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> m, n = wilson.grid.packed_cells(4, 4)
    >>> m.tolist()
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    >>> n.tolist()
    [0, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2]
    """
    validate_grid(Nf, Nt)
    m, n = np.nonzero(active_mask(Nf, Nt))
    return m, n


def pack(w: Array, Nf: int, Nt: int) -> Array:
    """``(..., Nf+1, Nt) -> (..., N)``: the active cells in band order."""
    m, n = packed_cells(Nf, Nt)
    w = np.asarray(w)
    if w.shape[-2:] != (Nf + 1, Nt):
        raise ValueError(f"w must end in ({Nf + 1}, {Nt}); got {w.shape}")
    return w[..., m, n]


def unpack(y: Array, Nf: int, Nt: int) -> Array:
    """``(..., N) -> (..., Nf+1, Nt)``, inactive edge pixels exactly 0."""
    m, n = packed_cells(Nf, Nt)
    y = np.asarray(y)
    if y.shape[-1] != m.size:
        raise ValueError(f"y must end in N = {m.size}; got {y.shape}")
    out = np.zeros((*y.shape[:-1], Nf + 1, Nt), dtype=y.dtype)
    out[..., m, n] = y
    return out


def edge_columns(Nt: int, lead: int, trail: int) -> tuple[int, ...]:
    """The first ``lead`` and last ``trail`` time columns of an ``Nt``-column grid.

    Examples
    --------
    >>> from domainia import wilson
    >>> wilson.grid.edge_columns(256, 8, 7)
    (0, 1, 2, 3, 4, 5, 6, 7, 249, 250, 251, 252, 253, 254, 255)
    >>> wilson.grid.edge_columns(256, True, 7)
    Traceback (most recent call last):
    TypeError: lead must be an integer, got bool
    """
    _require_int("lead", lead)
    _require_int("trail", trail)
    return tuple(range(lead)) + tuple(range(Nt - trail, Nt))


def endpoint_weights(n_rfft: int) -> Array:
    r"""Trapezoid weights :math:`\eta_k` of the one-sided sum: ``1/2`` at DC and
    Nyquist, ``1`` between (WDM Paper II, Eq. 51, eq:freq_edge_weight).

    Examples
    --------
    >>> from domainia import wilson
    >>> wilson.grid.endpoint_weights(5)
    array([0.5, 1. , 1. , 1. , 0.5])
    """
    c = np.ones(n_rfft)
    c[0] = 0.5
    c[-1] = 0.5
    return c
