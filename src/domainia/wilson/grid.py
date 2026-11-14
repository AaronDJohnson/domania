"""Time-frequency grid: the grid and input checks, pixel widths and centres, the
active-pixel mask and the folded layout.

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

The folded layout ``(Nf, Nt)`` holds exactly those $N$ cells: row ``i >= 1`` is band ``i``,
and row 0 holds band 0 in its even slots and band ``Nf`` in its odd ones (:func:`cells`).
:func:`fold` and :func:`unfold` convert between it and the padded ``(Nf+1, Nt)`` array.
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

__all__ = [
    "Grid",
    "active_mask",
    "band_frequencies",
    "cells",
    "delta_F",
    "delta_T",
    "endpoint_weights",
    "fold",
    "n_samples",
    "sanitize_psd",
    "time_pixel_centers",
    "unfold",
    "validate_dt",
    "validate_finite",
    "validate_grid",
]

#: The largest canonical-form residue an input may carry, as a fraction of its slice's peak.
RESIDUE = 1e-10


@dataclass(frozen=True)
class Grid:
    """A validated ``(Nf, Nt, dt)`` triple, with the derived quantities.

    Unpacking one ``Grid`` at each call site rules out swapping ``Nf`` and ``Nt``, which no
    check can see (both are positive even ints with the same product).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> g = wilson.Grid(Nf=8, Nt=16, dt=1.0)
    >>> g.N, g.dF, g.dT
    (128, 0.0625, 8.0)
    >>> x = np.random.default_rng(0).standard_normal(g.N)
    >>> wilson.forward_time(x, *g).shape          # unpacks as (Nf, Nt, dt)
    (8, 16)
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

    Raises
    ------
    TypeError
        If either argument is not an integer (``bool`` included).
    ValueError
        If either argument is not positive and even.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.validate_grid(8, 16)          # even and positive: fine
    >>> wilson.grid.validate_grid(9, 16)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    >>> wilson.grid.validate_grid(8.0, 16)
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
    """The package-wide policy for PSD samples ``values`` (``context`` names them in messages).

    An infinite sample means "this bin carries no measurable power" (sensitivity curves
    return ``inf`` at ``f = 0``), so it maps to ``0.0``; that keeps
    :func:`domainia.wilson.matrix.build_C_time` finite and positive semi-definite.  ``nan``
    has no physical reading and a negative PSD is a bug in the model, so both raise.

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
    """Total time-domain sample count $N = N_f N_t$, an ``int``.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer (``bool`` included).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.n_samples(8, 16)
    128
    """
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        # int() would silently truncate: n_samples(8.7, 16.2) == 128, and
        # bool is an int subclass, so n_samples(True, 16) would give 16.
        _require_int(name, val)
    return int(Nf) * int(Nt)


def delta_F(Nf: int, dt: float) -> float:
    """Frequency-pixel width $\\Delta F = 1/(2\\,\\delta t\\,N_f)$ in Hz, a ``float``.

    Raises
    ------
    TypeError
        If ``Nf`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` is not positive and even, if ``dt`` is not positive
        and finite, or if $2\\,\\delta t N_f$ is so small or so large that the
        width is not a positive finite double (an ``inf`` or ``0.0`` width would make
        the transform return exact zeros).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.delta_F(8, 1.0)
    0.0625
    >>> wilson.grid.delta_F(0, 1.0)
    Traceback (most recent call last):
    ValueError: Nf must be positive, got 0
    >>> wilson.grid.delta_F(8, 5e-324)          # doctest: +ELLIPSIS
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
    """Time-pixel width $\\Delta T = N_f\\,\\delta t$, a ``float``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.delta_T(8, 1.0)
    8.0
    """
    validate_dt(dt)
    return float(Nf) * float(dt)


def band_frequencies(Nf: int, dt: float) -> Array:
    """Band-center frequencies $f_m = m\\,\\Delta F$ for $m = 0,\\ldots,N_f$, shape ``(Nf+1,)``:
    ``freqs[0] == 0`` (DC) and ``freqs[Nf] == 1/(2*dt)`` (Nyquist).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> f = wilson.grid.band_frequencies(8, 1.0)
    >>> f.shape, float(f[0]), float(f[-1])   # DC .. Nyquist
    ((9,), 0.0, 0.5)
    """
    validate_grid(Nf, 2)  # only Nf is constrained here; Nt is a placeholder
    return np.arange(Nf + 1) * delta_F(Nf, dt)


def time_pixel_centers(Nf: int, Nt: int, dt: float) -> Array:
    """Pixel-center times $t_n = n\\,\\Delta T$ for $n = 0,\\ldots,N_t-1$, shape ``(Nt,)``.

    These are the times of the WDM lattice: the energy centroid of
    the basis function $g_{mn}$ sits at exactly $n\\,\\Delta T$.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.time_pixel_centers(8, 4, 1.0)
    array([ 0.,  8., 16., 24.])
    """
    validate_grid(Nf, Nt)
    return np.arange(Nt) * delta_T(Nf, dt)


def active_mask(Nf: int, Nt: int) -> BoolArray:
    """Boolean mask of shape ``(Nf+1, Nt)`` marking active (nonzero) pixels.

    - Interior bands ($1 \\le m \\le N_f - 1$): all $N_t$ time bins active.
    - DC band ($m=0$): only even $n$ are active ($N_t/2$ pixels).
    - Nyquist band ($m=N_f$): only $n$ with $(N_f + n)$ even are active
      ($N_t/2$ pixels).

    Total active count equals $N = N_f N_t$.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> mask = wilson.grid.active_mask(8, 16)
    >>> mask.shape, int(mask.sum())   # total active == N
    ((9, 16), 128)
    """
    validate_grid(Nf, Nt)
    mask = np.ones((Nf + 1, Nt), dtype=bool)
    n = np.arange(Nt)
    mask[0] = n % 2 == 0
    mask[Nf] = (Nf + n) % 2 == 0
    return mask


def cells(Nf: int, Nt: int) -> tuple[np.ndarray, np.ndarray]:
    """The band ``m`` and time ``n`` of each folded slot, two int arrays of shape ``(Nf, Nt)``.

    Slot ``(i, j)`` with ``i >= 1`` is band ``i`` at time ``j``.  Row 0 holds band 0 (DC) at
    time ``j`` in its even slots and band ``Nf`` (Nyquist) at time ``j - 1`` in its odd ones.
    A slot's frequency and time are ``band_frequencies(Nf, dt)[m]`` and
    ``time_pixel_centers(Nf, Nt, dt)[n]``.

    Examples
    --------
    >>> from domainia import wilson
    >>> m, n = wilson.grid.cells(4, 4)
    >>> m[0].tolist(), n[0].tolist()
    ([0, 4, 0, 4], [0, 0, 2, 2])
    """
    validate_grid(Nf, Nt)
    m, n = np.indices((Nf, Nt))
    m[0, 1::2] = Nf
    n[0, 1::2] -= 1
    return m, n


def _wdm_array(name: str, x: object, shape: tuple[int, int]) -> np.ndarray:
    """``x`` as an array, after the kind and trailing-shape checks of :func:`fold` and
    :func:`unfold`."""
    if isinstance(x, np.ma.MaskedArray):
        raise TypeError(f"{name} is a masked array; numpy.asarray would drop the mask, so fill "
                        "it first")
    a = np.asarray(x)
    require_dtype(name, a, "biufc", "a numeric or boolean array")
    if a.shape[-2:] != shape:
        raise ValueError(f"{name} has shape {a.shape}, but must end in {shape}")
    return a


def _fold_fix(shape: tuple[int, ...], Nf: int, Nt: int) -> str:
    """The fix a WDM array of ``shape`` needs if it is padded or flat, as the end of the
    kernels' wrong-shape message (``""`` for any other shape)."""
    if shape[-2:] == (Nf + 1, Nt):
        return "; fold a padded (..., Nf+1, Nt) array with wilson.grid.fold first"
    if shape[-1:] == (Nf * Nt,):
        return f"; reshape a flat (..., N) array with .reshape(..., {Nf}, {Nt}) first"
    return ""


def fold(P: object, Nf: int, Nt: int) -> np.ndarray:
    """The padded ``(..., Nf+1, Nt)`` array ``P`` in the folded layout ``(..., Nf, Nt)``:
    ``P[..., m, n]`` with ``m, n = cells(Nf, Nt)``, in ``P``'s dtype (a boolean mask folds too).

    Raises
    ------
    TypeError
        If ``P`` is masked, or neither numeric nor boolean.
    ValueError
        If the grid is invalid, ``P`` does not end in ``(Nf+1, Nt)``, or an inactive cell is
        non-finite or exceeds :data:`RESIDUE` of its slice's peak (``max |P|`` over the last two
        axes; for a boolean mask, any ``True``): fold drops it.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> R = np.arange(32.0).reshape(4, 8)
    >>> P = wilson.grid.unfold(R, 4, 8)
    >>> P.shape, bool(np.array_equal(wilson.grid.fold(P, 4, 8), R))
    ((5, 8), True)
    >>> P[0, 1] = 1.0
    >>> wilson.grid.fold(P, 4, 8)                  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: P: the inactive cell (m=0, n=1) holds 1.0; ...
    """
    m, n = cells(Nf, Nt)
    a = _wdm_array("P", P, (Nf + 1, Nt))
    v = np.abs(a.astype(np.result_type(a, 1.0), copy=False))
    peak = np.max(v, axis=(-2, -1), keepdims=True)
    bad = np.argwhere(~active_mask(Nf, Nt) & (~np.isfinite(v) | (v > RESIDUE * peak)))
    if bad.size:
        *lead, i, j = (int(k) for k in bad[0])
        raise ValueError(f"P{lead if lead else ''}: the inactive cell (m={i}, n={j}) holds "
                         f"{a[tuple(bad[0])].item()}; fold drops inactive cells, so each must be "
                         f"finite and at most {RESIDUE:g} of its slice's peak")
    return a[..., m, n]


def unfold(R: object, Nf: int, Nt: int) -> np.ndarray:
    """The folded ``(..., Nf, Nt)`` array ``R`` in the padded layout ``(..., Nf+1, Nt)``, in
    ``R``'s dtype, with the inactive cells exactly 0 (``False``); the inverse of :func:`fold`.
    Raises as :func:`fold` does for a wrong kind, grid or shape."""
    m, n = cells(Nf, Nt)
    a = _wdm_array("R", R, (Nf, Nt))
    P = np.zeros(a.shape[:-2] + (Nf + 1, Nt), a.dtype)
    P[..., m, n] = a
    return P


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
