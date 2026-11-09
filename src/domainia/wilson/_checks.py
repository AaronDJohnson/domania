"""Argument checks shared by :mod:`domainia.wilson` and the rest of domainia.

One rule for the whole package: an argument whose *type* (or an array whose *dtype*) can
never be valid in that place raises ``TypeError``; the right type with a wrong value (range,
shape, finiteness, integrality, an unknown option string) raises ``ValueError``.

Every kind message has the form ``"{name} must be <kind>, got <type or dtype>"``.

Private: this module imports only the standard library and NumPy, so both
``domainia.wilson`` and the core modules can import it without a cycle.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np

#: dtype kinds of a real numeric array: signed and unsigned integers and floats.
#: bool (``"b"``), complex (``"c"``), str/bytes (``"U"``/``"S"``) and object (``"O"``)
#: are wrong kinds wherever a real numeric array is taken.
REAL_KINDS = "iuf"


def require_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``; ``TypeError`` unless it is a Python or NumPy integer.

    ``bool`` and ``np.bool_`` are refused, and so is every float (``2.0`` included): ``int()``
    would truncate ``8.7`` to ``8`` silently, and ``True`` is an ``int`` subclass.

    >>> from domainia.wilson._checks import require_int
    >>> require_int("n", 3)
    3
    >>> require_int("n", 3.0)
    Traceback (most recent call last):
    TypeError: n must be an integer, got float
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def require_real(name: str, value: object) -> float:
    """Return ``value`` as a Python ``float``; ``TypeError`` unless it is a real number.

    Accepted: any :class:`numbers.Real` except ``bool``/``np.bool_`` -- ``int``, ``float``,
    every NumPy integer and floating scalar, :class:`fractions.Fraction`. Refused with
    ``TypeError``: ``bool``, ``str``, ``None``, complex, :class:`decimal.Decimal` and arrays
    of any shape, 0-d included. Range and finiteness are the caller's ``ValueError``; a real
    too large for a float (``10**400``) raises ``ValueError`` here.

    >>> from domainia.wilson._checks import require_real
    >>> require_real("dt", 1)
    1.0
    >>> require_real("dt", "0.5")
    Traceback (most recent call last):
    TypeError: dt must be a real number, got str
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"{name} is too large for a float") from None


def require_bool(name: str, value: object) -> bool:
    """Return ``value`` as a ``bool``; ``TypeError`` unless it is a ``bool`` or ``np.bool_``.

    A truthy ``"no"``, ``1`` or ``0.5`` is refused rather than read as a flag.

    >>> from domainia.wilson._checks import require_bool
    >>> require_bool("restrict", True)
    True
    >>> require_bool("restrict", "no")
    Traceback (most recent call last):
    TypeError: restrict must be a bool, got str
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


def require_str(name: str, value: object) -> str:
    """Return ``value``; ``TypeError`` unless it is a ``str``.

    Only the kind: whether the string is one of the choices stays the caller's
    ``ValueError``, with its own message listing them.

    >>> from domainia.wilson._checks import require_str
    >>> require_str("impl", 3)
    Traceback (most recent call last):
    TypeError: impl must be a string, got int
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


def require_instance(name: str, value: object, cls: type | tuple[type, ...], expected: str) -> Any:
    """Return ``value``; ``TypeError`` unless ``isinstance(value, cls)``.

    ``expected`` is the phrase after "must be", article included
    (``"a TimeDomain"``, ``"a CellSet or a WDMDomain"``).

    >>> from domainia.wilson._checks import require_instance
    >>> require_instance("src", 3, str, "a string domain")
    Traceback (most recent call last):
    TypeError: src must be a string domain, got int
    """
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be {expected}, got {type(value).__name__}")
    return value


def require_callable(name: str, value: object) -> Any:
    """Return ``value``; ``TypeError`` unless it is callable.

    >>> from domainia.wilson._checks import require_callable
    >>> require_callable("psd", 1.0)
    Traceback (most recent call last):
    TypeError: psd must be callable, got float
    """
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value).__name__}")
    return value


def require_dtype(name: str, a: object, kinds: str, expected: str) -> None:
    """``TypeError`` unless the dtype kind of ``a`` is one of ``kinds``.

    Reads ``a.dtype`` without converting ``a`` when it has one, so a JAX array or tracer
    passes through untouched; anything else (a list, a scalar) is read through
    :func:`numpy.asarray`. ``expected`` is the phrase after "must be", article included
    (``"a real numeric array"``). Use :data:`REAL_KINDS` for a real numeric array.

    >>> import numpy as np; from domainia.wilson._checks import require_dtype, REAL_KINDS
    >>> require_dtype("x", np.zeros(2), REAL_KINDS, "a real numeric array")
    >>> require_dtype("x", np.zeros(2, complex), REAL_KINDS, "a real numeric array")
    Traceback (most recent call last):
    TypeError: x must be a real numeric array, got dtype complex128
    """
    dtype = getattr(a, "dtype", None)
    dtype = np.asarray(a).dtype if dtype is None else np.dtype(dtype)
    if dtype.kind not in kinds:
        raise TypeError(f"{name} must be {expected}, got dtype {dtype}")


def index_array(name: str, value: object) -> np.ndarray:
    """Return ``value`` as an ``int64`` ndarray of indices.

    Integer dtypes pass. A float array whose entries are all finite integers (``[2.0]``) is
    accepted and converted. A float entry that is not integral or not finite raises
    ``ValueError``; a bool, str, bytes, object or complex array raises ``TypeError``.
    Range and shape are the caller's checks.

    >>> import numpy as np; from domainia.wilson._checks import index_array
    >>> index_array("m", [1, 2.0])
    array([1, 2])
    >>> index_array("m", [1.5])
    Traceback (most recent call last):
    ValueError: m must hold integer indices, got 1.5
    >>> index_array("m", [True])
    Traceback (most recent call last):
    TypeError: m must be an integer index array, got dtype bool
    """
    raw = np.asarray(value)
    kind = raw.dtype.kind
    if kind in "iu":
        return raw.astype(np.int64, copy=False)
    if kind != "f":
        raise TypeError(f"{name} must be an integer index array, got dtype {raw.dtype}")
    bad = ~np.isfinite(raw) | (raw != np.trunc(raw))
    if bad.any():
        first = raw.ravel()[int(np.flatnonzero(bad.ravel())[0])]
        raise ValueError(f"{name} must hold integer indices, got {float(first)!r}")
    return raw.astype(np.int64)


def validate_dt(dt: object) -> None:
    """Raise unless ``dt`` is a positive finite real number.

    A negative or zero sampling interval produces a nonsensical frequency grid
    (``delta_F`` flips sign or diverges) without raising anywhere, so the
    transform happily returns numbers that mean nothing.

    Parameters
    ----------
    dt : float
        Sampling interval.

    Raises
    ------
    TypeError
        If ``dt`` is not a real number (:func:`require_real`): ``None``, a ``bool``, a
        string, complex, a ``Decimal``, or an array of any shape, 0-d included.
        ``float()`` would take ``True`` as 1.0 and ``"0.5"`` as 0.5 silently, and drop
        the imaginary part of a NumPy complex scalar.
    ValueError
        If ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.validate_dt(0.5)
    >>> wilson.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    >>> wilson.validate_dt(True)
    Traceback (most recent call last):
    TypeError: dt must be a real number, got bool
    """
    value = require_real("dt", dt)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt!r}")


def validate_finite(values: object, context: str) -> None:
    """Raise ``ValueError`` if ``values`` holds a ``nan`` or an ``inf``.

    The WDM transform is *global* within a band: every coefficient is an
    inner product against a window spanning many samples, and the
    implementation routes each band through an FFT.  A single non-finite
    entry therefore does not stay local.  Measured on an $N_f=8$, $N_t=16$
    grid, one ``nan`` anywhere in the input series makes **all 144** output
    coefficients ``nan`` -- and numpy emits no warning on the way, because
    ``nan`` arithmetic inside an FFT is not an invalid operation.  The result
    is a coefficient grid that is entirely meaningless yet raises nothing,
    which is exactly the failure this rejects.

    Detector data with gaps routinely arrives with ``nan`` fills, so this is
    a live path rather than a theoretical one.  The check is one $O(N)$ pass
    against an $O(N\\log N_t)$ transform, so it is not measurable in the
    package's own benchmarks.

    Deliberately stricter than :func:`sanitize_psd`, which maps ``inf`` to
    ``0.0``: an infinite *PSD* sample has a physical reading ("no measurable
    power in this bin"), an infinite *data* sample has none.

    Parameters
    ----------
    values : array-like
        Data, coefficient, or spectrum array.  Complex input is checked on
        both components.
    context : str
        Name used in the error message.

    Raises
    ------
    ValueError
        If any entry is ``nan`` or infinite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.validate_finite([1.0, 2.0], 'x')      # all finite: fine
    >>> wilson.validate_finite([1.0, np.nan], 'x')   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: x: 1 of 2 entries are not finite (1 nan, 0 inf); first at index 1...
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    first = int(np.flatnonzero(bad.ravel())[0])
    raise ValueError(
        f"{context}: {int(np.count_nonzero(bad))} of {arr.size} entries are "
        f"not finite ({n_nan} nan, {int(np.count_nonzero(bad)) - n_nan} inf); "
        f"first at index {first}. A non-finite sample does not stay local -- "
        "it spreads through the transform's FFT to every coefficient, "
        "silently. Fill or excise the gaps before transforming."
    )
