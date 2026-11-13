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

    ``bool``, ``np.bool_`` and every float (``2.0`` included) are refused.

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


def require_instance(name: str, value: object, cls: type | tuple[type, ...], expected: str) -> Any:
    """Return ``value``; ``TypeError`` unless ``isinstance(value, cls)``.

    ``expected`` is the phrase after "must be", article included
    (``"a Window"``, ``"a string"``).

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

    Reads ``a.dtype`` when ``a`` has one, else ``numpy.asarray(a).dtype``.  ``expected`` is the
    phrase after "must be", article included (``"a real numeric array"``, with
    :data:`REAL_KINDS`).

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


def validate_dt(dt: object) -> None:
    """Raise unless the sampling interval ``dt`` is a positive finite real number.

    Raises
    ------
    TypeError
        If ``dt`` is not a real number (:func:`require_real`): ``None``, a ``bool``, a
        string, complex, a ``Decimal``, or an array of any shape, 0-d included.
    ValueError
        If ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.validate_dt(0.5)
    >>> wilson.grid.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    >>> wilson.grid.validate_dt(True)
    Traceback (most recent call last):
    TypeError: dt must be a real number, got bool
    """
    value = require_real("dt", dt)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt!r}")


def validate_finite(values: object, context: str) -> None:
    """Check that ``values`` holds no ``nan`` or ``inf`` (both parts of a complex array);
    ``context`` names it in the error message.

    One non-finite sample makes every coefficient of the transform ``nan``, silently, because
    each band goes through an FFT.  Stricter than
    :func:`domainia.wilson.grid.sanitize_psd`, which maps an infinite PSD sample to 0: an
    infinite *data* sample has no physical reading.

    Raises
    ------
    ValueError
        If any entry is ``nan`` or infinite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.validate_finite([1.0, 2.0], 'x')      # all finite: fine
    >>> wilson.grid.validate_finite([1.0, np.nan], 'x')   # doctest: +ELLIPSIS
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
