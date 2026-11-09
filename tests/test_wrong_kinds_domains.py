"""``domainia.domains`` refuses a wrong kind with ``TypeError`` and a wrong value with
``ValueError``: grids, cell sets, real scalars, ``units``, integers, index
arrays, the ``from_mask`` mask and channel names."""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import (CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain,
                              _cells_and_window, _cells_of)

GRID = wilson.Grid(8, 16, 0.5)
CELLS = CellSet.all_active(GRID)
FULL = WDMDomain(CELLS)
TIME = TimeDomain(GRID.N, GRID.dt)


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


#: Each slot that takes a ``wilson.Grid``, as a function of the value under test.
GRID_SLOTS = {
    "CellSet": lambda v: CellSet(v, (1,), (0,)),
    "CellSet.all_active": lambda v: CellSet.all_active(v),
    "CellSet.bands": lambda v: CellSet.bands(v, (1, 2)),
    "CellSet.from_mask": lambda v: CellSet.from_mask(v, np.ones(GRID.N, bool)),
    "WDMDomain.full": lambda v: WDMDomain.full(v),
}


@pytest.mark.parametrize("bad", [None, (8, 16, 0.5), CELLS, TIME],
                         ids=["None", "tuple", "CellSet", "TimeDomain"])
@pytest.mark.parametrize("slot", GRID_SLOTS)
def test_a_grid_of_the_wrong_kind_raises_type_error(slot, bad):
    """The class methods read ``grid.Nf`` or ``grid.N`` first: a bare ``AttributeError`` before.
    ``WDMDomain.full`` gets its check from ``CellSet.all_active``."""
    message = f"grid must be a domainia.wilson.Grid, got {type(bad).__name__}"
    with pytest.raises(TypeError, match=_exact(message)):
        GRID_SLOTS[slot](bad)


SUB_SLOTS = {
    "CellSet.index_of": lambda v: CELLS.index_of(v),
    "CellSet.complement": lambda v: CELLS.complement(v),
}


@pytest.mark.parametrize("bad", [None, FULL, GRID, (1, 2)],
                         ids=["None", "WDMDomain", "Grid", "tuple"])
@pytest.mark.parametrize("slot", SUB_SLOTS)
def test_a_sub_of_the_wrong_kind_raises_type_error(slot, bad):
    """``index_of`` read ``sub.grid`` first: a bare ``AttributeError`` before.  ``complement``
    gets its check from ``index_of``."""
    with pytest.raises(TypeError, match=_exact(f"sub must be a CellSet, got {type(bad).__name__}")):
        SUB_SLOTS[slot](bad)


@pytest.mark.parametrize("bad", [None, GRID, TIME], ids=["None", "Grid", "TimeDomain"])
def test_cells_of_the_wrong_kind_take_the_got_form(bad):
    kind = type(bad).__name__
    with pytest.raises(TypeError, match=_exact(f"cells must be a CellSet, got {kind}")):
        WDMDomain(bad)
    for fn in (_cells_and_window, _cells_of):
        with pytest.raises(TypeError, match=_exact(f"cells must be a CellSet or a WDMDomain, got {kind}")):
            fn(bad)


class _UnhashableWindow:
    """A Window by the protocol that cannot be hashed."""

    name = "unhashable"
    __hash__ = None  # type: ignore[assignment]

    def sample(self, Nf, Nt, dt):
        return wilson.MEYER.sample(Nf, Nt, dt)


def test_an_unhashable_window_takes_the_got_form():
    with pytest.raises(TypeError, match=r"^window must be hashable \(.*\), "
                                        r"got an unhashable _UnhashableWindow$"):
        WDMDomain(CELLS, window=_UnhashableWindow())


#: Each real-scalar slot, as a function of the value under test.
REAL_SLOTS = {
    "TimeDomain.dt": lambda v: TimeDomain(128, v),
    "TimeDomain.t0": lambda v: TimeDomain(128, 0.5, v),
    "FourierDomain.dt": lambda v: FourierDomain(128, v, 0, 10, "rfft"),
    "FourierDomain.sigma0": lambda v: FourierDomain(128, 0.5, 0, 10, "sigma0", sigma0=v),
    "FourierDomain.full.dt": lambda v: FourierDomain.full(128, v),
}


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (1.0 + 0j, "complex"),
    (np.complex64(1.0), "complex64"), (Decimal("0.5"), "Decimal"), (np.array(0.5), "ndarray"),
    (np.array([0.5]), "ndarray"),
], ids=["True", "np.True_", "str", "complex", "complex64", "Decimal", "0-d", "size-1"])
@pytest.mark.parametrize("slot", REAL_SLOTS)
def test_a_real_of_the_wrong_kind_raises_type_error(slot, bad, kind):
    """``float()`` took ``True`` as 1.0, ``"0.5"`` as 0.5, a ``Decimal`` and size-1 arrays,
    and dropped the imaginary part of a NumPy complex scalar with a ``ComplexWarning``."""
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be a real number, got {kind}")):
        REAL_SLOTS[slot](bad)


@pytest.mark.parametrize("slot", [s for s in REAL_SLOTS if s != "FourierDomain.sigma0"])
def test_none_is_a_wrong_kind_where_it_is_not_a_default(slot):
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be a real number, got NoneType")):
        REAL_SLOTS[slot](None)


def test_a_missing_sigma0_stays_a_value_error():
    """``sigma0=None`` is the documented default: with ``units="sigma0"`` it is a missing value."""
    with pytest.raises(ValueError, match="units='sigma0' needs sigma0"):
        REAL_SLOTS["FourierDomain.sigma0"](None)


@pytest.mark.parametrize("v", [np.float64(0.5), np.int64(2), np.float32(0.5), Fraction(1, 2), 2],
                         ids=["float64", "int64", "float32", "Fraction", "int"])
def test_real_scalars_are_taken_and_stored_as_float(v):
    """The domains store ``dt``, ``t0`` and ``sigma0`` as Python floats, so
    ``rfft_scale`` stays a Python float too."""
    t = TimeDomain(128, v, v)
    assert (type(t.dt), type(t.t0)) == (float, float) and (t.dt, t.t0) == (float(v), float(v))
    f = FourierDomain(128, v, 0, 10, "dt_rfft")
    assert type(f.dt) is float and type(f.rfft_scale) is float
    assert f == FourierDomain(128, float(v), 0, 10, "dt_rfft")
    s = FourierDomain(128, 0.5, 0, 10, "sigma0", sigma0=v)
    assert type(s.sigma0) is float and type(s.rfft_scale) is float and s.sigma0 == float(v)
    assert type(FourierDomain.full(128, v).dt) is float


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
@pytest.mark.parametrize("slot", ["TimeDomain.dt", "FourierDomain.dt", "FourierDomain.sigma0",
                                  "FourierDomain.full.dt"])
def test_reals_out_of_range_stay_value_errors(slot, bad):
    name = slot.split(".")[-1]
    with pytest.raises(ValueError, match=f"^{name} must be finite and > 0; got"):
        REAL_SLOTS[slot](bad)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_a_non_finite_t0_stays_a_value_error(bad):
    with pytest.raises(ValueError, match="^t0 must be finite; got"):
        TimeDomain(128, 0.5, bad)


@pytest.mark.parametrize(("bad", "kind"), [
    (None, "NoneType"), (3, "int"), (b"rfft", "bytes"), (True, "bool"), (["rfft"], "list"),
], ids=["None", "int", "bytes", "bool", "list"])
def test_units_of_the_wrong_kind_raise_type_error(bad, kind):
    """A non-string ``units`` was reported as ``units must be one of (...); got None``."""
    with pytest.raises(TypeError, match=_exact(f"units must be a string, got {kind}")):
        FourierDomain(128, 0.5, 0, 10, bad)


def test_an_unknown_units_string_stays_a_value_error():
    with pytest.raises(ValueError, match=r"^units must be one of \('rfft', 'dt_rfft', 'sigma0'\); "
                                         r"got 'hz'$"):
        FourierDomain(128, 0.5, 0, 10, "hz")
    assert FourierDomain(128, 0.5, 0, 10, np.str_("rfft")) == FourierDomain(128, 0.5, 0, 10, "rfft")


#: Each integer slot (``domains._int``, Task 1), as a function of the value under test.
INT_SLOTS = {
    "TimeDomain.N": lambda v: TimeDomain(v, 0.5),
    "FourierDomain.N": lambda v: FourierDomain(v, 0.5, 0, 10, "rfft"),
    "FourierDomain.k0": lambda v: FourierDomain(128, 0.5, v, 10, "rfft"),
    "FourierDomain.n": lambda v: FourierDomain(128, 0.5, 0, v, "rfft"),
    "FourierDomain.full.N": lambda v: FourierDomain.full(v, 0.5),
    "CellSet.drop_columns.lead": lambda v: CELLS.drop_columns(v, 1),
    "CellSet.drop_columns.trail": lambda v: CELLS.drop_columns(1, v),
    "CellSet.columns.lead": lambda v: CELLS.columns(v, 1),
    "CellSet.columns.trail": lambda v: CELLS.columns(1, v),
}


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), (2.0, "float"), (np.float64(2.0), "float64"), ("2", "str"), (None, "NoneType"),
])
@pytest.mark.parametrize("slot", INT_SLOTS)
def test_an_integer_of_the_wrong_kind_raises_type_error(slot, bad, kind):
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be an integer, got {kind}")):
        INT_SLOTS[slot](bad)


def test_numpy_integers_are_taken_and_stored_as_int():
    t = TimeDomain(np.int64(128), 0.5)
    f = FourierDomain(np.int64(128), 0.5, np.int64(0), np.int64(10), "rfft")
    assert type(t.N) is int and (type(f.N), type(f.k0), type(f.n)) == (int, int, int)
    assert f == FourierDomain(128, 0.5, 0, 10, "rfft") and type(FourierDomain.full(np.int64(128), 0.5).N) is int
    assert CELLS.drop_columns(np.int64(2), np.int64(1)) == CELLS.drop_columns(2, 1)
    assert CELLS.columns(np.int64(2), np.int64(1)) == CELLS.columns(2, 1)


#: Each index-array slot, as a function of a one-entry value.
INDEX_SLOTS = {
    "CellSet.m": lambda v: CellSet(GRID, v, [0]),
    "CellSet.n": lambda v: CellSet(GRID, [1], v),
    "CellSet.bands.bands": lambda v: CellSet.bands(GRID, v),
}


@pytest.mark.parametrize(("bad", "dtype"), [
    ([True], "bool"), (["2"], "<U1"), ([2 + 0j], "complex128"), ([None], "object"),
], ids=["bool", "str", "complex", "None"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_an_index_array_of_the_wrong_dtype_raises_type_error(slot, bad, dtype):
    """``CellSet`` reported these as ``ValueError``; ``CellSet.bands`` took ``[True]`` as band 1
    and ``["2"]`` as band 2."""
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be an integer index array, got dtype {dtype}")):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize(("bad", "first"), [([1.5], "1.5"), ([np.nan], "nan")], ids=["1.5", "nan"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_a_non_integral_index_raises_value_error(slot, bad, first):
    """``CellSet.bands`` truncated ``1.5`` to band 1."""
    name = slot.split(".")[-1]
    with pytest.raises(ValueError, match=_exact(f"{name} must hold integer indices, got {first}")):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_integer_and_integral_float_index_arrays_are_taken(slot):
    want = INDEX_SLOTS[slot]([2])
    for v in (np.array([2], np.int64), np.array([2], np.int32), np.array([2.0]), (np.float64(2.0),)):
        assert INDEX_SLOTS[slot](v) == want


def test_the_caller_index_arrays_stay_writeable():
    m, n = np.array([1, 2], np.int64), np.array([0, 0], np.int64)
    c = CellSet(GRID, m, n)
    assert m.flags.writeable and n.flags.writeable and not c.m_array.flags.writeable
    assert not np.shares_memory(c.m_array, m)


@pytest.mark.parametrize(("bad", "dtype"), [
    (np.ones(GRID.N, np.int64), "int64"), (np.arange(GRID.N), "int64"), (np.ones(GRID.N), "float64"),
    (np.ones(GRID.N, complex), "complex128"), (np.array(["1"] * GRID.N), "<U1"), (None, "object"),
], ids=["ones-int", "arange", "float", "complex", "str", "None"])
def test_a_mask_of_the_wrong_dtype_raises_type_error(bad, dtype):
    """``keep`` is a value array: its dtype is its kind.  An index array such as
    ``np.arange(N)`` is the likely mistake; it was a ``ValueError`` before."""
    with pytest.raises(TypeError, match=_exact(f"keep must be a boolean mask, got dtype {dtype}")):
        CellSet.from_mask(GRID, bad)


@pytest.mark.parametrize("bad", [np.ones(GRID.N - 1, bool), np.ones((GRID.Nf, GRID.Nt), bool)],
                         ids=["short", "2-d"])
def test_a_mask_of_the_wrong_shape_stays_a_value_error(bad):
    with pytest.raises(ValueError, match=_exact(f"keep must be a boolean mask of shape ({GRID.N},)")):
        CellSet.from_mask(GRID, bad)


def test_boolean_masks_are_taken():
    keep = CELLS.drop_columns(2, 1).keep_mask()
    want = CELLS.drop_columns(2, 1)
    assert CellSet.from_mask(GRID, keep) == want
    assert CellSet.from_mask(GRID, keep.tolist()) == want
    assert CellSet.from_mask(GRID, np.asarray(keep, np.bool_)) == want


@pytest.mark.parametrize(("bad", "kind"), [("AE", "str"), (np.str_("AE"), "str_")],
                         ids=["str", "np.str_"])
def test_channel_axis_refuses_a_bare_string(bad, kind):
    """``ChannelAxis("AE")`` used to split the string into the names ``("A", "E")``."""
    with pytest.raises(TypeError, match=_exact(f"names must be a tuple of channel names, got {kind}")):
        ChannelAxis(bad)


@pytest.mark.parametrize("bad", [("A", 3), ("A", ""), (b"A",)], ids=["int", "empty", "bytes"])
def test_channel_names_that_are_not_non_empty_strings_take_the_got_form(bad):
    with pytest.raises(TypeError, match=_exact(f"channel names must be non-empty strings, got {bad!r}")):
        ChannelAxis(bad)


def test_tuples_and_lists_of_names_are_taken():
    assert ChannelAxis(("A", "E")).names == ("A", "E") == ChannelAxis(["A", "E"]).names
    assert ChannelAxis().size == 0
    with pytest.raises(ValueError, match="unique"):
        ChannelAxis(("A", "A"))
