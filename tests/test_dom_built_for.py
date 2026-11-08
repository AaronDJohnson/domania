"""The ``built_for`` guard: a map's constants run only with the fields they were built for.

Seven maps build constants in ``__post_init__``.  Each records the fields its constants
derive from in ``built_for``, and refuses constants that arrive with another key (or none):
``dataclasses.replace(map, dst=...)`` raises instead of reusing the old domain's constants.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from dom_db_inputs import jax_or_skip, synthetic_record

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import (Compose, FourierCrop, FourierToWDM, FourierZeroPad, Restrict,
                           TimeToFourier, TimeToWDM, WDMToTime)

GRID = wilson.Grid(8, 16, 0.5)
N, DT = GRID.N, GRID.dt
TIME = TimeDomain(N, DT)
FULL = WDMDomain.full(GRID)
CUT = WDMDomain(FULL.cells.drop_columns(2, 1))
BANDS = WDMDomain(CellSet.bands(GRID, (2, 3)))
REC = FourierDomain(N, DT, 0, N // 2 + 1, "rfft")
PAD = FourierDomain(2 * N, DT, 0, N + 1, "rfft")
CROP = FourierDomain(N // 2, DT, 0, N // 4 + 1, "rfft")

MAPS = {
    "TimeToWDM": lambda: TimeToWDM(TIME, FULL),
    "WDMToTime": lambda: WDMToTime(FULL, TIME),
    "Restrict": lambda: Restrict(FULL, CUT),
    "FourierZeroPad": lambda: FourierZeroPad(REC, PAD),
    "FourierCrop": lambda: FourierCrop(REC, CROP, 8),
    "FourierToWDM": lambda: FourierToWDM(REC, BANDS),
    "TimeToFourier": lambda: TimeToFourier(TIME, 0.2),
}
KEYS = {
    "TimeToWDM": (TIME, FULL),
    "WDMToTime": (FULL, TIME),
    "Restrict": (FULL, CUT),
    "FourierZeroPad": (REC, PAD),
    "FourierCrop": (REC, CROP, 8),
    "FourierToWDM": (REC, BANDS, "dense", False),
    "TimeToFourier": (TIME, 0.2),
}
#: (map, field, new value): every field of every key, changed by ``dataclasses.replace``.
CHANGES = [
    ("TimeToWDM", "src", TimeDomain(N, DT, 1.0)),
    ("TimeToWDM", "dst", CUT),
    ("WDMToTime", "src", WDMDomain.full(GRID, window=wilson.Meyer(8))),
    ("WDMToTime", "dst", TimeDomain(N, DT, 1.0)),
    ("Restrict", "src", WDMDomain(FULL.cells.drop_columns(1, 0))),
    ("Restrict", "dst", WDMDomain(FULL.cells.drop_columns(3, 1))),
    ("FourierZeroPad", "src", FourierDomain(N, DT, 1, N // 2, "rfft")),
    ("FourierZeroPad", "dst", FourierDomain(2 * N, DT, 2, N - 1, "rfft")),
    ("FourierCrop", "src", FourierDomain(N, DT, 1, N // 2, "rfft")),
    ("FourierCrop", "dst", FourierDomain(N // 2, DT, 1, N // 4, "rfft")),
    ("FourierCrop", "start", 4),
    ("FourierToWDM", "src", FourierDomain(N, DT, 1, N // 2, "rfft")),
    ("FourierToWDM", "dst", WDMDomain(CellSet.bands(GRID, (3, 4)))),
    ("FourierToWDM", "impl", "fft"),
    ("FourierToWDM", "restrict", True),
    ("TimeToFourier", "src", TimeDomain(N, DT, 1.0)),
    ("TimeToFourier", "fmin", 0.5),
]


def _input(m, rng):
    """A vector on ``m.src``: samples, cells or complex bins."""
    if isinstance(m.src, TimeDomain):
        return rng.standard_normal(m.src.N)
    if isinstance(m.src, WDMDomain):
        return rng.standard_normal(m.src.size)
    return rng.standard_normal(m.src.n) + 1j * rng.standard_normal(m.src.n)


@pytest.mark.parametrize("name", MAPS)
def test_a_normal_build_records_its_key(name):
    assert MAPS[name]().built_for == KEYS[name]


@pytest.mark.parametrize(("name", "field", "value"), CHANGES)
def test_replace_that_changes_a_key_field_raises(name, field, value):
    m = MAPS[name]()
    with pytest.raises(ValueError, match="built for another src/dst: construct a new map"):
        dataclasses.replace(m, **{field: value})


@pytest.mark.parametrize("name", MAPS)
def test_replace_that_changes_nothing_keeps_working(name):
    m = MAPS[name]()
    r = dataclasses.replace(m)
    assert r.built_for == m.built_for
    x = _input(m, np.random.default_rng(0))
    np.testing.assert_array_equal(r.apply(x), m.apply(x))


@pytest.mark.parametrize("name", MAPS)
def test_constants_without_built_for_raise(name):
    with pytest.raises(ValueError, match="constants given without built_for"):
        dataclasses.replace(MAPS[name](), built_for=None)


@pytest.mark.parametrize("name", MAPS)
def test_built_for_is_not_in_the_repr(name):
    assert "built_for" not in repr(MAPS[name]())


def test_restrict_validates_its_domains_before_the_guard():
    """A window or grid mismatch keeps its own message, ``pos`` given and no ``built_for``."""
    R = Restrict(FULL, CUT)
    with pytest.raises(ValueError, match="different windows"):
        Restrict(FULL, WDMDomain(CUT.cells, window=wilson.Meyer(8)), pos=R.pos)
    other = WDMDomain(CellSet.all_active(wilson.Grid(8, 16, 0.25)))
    with pytest.raises(ValueError, match="another grid"):
        Restrict(FULL, other, pos=R.pos)


def test_wdm_to_time_adjoint_builds_its_forward_map_with_built_for():
    inv = WDMToTime(FULL, TIME)
    x = synthetic_record(N, 1)
    np.testing.assert_array_equal(inv.adjoint(x), TimeToWDM(TIME, FULL).apply(x) / DT)


def test_after_maps_carry_the_key_of_the_pair_they_were_built_for():
    """``W.after(inner)`` is built for ``(inner.src, W.dst, "dense", False)`` and still equals
    the composition."""
    rng = np.random.default_rng(2)
    ag = wilson.Grid(8, 16, 0.5)
    cells = WDMDomain(CellSet.bands(ag, tuple(range(1, 8))))
    W = FourierToWDM(FourierDomain(ag.N, ag.dt, 0, 64, "rfft"), cells)
    store = FourierDomain(160, ag.dt, 0, 81, "rfft")
    pad_grid = wilson.Grid(8, 20, 0.5)
    kept = WDMDomain(CellSet.bands(pad_grid, tuple(range(1, 8))).drop_columns(0, 4))
    win = FourierToWDM(FourierDomain(pad_grid.N, ag.dt, 0, pad_grid.N // 2 + 1, "rfft"),
                       kept).window
    Wp = FourierToWDM(win, kept)
    small = FourierDomain(ag.N, ag.dt, 1, 63, "rfft")
    for outer, inner in ((W, FourierCrop(store, W.src, 16)), (Wp, FourierZeroPad(small, win))):
        M = outer.after(inner)
        assert M.built_for == (inner.src, outer.dst, "dense", False)
        z = _input(inner, rng)
        got = M.apply(z)
        assert np.max(np.abs(got - Compose(outer, inner).apply(z))) <= 1e-12 * np.max(np.abs(got))
        assert dataclasses.replace(M).built_for == M.built_for
        with pytest.raises(ValueError, match="built for another src/dst"):
            dataclasses.replace(M, dst=WDMDomain(outer.dst.cells.drop_columns(1, 1)))


@pytest.mark.jax
@pytest.mark.parametrize("name", MAPS)
def test_maps_round_trip_as_pytrees_and_run_under_jit(name):
    """``on``, ``tree_unflatten`` and ``jax.jit`` rebuild the map from its leaves: the guard
    reads the static ``built_for`` only, so tracer leaves pass it."""
    jax, JB = jax_or_skip()
    m = MAPS[name]()
    on = m.on(JB)
    leaves, tree = jax.tree_util.tree_flatten(on)
    back = jax.tree_util.tree_unflatten(tree, leaves)
    assert back.built_for == on.built_for == m.built_for
    x = _input(m, np.random.default_rng(3))
    got = jax.jit(lambda mm, v: mm.apply(v))(on, JB.asarray(x))
    assert isinstance(got, jax.Array)
    want = m.apply(x)
    assert np.max(np.abs(np.asarray(got) - want)) <= 1e-12 * np.max(np.abs(want))
