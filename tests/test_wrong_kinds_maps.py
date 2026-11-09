"""``domainia.maps`` refuses a wrong kind with ``TypeError`` and a wrong value with
``ValueError``: domains, counts, index arrays, ``fmin``, ``restrict``, ``impl``,
``after``'s inner map and ``Compose``'s parts."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest
from dom_db_inputs import DenseCov, jax_or_skip

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import (BinSelect, Compose, CropTime, FourierCrop, FourierToWDM,
                           FourierZeroPad, Identity, Restrict, TimeToFourier, TimeToWDM,
                           WDMToTime, ZeroPadTime, crop_maps, domain_size, pad_maps)

GRID = wilson.Grid(8, 16, 0.5)
TIME = TimeDomain(GRID.N, GRID.dt)
FULL = WDMDomain.full(GRID)
CUT = WDMDomain(FULL.cells.drop_columns(2, 1))
BANDS = WDMDomain(CellSet.bands(GRID, tuple(range(1, 8))))
REC = FourierDomain(GRID.N, GRID.dt, 0, GRID.N // 2 + 1, "rfft")     # every bin of the record
WIN = FourierDomain(GRID.N, GRID.dt, 0, 64, "rfft")                  # the band windows of BANDS
STORE = FourierDomain(GRID.N, GRID.dt, 1, 63, "rfft")
LONG = FourierDomain(160, GRID.dt, 0, 81, "rfft")                    # a 160-sample record
CELLS = CellSet.all_active(GRID)                                     # not a domain

#: A good build of each map constructor; a test overrides one domain slot.
BUILDS = {
    "TimeToWDM": lambda src=TIME, dst=FULL: TimeToWDM(src, dst),
    "WDMToTime": lambda src=FULL, dst=TIME: WDMToTime(src, dst),
    "TimeToFourier": lambda src=TIME: TimeToFourier(src, 0.2),
    "CropTime": lambda src=TIME: CropTime(src, 8, 64),
    "ZeroPadTime": lambda src=TIME: ZeroPadTime(src, 32),
    "FourierZeroPad": lambda src=STORE, dst=LONG: FourierZeroPad(src, dst),
    "FourierCrop": lambda src=LONG, dst=WIN: FourierCrop(src, dst, 16),
    "FourierToWDM": lambda src=WIN, dst=BANDS: FourierToWDM(src, dst),
    "Restrict": lambda src=FULL, dst=CUT: Restrict(src, dst),
    "BinSelect": lambda src=REC, dst=(3, 10): BinSelect(src, dst),
    "Identity": lambda src=TIME: Identity(src),
}
ANY_DOMAIN = "a TimeDomain, a FourierDomain or a WDMDomain"
#: (map, slot, the expected kind, a value of another kind than expected).
DOMAIN_SLOTS = [
    ("TimeToWDM", "src", "a TimeDomain", FULL),
    ("TimeToWDM", "dst", "a WDMDomain", TIME),
    ("WDMToTime", "src", "a WDMDomain", TIME),
    ("WDMToTime", "dst", "a TimeDomain", FULL),
    ("TimeToFourier", "src", "a TimeDomain", FULL),
    ("CropTime", "src", "a TimeDomain", FULL),
    ("ZeroPadTime", "src", "a TimeDomain", FULL),
    ("FourierZeroPad", "src", "a FourierDomain", TIME),
    ("FourierZeroPad", "dst", "a FourierDomain", TIME),
    ("FourierCrop", "src", "a FourierDomain", TIME),
    ("FourierCrop", "dst", "a FourierDomain", TIME),
    ("FourierToWDM", "src", "a FourierDomain", TIME),
    ("FourierToWDM", "dst", "a WDMDomain", TIME),
    ("Restrict", "src", "a WDMDomain", CELLS),
    ("Restrict", "dst", "a WDMDomain", CELLS),
    ("BinSelect", "src", "a FourierDomain", TIME),
    ("BinSelect", "dst", "a FourierDomain or a (k0, n) tuple", [3, 10]),
    ("Identity", "src", ANY_DOMAIN, CELLS),
]


@pytest.mark.parametrize("which", ["None", "other kind"])
@pytest.mark.parametrize(("name", "slot", "expected", "other"), DOMAIN_SLOTS,
                         ids=[f"{n}.{s}" for n, s, _, _ in DOMAIN_SLOTS])
def test_a_domain_of_the_wrong_kind_raises_type_error(name, slot, expected, other, which):
    """Each map constructor checks its domains before it reads them: ``TypeError`` naming the
    slot and the expected class, not a bare ``AttributeError`` (or a silent build, for
    ``ZeroPadTime`` and ``Identity``)."""
    bad = None if which == "None" else other
    message = f"{slot} must be {expected}, got {type(bad).__name__}"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        BUILDS[name](**{slot: bad})


def test_the_other_kind_checks_take_the_got_form():
    with pytest.raises(TypeError, match=f"^dom must be {ANY_DOMAIN}, got CellSet$"):
        domain_size(CELLS)
    F = TimeToFourier(TIME, 0.2)
    with pytest.raises(TypeError, match="^dst must be a FourierDomain, got int$"):
        dataclasses.replace(F, dst=3)        # dst is derived: checked once it is filled
    with pytest.raises(TypeError, match=r"^PushedCovariance needs a covariance with a matvec "
                                        r"\(SupportsMatvec\) or a dense \(SupportsDense\) method, "
                                        r"got ndarray$"):
        Identity(TIME).push(np.eye(GRID.N))


#: Each integer slot, as a function of the value under test.
INT_SLOTS = {
    "CropTime.start": lambda v: CropTime(TIME, v, 64),
    "CropTime.n": lambda v: CropTime(TIME, 8, v),
    "ZeroPadTime.n_extra": lambda v: ZeroPadTime(TIME, v),
    "FourierCrop.start": lambda v: FourierCrop(LONG, WIN, v),
    "BinSelect.k0": lambda v: BinSelect(REC, (v, 10)),
    "BinSelect.n": lambda v: BinSelect(REC, (3, v)),
    "pad_maps.N": lambda v: pad_maps(v, 160, [1, 2], [1, 2]),
    "pad_maps.Np": lambda v: pad_maps(128, v, [1, 2], [1, 2]),
    "crop_maps.N": lambda v: crop_maps(v, 16, 128, [1, 2], [1, 2]),
    "crop_maps.start": lambda v: crop_maps(160, v, 128, [1, 2], [1, 2]),
    "crop_maps.Nc": lambda v: crop_maps(160, 16, v, [1, 2], [1, 2]),
}


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), (2.0, "float"), (np.float64(2.0), "float64"), ("2", "str"), (None, "NoneType"),
])
@pytest.mark.parametrize("slot", INT_SLOTS)
def test_an_integer_of_the_wrong_kind_raises_type_error(slot, bad, kind):
    """``operator.index`` took ``True``; ``int()`` (``BinSelect``) and the arithmetic of the
    Dirichlet kernels took ``True``, ``2.0`` and (``BinSelect``) ``"2"``."""
    name = slot.split(".")[1]
    with pytest.raises(TypeError, match=f"^{name} must be an integer, got {kind}$"):
        INT_SLOTS[slot](bad)


def test_numpy_integers_are_taken_and_stored_as_int():
    C = CropTime(TIME, np.int64(8), np.int32(64))
    assert (type(C.start), type(C.n)) == (int, int) and C.dst == CropTime(TIME, 8, 64).dst
    assert type(ZeroPadTime(TIME, np.int64(32)).n_extra) is int
    crop = FourierCrop(LONG, WIN, np.int64(16))
    assert type(crop.start) is int and crop.built_for == (LONG, WIN, 16)
    assert BinSelect(REC, (np.int64(3), np.int64(10))).dst == BinSelect(REC, (3, 10)).dst
    pairs = ((pad_maps(np.int64(128), np.int64(160), [1, 2], [1, 2]),
              pad_maps(128, 160, [1, 2], [1, 2])),
             (crop_maps(np.int64(160), np.int64(16), np.int64(128), [1, 2], [1, 2]),
              crop_maps(160, 16, 128, [1, 2], [1, 2])))
    for got, want in pairs:
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_counts_out_of_range_stay_value_errors():
    with pytest.raises(ValueError, match="not inside"):
        CropTime(TIME, -1, 64)
    with pytest.raises(ValueError, match="n_extra must be >= 0"):
        ZeroPadTime(TIME, -1)
    with pytest.raises(ValueError, match="not inside"):
        BinSelect(STORE, (0, 10))


#: Each index-array slot: the kernels' bin lists and the ``rows`` of every map that casts them.
INDEX_SLOTS = {
    "pad_maps.k_out": lambda v: pad_maps(128, 160, v, [1, 2]),
    "pad_maps.j_in": lambda v: pad_maps(128, 160, [1, 2], v),
    "crop_maps.k_out": lambda v: crop_maps(160, 16, 128, v, [1, 2]),
    "crop_maps.j_in": lambda v: crop_maps(160, 16, 128, [1, 2], v),
    "LinearMap.rows.idx": lambda v: Restrict(FULL, CUT).rows(v),
    "TimeToWDM.rows.idx": lambda v: TimeToWDM(TIME, FULL).rows(v),
    "TimeToFourier.rows.idx": lambda v: TimeToFourier(TIME, 0.2).rows(v),
    "FourierToWDM.rows.idx": lambda v: FourierToWDM(WIN, BANDS).rows(v),
    # a "cells" push reads push_rows, not map.rows: this reaches PushedCovariance.rows alone
    "PushedCovariance.rows.idx":
        lambda v: FourierToWDM(WIN, BANDS).push(DenseCov(np.eye(64))).rows(v),
}


@pytest.mark.parametrize(("bad", "dtype"), [
    ([True, False], "bool"), (["1"], "<U1"), ([1 + 0j], "complex128"), (None, "object"),
], ids=["bool", "str", "complex", "None"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_an_index_array_of_the_wrong_dtype_raises_type_error(slot, bad, dtype):
    name = slot.split(".")[-1]
    message = f"^{name} must be an integer index array, got dtype {dtype}$"
    with pytest.raises(TypeError, match=message):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize(("bad", "first"), [([1.5], "1.5"), ([1.0, np.nan], "nan")],
                         ids=["1.5", "nan"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_a_non_integral_index_raises_value_error(slot, bad, first):
    """``np.asarray(..., dtype=np.int64)`` truncated ``1.5`` to ``1`` silently."""
    name = slot.split(".")[-1]
    with pytest.raises(ValueError, match=f"^{name} must hold integer indices, got {first}$"):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_integer_and_integral_float_index_arrays_are_taken(slot):
    want = INDEX_SLOTS[slot]([1, 2])
    for v in (np.array([1, 2], dtype=np.int32), np.array([1.0, 2.0])):
        got = INDEX_SLOTS[slot](v)
        for g, w in zip(got if isinstance(got, tuple) else (got,),
                        want if isinstance(want, tuple) else (want,)):
            np.testing.assert_array_equal(g, w)


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), ("0.2", "str"), (None, "NoneType"), (0.2 + 0j, "complex"),
    (np.array(0.2), "ndarray"),
])
def test_fmin_of_the_wrong_kind_raises_type_error(bad, kind):
    """``fmin`` was stored raw: ``"0.2"`` gave the built_for key ``(src, "0.2")``,
    ``np.array(0.2)`` an unhashable static field, and ``True`` meant 1 Hz."""
    with pytest.raises(TypeError, match=f"^fmin must be a real number, got {kind}$"):
        TimeToFourier(TIME, bad)


def test_fmin_is_stored_as_a_float():
    for v in (0.2, np.float64(0.2), np.float32(0.25), 0, np.int64(0)):
        F = TimeToFourier(TIME, v)
        assert type(F.fmin) is float and F.built_for == (TIME, float(v))
        assert F.dst == TimeToFourier(TIME, float(v)).dst
    with pytest.raises(ValueError, match="leaves no interior Fourier frequencies"):
        TimeToFourier(TIME, 5.0)


@pytest.mark.parametrize(("bad", "kind"), [
    ("no", "str"), (1, "int"), (0.5, "float"), (None, "NoneType"),
])
def test_restrict_of_the_wrong_kind_raises_type_error(bad, kind):
    """A truthy ``"no"`` used to switch the atom truncation on."""
    with pytest.raises(TypeError, match=f"^restrict must be a bool, got {kind}$"):
        FourierToWDM(WIN, BANDS, restrict=bad)


def test_restrict_is_stored_as_a_bool_before_the_key_is_formed():
    m = FourierToWDM(WIN, BANDS, restrict=np.True_)
    assert m.restrict is True and m.built_for == (WIN, BANDS, "dense", True)


@pytest.mark.parametrize(("bad", "kind"), [
    (None, "NoneType"), (3, "int"), (True, "bool"), (b"dense", "bytes"),
])
def test_impl_of_the_wrong_kind_raises_type_error(bad, kind):
    with pytest.raises(TypeError, match=f"^impl must be a string, got {kind}$"):
        FourierToWDM(WIN, BANDS, bad)


def test_an_unknown_impl_stays_a_value_error():
    with pytest.raises(ValueError, match="impl must be 'dense' or 'fft'; got 'sparse'"):
        FourierToWDM(WIN, BANDS, "sparse")


@pytest.mark.parametrize("bad", [None, Identity(WIN), BinSelect(REC, WIN)],
                         ids=["None", "Identity", "BinSelect"])
def test_after_needs_a_zero_pad_or_a_crop(bad):
    with pytest.raises(TypeError, match=f"^inner must be a FourierZeroPad or a FourierCrop, "
                                        f"got {type(bad).__name__}$"):
        FourierToWDM(WIN, BANDS).after(bad)


@pytest.mark.parametrize(("outer", "inner", "slot", "kind"), [
    (None, Restrict(FULL, CUT), "outer", "NoneType"),
    (Restrict(FULL, CUT), FULL, "inner", "WDMDomain"),
    (3, "x", "outer", "int"),
])
def test_compose_needs_two_linear_maps(outer, inner, slot, kind):
    """``Compose`` used to accept any two objects and fail at first use."""
    with pytest.raises(TypeError, match=f"^{slot} must be a LinearMap, got {kind}$"):
        Compose(outer, inner)


def _every_map():
    """One map of each of the 12 classes, and an ``after()`` map."""
    W = FourierToWDM(WIN, BANDS)
    maps = [BUILDS[name]() for name in BUILDS]
    maps += [Compose(Restrict(FULL, CUT), TimeToWDM(TIME, FULL)),
             W.after(FourierCrop(LONG, WIN, 16))]
    return maps


@pytest.mark.jax
@pytest.mark.parametrize("leaf", [lambda x: None, lambda x: object()], ids=["None", "object"])
def test_jax_rebuilds_with_placeholder_leaves_still_work(leaf):
    """``jax.tree_util`` rebuilds a map through its constructor, with placeholders for the
    arrays it carries: the checks read only static fields and map arguments."""
    jax, _ = jax_or_skip()
    maps = _every_map()
    assert len({type(m) for m in maps}) == 12
    for m in maps:
        r = jax.tree_util.tree_map(leaf, m)
        assert type(r) is type(m) and (r.src, r.dst) == (m.src, m.dst)
        assert getattr(r, "built_for", None) == getattr(m, "built_for", None)
