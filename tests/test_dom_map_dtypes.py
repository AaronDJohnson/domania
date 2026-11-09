"""The maps compute in double (spec 2026-11-08 §1): a map that computes returns
float64/complex128 whatever the input's precision, integers included; a map that only
selects, slices or zero-fills returns its input's dtype (``BinSelect.adjoint``: the input's
complex counterpart).  One table over every public map method that takes data, on NumPy and
on JAX with x64 on."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from dom_db_inputs import DenseCov, jax_or_skip

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import (BinSelect, Compose, CropTime, FourierCrop, FourierToWDM,
                           FourierZeroPad, Identity, Restrict, TimeToFourier, TimeToWDM,
                           WDMToTime, ZeroPadTime, domain_size)

GRID = wilson.Grid(8, 16, 0.5)
TIME = TimeDomain(GRID.N, GRID.dt)
FULL = WDMDomain.full(GRID)
CUT = WDMDomain(FULL.cells.drop_columns(2, 1))
BANDS = WDMDomain(CellSet.bands(GRID, tuple(range(1, 8))))
REC = FourierDomain(GRID.N, GRID.dt, 0, GRID.N // 2 + 1, "rfft")     # every bin of the record
WIN = FourierDomain(GRID.N, GRID.dt, 0, 64, "rfft")                  # the band windows of BANDS
STORE = FourierDomain(GRID.N, GRID.dt, 1, 63, "rfft")
SUB = FourierDomain(GRID.N, GRID.dt, 8, 40, "rfft")                  # inside WIN: restrict=True
LONG = FourierDomain(160, GRID.dt, 0, 81, "rfft")                    # a 160-sample record

MAPS = {
    "Identity(time)": Identity(TIME),
    "Identity(bins)": Identity(REC),
    "Restrict": Restrict(FULL, CUT),
    "CropTime": CropTime(TIME, 8, 64),
    "ZeroPadTime": ZeroPadTime(TIME, 32),
    "BinSelect": BinSelect(REC, (3, 10)),
    "TimeToWDM": TimeToWDM(TIME, FULL),
    "WDMToTime": WDMToTime(FULL, TIME),
    "TimeToFourier": TimeToFourier(TIME, 0.2),
    "FourierZeroPad": FourierZeroPad(STORE, LONG),
    "FourierCrop": FourierCrop(LONG, WIN, 16),
    "FourierToWDM(dense)": FourierToWDM(WIN, BANDS),
    "FourierToWDM(fft)": FourierToWDM(WIN, BANDS, "fft"),
    "FourierToWDM(restrict)": FourierToWDM(SUB, BANDS, restrict=True),
    "after(crop)": FourierToWDM(WIN, BANDS).after(FourierCrop(LONG, WIN, 16)),
    "Compose": Compose(Restrict(FULL, CUT), TimeToWDM(TIME, FULL)),
}

REAL = ("float32", "float64", "int64")
COMPLEX = ("complex64", "complex128")


def same(*dtypes):
    return {d: d for d in dtypes}


def to(out, *dtypes):
    return {d: out for d in dtypes}


#: (map, method, {input dtype: output dtype}).  For ``push.rows`` the input dtype is the
#: covariance's: real for a "real" or "analysis" push, complex for "complex" and "cells".
TABLE = [
    # select, slice or zero-fill: the input's dtype
    ("Identity(time)", "apply", same(*REAL)),
    ("Identity(time)", "adjoint", same(*REAL)),
    ("Identity(bins)", "apply", same(*COMPLEX)),
    ("Identity(bins)", "adjoint", same(*COMPLEX)),
    ("Restrict", "apply", same(*REAL)),
    ("Restrict", "adjoint", same(*REAL)),
    ("CropTime", "apply", same(*REAL)),
    ("CropTime", "adjoint", same(*REAL)),
    ("ZeroPadTime", "apply", same(*REAL)),
    ("ZeroPadTime", "adjoint", same(*REAL)),
    ("BinSelect", "apply", same(*COMPLEX)),
    # ... except BinSelect.adjoint: the input's complex counterpart
    ("BinSelect", "adjoint", {"float32": "complex64", "float64": "complex128",
                              "int64": "complex128", **same(*COMPLEX)}),
    # maps that compute: float64 / complex128
    ("TimeToWDM", "apply", to("float64", *REAL)),
    ("TimeToWDM", "adjoint", to("float64", *REAL)),
    ("WDMToTime", "apply", to("float64", *REAL)),
    ("WDMToTime", "adjoint", to("float64", *REAL)),
    ("TimeToFourier", "apply", to("complex128", *REAL)),
    ("TimeToFourier", "apply_complex", to("complex128", *REAL, *COMPLEX)),
    ("TimeToFourier", "adjoint", to("float64", "float32", *COMPLEX)),
    ("FourierZeroPad", "apply", to("complex128", *COMPLEX)),
    ("FourierZeroPad", "adjoint", to("complex128", *COMPLEX)),
    ("FourierCrop", "apply", to("complex128", *COMPLEX)),
    ("FourierCrop", "apply_data", to("complex128", *COMPLEX)),
    ("FourierCrop", "adjoint", to("complex128", *COMPLEX)),
    ("FourierToWDM(dense)", "apply", to("float64", *COMPLEX)),
    ("FourierToWDM(dense)", "adjoint", to("complex128", *REAL)),
    ("FourierToWDM(fft)", "apply", to("float64", *COMPLEX)),
    ("FourierToWDM(fft)", "adjoint", to("complex128", *REAL)),
    ("FourierToWDM(restrict)", "apply", to("float64", *COMPLEX)),
    ("FourierToWDM(restrict)", "adjoint", to("complex128", *REAL)),
    ("after(crop)", "apply", to("float64", *COMPLEX)),
    ("after(crop)", "adjoint", to("complex128", *REAL)),
    ("Compose", "apply", to("float64", *REAL)),
    ("Compose", "adjoint", to("float64", *REAL)),
    # pushes: double through a map that computes; the covariance's dtype through
    # Identity(bins) and BinSelect; a "real" push reads matvec on float64 unit rows
    ("Identity(time)", "push.rows", to("float64", "float32", "float64")),
    ("Identity(bins)", "push.rows", same(*COMPLEX)),
    ("Restrict", "push.rows", to("float64", "float32", "float64")),
    ("CropTime", "push.rows", to("float64", "float32", "float64")),
    ("ZeroPadTime", "push.rows", to("float64", "float32", "float64")),
    ("BinSelect", "push.rows", same(*COMPLEX)),
    ("TimeToWDM", "push.rows", to("float64", "float32", "float64")),
    ("WDMToTime", "push.rows", to("float64", "float32", "float64")),
    ("TimeToFourier", "push.rows", to("complex128", "float32", "float64")),
    ("FourierToWDM(dense)", "push.rows", to("float64", *COMPLEX)),
    ("FourierToWDM(fft)", "push.rows", to("float64", *COMPLEX)),
    ("FourierToWDM(restrict)", "push.rows", to("float64", *COMPLEX)),
    ("after(crop)", "push.rows", to("float64", *COMPLEX)),
    ("Compose", "push.rows", to("float64", "float32", "float64")),
]
CASES = [(name, method, d, out) for name, method, table in TABLE for d, out in table.items()]
IDS = [f"{name}.{method}-{d}" for name, method, d, _ in CASES]


def _array(dtype, shape, rng):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    if dtype.kind in "ib":
        return rng.integers(-3, 4, shape).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


def _run(name, method, dtype, be, rng):
    """``method`` of ``MAPS[name]`` on an input of ``dtype`` (``be`` None: NumPy)."""
    m = MAPS[name]
    if method == "push.rows":
        A = _array(dtype, (domain_size(m.src),) * 2, rng)
        S = A @ np.conj(A).T                                    # symmetric / Hermitian
        if be is not None:
            m, S = m.on(be), be.asarray(S)
        return m.push(DenseCov(S)).rows([0, 1], be)
    x = _array(dtype, domain_size(m.dst if method == "adjoint" else m.src), rng)
    if be is not None:
        m, x = m.on(be), be.asarray(x)
    return getattr(m, method)(x)


@pytest.mark.parametrize(("name", "method", "dtype", "want"), CASES, ids=IDS)
def test_the_output_dtype_of_every_map_method(name, method, dtype, want, rng):
    out = _run(name, method, dtype, None, rng)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.dtype(want)


@pytest.mark.jax
@pytest.mark.parametrize(("name", "method", "dtype", "want"), CASES, ids=IDS)
def test_the_output_dtype_of_every_map_method_on_jax(name, method, dtype, want, rng):
    """JAX with x64 on agrees with NumPy row for row (the dtype is computed on the host)."""
    jax, be = jax_or_skip()
    out = _run(name, method, dtype, be, rng)
    if method != "push.rows":               # an "analysis" push returns a host array
        assert isinstance(out, jax.Array)
    assert np.dtype(out.dtype) == np.dtype(want)


#: The five methods that take a real array, and the name of that array.
REAL_TAKING = [("TimeToWDM", "apply", "x"), ("TimeToWDM", "adjoint", "y"),
               ("WDMToTime", "apply", "y"), ("WDMToTime", "adjoint", "x"),
               ("TimeToFourier", "apply", "x")]


@pytest.mark.parametrize("dtype", ["complex64", "complex128", "bool"])
@pytest.mark.parametrize(("name", "method", "arg"), REAL_TAKING,
                         ids=[f"{n}.{m}" for n, m, _ in REAL_TAKING])
def test_a_real_taking_method_refuses_a_complex_or_bool_input(name, method, arg, dtype, rng):
    """The float64 cast would drop an imaginary part with only a ``ComplexWarning``, and read
    a bool array as 0 and 1.  Before it, complex input was refused by accident, differently
    per backend (NumPy's rfft ``TypeError``, wilson's ``wmn`` check, JAX's ``ValueError``),
    and a bool array not at all."""
    with pytest.raises(TypeError, match=f"^{arg} must be a real numeric array, got dtype {dtype}$"):
        _run(name, method, dtype, None, rng)


@pytest.mark.jax
@pytest.mark.parametrize("dtype", ["complex64", "complex128", "bool"])
@pytest.mark.parametrize(("name", "method", "arg"), REAL_TAKING,
                         ids=[f"{n}.{m}" for n, m, _ in REAL_TAKING])
def test_a_real_taking_method_refuses_a_complex_or_bool_input_on_jax(name, method, arg, dtype,
                                                                      rng):
    _, be = jax_or_skip()
    with pytest.raises(TypeError, match=f"^{arg} must be a real numeric array, got dtype {dtype}$"):
        _run(name, method, dtype, be, rng)


@pytest.mark.jax
def test_the_conversion_runs_under_jit():
    """``require_dtype`` reads a tracer's dtype without converting it, and the float64 cast
    is traced."""
    jax, be = jax_or_skip()
    m = TimeToWDM(TIME, FULL)
    x = np.random.default_rng(1).standard_normal((2, GRID.N)).astype(np.float32)
    run = jax.jit(lambda mm, v: mm.apply(v))
    out = run(m.on(be), be.asarray(x))
    assert isinstance(out, jax.Array) and out.dtype == np.float64
    want = m.apply(x.astype(np.float64))
    np.testing.assert_allclose(np.asarray(out), want, rtol=0, atol=1e-13 * np.max(np.abs(want)))
    with pytest.raises(TypeError, match="^x must be a real numeric array, got dtype complex64"):
        run(m.on(be), be.asarray(x.astype(np.complex64)))


def test_large_dt_complex64_bins_give_finite_float64_cells_on_the_fft_path():
    """At ``dt = 1e4`` the window samples scale with ``sqrt(dt)``: complex64 bins of amplitude
    1e36 overflowed float32 inside ``forward_bands`` and came back NaN, with only
    ``RuntimeWarning``s, though the cells fit float32.  The fft path now runs in double, with
    no warning, and matches the dense path."""
    g = wilson.Grid(8, 16, 1e4)
    cells = WDMDomain(CellSet.bands(g, tuple(range(1, 8))))
    win = FourierDomain(g.N, g.dt, 0, 64, "rfft")
    rng = np.random.default_rng(3)
    z = ((rng.standard_normal(64) + 1j * rng.standard_normal(64)) * 1e36).astype(np.complex64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fft = FourierToWDM(win, cells, "fft").apply(z)
    dense = FourierToWDM(win, cells).apply(z)
    assert fft.dtype == np.float64 and np.all(np.isfinite(fft))
    assert np.max(np.abs(dense)) < np.finfo(np.float32).max
    np.testing.assert_allclose(fft, dense, rtol=0, atol=1e-12 * np.max(np.abs(dense)))


@pytest.mark.parametrize("dt", [0.5, np.float64(0.5)], ids=["float", "np.float64"])
def test_the_time_maps_do_not_follow_the_grid_dt_type(dt, rng):
    """``wilson.Grid`` keeps the raw ``dt`` type.  With ``np.float64`` the ``g.dt *`` of
    ``TimeToWDM.adjoint`` and the ``/ g.dt`` of ``WDMToTime.adjoint`` promoted float32 input to
    float64 (after a single-precision transform); with a Python float they did not.  Both now
    convert first, so the result is the float64 run's, bit for bit, whatever ``dt``'s type."""
    g = wilson.Grid(8, 16, dt)
    time, full = TimeDomain(g.N, 0.5), WDMDomain.full(g)
    y = rng.standard_normal(full.size).astype(np.float32)
    x = rng.standard_normal(g.N).astype(np.float32)
    got = (TimeToWDM(time, full).adjoint(y), WDMToTime(full, time).adjoint(x))
    want = (TimeToWDM(TIME, FULL).adjoint(y.astype(np.float64)),
            WDMToTime(FULL, TIME).adjoint(x.astype(np.float64)))
    for g_, w_ in zip(got, want):
        assert g_.dtype == np.float64
        np.testing.assert_array_equal(g_, w_)


def test_fourier_domains_store_python_floats():
    """Spec §1.5: ``FourierDomain`` stores ``dt`` and ``sigma0`` as Python floats, so
    ``rfft_scale`` is one too and no weak-type promotion depends on the caller's scalar."""
    d = FourierDomain(GRID.N, np.float64(0.5), 0, 65, "dt_rfft")
    s = FourierDomain(GRID.N, 0.5, 0, 65, "sigma0", sigma0=np.float32(2.5))
    assert type(d.dt) is float and type(d.rfft_scale) is float
    assert type(s.sigma0) is float and type(s.rfft_scale) is float
