"""Linear maps between domains and the lazy congruence they push.

A map is a frozen dataclass registered as a JAX pytree (``wilson.backend.pytree_dataclass``):
its host-built constants (index tables, atom matrices, Dirichlet kernels) are the pytree
*data*, its domains and sizes the static *meta*.  Pass a map to ``jax.jit`` as an
argument, never close over it, and call :meth:`LinearMap.on` once to move its constants
to the device.

Every constructor builds its constants in host NumPy float64 when they are not given
(the ``_BUILD`` sentinel), and records the fields they derive from in ``built_for`` (static
meta, left out of the ``repr``).  ``jax.tree_util`` rebuilds a map from its leaves without
recomputing anything.  Constants given to a constructor (``dataclasses.replace``, ``on``,
``tree_unflatten``) must come with a ``built_for`` equal to the map's own fields:
``dataclasses.replace(map, dst=...)`` raises ``ValueError`` instead of reusing constants
built for the old domain.

Map kinds (what ``apply`` takes and returns; ``adjoint`` is the adjoint for the real
inner product, ``<z, z'> = Re sum conj(z) z'`` on complex vectors):

==========  ==========================  ================================================
kind        apply                       push(cov) (``PushedCovariance``)
==========  ==========================  ================================================
"real"      real -> real                ``M Sigma M^T``; needs ``cov.matvec``
"analysis"  real -> complex bins        ``F Sigma F^H`` (complex Hermitian)
"cells"     complex bins -> real cells  ``1/2 Re(C M' C^H)``; needs ``cov.dense()``
"complex"   complex -> complex          ``A M' A^H`` for complex-linear maps (BinSelect)
"widely"    complex -> complex          not pushable (A z + B conj z; raises TypeError)
==========  ==========================  ================================================

A covariance without ``matvec`` -- every covariance on WDM cells -- cannot be pushed through
a real or analysis map: ``PushedCovariance.rows``/``matvec`` raise a ``TypeError`` (the
WDM -> time push ``W^T Lambda W / dt^2`` is not available).  Fix the
covariance at a noise state as a :class:`domainia.operators.CovarianceOperator` and see it
from time through :func:`domainia.translate.translate_operator` instead.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from domainia.wilson.backend import NUMPY, Backend, get_backend, pytree_dataclass, require_x64
from domainia.wilson.windows import Window

from ..domains import FourierDomain, TimeDomain, WDMDomain

__all__ = [
    "BinSelect",
    "Compose",
    "Identity",
    "LinearMap",
    "PushedCovariance",
    "Restrict",
    "SupportsDense",
    "SupportsMatvec",
    "domain_size",
    "on_backend",
]


class _Build:
    """Sentinel default of every constant field: "build it on the host"."""

    def __repr__(self) -> str:
        return "<build>"


_BUILD: Any = _Build()


def _built_for(m: Any, key: tuple, *, given: bool) -> None:
    """The ``built_for`` guard of a map whose constants derive from the fields ``key``.

    Constants about to be built (``given`` false) record ``key``.  Constants given to the
    constructor -- by ``dataclasses.replace``, :meth:`LinearMap.on`, ``jax.tree_util``
    unflatten or an internal caller -- must come with the ``built_for`` of the map they were
    built for, and it must equal ``key``; ``ValueError`` otherwise.  It reads static fields
    only, so it also runs under ``jax.jit``."""
    if not given:
        object.__setattr__(m, "built_for", key)
    elif m.built_for is None:
        raise ValueError(f"{type(m).__name__}: constants given without built_for: construct the "
                         "map from its fields")
    elif m.built_for != key:
        raise ValueError(f"{type(m).__name__}: constants built for another src/dst: construct a "
                         "new map")


def _last_axis(x: Any, n: int, what: str) -> None:
    """Refuse a vector whose last axis is not ``n`` long (shapes are static under ``jit``)."""
    if np.shape(x)[-1:] != (n,):
        raise ValueError(f"{what}: last axis must have length {n}; got shape {np.shape(x)}")


def _require_same_units(what: str, src: FourierDomain, dst: FourierDomain) -> None:
    """Refuse a map between bins in different units: ``(units, sigma0)`` must match, because
    no map rescales silently."""
    if (src.units, src.sigma0) != (dst.units, dst.sigma0):
        raise ValueError(f"{what} keeps the units: src has units={src.units!r}, "
                         f"sigma0={src.sigma0!r} but dst has units={dst.units!r}, "
                         f"sigma0={dst.sigma0!r}")


def domain_size(dom: Any) -> int:
    """Length of a vector on ``dom``: samples, bins (complex) or cells."""
    if isinstance(dom, TimeDomain):
        return int(dom.N)
    if isinstance(dom, FourierDomain):
        return int(dom.n)
    if isinstance(dom, WDMDomain):
        return len(dom.cells.m)
    raise TypeError(f"not a domain: {dom!r}")


def on_backend(obj: Any, be: Backend, fields: tuple[str, ...]) -> Any:
    """``obj`` with its constant ``fields`` (NumPy arrays, tuples of them, or nested maps)
    moved to ``be`` by ``be.asarray``: build once on
    the host, then ``device_put``.  NumPy returns ``obj`` itself.  Any other backend first
    calls ``require_x64()``: a ``Backend`` object kept from earlier would
    otherwise turn the float64 constants into float32 silently when x64 is off."""
    if be.name == "numpy":
        return obj
    require_x64()

    def move(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return be.asarray(v)
        if isinstance(v, tuple):
            return tuple(move(u) for u in v)
        if isinstance(v, LinearMap):
            return v.on(be)
        return v

    return dataclasses.replace(obj, **{k: move(getattr(obj, k)) for k in fields})


class SupportsMatvec(Protocol):
    """A real symmetric covariance on a map's source: ``matvec(X)`` is ``X Sigma`` row by
    row, ``(..., n) -> (..., n)``."""

    def matvec(self, X: Any, be: Backend | None = None) -> Any: ...


class SupportsDense(Protocol):
    """A complex Hermitian covariance ``M' = E[z z^H]`` of a map's source bins."""

    def dense(self, be: Backend | None = None) -> Any: ...


class LinearMap:
    """Base of every map.  Subclasses define ``src``, ``dst``, ``kind`` and ``apply``/``adjoint``."""

    kind: str = "real"

    @property
    def shape(self) -> tuple[int, int]:
        return domain_size(self.dst), domain_size(self.src)

    def _be(self, x: Any, be: Backend | None) -> Backend:
        """``be``, else JAX iff ``x`` or one of the map's constants is a ``jax.Array``."""
        if be is not None:
            return be
        consts = [getattr(self, f.name) for f in dataclasses.fields(self)]
        return get_backend(x, *(v for v in consts
                                if hasattr(v, "shape") and not isinstance(v, LinearMap)))

    def apply(self, x: Any, be: Backend | None = None) -> Any:
        raise NotImplementedError

    def adjoint(self, y: Any, be: Backend | None = None) -> Any:
        raise NotImplementedError

    def rows(self, idx: Any) -> np.ndarray:
        """Host dense rows ``M[idx, :]`` (``(k, n_src)``), from the adjoint of unit vectors."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        E = np.zeros((idx.size, self.shape[0]))
        E[np.arange(idx.size), idx] = 1.0
        out = np.asarray(self.adjoint(E, NUMPY))
        return np.conj(out) if np.iscomplexobj(out) else out

    def dense(self) -> np.ndarray:
        """The whole host matrix (tests and small grids only)."""
        return self.rows(np.arange(self.shape[0]))

    def push(self, cov: Any) -> "PushedCovariance":
        """``M Sigma M^T`` (or its complex counterpart, see the module table), lazily."""
        if self.kind == "widely":
            raise TypeError(
                f"{type(self).__name__} is widely linear (A z + B conj z): a circular bin "
                "covariance does not stay circular, so it cannot be pushed")
        return PushedCovariance(self, cov)

    def on(self, be: Backend) -> "LinearMap":
        """The same map with every array constant (and nested map) moved to ``be``."""
        names = tuple(f.name for f in dataclasses.fields(self)
                      if isinstance(getattr(self, f.name), (np.ndarray, LinearMap)))
        return on_backend(self, be, names)


@dataclass(frozen=True, eq=False)
class PushedCovariance:
    """``M Sigma M^T`` evaluated on demand: ``rows(idx)``, ``diag()``, ``dense()``.

    Rows follow the oracle order of each map kind: ``"real"`` --
    ``M (Sigma M[idx]^T)``, i.e. ``map.apply(cov.matvec(map.rows(idx)))`` (atoms, then
    the covariance, then the forward transform);
    ``"cells"`` -- ``0.5 * Re((C[idx] @ M') @ conj(C).T)``;
    ``"analysis"`` -- ``F[idx] Sigma F^H``; ``"complex"`` --
    ``(A M' A^H)[idx]``.

    A covariance that carries a Wilson ``window`` (``EvolutionaryCovariance``, a
    ``DiagonalCovariance`` over a ``BandAveragedVariance``, a ``PushedCovariance`` onto WDM
    cells) lives on WDM cells in that window's basis: on a map whose ``src`` is not a
    ``WDMDomain``, or is one with another window, it raises ``ValueError``.  A ``map`` that is
    not a :class:`LinearMap`, or a ``cov`` with neither ``matvec`` (:class:`SupportsMatvec`)
    nor ``dense`` (:class:`SupportsDense`), raises ``TypeError``.
    """

    map: LinearMap
    cov: Any

    def __post_init__(self) -> None:
        if not isinstance(self.map, LinearMap):
            raise TypeError(f"PushedCovariance needs a LinearMap, got {type(self.map).__name__}")
        if not (callable(getattr(self.cov, "matvec", None))
                or callable(getattr(self.cov, "dense", None))):
            raise TypeError(f"cannot push a {type(self.cov).__name__}: a covariance needs a matvec "
                            "(SupportsMatvec) or a dense (SupportsDense) method")
        src, window = getattr(self.map, "src", None), getattr(self.cov, "window", None)
        if isinstance(window, Window) and not isinstance(src, WDMDomain):
            raise ValueError(f"the covariance lives on WDM cells (window {window.name}) but the "
                             f"map's source domain is a {type(src).__name__}: push it through a "
                             "map from a WDMDomain")
        if isinstance(src, WDMDomain) and isinstance(window, Window) and window != src.window:
            raise ValueError(f"the covariance is built for the window {window.name} but the map's "
                             f"source domain uses {src.window.name}: push within one basis")

    @property
    def window(self) -> Window | None:
        """The Wilson window of the cells this covariance lives on (``map.dst.window``), or
        ``None`` when ``map.dst`` is not a ``WDMDomain``."""
        dst = getattr(self.map, "dst", None)
        return dst.window if isinstance(dst, WDMDomain) else None

    @property
    def size(self) -> int:
        return self.map.shape[0]

    def _matvec(self) -> Any:
        """``cov.matvec``, or a TypeError: a covariance on WDM cells (``DiagonalCovariance``,
        ``EvolutionaryCovariance``, ...) has no fixed-noise-state ``matvec``, so the
        WDM -> time push ``W^T Lambda W / dt^2`` is not available; the error names the
        route: :class:`domainia.operators.CovarianceOperator`, then
        :func:`domainia.translate.translate_operator`."""
        fn = getattr(self.cov, "matvec", None)
        if fn is None:
            raise TypeError(f"{type(self.cov).__name__} has no matvec: a push through a "
                            f"{self.map.kind!r} map needs a fixed-noise-state SupportsMatvec "
                            "(X -> X Sigma), so the WDM -> time push is not available; fix the "
                            "covariance at a noise state as a domainia.operators.CovarianceOperator "
                            "(a domainia.translate.Operator) and see it from time with "
                            "domainia.translate.translate_operator")
        return fn

    def rows(self, idx: Any, be: Backend | None = None) -> Any:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        kind = self.map.kind
        if kind == "real":
            matvec = self._matvec()
            R = self.map.rows(idx)
            return self.map.apply(matvec(R, be), be)
        if kind == "cells":
            return self.map.push_rows(self.cov.dense(be), idx, be)
        if kind == "complex":
            return self.map.push_rows(self.cov.dense(be), idx, be)
        if kind == "analysis":
            matvec = self._matvec()
            F = self.map.rows(idx)                                   # (k, N) complex
            Y = matvec(F.real, be) + 1j * matvec(F.imag, be)
            return np.conj(self.map.apply_complex(np.conj(Y), be))
        raise TypeError(f"cannot push through a {kind!r} map")

    def dense(self, be: Backend | None = None) -> Any:
        return self.rows(np.arange(self.size), be)

    def diag(self, be: Backend | None = None) -> Any:
        D = self.dense(be)
        be = be or get_backend(D)
        out = be.xp.diagonal(D)
        return be.xp.real(out) if be.xp.iscomplexobj(out) else out

    def matvec(self, X: Any, be: Backend | None = None) -> Any:
        """``X (M Sigma M^T)`` row by row (real kinds), so pushes compose."""
        if self.map.kind != "real":
            raise TypeError("matvec of a pushed covariance is defined for real maps only")
        return self.map.apply(self._matvec()(self.map.adjoint(X, be), be), be)


@pytree_dataclass(data=(), meta=("src",))
@dataclass(frozen=True, eq=False)
class Identity(LinearMap):
    """The identity on ``src`` (the sinusoid pad model's samples)."""

    src: Any

    @property
    def dst(self) -> Any:
        return self.src

    @property
    def kind(self) -> str:  # type: ignore[override]
        return "complex" if isinstance(self.src, FourierDomain) else "real"

    def apply(self, x, be=None):
        return x

    def adjoint(self, y, be=None):
        return y

    def push_rows(self, Mp, idx, be=None):
        return Mp[idx]


@pytree_dataclass(data=("outer", "inner"), meta=())
@dataclass(frozen=True, eq=False)
class Compose(LinearMap):
    """``outer . inner`` (``inner`` first).  ``inner.dst`` must equal ``outer.src``."""

    outer: Any
    inner: Any

    def __post_init__(self) -> None:
        if isinstance(self.outer, LinearMap) and isinstance(self.inner, LinearMap):
            a, b = self.inner.dst, self.outer.src
            if isinstance(a, WDMDomain) and isinstance(b, WDMDomain) and a.window != b.window:
                raise ValueError(f"cannot compose: inner.dst and outer.src use different windows "
                                 f"({a.window.name} vs {b.window.name})")
            if b != a:
                raise ValueError(f"cannot compose: {a!r} != {b!r}")

    @property
    def src(self) -> Any:
        return self.inner.src

    @property
    def dst(self) -> Any:
        return self.outer.dst

    @property
    def kind(self) -> str:  # type: ignore[override]
        if self.outer.kind == "real" and self.inner.kind == "real":
            return "real"
        return "composite"

    def apply(self, x, be=None):
        return self.outer.apply(self.inner.apply(x, be), be)

    def adjoint(self, y, be=None):
        return self.inner.adjoint(self.outer.adjoint(y, be), be)

    def rows(self, idx):
        # Row r of outer.inner is outer[r] inner = conj(inner^H conj(outer[r])): the same
        # conj convention as LinearMap.rows.  An all-real composite is untouched.
        R = self.outer.rows(idx)
        out = np.asarray(self.inner.adjoint(np.conj(R) if np.iscomplexobj(R) else R, NUMPY))
        return np.conj(out) if np.iscomplexobj(out) else out

    def push(self, cov):
        if self.kind != "real":
            raise TypeError("Compose.push is defined for real maps only "
                            f"({self.outer.kind} after {self.inner.kind})")
        return PushedCovariance(self, cov)


@pytree_dataclass(data=("pos",), meta=("src", "dst", "built_for"))
@dataclass(frozen=True, eq=False)
class Restrict(LinearMap):
    """The cells ``dst.cells`` (a subset, in its own order) of a vector on ``src.cells``.

    ``src`` and ``dst`` must share the grid and the Wilson window (``ValueError`` naming both
    windows otherwise): a restriction never changes the basis.  These checks read the domains
    only, so they run on every construction, ``pos`` given or not (``on``, ``tree_unflatten``,
    ``dataclasses.replace``); only the subset positions ``pos`` are built once.  The
    ``built_for`` guard runs after these checks, so a window or grid mismatch keeps its
    message."""

    src: WDMDomain
    dst: WDMDomain
    pos: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        if self.src.window != self.dst.window:
            raise ValueError(f"src and dst use different windows ({self.src.window.name} vs "
                             f"{self.dst.window.name}): restrict within one basis")
        if self.src.cells.grid != self.dst.cells.grid:
            raise ValueError(f"dst is not a subset of src: its cells live on another grid "
                             f"({self.dst.cells.grid} vs {self.src.cells.grid})")
        _built_for(self, (self.src, self.dst), given=self.pos is not _BUILD)
        if self.pos is _BUILD:
            where = {c: i for i, c in enumerate(zip(self.src.cells.m, self.src.cells.n))}
            missing = [c for c in zip(self.dst.cells.m, self.dst.cells.n) if c not in where]
            if missing:
                raise ValueError(f"dst is not a subset of src ({len(missing)} cells missing)")
            object.__setattr__(self, "pos", np.asarray(
                [where[c] for c in zip(self.dst.cells.m, self.dst.cells.n)], dtype=np.int64))

    def apply(self, x, be=None):
        return x[..., self.pos]

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        out = be.xp.zeros((*y.shape[:-1], self.shape[1]), dtype=y.dtype)
        return be.set(out, (..., self.pos), y, copy=False)


@pytree_dataclass(data=(), meta=("src", "dst"))
@dataclass(frozen=True, eq=False)
class BinSelect(LinearMap):
    """The bins of the sub-window ``dst`` of ``src`` (store -> window -> nominal).

    ``BinSelect(src, (k0, n))`` takes absolute bins ``[k0, k0 + n)`` of ``src``'s record.
    """

    src: FourierDomain
    dst: Any

    kind = "complex"

    def __post_init__(self) -> None:
        if isinstance(self.dst, tuple):
            k0, n = (int(v) for v in self.dst)
            object.__setattr__(self, "dst", dataclasses.replace(self.src, k0=k0, n=n))
        d, s = self.dst, self.src
        _require_same_units("BinSelect", s, d)
        if (d.N, d.dt) != (s.N, s.dt) or d.k0 < s.k0 or d.k0 + d.n > s.k0 + s.n:
            raise ValueError(f"bins [{d.k0}, {d.k0 + d.n}) are not inside [{s.k0}, {s.k0 + s.n})")

    @property
    def offset(self) -> int:
        return self.dst.k0 - self.src.k0

    def apply(self, z, be=None):
        return z[..., self.offset:self.offset + self.dst.n]

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        out = be.xp.zeros((*y.shape[:-1], self.src.n), dtype=complex)
        return be.set(out, (..., slice(self.offset, self.offset + self.dst.n)), y, copy=False)

    def push_rows(self, Mp, idx, be=None):
        sel = self.offset + np.arange(self.dst.n)
        return Mp[np.ix_(self.offset + np.asarray(idx), sel)]
