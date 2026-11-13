"""The window through the domainia layer: WDMDomain, maps, variances, oracles."""

from __future__ import annotations

import copy
import dataclasses
import pickle

import numpy as np
import pytest
from dom_db_inputs import DenseCov, jax_or_skip

from domainia import testing, wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import (Compose, FourierCrop, FourierToWDM, Identity, Restrict, TimeToWDM,
                           WDMToTime)
from domainia.spectra import GenericPSD
from domainia.testing import ALT_WINDOW
from domainia.variance import BandAveragedVariance, MidpointVariance
from domainia.wilson import MEYER

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
CELLS = CellSet.all_active(GRID)
TIME = TimeDomain(N, DT)
X = np.random.default_rng(31).standard_normal(N)
#: Both window names, in either order (the error messages name the two windows).
BOTH = r"(meyer\(d=4\).*wide-smoothstep|wide-smoothstep.*meyer\(d=4\))"


def _close(got, want, rtol: float = 1e-12) -> None:
    """``max |got - want| <= rtol * max |want|``."""
    err, scale = float(np.max(np.abs(got - want))), float(np.max(np.abs(want)))
    assert err <= rtol * scale, f"{err:.3e} > {rtol:.0e} * {scale:.3e}"


def _stationary_psd() -> tuple[GenericPSD, np.ndarray]:
    psd = GenericPSD(lambda f, t, nu: (1.0 + np.asarray(f, float) ** 2) ** nu[0] + 0.0 * np.asarray(t, float))
    return psd, np.array([-1.2])


def _roundtrip(obj):
    """A pickle round trip: an equal object that is not the original."""
    return pickle.loads(pickle.dumps(obj))


def _exact(psd: GenericPSD, nu: np.ndarray, window=ALT_WINDOW) -> np.ndarray:
    """The exact stationary covariance of all active cells with ``window`` (value-verified kernel)."""
    return wilson.matrix.build_C_wdm_active(NF, NT, DT, lambda f: psd(f, 0.0, nu), window=window)


class _UnhashableWindow:
    """A third-party Window that cannot be hashed (so it cannot be part of a domain's identity)."""

    name = "unhashable"
    __hash__ = None  # type: ignore[assignment]

    def sample(self, Nf, Nt, dt):
        return ALT_WINDOW.sample(Nf, Nt, dt)


# ---------------------------------------------------------------------------
# WDMDomain
# ---------------------------------------------------------------------------


def test_the_window_is_part_of_the_domain_identity():
    a, b, c = WDMDomain(CELLS), WDMDomain(CELLS, window=MEYER), WDMDomain(CELLS, window=ALT_WINDOW)
    assert a == b and hash(a) == hash(b) and a != c
    assert a.window is MEYER
    with pytest.raises(TypeError, match="Window"):
        WDMDomain(CELLS, window="meyer")
    with pytest.raises(TypeError, match="Window"):
        WDMDomain(CELLS, window=wilson.Meyer)                       # the class, not an instance


def test_an_unhashable_window_fails_at_construction():
    with pytest.raises(TypeError, match="hashable"):
        WDMDomain(CELLS, window=_UnhashableWindow())


@pytest.mark.parametrize("roundtrip", [copy.deepcopy, lambda d: pickle.loads(pickle.dumps(d))],
                         ids=["deepcopy", "pickle"])
def test_the_domain_survives_copy_and_pickle_with_its_window(roundtrip):
    dom = WDMDomain(CELLS, window=ALT_WINDOW)          # module-level profile: copy/pickle-stable
    back = roundtrip(dom)
    assert back == dom and hash(back) == hash(dom) and back.window == ALT_WINDOW
    assert back != WDMDomain(CELLS)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def test_time_maps_use_the_domain_window():
    dom = WDMDomain(CELLS, window=ALT_WINDOW)
    w = TimeToWDM(TIME, dom).apply(X)
    full = wilson.forward_time(X, NF, NT, DT, window=ALT_WINDOW)
    np.testing.assert_array_equal(w, full[np.asarray(CELLS.m), np.asarray(CELLS.n)])
    np.testing.assert_allclose(WDMToTime(dom, TIME).apply(w), X, rtol=0, atol=1e-12)
    y = np.random.default_rng(2).standard_normal(CELLS.size)
    assert float(np.dot(TimeToWDM(TIME, dom).apply(X), y)) == pytest.approx(
        float(np.dot(X, TimeToWDM(TIME, dom).adjoint(y))), rel=1e-12)


def test_time_map_batches_rows_and_inner_maps_use_the_domain_window():
    """The batched ``apply``, the atom ``rows``, the adjoint against the dense ``W`` of the window,
    and the ``TimeToWDM`` that ``WDMToTime.adjoint`` builds on its own ``src``."""
    dom = WDMDomain(CELLS, window=ALT_WINDOW)
    fwd = TimeToWDM(TIME, dom)
    m, n = np.asarray(CELLS.m), np.asarray(CELLS.n)
    X2 = np.stack([X, np.random.default_rng(3).standard_normal(N)])
    np.testing.assert_array_equal(fwd.apply(X2),
                                  wilson.forward_time_many(X2, NF, NT, DT, window=ALT_WINDOW)[:, m, n])
    Wa = wilson.matrix.active_W(NF, NT, DT, window=ALT_WINDOW)[CELLS.packed_positions()]
    idx = [0, 5, CELLS.size // 2, CELLS.size - 1]
    _close(fwd.rows(idx), Wa[idx])
    y = np.random.default_rng(4).standard_normal((2, CELLS.size))
    _close(fwd.adjoint(y), y @ Wa)                                          # W^T y on the cells
    np.testing.assert_array_equal(WDMToTime(dom, TIME).adjoint(X2), fwd.apply(X2) / DT)


def test_fourier_cells_use_the_domain_window():
    cells = CellSet.bands(GRID, (2, 3))
    dom = WDMDomain(cells, window=ALT_WINDOW)
    rec = FourierDomain(N, DT, 0, N // 2 + 1, "rfft")
    got = FourierToWDM(rec, dom).apply(np.fft.rfft(X))
    want = TimeToWDM(TIME, dom).apply(X)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.max(np.abs(got)))
    fft = FourierToWDM(rec, dom, "fft").apply(np.fft.rfft(X))
    assert np.max(np.abs(fft - got)) <= 1e-12 * np.max(np.abs(got))
    assert not np.allclose(got, FourierToWDM(rec, WDMDomain(cells)).apply(np.fft.rfft(X)))


def test_after_a_crop_uses_the_domain_window():
    """``after`` rebuilds the cell matrix from the window samples: on an ALT_WINDOW domain it is
    the composition with the crop, and the cropped record's ALT_WINDOW cells."""
    g = wilson.Grid(8, 16, 0.5)
    dom = WDMDomain(CellSet.bands(g, tuple(range(1, 8))), window=ALT_WINDOW)
    rng = np.random.default_rng(0)
    Z = np.zeros(81, complex)
    Z[8:57] = rng.standard_normal(49) + 1j * rng.standard_normal(49)       # no DC, no Nyquist
    xr = np.fft.irfft(Z, n=160)
    W = FourierToWDM(FourierDomain(g.N, g.dt, 0, 64, "rfft"), dom)        # the band windows
    crop = FourierCrop(FourierDomain(160, g.dt, 0, 81, "rfft"), W.src, 16)
    M = W.after(crop)
    assert M.dst == dom
    got = M.apply(Z)
    _close(got, Compose(W, crop).apply(Z))
    _close(got, TimeToWDM(TimeDomain(g.N, g.dt), dom).apply(xr[16:144]))
    meyer = FourierToWDM(W.src, WDMDomain(dom.cells))
    assert not np.allclose(got, meyer.after(crop).apply(Z))


def test_fourier_adjoint_rows_and_push_use_the_domain_window():
    """On an ALT_WINDOW domain: ``adjoint`` is the real-inner-product adjoint of ``apply``
    (``<z, z'> = Re sum conj(z) z'``, the maps module convention); ``rows`` are
    ``(2 / N) conj(rfft(g_p))`` for the time atoms ``g_p`` of ``active_W`` (so that
    ``Re(rows @ rfft(x)) = W x``); and the push of a circulant time covariance's bin
    covariance ``M' = F Sigma F^H`` is ``W Sigma W^T``."""
    cells = CellSet.bands(GRID, (2, 3))
    dom = WDMDomain(cells, window=ALT_WINDOW)
    rec = FourierDomain(N, DT, 0, N // 2 + 1, "rfft")
    F = FourierToWDM(rec, dom)
    rng = np.random.default_rng(9)
    z = rng.standard_normal(rec.n) + 1j * rng.standard_normal(rec.n)
    w = rng.standard_normal(cells.size)
    assert float(np.dot(F.apply(z), w)) == pytest.approx(float(np.real(np.vdot(z, F.adjoint(w)))),
                                                         rel=1e-12)
    W = testing.dense_W(cells, window=ALT_WINDOW)
    R = (2.0 / N) * np.conj(np.fft.rfft(W, axis=-1))
    idx = [0, 3, cells.size // 2, cells.size - 1]
    _close(F.rows(idx), R[idx])
    _close(F.adjoint(w), w @ np.conj(R))
    S = 1.0 / (1.0 + (4.0 * np.fft.rfftfreq(N, DT)) ** 2) + 0.1
    c = np.fft.irfft(S, n=N)
    Sigma = c[(np.arange(N)[:, None] - np.arange(N)[None, :]) % N]          # circulant, SPD
    Fm = np.fft.rfft(np.eye(N), axis=0)                                     # rfft(x) = Fm x
    Mp = Fm @ Sigma @ Fm.conj().T
    got = F.push(DenseCov(Mp)).dense()
    _close(got, W @ Sigma @ W.T)
    assert not np.allclose(got, FourierToWDM(rec, WDMDomain(cells)).push(DenseCov(Mp)).dense())


def test_maps_refuse_a_window_mismatch():
    meyer, alt = WDMDomain(CELLS), WDMDomain(CELLS.drop_columns(2, 2), window=ALT_WINDOW)
    with pytest.raises(ValueError, match="window"):
        Restrict(meyer, alt)
    with pytest.raises(ValueError):
        Compose(Restrict(WDMDomain(CELLS, window=ALT_WINDOW), alt), TimeToWDM(TIME, meyer))


def test_window_mismatch_errors_name_both_windows():
    meyer, alt = WDMDomain(CELLS), WDMDomain(CELLS.drop_columns(2, 2), window=ALT_WINDOW)
    with pytest.raises(ValueError, match=BOTH):
        Restrict(meyer, alt)
    with pytest.raises(ValueError, match=BOTH):
        Compose(Restrict(WDMDomain(CELLS, window=ALT_WINDOW), alt), TimeToWDM(TIME, meyer))
    same = WDMDomain(CELLS.drop_columns(2, 2), window=ALT_WINDOW)
    assert Restrict(WDMDomain(CELLS, window=ALT_WINDOW), same).dst == same          # one basis


def test_equal_but_distinct_windows_are_one_basis():
    """The window checks compare windows by value: a pickled ALT_WINDOW and ``Meyer()`` are
    other objects than the module singletons but the same basis, so Restrict, Compose and push
    accept them (and compute the same numbers)."""
    alt, cut = WDMDomain(CELLS, window=ALT_WINDOW), CELLS.drop_columns(2, 2)
    pos = CELLS.index_of(cut)
    alt_rt = _roundtrip(alt)
    meyer4 = wilson.Meyer()
    assert alt_rt.window == ALT_WINDOW and alt_rt.window is not ALT_WINDOW
    assert meyer4 == MEYER and meyer4 is not MEYER
    y = np.random.default_rng(5).standard_normal((2, CELLS.size))
    for src, dst in ((alt_rt, WDMDomain(cut, window=ALT_WINDOW)),
                     (WDMDomain(CELLS, window=meyer4), WDMDomain(cut))):
        np.testing.assert_array_equal(Restrict(src, dst).apply(y), y[:, pos])
    C = Compose(Restrict(alt, WDMDomain(cut, window=ALT_WINDOW)), TimeToWDM(TIME, alt_rt))
    np.testing.assert_array_equal(C.apply(X), TimeToWDM(TIME, alt).apply(X)[pos])
    # an evaluable chained push across equal windows: the restricted Meyer congruence
    S = DenseCov(np.diag(1.0 + np.arange(N) / N))
    inner = TimeToWDM(TIME, WDMDomain(CELLS, window=meyer4)).push(S)
    got = Restrict(WDMDomain(CELLS), WDMDomain(cut)).push(inner).dense()
    W = testing.dense_W(cut)
    _close(got, W @ S.S @ W.T)


def test_restrict_checks_the_windows_on_every_construction_route():
    """The window check reads the domains only, so ``pos`` given (``tree_unflatten``,
    ``on``, ``dataclasses.replace``) cannot build a cross-basis Restrict."""
    meyer, cut = WDMDomain(CELLS), CELLS.drop_columns(2, 2)
    alt_cut = WDMDomain(cut, window=ALT_WINDOW)
    R = Restrict(meyer, WDMDomain(cut))
    with pytest.raises(ValueError, match=BOTH):
        Restrict(meyer, alt_cut, pos=np.arange(cut.size))
    with pytest.raises(ValueError, match=BOTH):
        dataclasses.replace(R, dst=alt_cut)
    with pytest.raises(ValueError, match=BOTH):
        dataclasses.replace(R, src=WDMDomain(CELLS, window=ALT_WINDOW))
    np.testing.assert_array_equal(dataclasses.replace(R).pos, R.pos)            # same basis: fine


def test_a_pushed_covariance_lives_in_its_destination_window():
    """A covariance pushed onto WDM cells is in the basis of the map's ``dst.window``: a second
    push through a map of another window is refused, through one of the same window it is the
    restricted covariance."""
    S = DenseCov(np.diag(1.0 + np.arange(N) / N))
    meyer, alt = WDMDomain(CELLS), WDMDomain(CELLS, window=ALT_WINDOW)
    cut = CELLS.drop_columns(2, 2)
    inner = TimeToWDM(TIME, alt).push(S)
    assert inner.window is ALT_WINDOW and TimeToWDM(TIME, meyer).push(S).window is MEYER
    assert Identity(TIME).push(S).window is None
    with pytest.raises(ValueError, match=BOTH):
        Restrict(meyer, WDMDomain(cut)).push(inner)
    pos = CELLS.index_of(cut)
    got = Restrict(alt, WDMDomain(cut, window=ALT_WINDOW)).push(inner).dense()
    _close(got, inner.dense()[np.ix_(pos, pos)])
    W = testing.dense_W(cut, window=ALT_WINDOW)
    _close(got, W @ S.S @ W.T)


# ---------------------------------------------------------------------------
# Variances and oracles
# ---------------------------------------------------------------------------


def test_band_averaged_variance_uses_its_window():
    psd = GenericPSD(lambda f, t, nu: 1.0 + nu[0] * np.asarray(f, float) ** 2 + 0.0 * np.asarray(t, float))
    nu = np.array([3.0])
    full = wilson.noise.evolutionary_band_variance(NF, NT, DT, psd.bind(nu), window=ALT_WINDOW)
    got = BandAveragedVariance(psd, CELLS, window=ALT_WINDOW)(nu)
    np.testing.assert_array_equal(got, full[np.asarray(CELLS.m), np.asarray(CELLS.n)])
    assert not np.array_equal(got, BandAveragedVariance(psd, CELLS)(nu))
    # independent reference: a stationary band average is the exact windowed covariance diagonal
    exact = wilson.matrix.build_C_wdm_active(NF, NT, DT, lambda f: psd(f, 0.0, nu), window=ALT_WINDOW)
    want = np.diag(exact)[CELLS.packed_positions()]
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    with pytest.raises(TypeError, match="Window"):
        BandAveragedVariance(psd, CELLS, window="meyer")
    assert BandAveragedVariance(psd, CELLS).window is MEYER


def test_the_cell_set_consumers_take_a_wdm_domain_and_its_window():
    """Built on a ``WDMDomain`` they use its cells and its window, so a
    non-Meyer domain cannot fall back to Meyer; ``MidpointVariance`` reads the cells only."""
    psd, nu = _stationary_psd()
    cut = CELLS.drop_columns(2, 1)
    dom = WDMDomain(cut, window=ALT_WINDOW)
    bv = BandAveragedVariance(psd, dom)
    assert bv.window is ALT_WINDOW and bv.cells == cut
    np.testing.assert_array_equal(bv(nu), BandAveragedVariance(psd, cut, window=ALT_WINDOW)(nu))
    mv = MidpointVariance(psd, dom)
    assert mv.cells == cut
    np.testing.assert_array_equal(mv(nu), MidpointVariance(psd, cut)(nu))
    # the domain's own window passed again (equal, not identical) is accepted
    assert BandAveragedVariance(psd, dom, window=_roundtrip(ALT_WINDOW)).window is ALT_WINDOW


def test_a_wdm_domain_with_another_explicit_window_is_refused():
    psd, _ = _stationary_psd()
    alt = WDMDomain(CELLS, window=ALT_WINDOW)
    with pytest.raises(ValueError, match=r"^window meyer\(d=4\) is not the WDMDomain's window "
                                         r"wide-smoothstep: pass window=None"):
        BandAveragedVariance(psd, alt, window=MEYER)
    with pytest.raises(ValueError, match=r"window wide-smoothstep is not the WDMDomain's window meyer"):
        BandAveragedVariance(psd, WDMDomain(CELLS), window=ALT_WINDOW)
    with pytest.raises(TypeError, match="Window"):
        BandAveragedVariance(psd, alt, window="meyer")
    assert BandAveragedVariance(psd, CELLS, window=None).window is MEYER     # None on a CellSet


def test_testing_oracles_take_the_window():
    W = testing.dense_W(CELLS, window=ALT_WINDOW)
    np.testing.assert_array_equal(
        W, wilson.matrix.active_W(NF, NT, DT, window=ALT_WINDOW)[CELLS.packed_positions()])
    S = np.diag(1.0 + np.arange(N) / N)                  # not the identity: W I W^T = dt I for any window
    alt = testing.wdm_congruence(S, CELLS, window=ALT_WINDOW)
    np.testing.assert_allclose(alt, W @ S @ W.T, rtol=0, atol=1e-12)
    assert not np.allclose(alt, testing.wdm_congruence(S, CELLS))


def test_eq45_dense_oracle_takes_the_window():
    psd, nu = _stationary_psd()
    got = testing.eq45_dense(CELLS, psd.bind(nu), window=ALT_WINDOW)
    exact = _exact(psd, nu)
    assert np.max(np.abs(got - exact)) < 1e-10 * np.max(np.abs(exact))
    assert not np.allclose(got, testing.eq45_dense(CELLS, psd.bind(nu)))


# ---------------------------------------------------------------------------
# JAX: the window is static metadata (pytree meta of the maps and variances)
# ---------------------------------------------------------------------------


@pytest.mark.jax
def test_the_window_is_static_under_jit():
    """A map on an ALT_WINDOW domain keeps its window through ``on``, ``tree_unflatten`` and
    ``jax.jit``; one jitted function given a MEYER and an ALT_WINDOW map retraces for each (the
    window is part of the static key) and matches NumPy for both."""
    jax, JB = jax_or_skip()
    import jax.numpy as jnp

    apply = jax.jit(lambda mm, vv: mm.apply(vv))
    adjoint = jax.jit(lambda mm, vv: mm.adjoint(vv))
    for dom in (WDMDomain(CELLS), WDMDomain(CELLS, window=ALT_WINDOW)):
        fwd = TimeToWDM(TIME, dom)
        on = fwd.on(JB)
        leaves, tree = jax.tree_util.tree_flatten(on)
        assert on.dst.window is dom.window
        assert jax.tree_util.tree_unflatten(tree, leaves).dst.window is dom.window
        got = apply(on, JB.asarray(X))
        assert isinstance(got, jax.Array)
        _close(np.asarray(got), fwd.apply(X))
        y = np.asarray(got)
        _close(np.asarray(adjoint(on, jnp.asarray(y))), fwd.adjoint(y))
        inv = WDMToTime(dom, TIME)
        _close(np.asarray(apply(inv.on(JB), jnp.asarray(y))), inv.apply(y))
    fdom = WDMDomain(CellSet.bands(GRID, (2, 3)), window=ALT_WINDOW)
    rec = FourierDomain(N, DT, 0, N // 2 + 1, "rfft")
    z = np.fft.rfft(X)
    for impl in ("dense", "fft"):
        F = FourierToWDM(rec, fdom, impl)
        _close(np.asarray(apply(F.on(JB), jnp.asarray(z))), F.apply(z))
    psd, nu = _stationary_psd()
    v = BandAveragedVariance(psd, CELLS, window=ALT_WINDOW).on(JB)
    leaves, tree = jax.tree_util.tree_flatten(v)
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves).window is ALT_WINDOW
    mv = MidpointVariance(psd, CELLS).on(JB)
    leaves, tree = jax.tree_util.tree_flatten(mv)
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves).cells == CELLS
    np.testing.assert_array_equal(mv(nu), MidpointVariance(psd, CELLS)(nu))
