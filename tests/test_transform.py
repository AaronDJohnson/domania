"""domainia.transform: data arrays between time, frequency and WDM (domainia 3.0 spec,
sections 2 and 6.2)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from domainia import transform, wilson
from domainia.testing import ALT_WINDOW

REFERENCE = Path(__file__).resolve().parent / "data" / "translate_reference.npz"
KINDS = ("time", "frequency", "wdm")
N, DT, G = 128, 0.25, (8, 16)
#: Time data with leading axes (K, B) = (3, 2).
X = np.random.default_rng(20261113).standard_normal((3, 2, N))


def _kw(source, target, sg=G, tg=G, **kw):
    """transform's keywords for the 8x16 record: each WDM grid only on its WDM side."""
    return {"N": N, "dt": DT, "source_wdm": sg if source == "wdm" else None,
            "target_wdm": tg if target == "wdm" else None, **kw}


def _on(kind, **kw):
    """``X`` on ``kind``."""
    return transform(X, "time", kind, **_kw("time", kind, **kw))


def _close(got, want, rel):
    assert got.shape == want.shape and got.dtype == want.dtype
    assert np.max(np.abs(got - want)) <= rel * np.max(np.abs(want))


def test_agrees_with_the_recorded_translate_transform():
    """The six pairs on four WDM grids, time <-> frequency at odd N, one channel with no
    leading axis, and WDM -> WDM across two grids: each output within 1e-12 of its peak
    (measured: 0.0) and exactly canonical.  At odd N the last bin is not a Nyquist bin, and
    it keeps its imaginary part."""
    with np.load(REFERENCE) as z:
        for i, ((s, t), (n, dt, *g)) in enumerate(zip(z["routes"], z["grids"])):
            n, sg, tg = int(n), (int(g[0]), int(g[1])), (int(g[2]), int(g[3]))
            got = transform(z[f"x{i}"], s, t, N=n, dt=dt, source_wdm=sg if s == "wdm" else None,
                            target_wdm=tg if t == "wdm" else None)
            _close(got, z[f"y{i}"], 1e-12)
            if t == "frequency":
                assert not np.any(got[..., 0].imag), i
                assert np.all(got[..., -1].imag) if n % 2 else not np.any(got[..., -1].imag), i
            if t == "wdm":
                assert not np.any(got[..., ~wilson.grid.active_mask(*tg)]), i


def test_round_trips_leading_axes_and_parseval():
    """Every pair maps X's three layouts onto each other and back, with a non-default window;
    each row is the kernel's; the energy scales by dt on WDM and dt^2 N on frequency, also at
    odd N."""
    on = {k: _on(k, window=ALT_WINDOW) for k in KINDS}
    assert np.array_equal(on["time"], X) and on["wdm"].shape == (3, 2, 9, 16)
    _close(on["frequency"], DT * np.fft.rfft(X), 1e-15)
    _close(on["wdm"][2, 1], wilson.forward_time(X[2, 1], *G, DT, window=ALT_WINDOW), 1e-15)
    for s, t in itertools.permutations(KINDS, 2):
        y = transform(on[s], s, t, **_kw(s, t, window=ALT_WINDOW))
        _close(y, on[t], 1e-12)
        _close(transform(y, t, s, **_kw(t, s, window=ALT_WINDOW)), on[s], 1e-12)
    energy = np.sum(X ** 2, axis=-1)
    two = np.r_[1.0, np.full(N // 2 - 1, 2.0), 1.0]
    np.testing.assert_allclose(np.sum(on["wdm"] ** 2, axis=(-2, -1)), DT * energy, rtol=1e-12)
    np.testing.assert_allclose(np.sum(two * np.abs(on["frequency"]) ** 2, axis=-1),
                               DT ** 2 * N * energy, rtol=1e-12)
    odd = transform(X[..., :15], "time", "frequency", N=15, dt=DT)             # no Nyquist bin
    np.testing.assert_allclose(np.sum(two[:8] * np.abs(odd) ** 2, axis=-1),
                               DT ** 2 * 15 * np.sum(X[..., :15] ** 2, axis=-1), rtol=1e-12)


def test_a_residue_up_to_1e_10_of_the_peak_is_accepted_and_the_copy_is_canonical():
    """1e-12 of the slice's peak at DC, at Nyquist or in an inactive cell passes, and
    ``source == target`` returns a fresh canonical copy; 1e-6 is refused, naming the place."""
    for kind, at, name in (("frequency", (0,), r"bin 0 \(DC\)"), ("frequency", (64,), r"bin 64 \(Nyquist\)"),
                           ("wdm", (0, 1), r"cell \(m=0, n=1\)"), ("wdm", (8, 15), r"cell \(m=8, n=15\)")):
        x = _on(kind)
        unit = 1j if kind == "frequency" else 1.0
        peak = np.max(np.abs(x[2, 1]))
        r = x.copy()
        r[(2, 1, *at)] += 1e-12 * peak * unit
        same = transform(r, kind, kind, **_kw(kind, kind))
        assert np.array_equal(same, x) and not np.shares_memory(same, r)
        assert np.array_equal(transform(r, kind, "time", **_kw(kind, "time")),
                              transform(x, kind, "time", **_kw(kind, "time")))
        r[(2, 1, *at)] += 1e-6 * peak * unit
        with pytest.raises(ValueError, match=rf"^x\[2, 1\]: .*{name} is 1e-06 of the slice's peak"):
            transform(r, kind, "time", **_kw(kind, "time"))
    real = np.arange(65)                                                         # integers too
    assert np.array_equal(transform(real, "frequency", "frequency", **_kw("frequency", "frequency")),
                          real.astype(np.complex128))


def test_wdm_to_wdm_across_two_grids_is_the_chain_through_frequency():
    w = _on("wdm")
    regrid = _kw("frequency", "wdm", tg=(16, 8))
    chain = transform(transform(w, "wdm", "frequency", **_kw("wdm", "frequency")), "frequency", "wdm",
                      **regrid)
    _close(transform(w, "wdm", "wdm", **_kw("wdm", "wdm", tg=(16, 8))), chain, 1e-12)
    _close(chain, _on("wdm", tg=(16, 8)), 1e-12)


def test_every_refusal():
    t, w = X[0, 0], _on("wdm")[0, 0]
    zero_but_dc = np.zeros(65, complex)
    zero_but_dc[0] = 1e-300j
    cases = [
        (t, "fourier", "time", {}, ValueError, "source must be one of"),
        (t, "time", 3, {}, TypeError, "target must be a string"),
        (t, "time", "wdm", {"target_wdm": None}, ValueError, r"needs target_wdm=\(Nf, Nt\)"),
        (t, "time", "frequency", {"target_wdm": G}, ValueError, "target_wdm is only for"),
        (t, "time", "wdm", {"target_wdm": (8, 15)}, ValueError, "Nt must be even"),
        (t, "time", "wdm", {"target_wdm": (4, 16)}, ValueError, "N=128"),
        (t, "time", "wdm", {"N": 128.0}, TypeError, "N must be an integer"),
        (t, "time", "wdm", {"dt": 0.0}, ValueError, "dt must be"),
        (t[:-1], "time", "wdm", {}, ValueError, r"shape \(127,\)"),
        (w.T, "wdm", "time", {}, ValueError, r"shape \(16, 9\)"),
        (np.where(np.arange(N) == 3, np.nan, t), "time", "wdm", {}, ValueError, "not finite"),
        (t + 0j, "time", "frequency", {}, ValueError, "complex"),
        (w + 0j, "wdm", "time", {}, ValueError, "complex"),
        (t > 0, "time", "wdm", {}, TypeError, "numeric"),
        (np.array(["x"] * N), "time", "wdm", {}, TypeError, "numeric"),
        (t.astype(object), "time", "wdm", {}, TypeError, "numeric"),
        (np.ma.masked_array(t), "time", "wdm", {}, TypeError, "masked array"),
        (zero_but_dc, "frequency", "time", {}, ValueError, r"bin 0 \(DC\) is 1 of"),
    ]
    for x, s, d, kw, err, match in cases:
        with pytest.raises(err, match=match):
            transform(x, s, d, **{**_kw(s, d), **kw})
