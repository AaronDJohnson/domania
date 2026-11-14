"""Replay cases for the Meyer baseline (domainia 2.0 spec, section 5; 3.0 spec, section 6.1).

Every public function that consumes the window, on fixed grids and seeded inputs.
``wilson_cases`` covers the kernels and ``domainia_cases`` the cases recorded on the domainia
layer.  Each case is ``(name, fn)``: ``fn(kw)`` calls the function with the extra keywords
``kw`` (``{}``, or ``{"window": ...}``).

``data/meyer_baseline.npz`` was recorded once against the untouched code of commit
d991ac4 (the old ``wdm`` layout) by this module's own history: commit 24c4e55 holds
the recorder (``meyer_cases.py --write``), and its ``src/`` and ``wilson/src/`` are
d991ac4's byte for byte.  This version only replays the baseline, so refactored code
can never re-record it.  Re-recording means checking out 24c4e55 and running its
``wilson/tests/meyer_cases.py --write`` there.

The gate is ``tests/wilson/test_meyer_baseline.py``: it recomputes every case through
``domainia.wilson`` and compares with :func:`mismatches`, to :data:`RTOL` of each array's
scale.  The :data:`DROPPED` families, of code that domainia 3.0 deletes, are set aside.  The
cases of deleted classes keep their recorded names and are computed by the kept kernels
(:func:`domainia_cases`).  ``python tests/wilson/meyer_cases.py --check`` is the same replay
as a script, with the same comparator.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

GRIDS = ((8, 16), (16, 32), (32, 8))
DTS = (1.0, 0.25)
#: Dense O(N^2) outputs are recorded on this grid only (keeps the npz small).
DENSE_GRID = (8, 16)
BASELINE = Path(__file__).resolve().parent / "data" / "meyer_baseline.npz"
#: Case-name prefixes whose kernels take ``window=``.  The single source for
#: ``test_meyer_baseline.py`` and ``test_window_threading.py``.
THREADED = ("transform.", "matrix.", "stationary.", "noise.", "evolutionary.")
#: Baseline key prefixes of code that domainia 3.0 deletes: maps (54 keys), inner_product (24),
#: the band-limited entry points (18) and the active-cell matrices (4), which the folded layout
#: made ``build_W`` and ``build_C_wdm``.  They are not replayed.
DROPPED = ("domainia::TimeToWDM.", "domainia::WDMToTime.", "domainia::FourierToWDM.",
           "wilson::inner_product.", "wilson::transform.band_atoms::",
           "wilson::transform.forward_bands::", "wilson::transform.inverse_bands::",
           "wilson::matrix.active_W::", "wilson::matrix.build_C_wdm_active::")
#: The comparison: ``max |got - want| <= RTOL * s`` (see :func:`mismatches`).
RTOL = 1e-12
#: Families recorded padded ``(..., Nf+1, Nt)`` whose kernels now return the folded layout.
FOLDED = ("wilson::transform.forward", "wilson::transform.forward_many",
          "wilson::transform.forward_time", "wilson::transform.forward_time_many")
#: Dense families recorded with padded ``(Nf+1) Nt`` rows (and columns, ``build_C_wdm``), now
#: in folded flat order.
FOLDED_ROWS = ("wilson::matrix.build_W", "wilson::matrix.build_C_wdm")

Case = tuple[str, Callable[[dict], Any]]


def _family(key: str) -> str:
    """``"layer::case"``: the function that recorded ``key``."""
    return "::".join(key.split("::")[:2])


def load_baseline() -> dict[str, np.ndarray]:
    """The recorded baseline as ``{key: array}``.  The families whose kernels now return the
    folded WDM layout are converted here, never in :func:`mismatches` (folded WDM spec,
    section 5): the :data:`FOLDED` ones are gathered into the folded slots, and the
    :data:`FOLDED_ROWS` rows (and the covariance's columns) into folded flat order, without
    ``wilson.grid.fold``'s check of the inactive cells."""
    from domainia.wilson.grid import cells

    with np.load(BASELINE) as z:
        out = {k: z[k] for k in z.files}
    for k, v in out.items():
        fam = _family(k)
        if fam in FOLDED + FOLDED_ROWS:
            Nf, Nt = map(int, k.split("::")[2].split("x"))
            m, n = cells(Nf, Nt)
            p = (m * Nt + n).ravel()                # each folded slot's padded flat index
            out[k] = (v[..., m, n] if fam in FOLDED else v[p] if fam == "wilson::matrix.build_W"
                      else v[np.ix_(p, p)])
    return out


def mismatches(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> list[str]:
    """The keys of ``want`` that ``got`` does not reproduce.  ``want``'s :data:`DROPPED` keys are
    set aside; the other keys must be exactly ``got``'s.

    dtype and shape must match and integers must be equal.  Floats and complex numbers need
    ``max |got - want| <= RTOL * s``: ``s`` is ``max |want|`` over the array, or, for an all-zero
    array, over every array of the same function in ``want``.  A NaN fails.
    """
    want = {k: v for k, v in want.items() if not k.startswith(DROPPED)}
    if sorted(got) != sorted(want):
        raise AssertionError(f"baseline keys differ: {sorted(set(got) ^ set(want))[:10]}")
    scale: dict[str, float] = {}
    for k, w in want.items():
        scale[_family(k)] = max(scale.get(_family(k), 0.0), float(np.max(np.abs(w), initial=0.0)))
    bad = []
    for k, w in want.items():
        g = got[k]
        if g.dtype != w.dtype or g.shape != w.shape:
            bad.append(k)
        elif w.dtype.kind not in "fc":
            if not np.array_equal(g, w):
                bad.append(k)
        else:
            s = float(np.max(np.abs(w), initial=0.0)) or scale[_family(k)]
            if not np.max(np.abs(g - w), initial=0.0) <= RTOL * s:       # a NaN fails
                bad.append(k)
    return bad


def psd(f: Any) -> np.ndarray:
    """A strictly positive one-sided stationary PSD on any rfft grid."""
    f = np.asarray(f, dtype=float)
    return 1.0 / (1.0 + (4.0 * f) ** 2) + 0.1


def psd_ft(T: float, depth: float = 0.5) -> Callable[[Any, Any], np.ndarray]:
    """A strictly positive evolutionary PSD ``Sn(f, t)`` (elementwise, broadcasting)."""

    def Sn(f: Any, t: Any) -> np.ndarray:
        return psd(f) * (1.0 + depth * np.sin(2.0 * np.pi * np.asarray(t, dtype=float) / T))

    return Sn


def _seed(Nf: int, Nt: int, dt: float) -> int:
    return 1000 * Nf + Nt + int(round(100 * dt))


def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
    """The kernel window consumers of ``wl`` (``domainia.wilson``), and
    ``testing.empirical_variance`` under its recorded name."""
    from domainia import testing

    N = Nf * Nt
    T = N * dt
    rng = np.random.default_rng(_seed(Nf, Nt, dt))
    x = rng.standard_normal(N)
    X = rng.standard_normal((3, N))
    xr = np.fft.rfft(x)
    Xr = np.fft.rfft(X, axis=-1)
    S = psd(np.fft.rfftfreq(N, dt))
    Sn = psd_ft(T)
    chi = np.hanning(N)
    # Fixed WDM-domain inputs, computed once with the default (Meyer) window.
    w = wl.forward(xr, Nf, Nt, dt)
    Wm = wl.forward(Xr, Nf, Nt, dt)
    mask = wl.grid.active_mask(Nf, Nt)
    Tr, M, St, No, E = wl.transform, wl.matrix, wl.stationary, wl.noise, wl.evolutionary
    K0 = St.build_K_full_active(Nf, Nt, dt, psd)[0]
    B0 = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd)
    cases: list[Case] = [
        ("transform.forward", lambda kw: Tr.forward(xr, Nf, Nt, dt, **kw)),
        # The stacks (Xr, X, Wm) keep the recorded names of the batched entry points that
        # the kernels' leading axes replaced.
        ("transform.forward_many", lambda kw: Tr.forward(Xr, Nf, Nt, dt, **kw)),
        ("transform.forward_time", lambda kw: Tr.forward_time(x, Nf, Nt, dt, **kw)),
        ("transform.forward_time_many", lambda kw: Tr.forward_time(X, Nf, Nt, dt, **kw)),
        ("transform.inverse", lambda kw: Tr.inverse(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_many", lambda kw: Tr.inverse(Wm, Nf, Nt, dt, **kw)),
        ("transform.inverse_time", lambda kw: Tr.inverse_time(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_time_many", lambda kw: Tr.inverse_time(Wm, Nf, Nt, dt, **kw)),
        ("transform.base_atom_spectra", lambda kw: Tr.base_atom_spectra(Nf, Nt, dt, **kw)),
        ("transform.atom_spectrum", lambda kw: [Tr.atom_spectrum(m, n, Nf, Nt, dt, **kw)
                                                for m, n in ((1, 3), (0, 2), (Nf, 2))]),
        ("stationary.build_lambda_blocks_full_active",
         lambda kw: St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, **kw)),
        ("stationary.build_K_full_active", lambda kw: St.build_K_full_active(Nf, Nt, dt, psd, **kw)),
        ("stationary.psd_from_K", lambda kw: St.psd_from_K(K0, Nf, Nt, dt, **kw)),
        ("stationary.apply_lambda_from_psd",
         lambda kw: St.apply_lambda_from_psd(wl.grid.unfold(w, Nf, Nt), Nf, Nt, dt, S, **kw)),
        ("stationary.psd_from_lambda_blocks", lambda kw: St.psd_from_lambda_blocks(*B0, Nf, Nt, dt, **kw)),
        ("noise.band_averaged_variance", lambda kw: No.band_averaged_variance(Nf, Nt, dt, psd, **kw)),
        ("noise.evolutionary_band_variance",
         lambda kw: No.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)),
        ("noise.empirical_variance",
         lambda kw: testing.empirical_variance(Nf, Nt, dt, psd, 3, seed=11, **kw)),
        # w2 is a length-N time-domain weight (rho_p = sum_t w2(t) g_p(t)^2 / dt).
        ("evolutionary.energy_fraction", lambda kw: E.energy_fraction(x ** 2, Nf, Nt, dt, **kw)),
        ("evolutionary.apply_time_mask",
         lambda kw: E.apply_time_mask(wl.grid.unfold(Wm, Nf, Nt)[:, mask], chi, Nf, Nt, dt, **kw)),
        # nan (inactive cells) -> -1, as recorded; the fractions lie in [0, 1].
        ("evolutionary.wrap_fractions",
         lambda kw: np.nan_to_num(E.wrap_fractions(Nf, Nt, dt, **kw), nan=-1.0)),
        ("evolutionary.wrap_cut_columns", lambda kw: np.asarray(E.wrap_cut_columns(Nf, Nt, dt, **kw))),
        ("evolutionary.eq45_plan", lambda kw: E.eq45_plan(Nf, Nt, dt, **kw).base),
    ]
    if (Nf, Nt) == DENSE_GRID:
        cases += [
            ("matrix.build_W", lambda kw: M.build_W(Nf, Nt, dt, **kw)),
            ("matrix.build_C_wdm", lambda kw: M.build_C_wdm(Nf, Nt, dt, psd, **kw)),
            ("evolutionary.exact_covariance", lambda kw: E.exact_covariance(Nf, Nt, dt, Sn, **kw)),
            ("evolutionary.eq45_direct", lambda kw: E.eq45_direct(E.eq45_plan(Nf, Nt, dt, **kw), Sn)),
        ]
    return cases


def domainia_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
    """The cases recorded on the domainia layer, under their recorded names.  The classes
    that recorded them are computed by the kernels they wrapped: ``EvolutionaryCovariance``
    by ``eq45_plan`` (``P = 24``) + ``eq45_coeffs_interp`` + ``eq45_blocks_cheb`` and by
    ``eq45_direct``, ``BandAveragedVariance`` by ``noise.evolutionary_band_variance`` on the
    active cells.  ``kw`` goes to every window-taking call."""
    from domainia import testing

    N = Nf * Nt
    rng = np.random.default_rng(7 + _seed(Nf, Nt, dt))
    rng.standard_normal(5 * N + 2 * Nt)        # the recorded maps cases' inputs: Sigma unchanged
    Sn = psd_ft(N * dt, depth=0.4)
    E = wl.evolutionary
    mask = wl.grid.active_mask(Nf, Nt)
    m, n = np.nonzero(mask)

    def interp(kw: dict) -> Any:
        plan = E.eq45_plan(Nf, Nt, dt, **kw)
        return E.eq45_blocks_cheb(plan, E.eq45_coeffs_interp(Sn, plan.f, plan.t_max, plan.P,
                                                             k=plan.k_used))

    cases: list[Case] = [
        ("BandAveragedVariance", lambda kw: wl.noise.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)[m, n]),
        ("EvolutionaryCovariance.interp", interp),
    ]
    if (Nf, Nt) == DENSE_GRID:
        # A seeded non-identity SPD Sigma: with Sigma = I the congruence is dt I for any window.
        B = rng.standard_normal((N, N))
        Sigma = B @ B.T / N + np.eye(N)
        cases += [
            ("EvolutionaryCovariance.direct",
             lambda kw: E.eq45_direct(E.eq45_plan(Nf, Nt, dt, P=2, **kw), Sn)),
            ("testing.dense_W", lambda kw: testing.dense_W(Nf, Nt, dt, mask, **kw)),
            ("testing.wdm_congruence", lambda kw: testing.wdm_congruence(Sigma, Nf, Nt, dt, mask, **kw)),
            ("testing.eq45_dense", lambda kw: testing.eq45_dense(Nf, Nt, dt, mask, Sn, **kw)),
        ]
    return cases


def flatten(key: str, value: Any) -> dict[str, np.ndarray]:
    """``{key: array}`` for an array, scalar, list/tuple or dict output (recursively)."""
    if isinstance(value, dict):
        out: dict[str, np.ndarray] = {}
        for k in sorted(value, key=repr):
            out.update(flatten(f"{key}[{k!r}]", value[k]))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for i, v in enumerate(value):
            out.update(flatten(f"{key}[{i}]", v))
        return out
    return {key: np.asarray(value)}


def compute(wl: Any, wilson_kw: dict, domainia_kw: dict, *,
            only: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Every case on every grid and ``dt``.  ``only`` keeps the cases whose name starts
    with one of the prefixes (``"transform."``, ...)."""
    out: dict[str, np.ndarray] = {}
    for Nf, Nt in GRIDS:
        for dt in DTS:
            tag = f"{Nf}x{Nt}::dt={dt!r}"
            for layer, cases, kw in (("wilson", wilson_cases(wl, Nf, Nt, dt), wilson_kw),
                                     ("domainia", domainia_cases(wl, Nf, Nt, dt), domainia_kw)):
                for name, case in cases:
                    if only is None or name.startswith(only):
                        out.update(flatten(f"{layer}::{name}::{tag}", case(kw)))
    return out


def main() -> None:
    """``--check``: replay the baseline through the current ``domainia.wilson``, with
    :func:`mismatches` (the gate is ``tests/wilson/test_meyer_baseline.py``)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", required=True,
                   help="recompute and compare, to RTOL of each array's scale")
    p.parse_args()
    src = str(Path(__file__).resolve().parents[2] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from domainia import wilson

    out = compute(wilson, {}, {})
    stored = load_baseline()
    try:
        bad = mismatches(out, stored)
    except AssertionError as exc:
        raise SystemExit(str(exc)) from None
    if bad:
        raise SystemExit(f"{len(bad)} arrays differ from the baseline: {bad[:10]}")
    print(f"baseline reproduces: {len(out)} arrays ({len(stored) - len(out)} DROPPED keys set aside)")


if __name__ == "__main__":
    main()
