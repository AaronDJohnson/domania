"""Bit-identity cases for the Meyer window (domainia 2.0 spec, section 5).

Every public function that consumes the window, on fixed grids and seeded inputs.
``wilson_cases`` covers the kernels and ``domainia_cases`` the domainia layer.  Each
case is ``(name, fn)``: ``fn(kw)`` calls the function with the extra keywords ``kw``
(``{}`` on the old code, ``{"window": ...}`` after the refactor).

``data/meyer_baseline.npz`` was recorded once against the untouched code of commit
d991ac4 (the old ``wdm`` layout) by this module's own history: commit 24c4e55 holds
the recorder (``meyer_cases.py --write``), and its ``src/`` and ``wilson/src/`` are
d991ac4's byte for byte.  This version only replays the baseline, so refactored code
can never re-record it.  Re-recording means checking out 24c4e55 and running its
``wilson/tests/meyer_cases.py --write`` there.

The gate is ``tests/wilson/test_meyer_baseline.py``: it recomputes every case through
``domainia.wilson`` and requires ``np.array_equal`` (same dtype) on every array.
``python tests/wilson/meyer_cases.py --check`` is the same replay as a script.  The
arrays are bit-exact only in the recording environment (the ``wdmrel`` env's NumPy
pocketfft and BLAS on the machine that recorded them).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

GRIDS = ((8, 16), (16, 32), (32, 8))
DTS = (1.0, 0.25)
#: Dense O(N^2) outputs are recorded on this grid only (keeps the npz small).
DENSE_GRID = (8, 16)
BASELINE = Path(__file__).resolve().parent / "data" / "meyer_baseline.npz"

Case = tuple[str, Callable[[dict], Any]]


def psd(f: Any) -> np.ndarray:
    """A strictly positive one-sided stationary PSD on any rfft grid."""
    f = np.asarray(f, dtype=float)
    return 1.0 / (1.0 + (4.0 * f) ** 2) + 0.1


def psd_ft(T: float) -> Callable[[Any, Any], np.ndarray]:
    """A strictly positive evolutionary PSD ``Sn(f, t)`` (elementwise, broadcasting)."""

    def Sn(f: Any, t: Any) -> np.ndarray:
        return psd(f) * (1.0 + 0.5 * np.sin(2.0 * np.pi * np.asarray(t, dtype=float) / T))

    return Sn


def _seed(Nf: int, Nt: int, dt: float) -> int:
    return 1000 * Nf + Nt + int(round(100 * dt))


def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
    """The kernel window consumers of ``wl`` (old ``wdm`` or ``domainia.wilson``)."""
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
    bands = np.array([1, 2])
    # Fixed WDM-domain inputs, computed once with the default (Meyer) window.
    w = wl.forward_time(x, Nf, Nt, dt)
    Wm = wl.forward_time_many(X, Nf, Nt, dt)
    wb = wl.transform.forward_bands(xr, 0, bands, Nf, Nt, dt)
    mask = wl.active_mask(Nf, Nt)
    Tr, M, St, No, E, I = (wl.transform, wl.matrix, wl.stationary, wl.noise,
                           wl.evolutionary, wl.inner_product)
    K0 = St.build_K_full_active(Nf, Nt, dt, psd)[0]
    B0 = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd)
    cases: list[Case] = [
        ("transform.forward", lambda kw: Tr.forward(xr, Nf, Nt, dt, **kw)),
        ("transform.forward_many", lambda kw: Tr.forward_many(Xr, Nf, Nt, dt, **kw)),
        ("transform.forward_time", lambda kw: Tr.forward_time(x, Nf, Nt, dt, **kw)),
        ("transform.forward_time_many", lambda kw: Tr.forward_time_many(X, Nf, Nt, dt, **kw)),
        ("transform.inverse", lambda kw: Tr.inverse(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_many", lambda kw: Tr.inverse_many(Wm, Nf, Nt, dt, **kw)),
        ("transform.inverse_time", lambda kw: Tr.inverse_time(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_time_many", lambda kw: Tr.inverse_time_many(Wm, Nf, Nt, dt, **kw)),
        ("transform.base_atom_spectra", lambda kw: Tr.base_atom_spectra(Nf, Nt, dt, **kw)),
        ("transform.atom_spectrum", lambda kw: [Tr.atom_spectrum(m, n, Nf, Nt, dt, **kw)
                                                for m, n in ((1, 3), (0, 2), (Nf, 2))]),
        ("transform.band_atoms", lambda kw: Tr.band_atoms(bands, Nf, Nt, dt, **kw)),
        ("transform.forward_bands", lambda kw: Tr.forward_bands(xr, 0, bands, Nf, Nt, dt, **kw)),
        ("transform.inverse_bands", lambda kw: Tr.inverse_bands(wb, 0, N // 2 + 1, bands, Nf, Nt, dt,
                                                                **kw)),
        ("stationary.build_lambda_blocks_full_active",
         lambda kw: St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, **kw)),
        ("stationary.build_K_full_active", lambda kw: St.build_K_full_active(Nf, Nt, dt, psd, **kw)),
        ("stationary.psd_from_K", lambda kw: St.psd_from_K(K0, Nf, Nt, dt, **kw)),
        ("stationary.apply_lambda_from_psd", lambda kw: St.apply_lambda_from_psd(w, Nf, Nt, dt, S, **kw)),
        ("stationary.psd_from_lambda_blocks", lambda kw: St.psd_from_lambda_blocks(*B0, Nf, Nt, dt, **kw)),
        ("noise.band_averaged_variance", lambda kw: No.band_averaged_variance(Nf, Nt, dt, psd, **kw)),
        ("noise.evolutionary_band_variance",
         lambda kw: No.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)),
        ("noise.empirical_variance", lambda kw: No.empirical_variance(Nf, Nt, dt, psd, 3, seed=11, **kw)),
        # w2 is a length-N time-domain weight (rho_p = sum_t w2(t) g_p(t)^2 / dt).
        ("evolutionary.energy_fraction", lambda kw: E.energy_fraction(x ** 2, Nf, Nt, dt, **kw)),
        ("evolutionary.apply_time_mask", lambda kw: E.apply_time_mask(Wm[:, mask], chi, Nf, Nt, dt, **kw)),
        # nan (inactive cells) -> -1: np.array_equal treats nan != nan; the fractions lie in [0, 1].
        ("evolutionary.wrap_fractions",
         lambda kw: np.nan_to_num(E.wrap_fractions(Nf, Nt, dt, **kw), nan=-1.0)),
        ("evolutionary.wrap_cut_columns", lambda kw: np.asarray(E.wrap_cut_columns(Nf, Nt, dt, **kw))),
        ("evolutionary.eq45_plan", lambda kw: E.eq45_plan(Nf, Nt, dt, **kw).base),
        ("inner_product.gaussian", lambda kw: I.gaussian(w, Wm[0], Nf, Nt, dt, S, **kw)),
        ("inner_product.gaussian_from_func", lambda kw: I.gaussian_from_func(w, Wm[0], Nf, Nt, dt, psd, **kw)),
        ("inner_product.loglike_gaussian", lambda kw: I.loglike_gaussian(w, Wm[0], Nf, Nt, dt, S, **kw)),
        ("inner_product.loglike_gaussian_many",
         lambda kw: I.loglike_gaussian_many(Wm, Wm[::-1], Nf, Nt, dt, S, **kw)),
    ]
    if (Nf, Nt) == DENSE_GRID:
        cases += [
            ("matrix.build_W", lambda kw: M.build_W(Nf, Nt, dt, **kw)),
            ("matrix.active_W", lambda kw: M.active_W(Nf, Nt, dt, **kw)),
            ("matrix.build_C_wdm", lambda kw: M.build_C_wdm(Nf, Nt, dt, psd, **kw)),
            ("matrix.build_C_wdm_active", lambda kw: M.build_C_wdm_active(Nf, Nt, dt, psd, **kw)),
            ("evolutionary.exact_covariance", lambda kw: E.exact_covariance(Nf, Nt, dt, Sn, **kw)),
            ("evolutionary.eq45_direct", lambda kw: E.eq45_direct(E.eq45_plan(Nf, Nt, dt, **kw), Sn)),
        ]
    return cases


def domainia_cases(dm: Any, wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
    """The domainia-layer window consumers.  ``dm`` has the attributes ``domains``, ``maps``,
    ``spectra``, ``variance``, ``covariance`` and ``testing`` (the domainia modules).
    ``kw`` goes to ``WDMDomain`` and to the cell-based constructors."""
    g = wl.Grid(Nf, Nt, dt)
    N = g.N
    T = N * dt
    rng = np.random.default_rng(7 + _seed(Nf, Nt, dt))
    x = rng.standard_normal(N)
    X = rng.standard_normal((2, N))
    D = dm.domains
    cells = D.CellSet.all_active(g)
    band_cells = D.CellSet.bands(g, (1, 2))
    time = D.TimeDomain(N, dt)
    four = D.FourierDomain(N, dt, 0, N // 2 + 1, "rfft")
    y = rng.standard_normal(cells.size)
    xt = rng.standard_normal(N)                    # time-domain input of WDMToTime.adjoint
    wf = rng.standard_normal(band_cells.size)      # cell input of FourierToWDM.adjoint
    gen = dm.spectra.GenericPSD(
        lambda f, t, nu: psd(f) * (1.0 + nu[0] * np.sin(2.0 * np.pi * np.asarray(t, dtype=float) / T)))
    nu = np.array([0.4])
    Cov = dm.covariance.EvolutionaryCovariance

    def W(c: Any, kw: dict) -> Any:
        return D.WDMDomain(c, **kw)

    cases: list[Case] = [
        ("TimeToWDM.apply", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).apply(x)),
        ("TimeToWDM.apply_many", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).apply(X)),
        ("TimeToWDM.adjoint", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).adjoint(y)),
        ("TimeToWDM.rows", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).rows([0, 5, cells.size - 1])),
        ("WDMToTime.apply", lambda kw: dm.maps.WDMToTime(W(cells, kw), time).apply(y)),
        ("WDMToTime.adjoint", lambda kw: dm.maps.WDMToTime(W(cells, kw), time).adjoint(xt)),
        ("FourierToWDM.dense", lambda kw: dm.maps.FourierToWDM(four, W(band_cells, kw)).apply(np.fft.rfft(x))),
        ("FourierToWDM.fft", lambda kw: dm.maps.FourierToWDM(four, W(band_cells, kw), "fft").apply(np.fft.rfft(x))),
        ("FourierToWDM.adjoint", lambda kw: dm.maps.FourierToWDM(four, W(band_cells, kw)).adjoint(wf)),
        ("BandAveragedVariance", lambda kw: dm.variance.BandAveragedVariance(gen, cells, **kw)(nu)),
        ("EvolutionaryCovariance.interp", lambda kw: Cov(gen, cells, builder="interp", **kw).blocks(nu)),
    ]
    if (Nf, Nt) == DENSE_GRID:
        # A seeded non-identity SPD Sigma: with Sigma = I the congruence is dt I for any window.
        B = rng.standard_normal((N, N))
        Sigma = B @ B.T / N + np.eye(N)
        cases += [
            ("EvolutionaryCovariance.direct",
             lambda kw: Cov(gen, cells, builder="direct", assembly="dense", **kw).dense(nu)),
            ("testing.dense_W", lambda kw: dm.testing.dense_W(cells, **kw)),
            ("testing.wdm_congruence", lambda kw: dm.testing.wdm_congruence(Sigma, cells, **kw)),
            ("testing.eq45_dense", lambda kw: dm.testing.eq45_dense(cells, gen.bind(nu), **kw)),
        ]
    return cases


def flatten(key: str, value: Any) -> dict[str, np.ndarray]:
    """``{key: array}`` for an array, scalar, list/tuple or dict output (recursively)."""
    if value is None:
        return {f"{key}=None": np.zeros(0)}
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


def compute(wl: Any, dm: Any, wilson_kw: dict, domainia_kw: dict, *,
            only: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Every case on every grid and ``dt``.  ``only`` keeps the cases whose name starts
    with one of the prefixes (``"transform."``, ``"TimeToWDM."``, ...)."""
    out: dict[str, np.ndarray] = {}
    for Nf, Nt in GRIDS:
        for dt in DTS:
            tag = f"{Nf}x{Nt}::dt={dt!r}"
            for layer, cases, kw in (("wilson", wilson_cases(wl, Nf, Nt, dt), wilson_kw),
                                     ("domainia", domainia_cases(dm, wl, Nf, Nt, dt), domainia_kw)):
                for name, case in cases:
                    if only is None or name.startswith(only):
                        out.update(flatten(f"{layer}::{name}::{tag}", case(kw)))
    return out


def main() -> None:
    """``--check``: replay the baseline through the current ``domainia.wilson`` layout.

    The baseline was recorded at d991ac4 by this module's own history (24c4e55 holds
    the recorder); re-recording means checking out 24c4e55.  The gate is
    ``tests/wilson/test_meyer_baseline.py``; this is the same replay as a script.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", required=True,
                   help="recompute and compare bit for bit")
    p.parse_args()
    src = str(Path(__file__).resolve().parents[2] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import domainia.covariance
    import domainia.domains
    import domainia.maps
    import domainia.spectra
    import domainia.testing
    import domainia.variance
    from domainia import wilson

    dm = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                         variance=domainia.variance, covariance=domainia.covariance,
                         testing=domainia.testing)
    out = compute(wilson, dm, {}, {})
    with np.load(BASELINE) as z:
        stored = {k: z[k] for k in z.files}
    if sorted(stored) != sorted(out):
        raise SystemExit(f"baseline keys differ: {sorted(set(stored) ^ set(out))[:10]}")
    bad = [k for k in out if stored[k].dtype != out[k].dtype or not np.array_equal(stored[k], out[k])]
    if bad:
        raise SystemExit(f"{len(bad)} arrays differ from the baseline: {bad[:10]}")
    print(f"baseline reproduces: {len(out)} arrays")


if __name__ == "__main__":
    main()
