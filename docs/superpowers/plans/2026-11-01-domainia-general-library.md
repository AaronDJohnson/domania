# domainia 2.0 General Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn this copy of domainia 1.1.0 plus the bundled `wdm-paper` 1.1.0 into one general library, `domainia` 2.0.0. It moves data and covariance between the time, Fourier and Wilson (WDM) domains. The Wilson kernels live in `domainia.wilson`, and every kernel takes a `window=`.

**Architecture:** Restructure in place. `wilson/src/wdm` becomes `src/domainia/wilson`, and the paper-only and model-specific code is deleted. A new `domainia.wilson.windows` adds admissible windows (`Meyer(d)`, `RampWindow`, `SampledWindow`), and the window is threaded through every kernel that sampled `phi_centered`. A new `domainia.translate` moves `{channel: array}` dicts and lazily translated covariance operators between the full-record time, Fourier and WDM domains. Before any edit, a bit-identity baseline pins the Meyer outputs.

**Tech Stack:** Python ≥3.12, NumPy ≥2.1, SciPy ≥1.14.1, optional JAX ≥0.9, pytest ≥8, hatchling. The test env is `~/miniconda3/envs/wdmrel` (Python 3.12.4, jax 0.9.0.1).

**User decisions (already made):**
- Identity: "general library as above, but paper-only pieces can be removed entirely. THis is a copy of the repo containing this package among other things."
- Seed: the bundled wdm-paper code with its short names (`Nf`, `Nt`, `dt`); `paper.py` is dropped.
- Windows: compact support only (|f| < ΔF). "Meyer only via d, not flat": the flat region stays fixed at ΔF/4. `RampWindow` and `SampledWindow` are user windows.
- Boundary: domainia owns the array transforms and the lazy covariance change of basis. enchilada becomes a thin adapter in a separate follow-on spec and plan in the enchilada repo.
- Approach A: restructure in place.
- The middle tier (exact_time, vandermonde, exact_fourier, hermitian, pushed, edges, nuisance, spectra's model PSDs, sampling, likelihood, density) is deleted.
- The translate API takes and returns enchilada-style `{channel: array}` dicts.
- Version 2.0.0. Spec approved: `docs/superpowers/specs/2026-10-31-domainia-general-library-design.md`.

**Readings this plan fixes where the spec is silent:**
1. **Where `window` lives.** Maps (`TimeToWDM`, `WDMToTime`, `FourierToWDM`, `Restrict`) read it from their `WDMDomain`. Objects built on a `CellSet` (`EvolutionaryCovariance`, `BandAveragedVariance`, `testing.dense_W`, `testing.wdm_congruence`, `testing.eq45_dense`) take `window: Window = MEYER` as a keyword; it is pytree `meta` where the class is a pytree. `MidpointVariance` never sampled the window, so it gets no `window`. `Restrict` adds a window-equality check next to its grid check.
2. **`SampledWindow.fn` is dimensionless.** `fn(u)` with u = f/ΔF has flat value 1 and satisfies fn(u)² + fn(u − 1)² = 1 on [0, 1]. `sample` returns `fn(f/ΔF)/sqrt(ΔF)`, which is the Hz normalisation of `phi_centered`.
3. **Doctest sweep.** In `>>>`/`...` lines only the import lines change: `import numpy as np, wdm` becomes `import numpy as np; from domainia import wilson as wdm`, so the ~130 `wdm.f(...)` call lines stay valid. In every other line (prose, roles, error-message strings, expected output), `wdm.` becomes `domainia.wilson.`.
4. **`eq45_plan(..., window=MEYER)` is the only Eq. 45 entry point that takes a window.** `Eq45Plan.base` already holds the windowed atoms, and `eq45_gram_plan` and `eq45_direct` read `plan.base`. `Eq45Plan` gains no field.
5. **`window_bins` is deleted.** Only the deleted separable builders used it. `keep_mask` stays.
6. **Nothing spectrum-driven stays jittable.** With `bessel` and `gram` gone, `EvolutionaryCovariance.jittable` is always False. The `GenericPSD` variance branches are NumPy-only. JAX survives through the `linalg` kernels, `DenseCovariance`, `DiagonalCovariance` with callable evaluators, `BlockTridiagonalCovariance` and `ChannelStack`. Do not try to "fix" this.
7. **Test classes are re-derived from imports and assertions** (spec §1 allows this). The full per-file table is in Task 2. It differs from the spec's tentative lists:
   - `test_dom_keyed_jit` and `test_dom_factor_constants` → (a). Every test uses a deleted keyed combinator or nuisance class, and base jit stays covered by `test_dom_cov_base::test_jit_takes_the_constants_as_an_argument`.
   - `test_dom_g6_linalg` → (a). It is a golden gate.
   - `test_dom_eq45`, `test_dom_maps_cells` and `test_dom_variance` are deleted. Task 3 replaces them with synthetic property tests written in full.
8. **Bit-identity baseline scope.** Dense O(N²) outputs are recorded on the (8, 16) grid only, which keeps the `.npz` under ~3 MB. Every other output is recorded on all three grids and both `dt` values.
9. **`translate` routes every pair through the NumPy rfft hub** (`R = numpy.fft.rfft(x)`). Time→WDM is `forward_many(rfft(x))`, which `forward_time_many` already computes bit for bit, and WDM→Time is `irfft(inverse_many(w))`.

**Verification command (full suite):**
`PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q`
Below, `$PY` means `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P`.

**Git note:** the Claude Code sandbox blocks writes inside `.git`. Every `git add`, `git mv`, `git rm` and `git commit` must run with the sandbox disabled. End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## File structure (end state)

```
pyproject.toml                         domainia 2.0.0 (one distribution; wilson/pyproject.toml is gone)
README.md                              rewritten for the general library
src/domainia/
  __init__.py                          2.0 exports, no guard, never imports JAX
  wilson/                              ← wilson/src/wdm (git mv)
    __init__.py                        kernels + Window, Meyer, RampWindow, SampledWindow, MEYER, WindowError
    windows.py                         NEW: admissible windows, validation, LRU cache, sample_window
    filter.py                          nu_d, phi_centered, phi_fft_order (Meyer helpers, unchanged)
    backend.py grid.py special.py      unchanged except the doctest sweep
    transform.py matrix.py stationary.py noise.py evolutionary.py inner_product.py   window= threaded
  domains.py                           WDMDomain.window, FourierDomain.full, WDMDomain.full
  data.py linalg.py                    import rename only
  maps/{base,time,fourier}.py          window read from WDMDomain; Restrict checks it
  covariance/{base,diagonal,dense,blocktri,stack}.py   import rename only
  covariance/evolutionary.py           ← covariance/eq45.py: EvolutionaryCovariance (interp, direct)
  spectra.py                           GenericPSD, TabulatedSpectrum only
  variance.py                          GenericPSD branches only; BandAveragedVariance(window=)
  testing.py                           window= on the oracles; smoothstep, wide_profile, ALT_WINDOW, PointwiseOperator
  translate.py                         NEW: transform, log_coordinate_scale, Operator, TranslatedOperator, translate_operator
tests/
  conftest.py                          golden fixtures removed
  dom_db_inputs.py                     synthetic constants + DenseCov + jax_or_skip (no release data)
  test_dom_*.py                        survivors (Task 2 table) + new property tests (Task 3)
  test_translate.py test_translate_operator.py test_package.py   NEW
  wilson/                              ← wilson/tests (git mv), with vendored/ and oracle/
    meyer_cases.py data/meyer_baseline.npz test_meyer_baseline.py   NEW (Task 0, Task 1)
    test_windows.py test_window_threading.py                         NEW
```

Deleted: `wilson/` (after the moves), `src/domainia/{assets,nuisance,sampling,likelihood,density}.py`, `src/domainia/literal/`, `src/domainia/covariance/{exact_time,vandermonde,exact_fourier,hermitian,pushed,edges,eq45}.py`, and `wilson/src/wdm/paper.py`.

---

### Task 0: Record the Meyer bit-identity baseline (before any edit)

**Goal:** Record, with the untouched code at d991ac4 (old layout, `import wdm`), the outputs of every public window consumer on fixed grids and seeded inputs into `wilson/tests/data/meyer_baseline.npz`, together with the script that computes them.

**Files:**
- Create: `wilson/tests/meyer_cases.py`
- Create: `wilson/tests/data/meyer_baseline.npz` (generated)

**Acceptance Criteria:**
- [ ] `wilson/tests/meyer_cases.py --write` runs on the old layout and writes `wilson/tests/data/meyer_baseline.npz`.
- [ ] `--check` run straight after reports `baseline reproduces` (outputs are deterministic).
- [ ] No file under `src/` or `wilson/src/` changed (`git status --short src wilson/src` is empty).
- [ ] The `.npz` is ≤ 5 MB.

**Verify:** `cd ~/Documents/lisa/domainia && $PY wilson/tests/meyer_cases.py --check` → prints `baseline reproduces: <K> arrays`, with K > 300.

**Steps:**

- [ ] **Step 1: Confirm the tree is at the baseline**

Run: `git -C ~/Documents/lisa/domainia log --oneline -1 -- src wilson/src && git -C ~/Documents/lisa/domainia status --short`
Expected: the last commit touching `src`/`wilson/src` is `d991ac4`, and the working tree is clean.

- [ ] **Step 2: Write `wilson/tests/meyer_cases.py`**

```python
"""Bit-identity cases for the Meyer window (domainia 2.0 spec, section 5).

Every public function that consumes the window, on fixed grids and seeded inputs.
``wilson_cases`` covers the kernels and ``domainia_cases`` the domainia layer.  Each
case is ``(name, fn)``: ``fn(kw)`` calls the function with the extra keywords ``kw``
(``{}`` on the old code, ``{"window": ...}`` after the refactor).

Recorded once against the untouched code (commit d991ac4, the old ``wdm`` layout)::

    python wilson/tests/meyer_cases.py --write     # then --check: deterministic?

After the refactor, ``tests/wilson/test_meyer_baseline.py`` recomputes every case
through ``domainia.wilson`` and requires ``np.array_equal`` (same dtype) on every array.

When an input shape below is wrong for a function on the old code, fix the INPUT per
that function's docstring.  Never edit a library function in this task.
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
        ("stationary.psd_from_K",
         lambda kw: St.psd_from_K(St.build_K_full_active(Nf, Nt, dt, psd, **kw)[0], Nf, Nt, dt, **kw)),
        ("stationary.apply_lambda_from_psd", lambda kw: St.apply_lambda_from_psd(w, Nf, Nt, dt, S, **kw)),
        ("stationary.psd_from_lambda_blocks",
         lambda kw: St.psd_from_lambda_blocks(*St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, **kw),
                                              Nf, Nt, dt, **kw)),
        ("noise.band_averaged_variance", lambda kw: No.band_averaged_variance(Nf, Nt, dt, psd, **kw)),
        ("noise.evolutionary_band_variance",
         lambda kw: No.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)),
        ("noise.empirical_variance", lambda kw: No.empirical_variance(Nf, Nt, dt, psd, 3, seed=11, **kw)),
        ("evolutionary.energy_fraction", lambda kw: E.energy_fraction(w ** 2, Nf, Nt, dt, **kw)),
        ("evolutionary.apply_time_mask", lambda kw: E.apply_time_mask(Wm[:, mask], chi, Nf, Nt, dt, **kw)),
        ("evolutionary.wrap_fractions", lambda kw: E.wrap_fractions(Nf, Nt, dt, **kw)),
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
    gen = dm.spectra.GenericPSD(
        lambda f, t, nu: psd(f) * (1.0 + nu[0] * np.sin(2.0 * np.pi * np.asarray(t, dtype=float) / T)))
    nu = np.array([0.4])
    Cov = getattr(dm.covariance, "EvolutionaryCovariance", None) or dm.covariance.Eq45Covariance

    def W(c: Any, kw: dict) -> Any:
        return D.WDMDomain(c, **kw)

    cases: list[Case] = [
        ("TimeToWDM.apply", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).apply(x)),
        ("TimeToWDM.apply_many", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).apply(X)),
        ("TimeToWDM.adjoint", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).adjoint(y)),
        ("TimeToWDM.rows", lambda kw: dm.maps.TimeToWDM(time, W(cells, kw)).rows([0, 5, cells.size - 1])),
        ("WDMToTime.apply", lambda kw: dm.maps.WDMToTime(W(cells, kw), time).apply(y)),
        ("FourierToWDM.dense", lambda kw: dm.maps.FourierToWDM(four, W(band_cells, kw)).apply(np.fft.rfft(x))),
        ("FourierToWDM.fft", lambda kw: dm.maps.FourierToWDM(four, W(band_cells, kw), "fft").apply(np.fft.rfft(x))),
        ("BandAveragedVariance", lambda kw: dm.variance.BandAveragedVariance(gen, cells, **kw)(nu)),
        ("EvolutionaryCovariance.interp", lambda kw: Cov(gen, cells, builder="interp", **kw).blocks(nu)),
    ]
    if (Nf, Nt) == DENSE_GRID:
        cases += [
            ("EvolutionaryCovariance.direct",
             lambda kw: Cov(gen, cells, builder="direct", assembly="dense", **kw).dense(nu)),
            ("testing.dense_W", lambda kw: dm.testing.dense_W(cells, **kw)),
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


def _old_layout() -> tuple[Any, Any]:
    """``(wdm, domainia namespace)`` from the pre-refactor tree (``wilson/src`` + ``src``)."""
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "wilson" / "src")]
    import wdm  # noqa: PLC0415
    import domainia.covariance  # noqa: PLC0415
    import domainia.domains  # noqa: PLC0415
    import domainia.maps  # noqa: PLC0415
    import domainia.spectra  # noqa: PLC0415
    import domainia.testing  # noqa: PLC0415
    import domainia.variance  # noqa: PLC0415

    dm = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                         variance=domainia.variance, covariance=domainia.covariance,
                         testing=domainia.testing)
    return wdm, dm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true", help="record the baseline npz")
    p.add_argument("--check", action="store_true", help="recompute and compare bit for bit")
    args = p.parse_args()
    wl, dm = _old_layout()
    out = compute(wl, dm, {}, {})
    if args.write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(BASELINE, **out)
        print(f"wrote {len(out)} arrays to {BASELINE}")
    if args.check:
        with np.load(BASELINE) as z:
            stored = {k: z[k] for k in z.files}
        assert sorted(stored) == sorted(out), sorted(set(stored) ^ set(out))[:10]
        bad = [k for k in out if stored[k].dtype != out[k].dtype or not np.array_equal(stored[k], out[k])]
        assert not bad, bad[:10]
        print(f"baseline reproduces: {len(out)} arrays")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Record the baseline**

Run: `cd ~/Documents/lisa/domainia && $PY wilson/tests/meyer_cases.py --write`
Expected: `wrote <K> arrays to .../wilson/tests/data/meyer_baseline.npz`. If a case raises because an input shape guess is wrong, fix the input in `meyer_cases.py` per the function's docstring (likely candidates: `energy_fraction`'s `w2`, `apply_time_mask`'s `Y`, `inverse_bands`'s `w`), then rerun. Never edit library code here.

- [ ] **Step 4: Check determinism and size**

Run: `$PY wilson/tests/meyer_cases.py --check && ls -l wilson/tests/data/meyer_baseline.npz && git status --short src wilson/src`
Expected: `baseline reproduces: <K> arrays`, a file ≤ 5 MB, and no output from `git status`. If `--check` fails on a key, a BLAS call is nondeterministic. Record that key on `DENSE_GRID` only, or drop it with a comment saying why, then redo Step 3.

- [ ] **Step 5: Commit** (sandbox disabled)

```bash
git add wilson/tests/meyer_cases.py wilson/tests/data/meyer_baseline.npz
git commit -m "Record the Meyer bit-identity baseline at d991ac4

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: Move `wdm` into `domainia.wilson`; one 2.0.0 distribution

**Goal:** `wilson/src/wdm` becomes `src/domainia/wilson` and `wilson/tests` becomes `tests/wilson`. Every `wdm` import is rewritten, the release guard, sentinels and `paper.py` are gone, and `pyproject.toml` is 2.0.0. The wilson tests, the wilson doctests and the baseline replay all pass.

**Files:**
- Move: `wilson/src/wdm/*` → `src/domainia/wilson/`; `wilson/tests/*` → `tests/wilson/`
- Delete: `wilson/pyproject.toml`, `wilson/README.md`, `wilson/LICENSE`, `src/domainia/wilson/paper.py`, `tests/wilson/test_paper.py`, `tests/wilson/test_package_identity.py`, `tests/wilson/test_wd_rounding.py`
- Modify: `src/domainia/wilson/__init__.py`, `src/domainia/__init__.py`, every `src/domainia/**/*.py` and `tests/**/*.py` that imports `wdm`, `tests/wilson/test_lanes.py`, `tests/wilson/conftest.py`, `pyproject.toml`
- Create: `tests/wilson/test_meyer_baseline.py`

**Acceptance Criteria:**
- [ ] `wilson/` no longer exists, and `src/domainia/wilson/paper.py` does not exist.
- [ ] `grep -rnE "(^|[^A-Za-z0-9_/.-])wdm\." src` prints nothing, and `grep -rnE "^\s*(import wdm|from wdm)" src tests` prints nothing.
- [ ] `src/domainia/__init__.py` has no `__release__` check, and `domainia.wilson` defines no `__release__` or `__wdm_api__`.
- [ ] `$PY -m pytest -q tests/wilson src/domainia/wilson` passes with no collection errors.
- [ ] `tests/wilson/test_meyer_baseline.py` passes: every baseline array is reproduced bit for bit.

**Verify:** `$PY -m pytest -q tests/wilson src/domainia/wilson` → `N passed, M skipped`, no errors. (The `tests/test_dom_*` files are fixed in Task 2.)

**Steps:**

- [ ] **Step 1: Move the trees** (sandbox disabled)

```bash
cd ~/Documents/lisa/domainia
git mv wilson/src/wdm src/domainia/wilson
mkdir -p tests/wilson
for p in wilson/tests/* wilson/tests/.[!.]*; do [ -e "$p" ] && git mv "$p" tests/wilson/; done
git rm -q wilson/pyproject.toml wilson/README.md wilson/LICENSE
git rm -q src/domainia/wilson/paper.py tests/wilson/test_paper.py tests/wilson/test_package_identity.py tests/wilson/test_wd_rounding.py
ls wilson 2>/dev/null; rm -rf wilson
```
Expected: `ls wilson` prints nothing (or only untracked `__pycache__`, which `rm -rf` removes).

- [ ] **Step 2: Rewrite imports with a one-off codemod** (keep it in the scratchpad, do not commit it)

Save as `$TMPDIR/rename_wdm.py`:

```python
"""One-off: rewrite every ``wdm`` import/reference for the domainia.wilson move."""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PROMPT = re.compile(r"^\s*(>>>|\.\.\.)")
REF = re.compile(r"(?<![A-Za-z0-9_/.-])wdm\.(?=[A-Za-z_])")


def wilson_line(line: str) -> str:
    """src/domainia/wilson: doctest imports alias ``wdm``; everything else names domainia.wilson."""
    if PROMPT.match(line):
        line = line.replace("import numpy as np, wdm", "import numpy as np; from domainia import wilson as wdm")
        line = re.sub(r">>> import wdm\b", ">>> from domainia import wilson as wdm", line)
        line = re.sub(r"\bfrom wdm(\.| )", lambda m: "from domainia.wilson" + m.group(1), line)
        return line
    return REF.sub("domainia.wilson.", line)


def domainia_line(line: str) -> str:
    """src/domainia (outside wilson): ``wilson.`` everywhere."""
    line = re.sub(r"^(\s*)import wdm as _wdm\s*$", r"\1from domainia import wilson as _wdm", line)
    line = re.sub(r"^(\s*)import wdm\s*$", r"\1from domainia import wilson", line)
    line = re.sub(r"^(\s*)from wdm(\.\S+)? import", lambda m: f"{m.group(1)}from domainia.wilson{m.group(2) or ''} import", line)
    return REF.sub("wilson.", line)


def test_line(line: str) -> str:
    """tests: keep the local name ``wdm`` (an alias) so test bodies are untouched."""
    line = re.sub(r"^(\s*)import wdm\s*$", r"\1from domainia import wilson as wdm", line)
    line = re.sub(r"^(\s*)from wdm(\.\S+)? import", lambda m: f"{m.group(1)}from domainia.wilson{m.group(2) or ''} import", line)
    line = line.replace('"wdm.', '"domainia.wilson.').replace("'wdm.", "'domainia.wilson.")
    return line


for path in sorted(ROOT.rglob("*.py")):
    rel = path.relative_to(ROOT).as_posix()
    if "/oracle/" in f"/{rel}":
        continue                                    # frozen oracles never import wdm
    if rel.startswith("src/domainia/wilson/"):
        fix = wilson_line
    elif rel.startswith("src/domainia/"):
        fix = domainia_line
    elif rel.startswith("tests/"):
        fix = test_line
    else:
        continue
    text = path.read_text()
    new = "".join(fix(line) for line in text.splitlines(keepends=True))
    if new != text:
        path.write_text(new)
        print("rewrote", rel)
```

Run: `cd ~/Documents/lisa/domainia && ~/miniconda3/envs/wdmrel/bin/python -I $TMPDIR/rename_wdm.py .`
Expected: a list of rewritten files under `src/domainia` and `tests`.

- [ ] **Step 3: Remove the release guard and sentinels**

In `src/domainia/__init__.py`, delete these three lines. `domainia_line` rewrote the first to `from domainia import wilson as _wdm`:

```python
from domainia import wilson as _wdm

if getattr(_wdm, "__release__", None) != "wdm-paper-1" or not hasattr(_wdm, "forward_time_many"):
    raise ImportError(f"wdm resolved to {getattr(_wdm, '__file__', None)}: not the paper release")
```
Set `__version__ = "2.0.0"`. Replace the module docstring's second paragraph ("Importing ``domainia`` checks that ``wdm`` is the paper release ...") with: "Importing ``domainia`` never imports JAX: the JAX backend is reached only through :mod:`domainia.wilson.backend`." Drop the `# noqa: E402` comments on the imports below it.

In `src/domainia/wilson/__init__.py`:
- delete `paper,` from the `from . import (...)` submodule list and `"paper",` from `__all__`;
- delete the "Release sentinels" comment block and the three assignments `__version__ = "1.1.0"`, `__wdm_api__ = "functions/1"`, `__release__ = "wdm-paper-1"`;
- delete `"__release__"`, `"__version__"` and `"__wdm_api__"` from `__all__`;
- in the module docstring, delete every line that mentions `paper` or the release.

Run: `grep -rn "__release__\|__wdm_api__\|wdm-paper\|paper\b" src/domainia/wilson/__init__.py src/domainia/__init__.py`
Expected: no output.

- [ ] **Step 4: Fix the paths the move changed**

`tests/wilson/test_lanes.py`: replace the `PKG = ...` line and the `subprocess.run` arguments:

```python
REPO = Path(__file__).resolve().parents[2]


def test_numpy_only_lane_skips_every_jax_test_and_passes_the_rest():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO / "src"),
               WDM_TESTS_HIDE_JAX="1")
    out = subprocess.run(
        [sys.executable, "-P", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs",
         "tests/wilson/test_backend.py", "tests/wilson/test_transform_shim.py"],
        env=env, cwd=str(REPO), capture_output=True, text=True)
```
(The three `assert` lines stay unchanged.)

`tests/wilson/conftest.py`: in the docstring, change ``wdm-paper`` to ``domainia.wilson``. `ORACLE_DIR` is already relative to the file, so it still finds `tests/wilson/oracle`.

Run: `grep -rn "parents\[\|PKG\b\|wdm.*src\|\"src\"" tests/wilson/*.py tests/wilson/vendored/*.py`
Expected: every path you see resolves under the repo root (`parents[2]` from `tests/wilson/x.py`). Fix any `parents[1]`-relative path that expected `wilson/`. `test_evolutionary_kernels.py` points at the workspace lineage (`toy_models`, `wdm_dev`) and skips when those are missing. Leave it.

- [ ] **Step 5: Write the new `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "domainia"
version = "2.0.0"
description = "Move data and covariance between the time, Fourier and Wilson (WDM) time-frequency domains."
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Aaron Johnson" }]
dependencies = [
    "numpy>=2.1,<3",
    "scipy>=1.14.1,<2",
]

[project.optional-dependencies]
jax = ["jax>=0.9", "jaxlib>=0.9"]
test = ["pytest>=8"]

[tool.hatch.build.targets.wheel]
packages = ["src/domainia"]

[tool.pytest.ini_options]
# One config for the whole repo.  tests/ has no __init__.py (its modules import the
# sibling helpers dom_db_inputs.py by name); tests/wilson is a package whose conftest
# puts tests/wilson/oracle on sys.path.  Doctests run over domainia.wilson only.
testpaths = ["tests", "src/domainia/wilson"]
pythonpath = ["src"]
addopts = "-ra --doctest-modules -p no:cacheprovider"
norecursedirs = [".*", "__pycache__", "oracle", "data"]
xfail_strict = true
markers = [
    "jax: needs jax; skipped when jax is not importable",
    "slow: heavy constructions (opt-in: -m slow)",
]
```

- [ ] **Step 6: Write the baseline replay test `tests/wilson/test_meyer_baseline.py`**

```python
"""The default window reproduces the Meyer baseline recorded at d991ac4 bit for bit."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import domainia.covariance
import domainia.domains
import domainia.maps
import domainia.spectra
import domainia.testing
import domainia.variance
from domainia import wilson

from .meyer_cases import BASELINE, compute

DM = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                     variance=domainia.variance, covariance=domainia.covariance,
                     testing=domainia.testing)


@pytest.fixture(scope="module")
def baseline() -> dict[str, np.ndarray]:
    with np.load(BASELINE) as z:
        return {k: z[k] for k in z.files}


def _mismatches(got: dict, want: dict) -> list[str]:
    assert sorted(got) == sorted(want), sorted(set(got) ^ set(want))[:10]
    return [k for k in want if got[k].dtype != want[k].dtype or not np.array_equal(got[k], want[k])]


def test_default_window_reproduces_the_meyer_baseline_bit_for_bit(baseline):
    assert not _mismatches(compute(wilson, DM, {}, {}), baseline)
```

- [ ] **Step 7: Run the wilson suite and its doctests**

Run: `$PY -m pytest -q tests/wilson src/domainia/wilson`
Expected: all pass (about 390 passed, a handful skipped, no errors). Typical fixes:
- A doctest whose expected output names a non-builtin exception as `wdm.x.Error` was rewritten to `domainia.wilson.x.Error` by the codemod. That is correct, because the qualified name changed. If one still fails, fix the expected line.
- `test_lanes` paths: see Step 4.

If `test_meyer_baseline` fails, the move changed numerics, which is a bug. Diff a failing key against `git show d991ac4:<path>` before going on.

- [ ] **Step 8: Commit** (sandbox disabled)

```bash
git add -A src tests pyproject.toml
git commit -m "Move wdm into domainia.wilson; one domainia 2.0.0 distribution

Drop the paper release guard, sentinels and paper.py; rewrite every wdm import.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 2: Cut the paper-only and model-specific code; prune the tests

**Goal:** Delete the paper-only and model-specific domainia modules. Slim `spectra.py` and `variance.py`, turn `covariance/eq45.py` into `covariance/evolutionary.py` (`EvolutionaryCovariance`, builders `interp` and `direct`), and sort every test file per the table below. Afterwards the full suite passes with no collection errors.

**Files:**
- Delete: `src/domainia/{assets,nuisance,sampling,likelihood,density}.py`, `src/domainia/literal/`, `src/domainia/covariance/{exact_time,vandermonde,exact_fourier,hermitian,pushed,edges}.py`
- Move: `src/domainia/covariance/eq45.py` → `src/domainia/covariance/evolutionary.py` (rewritten)
- Modify: `src/domainia/spectra.py`, `src/domainia/variance.py`, `src/domainia/covariance/__init__.py`, `src/domainia/__init__.py`
- Delete (tests, class a): the 36 files listed in Step 6
- Modify (tests, class b): `tests/conftest.py`, `tests/dom_db_inputs.py`, `tests/test_dom_{cov_api,cov_base,ev_ported,linalg_factors,linalg_jax,maps_base,maps_dirichlet,maps_fourier_time,maps_wdm,spectra,db_imports}.py`
- Keep (tests, class c): `tests/test_dom_{cov_blocktri,data,domains,linalg_blocktri,linalg_pcg,testing}.py`, all of `tests/wilson/`

**Acceptance Criteria:**
- [ ] `ls src/domainia` shows only `__init__.py covariance data.py domains.py linalg.py maps spectra.py testing.py variance.py wilson`.
- [ ] `ls src/domainia/covariance` shows only `__init__.py base.py blocktri.py dense.py diagonal.py evolutionary.py stack.py`.
- [ ] The Step 9 grep for deleted names and golden fixtures prints nothing.
- [ ] The full verification command passes with no collection errors.
- [ ] The Meyer baseline replay still passes (it now uses `EvolutionaryCovariance`).

**Verify:** `$PY -m pytest -q` → all pass, `0 errors`.

**Steps:**

- [ ] **Step 1: Delete the modules** (sandbox disabled)

```bash
cd ~/Documents/lisa/domainia
git rm -rq src/domainia/literal
git rm -q src/domainia/{assets,nuisance,sampling,likelihood,density}.py
git rm -q src/domainia/covariance/{exact_time,vandermonde,exact_fourier,hermitian,pushed,edges}.py
git mv src/domainia/covariance/eq45.py src/domainia/covariance/evolutionary.py
```

- [ ] **Step 2: Slim `src/domainia/spectra.py`**

Delete the classes `LogLinearPSD`, `Envelope` and `SeparablePSD` entirely. Keep `TabulatedSpectrum` and `GenericPSD` byte for byte. Replace the module docstring and `__all__` with:

```python
"""Noise spectra.

* :class:`GenericPSD` -- any elementwise, broadcasting ``Sn(f, t, nu)``: what the
  evolutionary covariance and the WDM variances evaluate.
* :class:`TabulatedSpectrum` -- ``(C, n)`` values on absolute rfft bins of a record.
"""
```
```python
__all__ = ["GenericPSD", "TabulatedSpectrum"]
```
Then remove any import that is now unused (`ClassVar`/`Protocol`/`runtime_checkable`/`Backend`/`get_backend` if only the deleted classes used them). Check: `$PY -c "import ast,sys; ast.parse(open('src/domainia/spectra.py').read())"` and `$PY -c "import domainia.spectra as s; print(s.__all__)"`.

- [ ] **Step 3: Rewrite `src/domainia/variance.py`** (the GenericPSD branches, unchanged in arithmetic)

```python
"""Diagonal WDM variances of a :class:`~domainia.spectra.GenericPSD` ``S(f, t, nu)``.

* :class:`BandAveragedVariance` (WDM II Eq. 36) -- the window-weighted band average,
  ``domainia.wilson.noise.evolutionary_band_variance`` of the bound PSD at the cells;
* :class:`MidpointVariance` (Eq. 39) -- ``0.5 S(f_m, t_n)`` at the band centre
  ``f_m = m dF`` and the cell time ``t_n = n dT``.

Both are pytrees (``domainia.wilson.backend.pytree_dataclass``) so a
``DiagonalCovariance`` can carry them through ``jax.jit`` as arguments, but evaluating
them calls the PSD on the host: NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from domainia import wilson
from domainia.wilson.backend import Backend, pytree_dataclass

from .domains import CellSet
from .maps.base import _BUILD, on_backend
from .spectra import GenericPSD

__all__ = ["BandAveragedVariance", "MidpointVariance"]


def _cell_arrays(cells: CellSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = cells.grid
    m = np.asarray(cells.m, dtype=np.int64)
    n = np.asarray(cells.n, dtype=np.int64)
    return m, n, n * wilson.delta_T(g.Nf, g.dt)                     # t_n = n dT


def _require_generic(psd: Any, what: str) -> None:
    if not isinstance(psd, GenericPSD):
        raise TypeError(f"no {what} variance for {type(psd).__name__}: wrap the spectrum in "
                        "domainia.spectra.GenericPSD")


@pytree_dataclass(data=("t",), meta=("psd", "cells"))
@dataclass(frozen=True, eq=False)
class BandAveragedVariance:
    """Eq. 36: ``0.5 <S(f, t_n)>`` over the window squared around each cell's band."""

    psd: Any
    cells: CellSet
    t: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.t is not _BUILD:
            return
        _require_generic(self.psd, "band-averaged")
        object.__setattr__(self, "t", _cell_arrays(self.cells)[2])

    def on(self, be: Backend) -> "BandAveragedVariance":
        """The constants moved to ``be``."""
        return on_backend(self, be, ("t",))

    def __call__(self, nu: Any, *, channel: int = 0, be: Backend | None = None) -> Any:
        g = self.cells.grid
        m, n, _ = _cell_arrays(self.cells)
        full = wilson.noise.evolutionary_band_variance(g.Nf, g.Nt, g.dt, self.psd.bind(nu))
        return full[m, n]


@pytree_dataclass(data=("t",), meta=("psd", "cells"))
@dataclass(frozen=True, eq=False)
class MidpointVariance:
    """Eq. 39: ``0.5 S(f_m, t_n)``."""

    psd: Any
    cells: CellSet
    t: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.t is not _BUILD:
            return
        _require_generic(self.psd, "midpoint")
        object.__setattr__(self, "t", _cell_arrays(self.cells)[2])

    def on(self, be: Backend) -> "MidpointVariance":
        """The constants moved to ``be``."""
        return on_backend(self, be, ("t",))

    def __call__(self, nu: Any, *, channel: int = 0, be: Backend | None = None) -> Any:
        g = self.cells.grid
        m, _, t = _cell_arrays(self.cells)
        f = wilson.band_frequencies(g.Nf, g.dt)[m]
        return 0.5 * np.asarray(self.psd(f, t, nu))
```

- [ ] **Step 4: Rewrite `src/domainia/covariance/evolutionary.py`**

```python
r"""Evolutionary-spectrum covariance of WDM cells (WDM II Eq. 45) for any callable PSD.

.. math::  \Lambda_{pq} = \frac1T\,\mathrm{Re}\sum_k \eta_k\,S(f_k, \bar t_{pq})\,
           \hat g_p(k)\,\hat g_q^*(k),\qquad \bar t_{pq} = (t_{n_p} + t_{n_q})/2,

with :math:`\bar t` used literally, no lag truncation, and only :math:`|m - m'| \le 1`
(an admissible window's compact support makes every other block zero).

==========  =====================================================================
builder     evaluator
==========  =====================================================================
``interp``  ``eq45_coeffs_interp`` + ``eq45_blocks_cheb``: Chebyshev in time
``direct``  ``eq45_direct``: the dense direct sum (small grids and tests)
==========  =====================================================================

``assembly="blocktri"`` factorises the band blocks (:class:`BlockTridiagonalCovariance`,
``layout`` ``"ragged"`` | ``"padded"``); ``assembly="dense"`` assembles the matrix
(:class:`DenseCovariance`).  Both builders call the PSD on the host, so the covariance
runs eagerly on every backend (``jittable`` is always False).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from domainia.wilson.backend import Backend, get_backend
from domainia.wilson.evolutionary import eq45_blocks_cheb, eq45_coeffs_interp, eq45_direct, eq45_plan
from domainia.wilson.grid import packed_cells

from .base import Covariance
from .blocktri import BlockTridiagonalCovariance, blocks_to_dense, to_layout
from .dense import DenseCovariance

__all__ = ["BUILDERS", "EvolutionaryCovariance", "keep_mask"]

Builder = Literal["interp", "direct"]
BUILDERS: tuple[str, ...] = ("interp", "direct")


def keep_mask(cells: Any) -> np.ndarray:
    """Boolean mask over ``wilson.grid.packed_cells`` of ``cells.grid`` selecting ``cells``.

    Raises unless ``cells`` lists active cells in band order (increasing packed position),
    the order of every Eq. (45) block and of ``eq45_plan``'s kept cells.
    """
    g = cells.grid
    m_all, n_all = packed_cells(g.Nf, g.Nt)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(m_all, n_all, strict=True))}
    try:
        pos = np.array([index[int(a), int(b)] for a, b in zip(cells.m, cells.n, strict=True)], int)
    except KeyError as err:
        raise ValueError(f"cell {err.args[0]} is not an active cell of {g}") from None
    if pos.size and np.any(np.diff(pos) <= 0):
        raise ValueError("cells must be listed in band order (increasing packed position)")
    keep = np.zeros(m_all.size, bool)
    keep[pos] = True
    return keep


class _ChebBlocks:
    """``interp``: Chebyshev-in-time blocks on the plan's kept cells."""

    def __init__(self, psd: Any, grid: Any, keep: np.ndarray, P: int, t_max: float | None) -> None:
        self.psd, self.grid, self.keep, self.P, self.t_max = psd, grid, keep, P, t_max

    def host_constants(self) -> dict[str, Any]:
        g = self.grid
        return {"plan": eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=self.P, t_max=self.t_max)}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> tuple[list[Any], list[Any]]:
        plan = C["plan"]
        nu_h = np.asarray(nu)

        def Sn(f: Any, t: Any) -> Any:
            return self.psd(f, t, nu_h)

        coeffs = eq45_coeffs_interp(Sn, plan.f, plan.t_max, plan.P, k=plan.k_used)
        return eq45_blocks_cheb(plan, coeffs, be=be)


class _DirectMatrix:
    """``direct``: the dense direct sum."""

    def __init__(self, psd: Any, grid: Any, keep: np.ndarray, t_max: float | None) -> None:
        self.psd, self.grid, self.keep, self.t_max = psd, grid, keep, t_max

    def host_constants(self) -> dict[str, Any]:
        g = self.grid
        return {"plan": eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=2, t_max=self.t_max)}

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        nu_h = np.asarray(nu)
        return be.asarray(eq45_direct(C["plan"], lambda f, t: self.psd(f, t, nu_h)))


class _DenseFromBlocks:
    """Band blocks assembled into the dense matrix (``assembly="dense"``)."""

    def __init__(self, blocks: Any, sizes: tuple[int, ...]) -> None:
        self.blocks, self.sizes = blocks, sizes

    def host_constants(self) -> Any:
        return self.blocks.host_constants()

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = to_layout(*self.blocks.evaluate(C, nu, be), self.sizes, "ragged", be)
        return blocks_to_dense(diag, up, be)


class EvolutionaryCovariance(Covariance):
    """Eq. (45) on ``cells`` for a callable ``psd(f, t, nu)`` (``GenericPSD``).

    Parameters
    ----------
    psd
        Any callable, elementwise, broadcasting ``psd(f, t, nu)``.
    cells
        A ``CellSet`` in band order.
    builder
        ``"interp"`` (default) or ``"direct"`` (needs ``assembly="dense"``).
    assembly, layout
        ``"blocktri"`` (``layout`` ``"ragged"``, default, or ``"padded"``) or ``"dense"``
        (``chol_impl``, ``symmetrize`` as :class:`DenseCovariance`).
    t_max
        End of the Chebyshev time domain (default: the cells' grid ``T``).
    P
        Chebyshev order of ``interp`` (24).
    """

    def __init__(self, psd: Any, cells: Any, *, builder: Builder = "interp", assembly: str = "blocktri",
                 layout: str | None = None, t_max: float | None = None, P: int = 24,
                 chol_impl: Literal["numpy", "scipy"] = "numpy", symmetrize: bool = True) -> None:
        if builder not in BUILDERS:
            raise ValueError(f"unknown builder {builder!r}; expected one of {BUILDERS}")
        if not callable(psd):
            raise ValueError(f"builder {builder!r} needs a callable PSD psd(f, t, nu); "
                             f"{type(psd).__name__} is not callable")
        if assembly not in ("blocktri", "dense"):
            raise ValueError(f"assembly must be 'blocktri' or 'dense'; got {assembly!r}")
        if builder == "direct" and assembly != "dense":
            raise ValueError("builder 'direct' needs assembly='dense'")
        self.psd, self.cells, self.builder, self.assembly = psd, cells, builder, assembly
        self.n = len(cells.m)
        keep = keep_mask(cells)
        m_all, _ = packed_cells(cells.grid.Nf, cells.grid.Nt)
        sizes = tuple(int(np.sum(keep & (m_all == mm))) for mm in range(cells.grid.Nf + 1))
        blocks: Any = (_ChebBlocks(psd, cells.grid, keep, P, t_max) if builder == "interp"
                       else _DirectMatrix(psd, cells.grid, keep, t_max))
        self.sizes = sizes
        self.block_evaluator = blocks
        if assembly == "blocktri":
            self.impl: Covariance = BlockTridiagonalCovariance(blocks, sizes, layout=layout or "ragged")
        else:
            matrix = blocks if builder == "direct" else _DenseFromBlocks(blocks, sizes)
            self.impl = DenseCovariance(matrix, self.n, chol_impl=chol_impl, symmetrize=symmetrize)
        self.jittable = False

    @property
    def layout(self) -> str | None:
        """The block layout (``None`` for ``assembly="dense"``)."""
        return getattr(self.impl, "layout", None)

    def host_constants(self) -> Any:
        return self.impl.host_constants()

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._factor(C, nu, be)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._matrix(C, nu, be)

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
        """``(diag, up)`` from the constants ``C`` (``assembly="blocktri"`` only)."""
        if not isinstance(self.impl, BlockTridiagonalCovariance):
            raise ValueError("band_blocks needs assembly='blocktri'")
        return self.impl.band_blocks(C, nu, be)

    def blocks(self, nu: Any, be: Backend | None = None) -> Any:
        """The builder's raw output at ``nu``: ragged band blocks (``interp``) or the
        matrix (``direct``)."""
        be = be or get_backend(nu)
        C = self.constants(be)
        C = C["blocks"] if isinstance(self.impl, BlockTridiagonalCovariance) else C
        return self.block_evaluator.evaluate(C, be.asarray(nu), be)
```

- [ ] **Step 5: Rewrite the package exports**

`src/domainia/covariance/__init__.py`:

```python
"""Covariance operators on WDM cells: one protocol, one factor per noise state.

Modules: ``base`` (protocol, factors, ``Covariance``), ``diagonal``, ``dense``,
``blocktri``, ``evolutionary`` (Eq. 45 for any callable PSD), ``stack`` (``ChannelStack``).
"""

from .base import CholFactor, Covariance, DiagFactor, Evaluator, Factor, QuadForm, as_evaluator, quad_form
from .blocktri import BlockTriCovFactor, BlockTridiagonalCovariance
from .dense import DenseCovariance, dense_factor
from .diagonal import DiagonalCovariance
from .evolutionary import EvolutionaryCovariance, keep_mask
from .stack import ChannelStack, StackFactor

__all__ = [
    "BlockTriCovFactor",
    "BlockTridiagonalCovariance",
    "ChannelStack",
    "CholFactor",
    "Covariance",
    "DenseCovariance",
    "DiagFactor",
    "DiagonalCovariance",
    "Evaluator",
    "EvolutionaryCovariance",
    "Factor",
    "QuadForm",
    "StackFactor",
    "as_evaluator",
    "dense_factor",
    "keep_mask",
    "quad_form",
]
```

`src/domainia/__init__.py`: delete `assets` from `from . import assets, data, domains, linalg`, delete the line `from .assets import Artifact, ArtifactError`, and delete `"Artifact"`, `"ArtifactError"` and `"assets"` from `__all__`. In the docstring, delete the `domainia.assets` bullet. (Task 10 writes the final 2.0 `__init__`.)

Run: `$PY -c "import domainia, domainia.covariance, domainia.maps, domainia.variance, domainia.spectra, domainia.testing, domainia.linalg; import sys; print('jax' in sys.modules)"`
Expected: `False`.

- [ ] **Step 6: Delete the class (a) test files** (sandbox disabled)

Each of these exercises only deleted code or compares against release goldens. The general properties some of them carried come back in Task 3.

```bash
cd ~/Documents/lisa/domainia/tests
git rm -q dom_literal_cases.py dom_toys.py domtest_inputs.py \
  test_dom_assets.py test_dom_consistency.py test_dom_cov_vandermonde.py test_dom_density.py \
  test_dom_eq45.py test_dom_eq45_separable.py test_dom_exact_fourier_constants.py test_dom_exact_time.py \
  test_dom_factor_constants.py test_dom_g6_linalg.py test_dom_hermitian.py test_dom_keyed_jit.py \
  test_dom_likelihood.py test_dom_literal_congruence.py test_dom_literal_eq45.py test_dom_literal_masked.py \
  test_dom_literal_pad.py test_dom_literal_time.py test_dom_maps_cells.py test_dom_nuisance.py \
  test_dom_package.py test_dom_pad_mask.py test_dom_pad_reduce.py test_dom_projected_stack.py \
  test_dom_push_exact.py test_dom_pushed_rows.py test_dom_sampling.py test_dom_schur_edge.py \
  test_dom_schur_scaled_batching.py test_dom_variance.py test_dom_whittle.py test_dom_whittle_apply_data.py \
  test_dom_zero_padded.py
```

Classification record (re-derived from imports and assertions; every file in `tests/` and `tests/wilson/` appears once):

| File | Class | Reason / action |
|---|---|---|
| `dom_literal_cases.py`, `test_dom_literal_{congruence,eq45,masked,pad,time}.py` | a | `domainia.literal` |
| `test_dom_assets.py` | a | `assets` |
| `test_dom_package.py` | a | release sentinels; replaced by `tests/test_package.py` (Task 10) |
| `test_dom_{cov_vandermonde,density,exact_fourier_constants,exact_time,hermitian,likelihood,nuisance,pushed_rows,push_exact,sampling,schur_edge,schur_scaled_batching,whittle,whittle_apply_data,zero_padded,eq45_separable}.py` | a | subject is a deleted module or builder |
| `test_dom_pad_mask.py`, `test_dom_pad_reduce.py` | a | `ZeroPadded`/`PadMask` (edges) and GB goldens |
| `test_dom_projected_stack.py` | a | nuisance projections; its one clean test (`ChannelStack`) moves to `test_dom_cov_stack.py` (Task 3) |
| `test_dom_consistency.py` | a | likelihood/edges consistency; its clean `DiagonalCovariance.cells` check moves to Task 3 |
| `test_dom_keyed_jit.py` | a | every test uses `FlatMarginal`/`PushedRows` (the keyed combinators are gone) |
| `test_dom_factor_constants.py` | a | parametrised over `ExactTimeCovariance`/`SchurEdge`/`ZeroPadded`/nuisance |
| `test_dom_g6_linalg.py` | a | golden gate (`golden_io`, `same_build`); the properties live in `test_dom_linalg_blocktri.py` |
| `test_dom_eq45.py` | a | bessel/gram builders + goldens; replaced by `test_dom_evolutionary_cov.py` (Task 3) |
| `test_dom_maps_cells.py` | a | 9/10 tests read GB goldens; replaced by `test_dom_maps_fourier_wdm.py` (Task 3) |
| `test_dom_variance.py` | a | every test reads goldens or `LogLinearPSD`; replaced by `test_dom_variance_generic.py` (Task 3) |
| `dom_toys.py`, `domtest_inputs.py` | a | only fed deleted tests and goldens |
| `dom_db_inputs.py`, `conftest.py` | b | rewritten without release data (Steps 7–8) |
| `test_dom_{cov_api,cov_base,ev_ported,linalg_factors,linalg_jax,maps_base,maps_dirichlet,maps_fourier_time,maps_wdm,spectra,db_imports}.py` | b | prune the listed tests (Step 8) |
| `test_dom_{cov_blocktri,data,domains,linalg_blocktri,linalg_pcg,testing}.py` | c | import rename only (done in Task 1) |
| `tests/wilson/**` | c | import rename only (done in Task 1); `test_paper`, `test_package_identity`, `test_wd_rounding` (a) removed in Task 1 |

- [ ] **Step 7: Rewrite `tests/dom_db_inputs.py`** (synthetic; no release data)

```python
"""Shared synthetic inputs of the domainia tests: the sinusoid grid numbers (a realistic
test size), a dense covariance stand-in and the JAX helper.  Nothing here reads release
data or goldens."""

from __future__ import annotations

import numpy as np
import pytest

#: The toy sinusoid grid: N = 16384, dt = 2^-10, T = 16 s; cut and pad column counts.
SIN_NF, SIN_NT, SIN_DT, SIN_T = 64, 256, 2.0**-10, 16.0
SIN_CUT, SIN_PAD_COLUMNS, SIN_FMIN = (6, 5), 16, 4.0

__all__ = ["SIN_CUT", "SIN_DT", "SIN_FMIN", "SIN_NF", "SIN_NT", "SIN_PAD_COLUMNS", "SIN_T",
           "DenseCov", "jax_or_skip", "synthetic_record"]


def jax_or_skip():
    """``(jax, JAX backend)`` with float64 on; skips in the NumPy-only lane."""
    jax = pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
    from domainia.wilson.backend import enable_x64, jax_backend

    enable_x64()
    return jax, jax_backend()


def synthetic_record(N: int, seed: int = 0) -> np.ndarray:
    """A seeded white-noise record of length ``N``."""
    return np.random.default_rng(seed).standard_normal(N)


class DenseCov:
    """A dense covariance stand-in: ``matvec`` (real, row-batched) and ``dense``."""

    def __init__(self, S: np.ndarray) -> None:
        self.S = np.asarray(S)

    def matvec(self, X, be=None):
        return X @ self.S

    def dense(self, be=None):
        return self.S
```

- [ ] **Step 8: Prune the class (b) files**

In each file, delete exactly the listed tests. Then delete imports, module constants, fixtures and helpers that no remaining test uses. A helper used by a remaining test stays. If a remaining test then fails with `NameError` on a deleted name, delete it too and name it in the commit message.

- `tests/conftest.py`: delete `import importlib.util`, the constants `TESTS`/`PKG`/`REL`/`GOLDEN`/`DATA`/`COMMON` (keep any that `_HideJax` or the fixtures still use), and the fixtures `golden_io` and `same_build`. Replace the module docstring with: `"""Test configuration for domainia: the NumPy-only lane switch and the backend / rng fixtures."""`
- `tests/test_dom_cov_api.py`:
  - delete `from domainia.nuisance import MarginalFactor, ProjectedFactor` and `WDM_SRC`;
  - set `EXPORTS = {"BlockTriCovFactor", "BlockTridiagonalCovariance", "ChannelStack", "CholFactor", "Covariance", "DenseCovariance", "DiagFactor", "DiagonalCovariance", "Evaluator", "EvolutionaryCovariance", "Factor", "QuadForm", "StackFactor", "as_evaluator", "dense_factor", "keep_mask", "quad_form"}`;
  - in `test_import_does_not_import_jax` use `code = ("import sys; import domainia.covariance; assert 'jax' not in sys.modules, sorted(m for m in sys.modules if m.startswith('jax'))")` and `env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"}`;
  - set `SUPPORTED = {cov.DiagFactor: {"solve", "whiten", "quad"}, cov.CholFactor: {"solve", "whiten", "quad"}, cov.BlockTriCovFactor: {"solve", "whiten", "quad"}, cov.StackFactor: {"solve", "whiten", "quad"}}`;
  - in `test_every_concrete_factor_class_is_in_the_table` the assertion becomes `assert exported == set(SUPPORTED)`;
  - drop "(and ``domainia.nuisance``)" from the module docstring.
- `tests/test_dom_cov_base.py`: delete `test_variances_from_callables_and_evaluators`, `test_g6_sinusoid_band_and_mid_raw` and `test_g6_gb_band_and_mid_raw`, plus `import domtest_inputs as T`.
- `tests/test_dom_ev_ported.py`: delete `test_schur_edge_exact_matches_the_dense_matrix` and `test_padded_marginal_is_the_data_block_of_the_extended_covariance`; drop `padded_factor` and `schur_factor` from the `domainia.covariance` import.
- `tests/test_dom_linalg_factors.py`: delete `test_factor_reproduces_direct_inversion`, `test_schur_solve_form_matches_dense`, `test_schur_solve_form_is_wt_schur_bit_for_bit`, `test_schur_scaled_form_is_gb_edge_schur_bit_for_bit`, `test_schur_scaled_solve_is_wg_diagonal_solve`, `test_schur_failures`, `test_padded_marginal_matches_the_dense_data_covariance`, `test_padded_factor_is_wt_padded_marginal_bit_for_bit`, `test_padded_with_a_diagonal_base_and_failures` and `test_the_duplicate_family_is_gone`, together with the fixtures `hybrid` and `padded` and the helpers only they use (`ragged`, `consts`, `PAD`, `EXT`, `NP`, `_to_time`, `_pad_consts`). Keep `test_cblock_is_gb_block_factor_bit_for_bit` and `test_cblock_failure_is_flagged` and their helpers. Update the module docstring's first paragraph so it no longer mentions Schur or padded factors.
- `tests/test_dom_linalg_jax.py`: delete `test_dense_cblock_schur_and_padded_factors_on_jax`; drop `padded_factor`, `schur_factor` from the import. In the subprocess near line 53, set the `PYTHONPATH` to `str(TESTS.parent / "src")` only.
- `tests/test_dom_maps_base.py`: delete `test_store_windows_are_the_prepared_bins`; the import becomes `from dom_db_inputs import DenseCov, jax_or_skip`.
- `tests/test_dom_maps_dirichlet.py`: delete `test_gb_whittle_cut_data_and_template_sites_agree`; the import becomes `from dom_db_inputs import DenseCov, jax_or_skip`.
- `tests/test_dom_maps_fourier_time.py`: delete `test_sinusoid_whittle_raw_bins_are_bitwise`, `test_sinusoid_whittle_cut_crop_and_bins_are_bitwise` and `test_jax_parity`; keep only the `SIN_*`/`DenseCov` names still used in the `dom_db_inputs` import.
- `tests/test_dom_maps_wdm.py`: delete `test_g6_sinusoid_cells_to_rounding`, `test_g6_sinusoid_padding_impulses_Wp_to_rounding` and `test_jax_parity_under_jit_with_the_map_as_argument`; drop `DATA`, `REL` and every golden helper from the `dom_db_inputs` import.
- `tests/test_dom_spectra.py`: delete `test_loglinear_L_and_alpha_are_the_wt_formulas`, `test_store_spectra_on_the_window_are_the_prepared_spectra` and `test_separable_psd_validates_its_components`; the spectra import becomes `from domainia.spectra import GenericPSD, TabulatedSpectrum`.
- `tests/test_dom_db_imports.py`: set `EXPORTS["domainia.spectra"] = ["GenericPSD", "TabulatedSpectrum"]`; the subprocess `PYTHONPATH` becomes `os.pathsep.join([str(PKG / "src"), os.environ.get("PYTHONPATH", "")])`.

- [ ] **Step 9: Check for leftovers**

Run:
```bash
cd ~/Documents/lisa/domainia
grep -rnE "LogLinearPSD|SeparablePSD|TableEnvelope|\bEnvelope\b|ExactTimeCovariance|OperatorFactor|CovOperator|VandermondeExactTime|FCFactor|ExactFourierCovariance|HermitianCovariance|HermitianFactor|PushedRows|CHEB_TOL|cheb_order|SchurEdge|SchurFactor|ZeroPadded|PadMask|PaddedFactor|padded_factor|schur_factor|Eq45Covariance|window_bins|domainia\.(nuisance|likelihood|density|sampling|literal|assets)|gb_case|GB_CASES|sinusoid_g6|sinusoid_data|fingerprint_gap|same_build|golden_io|sha256_array|read_json|to_float|domtest_inputs|dom_toys|scripts\.golden|builder=\"(bessel|gram|separable)\"" src tests
```
Expected: no output. A hit in a prose docstring of a surviving module, such as `covariance/base.py` mentioning `ProjectedFactor` or `domainia.sampling`, gets its sentence reworded or deleted. Rewording counts as part of this step.

- [ ] **Step 10: Run the whole suite**

Run: `$PY -m pytest -q`
Expected: all pass, 0 errors. That includes `tests/wilson/test_meyer_baseline.py`: `compute` picks `EvolutionaryCovariance` by name, and its arithmetic is the old `interp`/`direct` path.

- [ ] **Step 11: Commit** (sandbox disabled)

```bash
git add -A src tests
git commit -m "Cut paper-only and model-specific code; EvolutionaryCovariance; prune tests

Delete literal/, assets, nuisance, sampling, likelihood, density and the
exact_time/vandermonde/exact_fourier/hermitian/pushed/edges covariances. spectra
keeps GenericPSD and TabulatedSpectrum; variance keeps the GenericPSD branches;
eq45.py becomes evolutionary.py (interp, direct). Test files are reclassified
from their imports and assertions (table in the plan).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Synthetic property tests for the dropped golden coverage

**Goal:** Assert, on seeded synthetic inputs, the general properties the deleted golden tests carried: the Fourier→WDM cell maps, the generic variances and the evolutionary covariance builders, plus the two clean tests carried over from deleted files (`ChannelStack` and `DiagonalCovariance.cells`).

**Files:**
- Create: `tests/test_dom_maps_fourier_wdm.py`, `tests/test_dom_variance_generic.py`, `tests/test_dom_evolutionary_cov.py`, `tests/test_dom_cov_stack.py`

**Acceptance Criteria:**
- [ ] All four files pass under the verification command (the JAX test runs in the wdmrel env).
- [ ] No library code changes in this task (`git diff --stat src` is empty).

**Verify:** `$PY -m pytest -q tests/test_dom_maps_fourier_wdm.py tests/test_dom_variance_generic.py tests/test_dom_evolutionary_cov.py tests/test_dom_cov_stack.py` → all pass.

**Steps:**

- [ ] **Step 1: Write `tests/test_dom_maps_fourier_wdm.py`**

```python
"""FourierToWDM on synthetic records (replaces the GB-golden test_dom_maps_cells)."""

from __future__ import annotations

import numpy as np
import pytest
from dom_db_inputs import jax_or_skip, synthetic_record

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import FourierToWDM, TimeToWDM

NF, NT, DT = 16, 32, 0.25
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
BANDS = (3, 4, 5)


def _record() -> FourierDomain:
    return FourierDomain(N, DT, 0, N // 2 + 1, "rfft")


def _cells() -> CellSet:
    return CellSet.bands(GRID, BANDS)


def test_cells_are_the_record_scale_times_the_time_cells():
    x = synthetic_record(N, 3)
    got = FourierToWDM(_record(), WDMDomain(_cells())).apply(np.fft.rfft(x))
    want = TimeToWDM(TimeDomain(N, DT), WDMDomain(_cells())).apply(x)
    np.testing.assert_allclose(got, np.sqrt(N / (2 * DT)) * want, rtol=0, atol=1e-12 * np.max(np.abs(got)))


def test_fft_impl_matches_dense():
    z = np.fft.rfft(synthetic_record(N, 4))
    d = FourierToWDM(_record(), WDMDomain(_cells())).apply(z)
    f = FourierToWDM(_record(), WDMDomain(_cells()), "fft").apply(z)
    assert np.max(np.abs(f - d)) <= 1e-12 * np.max(np.abs(d))


def test_band_window_bins_feed_the_same_cells():
    z = np.fft.rfft(synthetic_record(N, 5))
    full = FourierToWDM(_record(), WDMDomain(_cells()))
    win = full.window
    sub = FourierToWDM(win, WDMDomain(_cells()))
    np.testing.assert_array_equal(sub.apply(z[win.k0:win.k0 + win.n]), full.apply(z))


def test_restrict_truncates_the_atoms_to_a_sub_window():
    full = FourierToWDM(_record(), WDMDomain(_cells()))
    win = full.window
    inner = FourierDomain(N, DT, win.k0 + 2, win.n - 4, "rfft")
    with pytest.raises(ValueError, match="restrict=True"):
        FourierToWDM(inner, WDMDomain(_cells()))
    m = FourierToWDM(inner, WDMDomain(_cells()), restrict=True)
    np.testing.assert_array_equal(m.C, full.C[:, 2:win.n - 2])


def test_adjoint_is_the_real_inner_product_adjoint():
    m = FourierToWDM(_record(), WDMDomain(_cells()))
    rng = np.random.default_rng(6)
    z = rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1)
    w = rng.standard_normal(_cells().size)
    lhs = float(np.dot(m.apply(z), w))
    rhs = float(np.real(np.vdot(m.adjoint(w), z)))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_edge_bands_are_refused():
    with pytest.raises(ValueError, match="interior bands"):
        FourierToWDM(_record(), WDMDomain(CellSet.bands(GRID, (0, 1))))


@pytest.mark.jax
def test_jax_parity_with_the_map_as_a_jit_argument():
    jax, be = jax_or_skip()
    import jax.numpy as jnp

    m = FourierToWDM(_record(), WDMDomain(_cells()))
    z = np.fft.rfft(synthetic_record(N, 7))
    want = m.apply(z)
    got = jax.jit(lambda mm, zz: mm.apply(zz))(m.on(be), jnp.asarray(z))
    np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-12 * np.max(np.abs(want)))
```

- [ ] **Step 2: Write `tests/test_dom_variance_generic.py`**

```python
"""BandAveragedVariance and MidpointVariance of a GenericPSD (replaces the golden
test_dom_variance), and DiagonalCovariance over them."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import DiagonalCovariance
from domainia.domains import CellSet
from domainia.spectra import GenericPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
T = N * DT
GRID = wilson.Grid(NF, NT, DT)
CELLS = CellSet.all_active(GRID)
CUT = CELLS.drop_columns(3, 1)
PSD = GenericPSD(lambda f, t, nu: (1.0 + np.asarray(f, float) ** 2) ** nu[0]
                 * (1.0 + nu[1] * np.asarray(t, float) / T))
NU = np.array([-0.7, 0.3])


def test_band_average_is_the_kernel_band_variance_at_the_cells():
    full = wilson.noise.evolutionary_band_variance(NF, NT, DT, PSD.bind(NU))
    got = BandAveragedVariance(PSD, CELLS)(NU)
    np.testing.assert_array_equal(got, full[np.asarray(CELLS.m), np.asarray(CELLS.n)])


def test_midpoint_is_half_the_psd_at_the_band_centre_and_cell_time():
    m, n = np.asarray(CELLS.m), np.asarray(CELLS.n)
    f = wilson.band_frequencies(NF, DT)[m]
    t = n * wilson.delta_T(NF, DT)
    np.testing.assert_array_equal(MidpointVariance(PSD, CELLS)(NU), 0.5 * PSD(f, t, NU))


@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
def test_cut_cells_are_the_full_vector_restricted(cls):
    full = cls(PSD, CELLS)(NU)
    np.testing.assert_array_equal(cls(PSD, CUT)(NU), full[CELLS.index_of(CUT)])


@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
def test_only_generic_psds_are_accepted(cls):
    with pytest.raises(TypeError, match="GenericPSD"):
        cls(object(), CELLS)


def test_diagonal_covariance_of_a_variance_carries_its_cells_and_factors():
    cov = DiagonalCovariance(BandAveragedVariance(PSD, CUT), CUT.size)
    assert cov.cells == CUT
    v = BandAveragedVariance(PSD, CUT)(NU)
    F = cov.factor(NU)
    b = np.random.default_rng(1).standard_normal(CUT.size)
    assert float(F.logdet) == pytest.approx(float(np.sum(np.log(v))), rel=1e-13)
    assert float(F.quad(b)) == pytest.approx(float(np.sum(b * b / v)), rel=1e-13)


def test_callable_and_array_variances_carry_no_cells():
    assert DiagonalCovariance(np.ones(3), 3).cells is None
    assert DiagonalCovariance(lambda nu, be: be.asarray(np.ones(3)), 3).cells is None
```

- [ ] **Step 3: Write `tests/test_dom_evolutionary_cov.py`**

```python
"""EvolutionaryCovariance (WDM II Eq. 45) for a callable PSD: builders, assemblies and
the stationary limit (replaces test_dom_eq45)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import EvolutionaryCovariance, keep_mask
from domainia.covariance.blocktri import blocks_to_dense, to_layout
from domainia.domains import CellSet
from domainia.spectra import GenericPSD
from domainia.wilson.backend import NUMPY
from domainia.wilson.evolutionary import eq45_blocks_cheb, eq45_coeffs_interp, eq45_plan
from domainia.wilson.matrix import build_C_wdm_active

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
T_REC = N * DT
NU = np.array([-1.7, 0.9])


def _drifting_power_law(f, t, nu):
    """``max(f, 0.5) ** (a_c + a_d (t / T - 1/2))``: a drifting power law as a plain callable."""
    alpha = nu[0] + nu[1] * (np.asarray(t, float) / T_REC - 0.5)
    return np.exp(alpha * np.log(np.maximum(np.asarray(f, float), 0.5)))


PSD = GenericPSD(_drifting_power_law)


def _cells(Nt: int = NT) -> CellSet:
    return CellSet.all_active(wilson.Grid(NF, Nt, DT))


def _dense_blocks(cov: EvolutionaryCovariance, nu: np.ndarray) -> np.ndarray:
    d, u = to_layout(*cov.blocks(nu), cov.sizes, "ragged", NUMPY)
    return np.asarray(blocks_to_dense(d, u, NUMPY))


@pytest.mark.parametrize("which", ["all", "cut", "padded"])
def test_interp_matches_the_direct_sum(which):
    Nt = NT + 4 if which == "padded" else NT
    cells = _cells(Nt)
    if which == "cut":
        cells = cells.drop_columns(3, 1)
    t_max = Nt * NF * DT if which == "padded" else None
    ref = EvolutionaryCovariance(PSD, cells, builder="direct", assembly="dense", t_max=t_max).dense(NU)
    got = _dense_blocks(EvolutionaryCovariance(PSD, cells, builder="interp", t_max=t_max), NU)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_assemblies_agree(rng):
    cells = _cells()
    blk = EvolutionaryCovariance(PSD, cells)
    den = EvolutionaryCovariance(PSD, cells, assembly="dense", chol_impl="scipy", symmetrize=False)
    y = rng.standard_normal(N)
    Fb, Fd = blk.factor(NU), den.factor(NU)
    assert np.array_equal(blk.dense(NU), den.dense(NU))
    assert float(Fb.logdet) == pytest.approx(float(Fd.logdet), rel=1e-13)
    assert float(Fb.quad(y)) == pytest.approx(float(Fd.quad(y)), rel=1e-12)


def test_interp_is_the_kernel_chebyshev_path():
    plan = eq45_plan(NF, NT, DT)
    want = eq45_blocks_cheb(plan, eq45_coeffs_interp(lambda f, t: PSD(f, t, NU), plan.f, plan.t_max,
                                                     plan.P, k=plan.k_used))
    got = EvolutionaryCovariance(PSD, _cells()).blocks(NU)
    assert all(np.array_equal(a, b) for a, b in zip(got[0], want[0], strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(got[1], want[1], strict=True))


@pytest.mark.parametrize("ac", [-2.0, -0.7])
def test_stationary_limit_is_the_exact_covariance(ac):
    nu = np.array([ac, 0.0])
    exact = build_C_wdm_active(NF, NT, DT, lambda f: PSD(f, 0.0, nu))
    got = _dense_blocks(EvolutionaryCovariance(PSD, _cells()), nu)
    assert np.max(np.abs(got - exact)) < 1e-10 * np.max(np.abs(exact))


def test_arguments_are_checked():
    cells = _cells()
    with pytest.raises(ValueError, match="unknown builder"):
        EvolutionaryCovariance(PSD, cells, builder="bessel")
    with pytest.raises(ValueError, match="callable"):
        EvolutionaryCovariance(object(), cells)
    with pytest.raises(ValueError, match="assembly"):
        EvolutionaryCovariance(PSD, cells, assembly="banded")
    with pytest.raises(ValueError, match="direct"):
        EvolutionaryCovariance(PSD, cells, builder="direct")
    assert not EvolutionaryCovariance(PSD, cells).jittable


def test_keep_mask_selects_the_cells_in_band_order():
    cells = _cells().drop_columns(2, 2)
    keep = keep_mask(cells)
    assert int(keep.sum()) == cells.size
    np.testing.assert_array_equal(np.flatnonzero(keep), cells.packed_positions())
```

- [ ] **Step 4: Write `tests/test_dom_cov_stack.py`** (carried over from the deleted `test_dom_projected_stack.py`)

```python
"""ChannelStack: one covariance per channel, factors split and summed."""

from __future__ import annotations

import numpy as np
import pytest

from domainia.covariance import ChannelStack, DiagonalCovariance, StackFactor


def test_channel_stack_splits_and_sums(backend):
    covs = [DiagonalCovariance(lambda nu, be, s=s: be.xp.exp(nu[0]) * be.asarray(np.full(3, s)), 3)
            for s in (1.0, 2.0)]
    st = ChannelStack(covs, 2)
    nu = np.array([0.1, 9.0, 0.2, 9.0])
    F = st.factor(backend.asarray(nu), backend, jit=False)
    assert isinstance(F, StackFactor) and len(F.channels) == 2 and st.n == 6 and F.n_u == 0
    assert st.n_u == 0 and st.state_key(nu) is None
    want = 3 * 0.1 + 3 * (np.log(2.0) + 0.2)
    assert float(F.logdet) == pytest.approx(want, rel=1e-14)
    b = np.ones((2, 3))
    assert float(F.quad(backend.asarray(b))) == pytest.approx(3 / np.exp(0.1) + 3 / (2 * np.exp(0.2)),
                                                             rel=1e-14)
    with pytest.raises(ValueError, match="entries"):
        st.factor(np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        ChannelStack(covs, -1)
```

- [ ] **Step 5: Run them**

Run: `$PY -m pytest -q tests/test_dom_maps_fourier_wdm.py tests/test_dom_variance_generic.py tests/test_dom_evolutionary_cov.py tests/test_dom_cov_stack.py`
Expected: all pass. These assert existing behaviour, so a failure points at a wrong test premise. Read the relevant docstring (`FourierToWDM`, `evolutionary_band_variance`) and fix the test, never the library. Record any premise you corrected in the commit message.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
git add tests/test_dom_maps_fourier_wdm.py tests/test_dom_variance_generic.py tests/test_dom_evolutionary_cov.py tests/test_dom_cov_stack.py
git commit -m "Synthetic property tests for the coverage the golden tests carried

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 4: Admissible windows (`domainia.wilson.windows`)

**Goal:** Add `Window`, `Meyer(d)`, `RampWindow`, `SampledWindow`, `MEYER`, `WindowError` and `sample_window`. Sampling is validated, LRU-cached and read-only, and `Meyer(d).sample` equals `phi_centered(..., d)` bit for bit. Add the test windows `smoothstep`, `wide_profile` and `ALT_WINDOW` to `domainia.testing`. The kernels are not threaded yet.

**Files:**
- Create: `src/domainia/wilson/windows.py`, `tests/wilson/test_windows.py`
- Modify: `src/domainia/wilson/__init__.py` (exports), `src/domainia/testing.py` (test windows)

**Acceptance Criteria:**
- [ ] `Meyer(d).sample(Nf, Nt, dt)` is `np.array_equal` to `filter.phi_centered(Nf, Nt, dt, d)` for d = 1…8 on every test grid.
- [ ] A repeat `sample` call returns the same object, and the array is not writeable.
- [ ] Windows that break evenness, support (inside or outside the slab), the partition of unity or the ramp rule each raise `WindowError`. The message contains the condition name, the window name and a bin or point.
- [ ] `sample_window` raises `TypeError` for a non-window and validates third-party `Window` implementations.
- [ ] `MEYER == Meyer() == Meyer(4)` with equal hashes; `RampWindow`/`SampledWindow` compare by `(function identity, name)`.

**Verify:** `$PY -m pytest -q tests/wilson/test_windows.py` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests `tests/wilson/test_windows.py`**

```python
"""Admissible Wilson windows: Meyer(d), RampWindow, SampledWindow (spec section 2)."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from domainia import wilson
from domainia.testing import ALT_WINDOW, smoothstep, wide_profile
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow, Window, WindowError
from domainia.wilson.filter import nu_d, phi_centered
from domainia.wilson.windows import sample_window

GRIDS = [(8, 16, 1.0), (16, 32, 0.25), (32, 8, 2.0), (64, 256, 2.0**-10)]


@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
@pytest.mark.parametrize("d", range(1, 9))
def test_meyer_sample_is_phi_centered_bit_for_bit(Nf, Nt, dt, d):
    assert np.array_equal(Meyer(d).sample(Nf, Nt, dt), phi_centered(Nf, Nt, dt, d))


def test_samples_are_cached_and_read_only():
    a = MEYER.sample(8, 16, 1.0)
    assert MEYER.sample(8, 16, 1.0) is a
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0] = 1.0


def test_meyer_identity():
    assert MEYER == Meyer() == Meyer(4) and hash(MEYER) == hash(Meyer(4))
    assert Meyer(5) != MEYER
    assert MEYER.name == "meyer(d=4)"
    assert isinstance(MEYER, Window)


@pytest.mark.parametrize("d, err", [(0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)])
def test_meyer_rejects_a_bad_d(d, err):
    with pytest.raises(err):
        Meyer(d)


def test_a_ramp_window_with_the_meyer_ramp_is_meyer_bit_for_bit():
    w = RampWindow(partial(nu_d, d=4), "nu4")
    for Nf, Nt, dt in GRIDS:
        assert np.array_equal(w.sample(Nf, Nt, dt), MEYER.sample(Nf, Nt, dt))


@pytest.mark.parametrize("window", [MEYER, Meyer(1), Meyer(8), RampWindow(smoothstep, "smoothstep"),
                                    ALT_WINDOW], ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_admissible_windows_are_even_supported_and_a_partition_of_unity(window, Nf, Nt, dt):
    phi = window.sample(Nf, Nt, dt)
    h = Nt // 2
    inv_dF = 1.0 / wilson.delta_F(Nf, dt)
    assert phi.shape == (Nt,) and phi.dtype == np.float64 and phi[0] == 0.0
    assert np.array_equal(phi[h + 1:], phi[h - 1:0:-1])
    np.testing.assert_allclose(phi[h:] ** 2 + phi[:h] ** 2, inv_dF, rtol=1e-12, atol=0)


def test_the_alt_window_differs_from_meyer():
    assert not np.allclose(ALT_WINDOW.sample(8, 16, 1.0), MEYER.sample(8, 16, 1.0))


def test_evenness_failure_names_the_condition_window_and_bin():
    w = SampledWindow(lambda u: wide_profile(u) * np.where(np.asarray(u) < 0, 1.0 - 1e-3, 1.0), "lopsided")
    with pytest.raises(WindowError, match=r"lopsided.*evenness.*bin"):
        w.sample(8, 16, 1.0)


def test_support_failure_at_the_slab_edge_is_caught_at_construction():
    # A SampledWindow can never reach the sample-time bin-0 support branch: psi(+-1) is
    # checked at construction.  That branch is exercised by `Flat` in
    # test_sample_window_validates_third_party_windows; do not re-add a boxcar here.
    with pytest.raises(WindowError, match=r"boxcar.*support"):
        SampledWindow(lambda u: np.where(np.abs(np.asarray(u)) <= 1.0, 0.5, 0.0), "boxcar")


def test_support_failure_outside_the_slab_is_caught_at_construction():
    def leaky(u):
        a = np.abs(np.asarray(u, dtype=float))
        return wide_profile(u) + np.where((a >= 1.25) & (a <= 1.75), 1e-3, 0.0)

    with pytest.raises(WindowError, match=r"leaky.*support"):
        SampledWindow(leaky, "leaky")


def test_partition_failure():
    w = SampledWindow(lambda u: 1.01 * wide_profile(u), "loud")
    with pytest.raises(WindowError, match=r"loud.*partition of unity.*bin"):
        w.sample(8, 16, 1.0)


@pytest.mark.parametrize("ramp", [lambda x: np.asarray(x) ** 2,            # nu(x) + nu(1-x) != 1
                                  lambda x: 0.5 + 0.0 * np.asarray(x)])    # nu(0) != 0
def test_ramp_failures(ramp):
    with pytest.raises(WindowError, match="ramp"):
        RampWindow(ramp, "bad-ramp")


def test_windows_compare_by_function_and_name():
    assert SampledWindow(wide_profile, "a") == SampledWindow(wide_profile, "a")
    assert hash(SampledWindow(wide_profile, "a")) == hash(SampledWindow(wide_profile, "a"))
    assert SampledWindow(wide_profile, "a") != SampledWindow(wide_profile, "b")
    assert SampledWindow(wide_profile, "a") != SampledWindow(lambda u: wide_profile(u), "a")
    assert RampWindow(smoothstep, "s") == RampWindow(smoothstep, "s")


@pytest.mark.parametrize("make", [lambda: SampledWindow(wide_profile, ""), lambda: SampledWindow(3, "x"),
                                  lambda: RampWindow(smoothstep, None), lambda: RampWindow("x", "x")])
def test_bad_names_and_functions_are_type_errors(make):
    with pytest.raises(TypeError):
        make()


def test_sample_window_rejects_non_windows():
    with pytest.raises(TypeError, match="Window"):
        sample_window("meyer", 8, 16, 1.0)


def test_sample_window_validates_third_party_windows():
    class Flat:
        name = "flat"

        def sample(self, Nf, Nt, dt):
            return np.ones(Nt)

    with pytest.raises(WindowError, match=r"flat.*support.*bin 0"):
        sample_window(Flat(), 8, 16, 1.0)


def test_sample_window_returns_builtin_samples_unchanged():
    assert sample_window(MEYER, 8, 16, 1.0) is MEYER.sample(8, 16, 1.0)


def test_an_invalid_grid_is_a_value_error():
    with pytest.raises(ValueError):
        MEYER.sample(7, 16, 1.0)
    with pytest.raises(ValueError):
        MEYER.sample(8, 16, -1.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest -q tests/wilson/test_windows.py`
Expected: collection error, `ImportError: cannot import name 'ALT_WINDOW' from 'domainia.testing'` (and `MEYER` from `domainia.wilson`).

- [ ] **Step 3: Write `src/domainia/wilson/windows.py`**

```python
"""Admissible Wilson windows.

A Wilson basis on the half-shifted lattice is orthonormal when its window
:math:`\\hat\\phi` is *admissible* on the slab :math:`f \\in [-\\Delta F, \\Delta F)`,
:math:`\\Delta F = 1/(2 N_f \\delta t)`:

1. real and even;
2. zero for :math:`|f| \\ge \\Delta F` (compact support inside one slab: the transform
   stays exact and every band covariance block-tridiagonal);
3. a partition of unity on the lattice,
   :math:`\\hat\\phi(f)^2 + \\hat\\phi(f - \\Delta F)^2 = 1/\\Delta F` on :math:`[0, \\Delta F]`
   (the Hz normalisation of :func:`domainia.wilson.filter.phi_centered`).

The WDM transform is the Wilson basis with the Meyer window, :data:`MEYER`.  Every
kernel takes ``window=`` and samples it through :func:`sample_window`.  Built-in
windows validate each ``(window, Nf, Nt, dt)`` once, cache the read-only samples in a
bounded LRU, and raise :class:`WindowError` (naming the condition, the window and the
worst bin) when a condition fails.

Examples
--------
>>> import numpy as np; from domainia import wilson
>>> phi = wilson.MEYER.sample(8, 16, 1.0)
>>> bool(np.array_equal(phi, wilson.filter.phi_centered(8, 16, 1.0)))
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .filter import phi_centered
from .grid import delta_F, validate_dt, validate_grid

__all__ = ["MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError", "sample_window"]

type Array = NDArray[np.float64]

#: Relative tolerance of the partition-of-unity check (relative to 1/dF).
PARTITION_RTOL = 1e-12
#: Absolute tolerance of the ramp symmetry check nu(x) + nu(1 - x) = 1.
RAMP_ATOL = 1e-12
_RAMP_POINTS = 257
_SUPPORT_POINTS = 32            # per side: 64 points of |u| in [1, 2]
_CACHE_SIZE = 256


class WindowError(ValueError):
    """An inadmissible window.  The message names the condition, the window and the bin."""


@runtime_checkable
class Window(Protocol):
    """A Wilson window: frozen and hashable (static under ``jax.jit``)."""

    @property
    def name(self) -> str: ...

    def sample(self, Nf: int, Nt: int, dt: float) -> Array:
        """Centred ``(Nt,)`` samples in ``phi_centered``'s ordering and Hz normalisation."""
        ...


def _fail(name: str, condition: str, Nf: int, Nt: int, dt: float, detail: str) -> None:
    raise WindowError(f"window {name!r} fails the {condition} condition on the "
                      f"(Nf={Nf}, Nt={Nt}, dt={dt!r}) slab: {detail}")


def _as_samples(name: str, raw: Any, Nf: int, Nt: int, dt: float) -> np.ndarray:
    a = np.asarray(raw)
    if a.dtype.kind not in "fiu":
        _fail(name, "real", Nf, Nt, dt, f"samples have dtype {a.dtype}")
    return np.array(a, dtype=np.float64)                         # a fresh float64 copy


def _check_samples(name: str, phi: np.ndarray, Nf: int, Nt: int, dt: float) -> None:
    """The admissibility conditions on the sampled slab (module docstring)."""
    if phi.shape != (Nt,):
        _fail(name, "shape", Nf, Nt, dt, f"expected ({Nt},) samples, got {phi.shape}")
    if not np.all(np.isfinite(phi)):
        k = int(np.flatnonzero(~np.isfinite(phi))[0])
        _fail(name, "finiteness", Nf, Nt, dt, f"bin {k} is {phi[k]!r}")
    h = Nt // 2
    if phi[0] != 0.0:
        _fail(name, "support", Nf, Nt, dt, f"bin 0 (f = -dF) is {phi[0]!r}, not 0")
    diff = np.abs(phi[h + 1:] - phi[h - 1:0:-1])
    if diff.size and np.max(diff) > 0.0:
        j = int(np.argmax(diff)) + 1
        _fail(name, "evenness", Nf, Nt, dt,
              f"worst at bin {h + j}: {phi[h + j]!r} but its mirror bin {h - j} is {phi[h - j]!r}")
    inv_dF = 1.0 / delta_F(Nf, dt)
    err = np.abs(phi[h:] ** 2 + phi[:h] ** 2 - inv_dF)
    j = int(np.argmax(err))
    if err[j] > PARTITION_RTOL * inv_dF:
        _fail(name, "partition of unity", Nf, Nt, dt,
              f"worst at bin {h + j}: phi(f)^2 + phi(f - dF)^2 = {phi[h + j] ** 2 + phi[j] ** 2!r}, "
              f"1/dF = {inv_dF!r} (relative error {err[j] / inv_dF:.3e})")


@lru_cache(maxsize=_CACHE_SIZE)
def _checked(window: _Admissible, Nf: int, Nt: int, dt: float) -> Array:
    phi = _as_samples(window.name, window._samples(Nf, Nt, dt), Nf, Nt, dt)
    _check_samples(window.name, phi, Nf, Nt, dt)
    phi.setflags(write=False)
    return phi


class _Admissible:
    """The built-in windows' ``sample``: validate the grid, then the cached checked samples."""

    __slots__ = ()
    name: str

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        raise NotImplementedError

    def sample(self, Nf: int, Nt: int, dt: float) -> Array:
        """Validated, read-only, cached samples (centred, Hz normalisation)."""
        validate_grid(Nf, Nt)
        validate_dt(dt)
        return _checked(self, int(Nf), int(Nt), float(dt))


def _check_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError(f"window name must be a non-empty string; got {name!r}")


def _check_callable(what: str, fn: Any) -> None:
    if not callable(fn):
        raise TypeError(f"{what} must be callable; got {type(fn).__name__}")


def _meyer_geometry(Nf: int, Nt: int, dt: float) -> tuple[np.ndarray, float, float, float]:
    """``(|f|, A, B, amp)`` exactly as ``phi_centered`` computes them."""
    dF = delta_F(Nf, dt)
    T = Nf * Nt * dt
    f = (np.arange(Nt) - Nt // 2) / T
    return np.abs(f), dF / 4.0, dF / 2.0, 1.0 / np.sqrt(dF)


@dataclass(frozen=True)
class Meyer(_Admissible):
    """The Meyer window: flat to dF/4, ``cos(pi/2 nu_d)`` rolloff to 3 dF/4, ``nu_d = I_x(d, d)``.

    ``Meyer(d).sample`` is :func:`domainia.wilson.filter.phi_centered` ``(Nf, Nt, dt, d)``
    bit for bit; ``Meyer()`` (d = 4) is the WDM transform's window.
    """

    d: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.d, bool) or not isinstance(self.d, (int, np.integer)):
            raise TypeError(f"Meyer d must be an int; got {type(self.d).__name__}")
        if self.d < 1:
            raise ValueError(f"Meyer d must be >= 1; got {self.d}")
        object.__setattr__(self, "d", int(self.d))

    @property
    def name(self) -> str:
        return f"meyer(d={self.d})"

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        return phi_centered(Nf, Nt, dt, self.d)


@dataclass(frozen=True)
class RampWindow(_Admissible):
    """Meyer's geometry with a user ramp ``nu`` on [0, 1]: ``nu(0) = 0``, ``nu(1) = 1`` and
    ``nu(x) + nu(1 - x) = 1`` (checked on 257 points at construction).  Two RampWindows are
    equal when they hold the same function object and name."""

    ramp: Callable[[Array], Array]
    name: str

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_callable("ramp", self.ramp)
        x = np.linspace(0.0, 1.0, _RAMP_POINTS)                    # dyadic: 1 - x[k] == x[-1-k]
        nu = np.asarray(self.ramp(x))
        if nu.shape != x.shape or nu.dtype.kind not in "fiu" or not np.all(np.isfinite(nu)):
            raise WindowError(f"window {self.name!r} fails the ramp condition: the ramp must map "
                              f"{x.size} points to {x.size} finite real values")
        nu = nu.astype(np.float64)
        if nu[0] != 0.0 or nu[-1] != 1.0:
            raise WindowError(f"window {self.name!r} fails the ramp condition: nu(0) = {nu[0]!r} "
                              f"and nu(1) = {nu[-1]!r}; need exactly 0 and 1")
        err = np.abs(nu + nu[::-1] - 1.0)
        k = int(np.argmax(err))
        if err[k] > RAMP_ATOL:
            raise WindowError(f"window {self.name!r} fails the ramp condition: nu(x) + nu(1 - x) - 1 "
                              f"= {err[k]:.3e} at x = {x[k]!r}")

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        f_abs, A, B, amp = _meyer_geometry(Nf, Nt, dt)
        nu = np.asarray(self.ramp(np.clip((f_abs - A) / B, 0.0, 1.0)), dtype=np.float64)
        out = np.where(f_abs < A, amp, 0.0)
        rolloff = (f_abs >= A) & (f_abs < A + B)
        return np.where(rolloff, amp * np.cos(np.pi / 2.0 * nu), out)


@dataclass(frozen=True)
class SampledWindow(_Admissible):
    """Any dimensionless profile ``psi(u)``, ``u = f / dF``: flat value 1,
    ``psi(u)^2 + psi(u - 1)^2 = 1`` on [0, 1] and ``psi = 0`` for ``|u| >= 1``.

    ``sample`` returns ``psi(f / dF) / sqrt(dF)`` (the Hz normalisation).  ``psi`` is also
    evaluated on 64 points of ``|u|`` in [1, 2] at construction and must be exactly 0
    there: mass outside the slab would be truncated silently.  Two SampledWindows are equal
    when they hold the same function object and name.
    """

    fn: Callable[[Array], Array]
    name: str

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_callable("fn", self.fn)
        u = np.linspace(1.0, 2.0, _SUPPORT_POINTS)
        u = np.concatenate([-u[::-1], u])
        psi = np.asarray(self.fn(u))
        if psi.shape != u.shape:
            raise WindowError(f"window {self.name!r} fails the support condition: fn maps {u.size} "
                              f"points to shape {psi.shape}")
        bad = np.flatnonzero(psi != 0.0)
        if bad.size:
            k = int(bad[0])
            raise WindowError(f"window {self.name!r} fails the support condition: psi({u[k]!r}) = "
                              f"{psi[k]!r}, but a window must vanish for |f| >= dF")

    def _samples(self, Nf: int, Nt: int, dt: float) -> Any:
        dF = delta_F(Nf, dt)
        T = Nf * Nt * dt
        u = (np.arange(Nt) - Nt // 2) / T / dF
        return np.asarray(self.fn(u), dtype=np.float64) / np.sqrt(dF)


def sample_window(window: Any, Nf: int, Nt: int, dt: float) -> Array:
    """The kernels' single entry point: the validated, read-only samples of ``window``.

    Built-in windows return their cached samples.  Any other object implementing
    :class:`Window` is validated on every call; anything else raises ``TypeError``.
    """
    if isinstance(window, _Admissible):
        return window.sample(Nf, Nt, dt)
    if not isinstance(window, Window):
        raise TypeError(f"window must be a domainia.wilson Window (Meyer, RampWindow, "
                        f"SampledWindow); got {type(window).__name__}")
    validate_grid(Nf, Nt)
    validate_dt(dt)
    name = str(window.name)
    phi = _as_samples(name, window.sample(Nf, Nt, dt), int(Nf), int(Nt), float(dt))
    _check_samples(name, phi, int(Nf), int(Nt), float(dt))
    phi.setflags(write=False)
    return phi


#: The WDM transform's window: Meyer with d = 4.
MEYER = Meyer()
```

Note: `_checked` names `_Admissible` in its annotation before the class is defined. That is fine because of `from __future__ import annotations`.

- [ ] **Step 4: Export the windows from `domainia.wilson`**

In `src/domainia/wilson/__init__.py`, add `windows,` to the `from . import (...)` submodule list. Below the `from .transform import (...)` block, add:

```python
from .windows import MEYER, Meyer, RampWindow, SampledWindow, Window, WindowError
```
and add `"MEYER"`, `"Meyer"`, `"RampWindow"`, `"SampledWindow"`, `"Window"`, `"WindowError"`, `"windows"` to `__all__`. In the module docstring, add a "Windows (in :mod:`domainia.wilson.windows`)" bullet listing them.

- [ ] **Step 5: Add the test windows to `src/domainia/testing.py`**

Add `from domainia.wilson.windows import SampledWindow` to the imports, and `"ALT_WINDOW"`, `"smoothstep"`, `"wide_profile"` to `__all__`. Append:

```python
def smoothstep(x: Any) -> Array:
    """``3 x^2 - 2 x^3`` on [0, 1] (clipped): a ramp with ``nu(x) + nu(1 - x) = 1``."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def wide_profile(u: Any) -> Array:
    """A non-Meyer admissible profile ``psi(u)``, ``u = f / dF``: 1 for ``|u| < 1/8``, a
    smoothstep cosine rolloff to 0 at ``|u| = 7/8``, 0 beyond.  On [0, 1],
    ``psi(u)^2 + psi(u - 1)^2 = 1`` because ``smoothstep(1 - x) = 1 - smoothstep(x)``."""
    a = np.abs(np.asarray(u, dtype=float))
    x = np.clip((a - 0.125) / 0.75, 0.0, 1.0)
    return np.where(a < 0.125, 1.0, np.where(a < 0.875, np.cos(np.pi / 2.0 * smoothstep(x)), 0.0))


#: A non-default admissible window for tests (other flat width, other ramp than Meyer).
ALT_WINDOW = SampledWindow(wide_profile, "wide-smoothstep")
```

- [ ] **Step 6: Run the tests**

Run: `$PY -m pytest -q tests/wilson/test_windows.py src/domainia/wilson/windows.py`
Expected: all pass, the module doctest included. If only `test_a_ramp_window_with_the_meyer_ramp_is_meyer_bit_for_bit` fails, `RampWindow._samples` does not repeat `phi_centered`'s arithmetic. Open `src/domainia/wilson/filter.py::phi_centered` and mirror its operations in `_meyer_geometry` and `RampWindow._samples`, in the same order. Never change `Meyer._samples`, which delegates to `phi_centered`.

- [ ] **Step 7: Run the whole suite and commit** (sandbox disabled)

Run: `$PY -m pytest -q` → all pass.

```bash
git add src/domainia/wilson/windows.py src/domainia/wilson/__init__.py src/domainia/testing.py tests/wilson/test_windows.py
git commit -m "Admissible Wilson windows: Meyer(d), RampWindow, SampledWindow

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Thread `window=` through the transform and matrix kernels

**Goal:** Every function in `domainia.wilson.transform` and `domainia.wilson.matrix` that reaches the window takes the keyword `window: Window = MEYER` and samples it through `sample_window`. With the default, every output stays bit-identical to the baseline. Any admissible window gives an orthonormal basis with block-tridiagonal stationary covariance.

**Files:**
- Modify: `src/domainia/wilson/transform.py`, `src/domainia/wilson/matrix.py`
- Create: `tests/wilson/test_window_threading.py`
- Modify: `tests/wilson/test_meyer_baseline.py`

**Acceptance Criteria:**
- [ ] Every function listed in Step 3 accepts `window=` and passes it by keyword to every window consumer it calls.
- [ ] `test_no_kernel_call_drops_the_window` passes for `wilson/transform.py` and `wilson/matrix.py`: no call to a consumer without `window=` (or `base=`), and no `phi_centered(` call.
- [ ] A counting third-party window is sampled by every `transform.*` and `matrix.*` baseline case.
- [ ] Explicit `window=MEYER` reproduces those baseline arrays bit for bit, and the default still does.
- [ ] For `MEYER`, `Meyer(1)`, `Meyer(8)`, `RampWindow(smoothstep)` and `ALT_WINDOW` on three grids: `A Aᵀ = dt·I` (atol 1e-12·dt), time round trip ≤ 1e-12, ‖w‖² = dt‖x‖² (rel 1e-12), and the stationary covariance vanishes (≤ 1e-12 relative) for |m − m'| > 1.

**Verify:** `$PY -m pytest -q tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests `tests/wilson/test_window_threading.py`**

```python
"""Every window consumer samples the window it is given, and every admissible window
gives an orthonormal Wilson basis with block-tridiagonal stationary covariance."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import domainia
from domainia import wilson
from domainia.testing import ALT_WINDOW, smoothstep
from domainia.wilson import MEYER, Meyer, RampWindow

from .meyer_cases import psd, wilson_cases

#: Case-name prefixes whose kernels take ``window=`` (extended task by task).
THREADED = ("transform.", "matrix.")
#: Source files (relative to src/domainia) whose every consumer call must pass ``window=``.
SCANNED = ("wilson/transform.py", "wilson/matrix.py")

#: Every function that samples the window, directly or through another consumer.
CONSUMERS = frozenset({
    "_gather_bands", "forward", "forward_many", "forward_time", "forward_time_many", "inverse",
    "inverse_many", "inverse_time", "inverse_time_many", "base_atom_spectra", "atom_spectrum",
    "band_atoms", "forward_bands", "inverse_bands",
    "build_W", "active_W", "build_C_wdm", "build_C_wdm_active",
    "_slab_K", "build_lambda_blocks_full_active", "build_K_full_active", "psd_from_K",
    "apply_lambda_from_psd", "psd_from_lambda_blocks", "_forward", "_inverse", "_forward_wdm",
    "_inverse_wdm",
    "band_averaged_variance", "evolutionary_band_variance", "empirical_variance",
    "energy_fraction", "apply_time_mask", "wrap_fractions", "wrap_cut_columns", "exact_covariance",
    "eq45_plan",
    "gaussian", "gaussian_from_func", "loglike_gaussian", "loglike_gaussian_many",
    "dense_W", "wdm_congruence", "eq45_dense", "_dense_atoms",
})
WINDOWS = [MEYER, Meyer(1), Meyer(8), RampWindow(smoothstep, "smoothstep"), ALT_WINDOW]
GRIDS = [(8, 16, 1.0), (16, 8, 0.25), (4, 32, 2.0)]


class CountingWindow:
    """A third-party Window (not a built-in) that counts its ``sample`` calls."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    @property
    def name(self) -> str:
        return f"counting({self.inner.name})"

    def sample(self, Nf, Nt, dt):
        self.calls += 1
        return self.inner.sample(Nf, Nt, dt)


def _missing_window(path: Path) -> list[tuple[int, str]]:
    """``(line, name)`` of every consumer call without ``window=``/``base=``, and every
    ``phi_centered`` call."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name == "phi_centered" or (
                name in CONSUMERS and not any(k.arg in ("window", "base") for k in node.keywords)):
            bad.append((node.lineno, name))
    return bad


@pytest.mark.parametrize("rel", SCANNED)
def test_no_kernel_call_drops_the_window(rel):
    assert _missing_window(Path(domainia.__file__).parent / rel) == []


_CASES = [c for c in wilson_cases(wilson, 8, 16, 1.0) if c[0].startswith(THREADED)]


@pytest.mark.parametrize("case", [c for _, c in _CASES], ids=[n for n, _ in _CASES])
def test_every_kernel_samples_the_window_it_is_given(case):
    w = CountingWindow(ALT_WINDOW)
    case({"window": w})
    assert w.calls > 0


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_every_admissible_window_gives_an_orthonormal_basis(window, Nf, Nt, dt):
    A = wilson.matrix.active_W(Nf, Nt, dt, window=window)
    np.testing.assert_allclose(A @ A.T, dt * np.eye(A.shape[0]), rtol=0, atol=1e-12 * dt)


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_time_round_trip_and_energy(window, Nf, Nt, dt):
    x = np.random.default_rng(7).standard_normal(Nf * Nt)
    w = wilson.forward_time(x, Nf, Nt, dt, window=window)
    np.testing.assert_allclose(wilson.inverse_time(w, Nf, Nt, dt, window=window), x, rtol=0, atol=1e-12)
    assert float(np.sum(w ** 2)) == pytest.approx(dt * float(np.sum(x ** 2)), rel=1e-12)
    many = wilson.forward_time_many(np.stack([x, -x]), Nf, Nt, dt, window=window)
    np.testing.assert_allclose(many[0], w, rtol=0, atol=1e-13 * float(np.max(np.abs(w))))


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
def test_stationary_covariance_is_block_tridiagonal(window):
    Nf, Nt, dt = 8, 16, 1.0
    C = wilson.matrix.build_C_wdm(Nf, Nt, dt, psd, window=window).reshape(Nf + 1, Nt, Nf + 1, Nt)
    far = max(float(np.max(np.abs(C[m, :, k, :])))
              for m in range(Nf + 1) for k in range(Nf + 1) if abs(m - k) > 1)
    assert far <= 1e-12 * float(np.max(np.abs(C)))


def test_a_different_window_gives_a_different_basis():
    x = np.random.default_rng(8).standard_normal(128)
    a = wilson.forward_time(x, 8, 16, 1.0)
    b = wilson.forward_time(x, 8, 16, 1.0, window=ALT_WINDOW)
    assert float(np.max(np.abs(a - b))) > 1e-3 * float(np.max(np.abs(a)))
```

Append to `tests/wilson/test_meyer_baseline.py`:

```python
#: Case-name prefixes whose kernels take ``window=`` (extended task by task).
THREADED = ("transform.", "matrix.")


def test_explicit_meyer_reproduces_the_baseline_bit_for_bit(baseline):
    got = compute(wilson, DM, {"window": wilson.MEYER}, {}, only=THREADED)
    want = {k: v for k, v in baseline.items()
            if k.startswith("wilson::") and k.split("::")[1].startswith(THREADED)}
    assert want and not _mismatches(got, want)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest -q tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'window'`, and the AST test lists the `phi_centered` calls.

- [ ] **Step 3: Thread the window through `transform.py`**

Replace the import `from .filter import phi_centered` with `from .windows import MEYER, Window, sample_window`. Then, per function (keep each signature's existing parameters and order; `window` is always keyword-only, appended after the last keyword):

| Function | Signature change | Body change |
|---|---|---|
| `_gather_bands` | `(x_rfft, Nf, Nt, dt, be=NUMPY, *, window: Window = MEYER)` | `phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)          # host constant` |
| `forward` | add `window: Window = MEYER,` after `be: Backend \| None = None,` | `return _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)` |
| `forward_many` | same | `return _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)` |
| `forward_time_many` | same | `return forward_many(be.fft.rfft(x, axis=-1), Nf, Nt, dt, be=be, window=window)` |
| `forward_time` | same | `return forward(be.fft.rfft(x), Nf, Nt, dt, be=be, window=window)` |
| `inverse` | same | `phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)` |
| `inverse_many` | same | `phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)` |
| `inverse_time_many` | same | `return be.fft.irfft(inverse_many(wmn, Nf, Nt, dt, be=be, window=window), n=N, axis=-1)` |
| `inverse_time` | same | `x_rfft = inverse(wmn, Nf, Nt, dt, be=be, window=window)` |
| `base_atom_spectra` | `(Nf, Nt, dt, *, window: Window = MEYER)` | `out[m, a] = dt * inverse_many(e[None], Nf, Nt, dt, window=window)[0]` |
| `atom_spectrum` | add `window: Window = MEYER` after `base=None` | `base = base_atom_spectra(Nf, Nt, dt, window=window) if base is None else base` |
| `band_atoms` | `(bands, Nf, Nt, dt, *, window: Window = MEYER)` | `phi = sample_window(window, Nf, Nt, dt)` |
| `forward_bands` | add `window: Window = MEYER` after `be=None` | `phi = sample_window(window, Nf, Nt, dt)` |
| `inverse_bands` | add `window: Window = MEYER` after `be=None` | `phi = sample_window(window, Nf, Nt, dt)` |

If `base_atom_spectra` or any other function is wrapped in `functools.lru_cache`, keep the decorator: `window` is hashable and becomes part of the key.

In each public function's docstring `Parameters` section, add:

```
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.
```
In `band_atoms`' docstring, replace the sentence about "window ``phi_centered(Nf, Nt, dt)`` (the Hz window; ...)" with "``window``'s samples (Hz normalisation)".

jit check: `grep -n "@.*jit\|jax\.jit(\|partial(jit" src/domainia/wilson/*.py` prints only docstring lines at the baseline (no kernel is jitted). If a jitted callable gains `window`, add `"window"` to its `static_argnames`, because windows are hashable and static.

Read-only check: the samples are read-only. Run `grep -n "phi_c\[\|phi\[.*\] *=\|phi_c *\*=\|phi *\*=" src/domainia/wilson/transform.py`. Expected: no in-place writes. If one exists, make it a copy first (`phi = sample_window(...).copy()`).

- [ ] **Step 4: Thread the window through `matrix.py`**

Add `from .windows import MEYER, Window` to the imports.

| Function | Signature change | Body change |
|---|---|---|
| `build_W` | `(Nf, Nt, dt, *, window: Window = MEYER)` | `row = dt * inverse_time(seed, Nf, Nt, dt, window=window)` |
| `active_W` | `(Nf, Nt, dt, *, window: Window = MEYER)` | `W = build_W(Nf, Nt, dt, window=window)` |
| `build_C_wdm` | `(Nf, Nt, dt, Sn_func, *, window: Window = MEYER)` | `W = build_W(Nf, Nt, dt, window=window)` |
| `build_C_wdm_active` | `(Nf, Nt, dt, Sn_func, *, window: Window = MEYER)` | `Wa = active_W(Nf, Nt, dt, window=window)` |

Add the same `window` docstring entry to each one.

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest -q tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py tests/wilson/test_windows.py`
Expected: all pass. If a baseline key mismatches, the edit changed arithmetic. Typical causes are a re-associated expression or a dtype change: `sample_window(...).astype(rdtype)` must replace `phi_centered(...).astype(rdtype)` one for one.

- [ ] **Step 6: Run the whole suite and commit** (sandbox disabled)

Run: `$PY -m pytest -q` → all pass (doctests included).

```bash
git add src/domainia/wilson/transform.py src/domainia/wilson/matrix.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py
git commit -m "Thread window= through the transform and matrix kernels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Thread `window=` through the stationary, noise, evolutionary and inner-product kernels

**Goal:** Every remaining kernel in `domainia.wilson` that reaches the window takes `window: Window = MEYER` and passes it on by keyword. Every kernel baseline array stays bit-identical with the default window and with explicit `MEYER`.

**Files:**
- Modify: `src/domainia/wilson/{stationary,noise,evolutionary,inner_product}.py`
- Modify: `tests/wilson/test_window_threading.py`, `tests/wilson/test_meyer_baseline.py` (extend `THREADED`/`SCANNED`)

**Acceptance Criteria:**
- [ ] `test_no_kernel_call_drops_the_window` passes for every `src/domainia/wilson/*.py` except `filter.py` and `windows.py`.
- [ ] A counting window is sampled by every `wilson_cases` case.
- [ ] Explicit `window=MEYER` reproduces every `wilson::` baseline array bit for bit.
- [ ] `grep -rn "phi_centered(" src/domainia | grep -v "wilson/filter.py\|wilson/windows.py"` prints only docstring lines.

**Verify:** `$PY -m pytest -q tests/wilson` → all pass.

**Steps:**

- [ ] **Step 1: Extend the tests (they fail first)**

In `tests/wilson/test_window_threading.py` set:

```python
THREADED = ("transform.", "matrix.", "stationary.", "noise.", "evolutionary.", "inner_product.")
SCANNED = ("wilson/transform.py", "wilson/matrix.py", "wilson/stationary.py", "wilson/noise.py",
           "wilson/evolutionary.py", "wilson/inner_product.py", "wilson/grid.py", "wilson/backend.py",
           "wilson/special.py")
```
In `tests/wilson/test_meyer_baseline.py` set `THREADED` to the same tuple.

Run: `$PY -m pytest -q tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py`
Expected: FAIL (`unexpected keyword argument 'window'` and AST hits in stationary/noise/evolutionary/inner_product).

- [ ] **Step 2: `stationary.py`**

Replace `from .filter import phi_centered` with `from .windows import MEYER, Window, sample_window`.

| Function | Signature change | Body change |
|---|---|---|
| `_slab_K` | `(Nf, Nt, dt, sigma2_rfft, *, window: Window = MEYER)` | `phi_c = sample_window(window, Nf, Nt, dt)` |
| `build_lambda_blocks_full_active` | add `window: Window = MEYER` after `rtol` | `K_diag, K_off = _slab_K(Nf, Nt, dt, sigma2_rfft, window=window)` |
| `build_K_full_active` | add `window: Window = MEYER` after `rtol` | `return _slab_K(Nf, Nt, dt, sigma2_rfft, window=window)` |
| `psd_from_K` | `(K_diag, Nf, Nt, dt, *, window: Window = MEYER)` | `phi_c = sample_window(window, Nf, Nt, dt)` |
| `apply_lambda_from_psd` | add `window: Window = MEYER` after `be` | `y = _inverse(x_wmn, Nf, Nt, dt, be=be, window=window)` and `return _forward(z, Nf, Nt, dt, be=be, window=window)` |
| `psd_from_lambda_blocks` | `(diag_blocks, offd_blocks, n_active, Nf, Nt, dt, *, window: Window = MEYER)` | `bar_w = _forward_wdm(ones_rfft, Nf, Nt, dt, window=window)` and `X = _inverse_wdm(u, Nf, Nt, dt, window=window)` |

Read-only check: `psd_from_K` divides by the window. Make sure it never writes into `phi_c` (`grep -n "phi_c\[.*\] *=\|phi_c *[*/+-]=" src/domainia/wilson/stationary.py` → no output; otherwise copy first). In the `stationary.py:228` error-message string, `domainia.wilson.matrix.build_W(...)` (rewritten in Task 1) stays as is.

- [ ] **Step 3: `noise.py`**

Replace `from .filter import phi_centered` with `from .windows import MEYER, Window, sample_window`.

| Function | Signature change | Body change |
|---|---|---|
| `band_averaged_variance` | `(Nf, Nt, dt, Sn_func, *, window: Window = MEYER)` | `phi = sample_window(window, Nf, Nt, dt)` |
| `empirical_variance` | add `*, window: Window = MEYER` after `seed=None` | `w = forward(x_rfft, Nf, Nt, dt, window=window)` |
| `evolutionary_band_variance` | add `window: Window = MEYER` after `bands=None` | `phi2 = sample_window(window, Nf, Nt, dt) ** 2`; reword the comment "The Meyer window has compact support" to "An admissible window has compact support" |

- [ ] **Step 4: `evolutionary.py`**

Add `from .windows import MEYER, Window` to the imports.

| Function | Signature change | Body change |
|---|---|---|
| `energy_fraction` | `(w2, Nf, Nt, dt, *, window: Window = MEYER)` | `base = base_atom_spectra(Nf, Nt, dt, window=window)` |
| `apply_time_mask` | `(Y, chi, Nf, Nt, dt, *, window: Window = MEYER)` | `inverse_time_many(full, Nf, Nt, dt, window=window)` and `forward_time_many(X, Nf, Nt, dt, window=window)` |
| `wrap_fractions` | add `window: Window = MEYER` after `bands=None` | `g2 = inverse_time_many(e, Nf, Nt, dt, window=window)[0] ** 2` |
| `wrap_cut_columns` | add `window: Window = MEYER` after `bands=None` | `wrap_fractions(Nf, Nt, dt, bands=bands, window=window)` |
| `exact_covariance` | `(Nf, Nt, dt, Sn, *, window: Window = MEYER)` | `forward_time_many(np.ascontiguousarray(G.T), Nf, Nt, dt, window=window)` |
| `eq45_plan` | add `window: Window = MEYER` after `t_max=None` | `base = base_atom_spectra(Nf, Nt, dt, window=window)` |

`eq45_gram_plan` and `eq45_direct` read `plan.base` (already windowed), so they do not change. `eq45_gram_plan`'s `atom_spectrum(..., base=plan.base)` passes the AST check through `base=`. Add a sentence to `eq45_plan`'s docstring: "``window`` sets the atoms (``plan.base``); every Eq. (45) function downstream of the plan inherits it."

- [ ] **Step 5: `inner_product.py`**

Add `from .windows import MEYER, Window`.

| Function | Signature change | Body change |
|---|---|---|
| `gaussian` | `(w_a, w_b, Nf, Nt, dt, Sn_array, *, window: Window = MEYER)` | `a_rfft = inverse(w_a, Nf, Nt, dt, window=window)`, `b_rfft = inverse(w_b, Nf, Nt, dt, window=window)` |
| `gaussian_from_func` | `(w_a, w_b, Nf, Nt, dt, Sn_func, *, window: Window = MEYER)` | pass `window=window` to the `gaussian(...)` (or `inverse`) call(s) in its body |
| `loglike_gaussian_many` | add `window: Window = MEYER` after `include_logdet` | `r_rfft = inverse_many(np.ascontiguousarray(residual), Nf, Nt, dt, window=window)` |
| `loglike_gaussian` | add `window: Window = MEYER` after `include_logdet` | `r_rfft = inverse(_residual(w_d, w_h), Nf, Nt, dt, window=window)` |

Add the `window` docstring entry from Task 5 to every function touched in Steps 2–5.

- [ ] **Step 6: Run the tests**

Run: `$PY -m pytest -q tests/wilson`
Expected: all pass. If the AST test reports a consumer call this plan does not list, first check that the callee really is one of the window consumers. If it is, it is a real missed thread: add `window=window`, plus the parameter on its enclosing function if needed. If the callee is an unrelated function that shares a `CONSUMERS` name (for example a local `_forward` substitution helper), that is a naming collision. Narrow the scan (skip calls whose callee is defined in the scanned file and never reaches the window) and do not add `window=` to it.

- [ ] **Step 7: Run the whole suite and commit** (sandbox disabled)

Run: `$PY -m pytest -q` → all pass.

```bash
git add src/domainia/wilson tests/wilson
git commit -m "Thread window= through the stationary, noise, evolutionary and inner-product kernels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Thread the window through the domainia layer

**Goal:** `WDMDomain` gains `window: Window = MEYER`, which takes part in equality and hashing. The maps read the window from their `WDMDomain`, `Restrict` refuses a window mismatch, and `BandAveragedVariance`, `EvolutionaryCovariance` and the `testing` oracles take `window=`. Every domainia baseline array stays bit-identical with the default window and with explicit `MEYER`.

**Files:**
- Modify: `src/domainia/domains.py`, `src/domainia/maps/{base,time,fourier}.py`, `src/domainia/variance.py`, `src/domainia/covariance/evolutionary.py`, `src/domainia/testing.py`
- Create: `tests/test_dom_windows.py`
- Modify: `tests/wilson/test_window_threading.py`, `tests/wilson/test_meyer_baseline.py`

**Acceptance Criteria:**
- [ ] `WDMDomain(cells) == WDMDomain(cells, window=MEYER)`, both differ from `WDMDomain(cells, window=ALT_WINDOW)`, and equal domains hash equally. A non-window raises `TypeError`.
- [ ] `TimeToWDM`, `WDMToTime` and `FourierToWDM` on an `ALT_WINDOW` domain match the kernels called with `window=ALT_WINDOW`.
- [ ] `Restrict` between domains with different windows raises `ValueError` naming both windows, and so does `Compose` across a window mismatch.
- [ ] `EvolutionaryCovariance(..., window=ALT_WINDOW)` matches `build_C_wdm_active(..., window=ALT_WINDOW)` in the stationary limit (≤ 1e-10 relative).
- [ ] The AST scan covers `domains.py`, `maps/*.py`, `variance.py`, `covariance/*.py`, `testing.py` and `translate.py` (once it exists), and passes.
- [ ] A counting window is sampled by every `domainia_cases` case. Explicit `MEYER` on both layers reproduces the whole baseline bit for bit.

**Verify:** `$PY -m pytest -q tests/test_dom_windows.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests `tests/test_dom_windows.py`**

```python
"""The window through the domainia layer: WDMDomain, maps, variances, covariance, oracles."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import testing, wilson
from domainia.covariance import EvolutionaryCovariance
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import Compose, FourierToWDM, Restrict, TimeToWDM, WDMToTime
from domainia.spectra import GenericPSD
from domainia.testing import ALT_WINDOW
from domainia.variance import BandAveragedVariance
from domainia.wilson import MEYER

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
CELLS = CellSet.all_active(GRID)
TIME = TimeDomain(N, DT)
X = np.random.default_rng(31).standard_normal(N)


def test_the_window_is_part_of_the_domain_identity():
    a, b, c = WDMDomain(CELLS), WDMDomain(CELLS, window=MEYER), WDMDomain(CELLS, window=ALT_WINDOW)
    assert a == b and hash(a) == hash(b) and a != c
    assert a.window is MEYER
    with pytest.raises(TypeError, match="Window"):
        WDMDomain(CELLS, window="meyer")


def test_time_maps_use_the_domain_window():
    dom = WDMDomain(CELLS, window=ALT_WINDOW)
    w = TimeToWDM(TIME, dom).apply(X)
    full = wilson.forward_time(X, NF, NT, DT, window=ALT_WINDOW)
    np.testing.assert_array_equal(w, full[np.asarray(CELLS.m), np.asarray(CELLS.n)])
    np.testing.assert_allclose(WDMToTime(dom, TIME).apply(w), X, rtol=0, atol=1e-12)
    y = np.random.default_rng(2).standard_normal(CELLS.size)
    assert float(np.dot(TimeToWDM(TIME, dom).apply(X), y)) == pytest.approx(
        float(np.dot(X, TimeToWDM(TIME, dom).adjoint(y))), rel=1e-12)


def test_fourier_cells_use_the_domain_window():
    cells = CellSet.bands(GRID, (2, 3))
    dom = WDMDomain(cells, window=ALT_WINDOW)
    rec = FourierDomain(N, DT, 0, N // 2 + 1, "rfft")
    got = FourierToWDM(rec, dom).apply(np.fft.rfft(X))
    want = np.sqrt(N / (2 * DT)) * TimeToWDM(TIME, dom).apply(X)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.max(np.abs(got)))
    fft = FourierToWDM(rec, dom, "fft").apply(np.fft.rfft(X))
    assert np.max(np.abs(fft - got)) <= 1e-12 * np.max(np.abs(got))
    assert not np.allclose(got, FourierToWDM(rec, WDMDomain(cells)).apply(np.fft.rfft(X)))


def test_maps_refuse_a_window_mismatch():
    meyer, alt = WDMDomain(CELLS), WDMDomain(CELLS.drop_columns(2, 2), window=ALT_WINDOW)
    with pytest.raises(ValueError, match="window"):
        Restrict(meyer, alt)
    with pytest.raises(ValueError):
        Compose(Restrict(WDMDomain(CELLS, window=ALT_WINDOW), alt), TimeToWDM(TIME, meyer))


def test_band_averaged_variance_uses_its_window():
    psd = GenericPSD(lambda f, t, nu: 1.0 + nu[0] * np.asarray(f, float) ** 2 + 0.0 * np.asarray(t, float))
    nu = np.array([3.0])
    full = wilson.noise.evolutionary_band_variance(NF, NT, DT, psd.bind(nu), window=ALT_WINDOW)
    got = BandAveragedVariance(psd, CELLS, window=ALT_WINDOW)(nu)
    np.testing.assert_array_equal(got, full[np.asarray(CELLS.m), np.asarray(CELLS.n)])
    assert not np.array_equal(got, BandAveragedVariance(psd, CELLS)(nu))


def test_evolutionary_covariance_stationary_limit_with_another_window():
    psd = GenericPSD(lambda f, t, nu: (1.0 + np.asarray(f, float) ** 2) ** nu[0] + 0.0 * np.asarray(t, float))
    nu = np.array([-1.2])
    exact = wilson.matrix.build_C_wdm_active(NF, NT, DT, lambda f: psd(f, 0.0, nu), window=ALT_WINDOW)
    got = EvolutionaryCovariance(psd, CELLS, builder="direct", assembly="dense", window=ALT_WINDOW).dense(nu)
    assert np.max(np.abs(got - exact)) < 1e-10 * np.max(np.abs(exact))
    assert EvolutionaryCovariance(psd, CELLS, window=ALT_WINDOW).window is ALT_WINDOW


def test_testing_oracles_take_the_window():
    W = testing.dense_W(CELLS, window=ALT_WINDOW)
    np.testing.assert_array_equal(W, wilson.matrix.active_W(NF, NT, DT, window=ALT_WINDOW)[CELLS.packed_positions()])
    np.testing.assert_allclose(testing.wdm_congruence(np.eye(N), CELLS, window=ALT_WINDOW),
                               DT * np.eye(CELLS.size), rtol=0, atol=1e-12)
```

In `tests/wilson/test_window_threading.py`, extend `SCANNED` with `"domains.py", "maps/base.py", "maps/time.py", "maps/fourier.py", "variance.py", "covariance/evolutionary.py", "covariance/base.py", "covariance/diagonal.py", "covariance/dense.py", "covariance/blocktri.py", "covariance/stack.py", "testing.py", "linalg.py", "data.py"`. Then append:

```python
from types import SimpleNamespace  # noqa: E402

import domainia.covariance  # noqa: E402
import domainia.domains  # noqa: E402
import domainia.maps  # noqa: E402
import domainia.spectra  # noqa: E402
import domainia.testing  # noqa: E402
import domainia.variance  # noqa: E402

from .meyer_cases import domainia_cases  # noqa: E402

_DM = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                      variance=domainia.variance, covariance=domainia.covariance, testing=domainia.testing)
_DCASES = domainia_cases(_DM, wilson, 8, 16, 1.0)


@pytest.mark.parametrize("case", [c for _, c in _DCASES], ids=[n for n, _ in _DCASES])
def test_every_domainia_consumer_samples_the_window_it_is_given(case):
    w = CountingWindow(ALT_WINDOW)
    case({"window": w})
    assert w.calls > 0
```

Append to `tests/wilson/test_meyer_baseline.py`:

```python
def test_explicit_meyer_on_both_layers_reproduces_the_whole_baseline(baseline):
    got = compute(wilson, DM, {"window": wilson.MEYER}, {"window": wilson.MEYER})
    assert not _mismatches(got, baseline)
```

Run: `$PY -m pytest -q tests/test_dom_windows.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py`
Expected: FAIL (`WDMDomain.__init__() got an unexpected keyword argument 'window'`, AST hits in maps/variance/testing).

- [ ] **Step 2: `domains.py`**

Add `from domainia.wilson.windows import MEYER, Window` to the imports. Replace the `WDMDomain` class header, fields and `__post_init__` with:

```python
@dataclass(frozen=True)
class WDMDomain:
    """A cell set, the constant ``scale`` that relates its cells to the Wilson
    coefficients, and the Wilson ``window`` of the basis (default :data:`MEYER`: WDM).

    The window is part of the domain's identity (equality and hashing), so two maps or
    covariances built for different windows never compose silently.
    """

    cells: CellSet
    scale: float = 1.0
    window: Window = MEYER

    def __post_init__(self) -> None:
        if not isinstance(self.cells, CellSet):
            raise TypeError(f"cells must be a CellSet; got {type(self.cells).__name__}")
        object.__setattr__(self, "scale", _positive_float("scale", self.scale))
        if not isinstance(self.window, Window):
            raise TypeError(f"window must be a domainia.wilson Window; got {type(self.window).__name__}")
```
Keep the existing `scale` explanation paragraph (gap G4) in the docstring, below the new first paragraph, and keep `size` and `require_unit_scale` unchanged.

- [ ] **Step 3: The maps**

`maps/time.py`:
- `TimeToWDM.apply`: `wilson.forward_time(x, g.Nf, g.Nt, g.dt, be=be, window=self.dst.window)` and `wilson.forward_time_many(X, g.Nf, g.Nt, g.dt, be=be, window=self.dst.window)`.
- `TimeToWDM.adjoint`: `wilson.inverse_time_many(self._scatter(y, be), g.Nf, g.Nt, g.dt, be=be, window=self.dst.window)`.
- `TimeToWDM.rows`: `base = base_atom_spectra(g.Nf, g.Nt, g.dt, window=self.dst.window)`. The `atom_spectrum(..., base=base)` call is unchanged.
- `WDMToTime.apply`: `wilson.inverse_time_many(full, g.Nf, g.Nt, g.dt, be=be, window=self.src.window)`.

`maps/fourier.py`:
- Replace `from domainia.wilson.filter import phi_centered` with `from domainia.wilson.windows import sample_window`.
- `_dense_atoms(geo, Nf, dt, window)`: `atoms = band_atoms(bands, Nf, Nt, dt, window=window)`.
- `FourierToWDM.__post_init__`: `U = _dense_atoms(geo, g.Nf, g.dt, window=self.dst.window)[geo["sel"]]`.
- `FourierToWDM.apply` (fft): `wilson.transform.forward_bands(z, self.src.k0, geo["bands"], g.Nf, g.Nt, g.dt, be=be, window=self.dst.window)`.
- `FourierToWDM.after`: `phi = sample_window(self.dst.window, g.Nf, g.Nt, g.dt)`.
- Add to the `FourierToWDM.window` property docstring: "(These are band-window *bins*. The Wilson window of the cells is ``dst.window``.)"

`maps/base.py`, `Restrict.__post_init__`: right before the existing `missing`/grid check, add:

```python
            if self.src.window != self.dst.window:
                raise ValueError(f"src and dst use different windows ({self.src.window.name} vs "
                                 f"{self.dst.window.name}): restrict within one basis")
```

- [ ] **Step 4: `variance.py`, `covariance/evolutionary.py`, `testing.py`**

`variance.py`:
- import `from domainia.wilson.windows import MEYER, Window`;
- `BandAveragedVariance` decorator becomes `@pytree_dataclass(data=("t",), meta=("psd", "cells", "window"))`;
- add the field `window: Window = field(default=MEYER, kw_only=True)` after `t`;
- in `__post_init__`, after `_require_generic`, add `if not isinstance(self.window, Window): raise TypeError(f"window must be a domainia.wilson Window; got {type(self.window).__name__}")`;
- in `__call__`, the call becomes `wilson.noise.evolutionary_band_variance(g.Nf, g.Nt, g.dt, self.psd.bind(nu), window=self.window)`;
- in the module docstring, add: "``BandAveragedVariance(..., window=)`` averages over that window (default Meyer); the midpoint variance is window-free."

`covariance/evolutionary.py`:
- import `from domainia.wilson.windows import MEYER, Window`;
- add `window: Window = MEYER` as the last keyword of `EvolutionaryCovariance.__init__`, store `self.window = window`, and pass it to `_ChebBlocks(psd, cells.grid, keep, P, t_max, window)` and `_DirectMatrix(psd, cells.grid, keep, t_max, window)`;
- both helper classes store `window` and call `eq45_plan(g.Nf, g.Nt, g.dt, self.keep, P=..., t_max=self.t_max, window=self.window)`;
- add to the class docstring: "``window``: the Wilson window of the cells' basis (default :data:`MEYER`)."

`testing.py`:
- `dense_W(cells, *, window=MEYER)` → `wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=window)[cells.packed_positions()]`;
- `wdm_congruence(Sigma, cells, *, window=MEYER)` → `W = dense_W(cells, window=window)`;
- `eq45_dense(cells, Sn, *, P=24, t_max=None, window=MEYER)` → `eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max, window=window)`;
- import `MEYER` alongside `SampledWindow`.

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest -q tests/test_dom_windows.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py`
Expected: all pass. If the AST scan flags another consumer call in a domainia module, check what the callee is. A real kernel call gets `window` from the nearest `WDMDomain` or a `window=` parameter. An unrelated function that shares a `CONSUMERS` name (`linalg.py`, `covariance/*.py`, `data.py` may define local `_forward`/`solve` helpers) is a naming collision. Narrow the scan for it, and do not add `window=`.

- [ ] **Step 6: Run the whole suite and commit** (sandbox disabled)

Run: `$PY -m pytest -q` → all pass.

```bash
git add src/domainia tests
git commit -m "WDMDomain.window; maps, variances, EvolutionaryCovariance and oracles take the window

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 8: `domainia.translate.transform` and the full-record domains

**Goal:** Add `FourierDomain.full(N, dt)` and `WDMDomain.full(grid, window=MEYER)`. Add `domainia.translate` with `transform(channel_data, source, target)` and `log_coordinate_scale(domain)`. `transform` moves `{channel: ndarray}` dicts between the full-record time, Fourier and WDM domains through the NumPy rfft hub and refuses every malformed input.

**Files:**
- Create: `src/domainia/translate.py`, `tests/test_translate.py`
- Modify: `src/domainia/domains.py` (the two `full` constructors), `tests/wilson/test_window_threading.py` (`SCANNED` gains `"translate.py"`)

**Acceptance Criteria:**
- [ ] `FourierDomain.full(N, dt) == FourierDomain(N, dt, 0, N // 2 + 1, "dt_rfft")` and `WDMDomain.full(grid, window=w) == WDMDomain(CellSet.all_active(grid), window=w)`.
- [ ] For all 16 ordered pairs of {time, Fourier, WDM-Meyer (8, 16), WDM-`ALT_WINDOW` (16, 8)}, `transform` agrees with the route through time and round-trips to ≤ 1e-12 relative.
- [ ] Energies match `exp(2 · log_coordinate_scale(domain)) · ‖x‖²` to 1e-12 relative. Fourier energy uses the real-coordinate weights: 1 at DC and at an even-`N` Nyquist bin, 2 elsewhere.
- [ ] Each of the 26 refusal cases raises the listed `TypeError`/`ValueError` with the listed message fragment.
- [ ] Outputs are fresh `float64`/`complex128` arrays in input channel order, with Fourier endpoints exactly real and inactive WDM cells exactly 0.

**Verify:** `$PY -m pytest -q tests/test_translate.py tests/wilson/test_window_threading.py` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests `tests/test_translate.py`**

```python
"""domainia.translate.transform: {channel: array} between full-record domains (spec section 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.testing import ALT_WINDOW
from domainia.translate import log_coordinate_scale, transform
from domainia.wilson import MEYER

NF, NT, DT = 8, 16, 0.25
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
TIME = TimeDomain(N, DT)
FOURIER = FourierDomain.full(N, DT)
WDM = WDMDomain.full(GRID)
WDM_ALT = WDMDomain.full(wilson.Grid(16, 8, DT), window=ALT_WINDOW)
DOMAINS = {"time": TIME, "fourier": FOURIER, "wdm": WDM, "wdm-alt": WDM_ALT}
MASK = wilson.active_mask(NF, NT)
_rng = np.random.default_rng(41)
X = {"A": _rng.standard_normal(N), "E": _rng.standard_normal(N)}


def _on(domain):
    return transform(X, TIME, domain)


def _close(got, want, rel=1e-12):
    assert list(got) == list(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=rel * float(np.max(np.abs(want[k]))))


def _energy(domain, v):
    if isinstance(domain, FourierDomain):
        w = np.full(v.shape[-1], 2.0)
        w[0] = 1.0
        if domain.N % 2 == 0:
            w[-1] = 1.0
        return float(np.sum(w * np.abs(v) ** 2))
    return float(np.sum(np.asarray(v) ** 2))


def _with(a, idx, value):
    a = np.array(a)
    a[idx] = value
    return a


def test_the_full_domains():
    assert FOURIER == FourierDomain(N, DT, 0, N // 2 + 1, "dt_rfft")
    assert WDM == WDMDomain(CellSet.all_active(GRID)) and WDM.window is MEYER and WDM.scale == 1.0
    assert WDM_ALT.window is ALT_WINDOW and WDM_ALT != WDMDomain.full(wilson.Grid(16, 8, DT))


def test_the_definitions():
    _close(_on(FOURIER), {k: DT * np.fft.rfft(v) for k, v in X.items()}, rel=1e-14)
    _close(_on(WDM), {k: wilson.forward_time(v, NF, NT, DT) for k, v in X.items()}, rel=1e-14)
    _close(_on(WDM_ALT), {k: wilson.forward_time(v, 16, 8, DT, window=ALT_WINDOW) for k, v in X.items()},
           rel=1e-14)


@pytest.mark.parametrize("src", DOMAINS)
@pytest.mark.parametrize("dst", DOMAINS)
def test_every_pair_round_trips_and_agrees_with_the_route_through_time(src, dst):
    a, b = DOMAINS[src], DOMAINS[dst]
    there = transform(_on(a), a, b)
    _close(there, _on(b))
    _close(transform(there, b, a), _on(a))


@pytest.mark.parametrize("name", DOMAINS)
def test_parseval_scales_match_log_coordinate_scale(name):
    d = DOMAINS[name]
    for k, x in X.items():
        want = math.exp(2.0 * log_coordinate_scale(d)) * float(np.sum(x ** 2))
        assert _energy(d, _on(d)[k]) == pytest.approx(want, rel=1e-12)


def test_log_coordinate_scale_values():
    assert log_coordinate_scale(TIME) == 0.0
    assert log_coordinate_scale(WDM) == pytest.approx(0.5 * math.log(DT), rel=1e-15)
    assert log_coordinate_scale(WDM_ALT) == log_coordinate_scale(WDM)
    assert log_coordinate_scale(FOURIER) == pytest.approx(math.log(DT) + 0.5 * math.log(N), rel=1e-15)


def test_an_odd_record_has_no_nyquist_bin():
    t, f = TimeDomain(15, 0.5), FourierDomain.full(15, 0.5)
    x = {"A": np.random.default_rng(3).standard_normal(15)}
    y = transform(x, t, f)
    assert y["A"].shape == (8,) and y["A"][-1].imag != 0.0
    _close(transform(y, f, t), x)


def test_outputs_are_fresh_exact_and_in_input_order():
    src = {"E": X["E"], "A": X["A"]}
    same = transform(src, TIME, TIME)
    assert list(same) == ["E", "A"] and not np.shares_memory(same["A"], X["A"])
    f = transform(src, TIME, FOURIER)
    assert all(v.dtype == np.complex128 and v[0].imag == 0.0 and v[-1].imag == 0.0 for v in f.values())
    w = transform(f, FOURIER, WDM)
    assert list(w) == ["E", "A"]
    assert all(v.dtype == np.float64 and np.all(v[~MASK] == 0.0) for v in w.values())
    assert transform({"A": np.arange(N)}, TIME, TIME)["A"].dtype == np.float64
    assert transform({"A": X["A"].astype(np.float32)}, TIME, WDM)["A"].dtype == np.float64


_FOUR = _on(FOURIER)["A"]
_WDMV = _on(WDM)["A"]
REFUSALS = {
    "empty": ({}, TIME, WDM, ValueError, "empty"),
    "list": ([X["A"]], TIME, WDM, TypeError, "mapping"),
    "array": (X["A"], TIME, WDM, TypeError, "mapping"),
    "source-kind": (X, "time", WDM, TypeError, "domain"),
    "target-kind": (X, TIME, None, TypeError, "domain"),
    "partial-fourier": ({"A": _FOUR[1:11]}, FourierDomain(N, DT, 1, 10, "dt_rfft"), TIME, ValueError,
                        "FourierDomain.full"),
    "fourier-units": ({"A": _FOUR}, FourierDomain(N, DT, 0, N // 2 + 1, "rfft"), TIME, ValueError,
                      "FourierDomain.full"),
    "partial-wdm": (X, TIME, WDMDomain(CellSet.bands(GRID, (2, 3))), ValueError, "WDMDomain.full"),
    "wdm-scale": (X, TIME, WDMDomain(CellSet.all_active(GRID), scale=2.0), ValueError, "scale"),
    "N": (X, TIME, FourierDomain.full(N + 2, DT), ValueError, "N="),
    "dt": (X, TIME, WDMDomain.full(wilson.Grid(NF, NT, 2 * DT)), ValueError, "dt="),
    "t0": (X, TIME, TimeDomain(N, DT, t0=1.0), ValueError, "t0"),
    "shape": ({"A": X["A"][:-1]}, TIME, WDM, ValueError, "shape"),
    "channel-shapes": ({"A": X["A"], "E": np.zeros(N + 1)}, TIME, WDM, ValueError, "shape"),
    "nan": ({"A": _with(X["A"], 3, np.nan)}, TIME, WDM, ValueError, "finite"),
    "inf-fourier": ({"A": _with(_FOUR, 5, np.inf)}, FOURIER, TIME, ValueError, "finite"),
    "complex-time": ({"A": X["A"] + 0j}, TIME, WDM, ValueError, "complex"),
    "complex-wdm": ({"A": _WDMV + 0j}, WDM, TIME, ValueError, "complex"),
    "strings": ({"A": np.array(["x"] * N)}, TIME, WDM, TypeError, "numeric"),
    "booleans": ({"A": np.ones(N, dtype=bool)}, TIME, WDM, TypeError, "numeric"),
    "name-kind": ({1: X["A"]}, TIME, WDM, TypeError, "string"),
    "name-empty": ({"": X["A"]}, TIME, WDM, ValueError, "non-empty"),
    "dc": ({"A": _with(_FOUR, 0, _FOUR[0] + 1e-300j)}, FOURIER, TIME, ValueError, "DC"),
    "nyquist": ({"A": _with(_FOUR, -1, _FOUR[-1] + 1j)}, FOURIER, TIME, ValueError, "Nyquist"),
    "inactive-dc": ({"A": _with(_WDMV, (0, 1), 1e-300)}, WDM, TIME, ValueError, "inactive"),
    "inactive-nyquist": ({"A": _with(_WDMV, (NF, 1), 1.0)}, WDM, TIME, ValueError, "inactive"),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_every_refusal_rule(case):
    data, src, dst, err, match = REFUSALS[case]
    with pytest.raises(err, match=match):
        transform(data, src, dst)


def test_a_non_finite_result_is_refused():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            transform({"A": np.full(N, 1e308)}, TIME, FOURIER)
```

`REFUSALS` holds 26 cases.

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest -q tests/test_translate.py`
Expected: collection error, `ModuleNotFoundError: No module named 'domainia.translate'`.

- [ ] **Step 3: Add the two constructors to `src/domainia/domains.py`**

In `FourierDomain`, after `__post_init__`:

```python
    @classmethod
    def full(cls, N: int, dt: float) -> FourierDomain:
        """Every rfft bin of an ``N``-sample record in ``"dt_rfft"`` units (``dt * numpy.fft.rfft(x)``):
        the Fourier domain of :func:`domainia.translate.transform`."""
        N = _int("N", N, minimum=1)
        return cls(N, dt, 0, N // 2 + 1, "dt_rfft")
```

In `WDMDomain`, after `__post_init__`:

```python
    @classmethod
    def full(cls, grid: wilson.Grid, window: Window = MEYER) -> WDMDomain:
        """Every active cell of ``grid``, scale 1, Wilson ``window``: the WDM domain of
        :func:`domainia.translate.transform` (dense ``(Nf + 1, Nt)`` layout there)."""
        return cls(CellSet.all_active(grid), window=window)
```

In the module docstring's `FourierDomain` and `WDMDomain` bullets, add "``.full(...)``: the full-record domain of :mod:`domainia.translate`".

- [ ] **Step 4: Write `src/domainia/translate.py`**

```python
"""Move full-record data between the time, Fourier and WDM domains (spec section 3).

NumPy only.  Data are ``{channel: ndarray}`` dicts (insertion order kept) on one of the
full-record domains:

* ``TimeDomain(N, dt, t0)``: real ``(N,)`` per channel;
* ``FourierDomain.full(N, dt)``: complex ``(N // 2 + 1,)`` per channel, ``dt * numpy.fft.rfft(x)``;
* ``WDMDomain.full(Grid(Nf, Nt, dt), window)``: real dense ``(Nf + 1, Nt)`` per channel, the
  Wilson coefficients of ``window``, inactive cells exactly 0.

Every pair goes through the rfft hub ``R = numpy.fft.rfft(x)``: time ``rfft``/``irfft``,
Fourier ``X / dt``/``dt R``, WDM ``wilson.inverse_many``/``wilson.forward_many``.  Partial
or packed cell sets stay with :mod:`domainia.maps`.

In the real coordinates (time samples; WDM cells; Fourier ``Re X_0``, ``sqrt(2) Re/Im X_k``,
``Re X_{N/2}``) every domain is the time record scaled by ``exp(log_coordinate_scale(domain))``:
1 for time, ``sqrt(dt)`` for WDM (any admissible window), ``dt sqrt(N)`` for Fourier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from domainia import wilson
from domainia.domains import FourierDomain, TimeDomain, WDMDomain

__all__ = ["log_coordinate_scale", "transform"]

type Domain = TimeDomain | FourierDomain | WDMDomain


def _record(domain: Any, role: str) -> tuple[int, float]:
    """``(N, dt)`` of a full-record domain; refuses every other domain."""
    if isinstance(domain, TimeDomain):
        return domain.N, domain.dt
    if isinstance(domain, FourierDomain):
        if domain.units != "dt_rfft" or domain.k0 != 0 or domain.n != domain.N // 2 + 1:
            raise ValueError(f"{role}: translate handles only FourierDomain.full(N, dt) (every rfft "
                             f"bin, dt_rfft units); got {domain!r}")
        return domain.N, domain.dt
    if isinstance(domain, WDMDomain):
        g = domain.cells.grid
        if domain.cells.size != g.N:
            raise ValueError(f"{role}: translate handles only WDMDomain.full(grid, window) (every "
                             f"active cell, scale 1); got {domain.cells.size} of {g.N} cells")
        if domain.scale != 1.0:
            raise ValueError(f"{role}: translate handles only WDMDomain.full(grid, window) (every "
                             f"active cell, scale 1); got scale={domain.scale!r}")
        return g.N, g.dt
    raise TypeError(f"{role} must be a full-record domain (TimeDomain, FourierDomain.full or "
                    f"WDMDomain.full); got {type(domain).__name__}")


def _check_pair(source: Any, target: Any) -> None:
    Ns, dts = _record(source, "source")
    Nt, dtt = _record(target, "target")
    if Ns != Nt or dts != dtt:
        raise ValueError(f"source and target describe different records: N={Ns}, dt={dts!r} vs "
                         f"N={Nt}, dt={dtt!r}")
    if isinstance(source, TimeDomain) and isinstance(target, TimeDomain) and source.t0 != target.t0:
        raise ValueError(f"source and target time domains start at different t0 ({source.t0!r} vs "
                         f"{target.t0!r}): transform does not shift records")


def _shape(domain: Domain) -> tuple[int, ...]:
    if isinstance(domain, TimeDomain):
        return (domain.N,)
    if isinstance(domain, FourierDomain):
        return (domain.N // 2 + 1,)
    g = domain.cells.grid
    return (g.Nf + 1, g.Nt)


def _active(domain: WDMDomain) -> np.ndarray:
    g = domain.cells.grid
    return wilson.active_mask(g.Nf, g.Nt)


def _stack(channel_data: Any, domain: Domain) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate ``channel_data`` on ``domain``; ``(names, (C, *shape) float64/complex128)``."""
    if not isinstance(channel_data, Mapping):
        raise TypeError(f"channel_data must be a mapping {{channel: array}}; got "
                        f"{type(channel_data).__name__}")
    if not channel_data:
        raise ValueError("channel_data is empty: pass at least one channel")
    shape = _shape(domain)
    kind = type(domain).__name__
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for name, value in channel_data.items():
        if not isinstance(name, str):
            raise TypeError(f"channel names must be strings; got {name!r}")
        if not name:
            raise ValueError("channel names must be non-empty")
        a = np.asarray(value)
        if a.dtype.kind not in "iufc":
            raise TypeError(f"channel {name!r}: values must be numeric; got dtype {a.dtype}")
        if a.dtype.kind == "c" and not isinstance(domain, FourierDomain):
            raise ValueError(f"channel {name!r}: complex values on a real {kind}")
        if a.shape != shape:
            raise ValueError(f"channel {name!r}: shape {a.shape}, but this {kind} needs shape {shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"channel {name!r}: values must be finite")
        if isinstance(domain, FourierDomain):
            a = a.astype(np.complex128)
            ends = ((0, "DC"), (shape[0] - 1, "Nyquist")) if domain.N % 2 == 0 else ((0, "DC"),)
            for k, label in ends:
                if a[k].imag != 0.0:
                    raise ValueError(f"channel {name!r}: rfft bin {k} ({label}) must be real; its "
                                     f"imaginary part is {a[k].imag!r}")
        else:
            a = a.astype(np.float64)
            if isinstance(domain, WDMDomain):
                bad = (a != 0.0) & ~_active(domain)
                if np.any(bad):
                    m, n = (int(i) for i in np.argwhere(bad)[0])
                    raise ValueError(f"channel {name!r}: inactive WDM cell (m={m}, n={n}) is "
                                     f"{a[m, n]!r}; inactive cells must be exactly 0")
        names.append(name)
        arrays.append(a)
    return tuple(names), np.stack(arrays)


def _to_rfft(X: np.ndarray, domain: Domain) -> np.ndarray:
    if isinstance(domain, TimeDomain):
        return np.fft.rfft(X, axis=-1)
    if isinstance(domain, FourierDomain):
        return X / domain.dt
    g = domain.cells.grid
    return wilson.inverse_many(X, g.Nf, g.Nt, g.dt, window=domain.window)


def _from_rfft(R: np.ndarray, domain: Domain) -> np.ndarray:
    if isinstance(domain, TimeDomain):
        return np.fft.irfft(R, n=domain.N, axis=-1)
    if isinstance(domain, FourierDomain):
        return domain.dt * R
    g = domain.cells.grid
    return wilson.forward_many(R, g.Nf, g.Nt, g.dt, window=domain.window)


def _finish(names: tuple[str, ...], Y: np.ndarray, domain: Domain) -> dict[str, np.ndarray]:
    if isinstance(domain, FourierDomain):
        out = np.array(Y, dtype=np.complex128)
        out[..., 0] = out[..., 0].real
        if domain.N % 2 == 0:
            out[..., -1] = out[..., -1].real
    else:
        out = np.array(Y, dtype=np.float64)
        if isinstance(domain, WDMDomain):
            out[..., ~_active(domain)] = 0.0
    if not np.all(np.isfinite(out)):
        raise ValueError(f"the result on the {type(domain).__name__} is not finite (overflow?)")
    return {name: out[i].copy() for i, name in enumerate(names)}


def transform(channel_data: Mapping[str, Any], source: Domain, target: Domain) -> dict[str, np.ndarray]:
    """``channel_data`` on ``source`` re-expressed on ``target`` (module docstring).

    The same domain returns validated copies.  WDM to WDM across another ``(Nf, Nt)`` or
    window goes through the spectrum.  Raises ``TypeError`` for a non-mapping, a non-domain or
    non-numeric values, and ``ValueError`` for a partial domain, another record (``N``,
    ``dt``, or ``t0`` between time domains), an empty dict, a wrong shape, non-finite
    values, complex values on a real domain, a non-real DC (or even-``N`` Nyquist) bin, a
    non-zero inactive WDM cell, or a non-finite result.
    """
    _check_pair(source, target)
    names, X = _stack(channel_data, source)
    Y = X if source == target else _from_rfft(_to_rfft(X, source), target)
    return _finish(names, Y, target)


def log_coordinate_scale(domain: Domain) -> float:
    """``log`` of the factor between ``domain``'s real coordinates and the time samples:
    0 for time, ``log(dt) / 2`` for WDM, ``log(dt) + log(N) / 2`` for Fourier."""
    N, dt = _record(domain, "domain")
    if isinstance(domain, TimeDomain):
        return 0.0
    if isinstance(domain, WDMDomain):
        return 0.5 * math.log(dt)
    return math.log(dt) + 0.5 * math.log(N)
```

- [ ] **Step 5: Extend the AST scan**

In `tests/wilson/test_window_threading.py`, append `"translate.py"` to `SCANNED`. `translate.py` calls `inverse_many`/`forward_many` only with `window=`.

- [ ] **Step 6: Run the tests**

Run: `$PY -m pytest -q tests/test_translate.py tests/wilson/test_window_threading.py`
Expected: all pass.

- [ ] **Step 7: Run the whole suite and commit** (sandbox disabled)

Run: `$PY -m pytest -q` → all pass.

```bash
git add src/domainia/translate.py src/domainia/domains.py tests/test_translate.py tests/wilson/test_window_threading.py
git commit -m "domainia.translate.transform between full-record time, Fourier and WDM domains

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Covariance operators across domains (`Operator`, `TranslatedOperator`, `translate_operator`)

**Goal:** Add the runtime-checkable `Operator` protocol, the lazy `TranslatedOperator(native, target)` and `translate_operator(op, target)` to `domainia.translate`. Add `PointwiseOperator` to `domainia.testing` as the minimal native operator for tests.

**Files:**
- Modify: `src/domainia/translate.py`, `src/domainia/testing.py`
- Create: `tests/test_translate_operator.py`

**Acceptance Criteria:**
- [ ] For every native (time, WDM-Meyer, WDM-ALT) × target (time, Fourier, WDM-Meyer, WDM-ALT), `quadratic_form` agrees with the native one (1e-10 relative).
- [ ] For every such pair, `log_determinant` equals native + 2·dof·Δ, and `solve(apply(y))` and `apply(solve(y))` equal `project(y)` (1e-10).
- [ ] A time operator seen in WDM equals `A diag(σ²) Aᵀ` with `A = active_W` (apply and slogdet). A WDM operator seen in time equals `Aᵀ diag(σ²) A / dt²`.
- [ ] Translating a `TranslatedOperator` re-targets its native operator, and a native target returns the native operator itself (`is`). `TranslatedOperator`'s only dataclass fields are `native` and `target`.
- [ ] A non-operator or non-domain raises `TypeError`. Another record, `t0`, a partial domain or wrong channel names raise `ValueError`.

**Verify:** `$PY -m pytest -q tests/test_translate_operator.py` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests `tests/test_translate_operator.py`**

```python
"""domainia.translate operators: a native covariance seen from another domain (spec section 3)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.testing import ALT_WINDOW, PointwiseOperator
from domainia.translate import (
    Operator,
    TranslatedOperator,
    log_coordinate_scale,
    transform,
    translate_operator,
)

NF, NT, DT = 8, 16, 0.25
N = NF * NT
TIME = TimeDomain(N, DT)
FOURIER = FourierDomain.full(N, DT)
WDM = WDMDomain.full(wilson.Grid(NF, NT, DT))
WDM_ALT = WDMDomain.full(wilson.Grid(16, 8, DT), window=ALT_WINDOW)
TARGETS = {"time": TIME, "fourier": FOURIER, "wdm": WDM, "wdm-alt": WDM_ALT}
_rng = np.random.default_rng(43)
X = {"A": _rng.standard_normal(N), "E": _rng.standard_normal(N)}


def _variances(shape):
    return {"A": 0.5 + _rng.random(shape), "E": 0.5 + _rng.random(shape)}


NATIVES = {
    "time": PointwiseOperator(TIME, _variances((N,))),
    "wdm": PointwiseOperator(WDM, _variances((NF + 1, NT))),
    "wdm-alt": PointwiseOperator(WDM_ALT, _variances((17, 8))),
}
PAIRS = [(a, b) for a in NATIVES for b in TARGETS]


def _y(domain):
    return transform(X, TIME, domain)


def _close(got, want, rel=1e-10):
    assert list(got) == list(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=rel * float(np.max(np.abs(want[k]))))


def test_the_operators_satisfy_the_protocol():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    assert isinstance(op, Operator) and isinstance(t, Operator) and isinstance(t, TranslatedOperator)
    assert t.domain == WDM and t.channel_names == ("A", "E") and t.degrees_of_freedom == 2 * N
    assert [f.name for f in dataclasses.fields(TranslatedOperator)] == ["native", "target"]


@pytest.mark.parametrize("native, target", PAIRS)
def test_quadratic_forms_agree_across_domains(native, target):
    op = NATIVES[native]
    t = translate_operator(op, TARGETS[target])
    assert t.quadratic_form(_y(TARGETS[target])) == pytest.approx(op.quadratic_form(_y(op.domain)), rel=1e-10)


@pytest.mark.parametrize("native, target", PAIRS)
def test_the_log_determinant_carries_the_coordinate_jacobian(native, target):
    op, d = NATIVES[native], TARGETS[target]
    delta = log_coordinate_scale(d) - log_coordinate_scale(op.domain)
    want = op.log_determinant() + 2 * op.degrees_of_freedom * delta
    assert translate_operator(op, d).log_determinant() == pytest.approx(want, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("native, target", PAIRS)
def test_solve_inverts_apply_up_to_the_projection(native, target):
    t = translate_operator(NATIVES[native], TARGETS[target])
    y = _y(TARGETS[target])
    _close(t.solve(t.apply(y)), t.project(y))
    _close(t.apply(t.solve(y)), t.project(y))


def test_a_time_operator_seen_in_wdm_is_the_dense_congruence():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    A = wilson.matrix.active_W(NF, NT, DT)
    m, n = wilson.grid.packed_cells(NF, NT)
    y = _y(WDM)
    np.testing.assert_allclose(A @ X["A"], y["A"][m, n], rtol=0, atol=1e-12)
    out = t.apply(y)
    for k in ("A", "E"):
        want = A @ (op.variances[k] * (A.T @ y[k][m, n]))
        np.testing.assert_allclose(out[k][m, n], want, rtol=0, atol=1e-10 * np.max(np.abs(want)))
    dense = sum(np.linalg.slogdet(A @ np.diag(op.variances[k]) @ A.T)[1] for k in ("A", "E"))
    assert t.log_determinant() == pytest.approx(dense, rel=1e-10)


def test_a_wdm_operator_seen_in_time_is_the_dense_congruence():
    op = NATIVES["wdm"]
    t = translate_operator(op, TIME)
    A = wilson.matrix.active_W(NF, NT, DT)
    m, n = wilson.grid.packed_cells(NF, NT)
    out = t.apply(X)
    for k in ("A", "E"):
        want = A.T @ (op.variances[k][m, n] * (A @ X[k])) / DT**2
        np.testing.assert_allclose(out[k], want, rtol=0, atol=1e-10 * np.max(np.abs(want)))


def test_chains_collapse_and_a_native_target_returns_the_native_operator():
    op = NATIVES["wdm"]
    assert translate_operator(op, WDM) is op
    t1 = translate_operator(op, FOURIER)
    t2 = translate_operator(t1, WDM_ALT)
    assert isinstance(t2, TranslatedOperator) and t2.native is op and t2.target == WDM_ALT
    assert translate_operator(t1, WDM) is op
    y = _y(WDM_ALT)
    _close(t2.apply(y), translate_operator(op, WDM_ALT).apply(y), rel=1e-14)


def test_refusals():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    with pytest.raises(TypeError, match="Operator"):
        translate_operator("not an operator", WDM)
    with pytest.raises(TypeError, match="domain"):
        translate_operator(op, "wdm")
    with pytest.raises(TypeError, match="native"):
        TranslatedOperator(t, FOURIER)
    with pytest.raises(ValueError, match="N="):
        translate_operator(op, FourierDomain.full(N + 2, DT))
    with pytest.raises(ValueError, match="t0"):
        translate_operator(op, TimeDomain(N, DT, t0=2.0))
    with pytest.raises(ValueError, match="WDMDomain.full"):
        translate_operator(op, WDMDomain(CellSet.bands(WDM.cells.grid, (2, 3))))
    with pytest.raises(ValueError, match="channels"):
        t.apply({"A": _y(WDM)["A"]})
    with pytest.raises(TypeError, match="mapping"):
        t.quadratic_form([_y(WDM)["A"]])


def test_the_pointwise_test_operator_refuses_bad_input():
    with pytest.raises(TypeError, match="TimeDomain or WDMDomain"):
        PointwiseOperator(FOURIER, {"A": np.ones(N // 2 + 1)})
    with pytest.raises(ValueError, match="WDMDomain.full"):
        PointwiseOperator(WDMDomain(CellSet.bands(WDM.cells.grid, (2, 3))), {"A": np.ones((NF + 1, NT))})
    with pytest.raises(ValueError, match="positive"):
        PointwiseOperator(TIME, {"A": -np.ones(N)})
    with pytest.raises(ValueError, match="shape"):
        PointwiseOperator(TIME, {"A": np.ones(N + 1)})
    with pytest.raises(ValueError, match="channel"):
        PointwiseOperator(TIME, {})
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest -q tests/test_translate_operator.py`
Expected: collection error, `ImportError: cannot import name 'PointwiseOperator' from 'domainia.testing'`.

- [ ] **Step 3: Add the operators to `src/domainia/translate.py`**

Add `from dataclasses import dataclass` and `from typing import Any, Protocol, runtime_checkable` to the imports. Replace `__all__` with:

```python
__all__ = ["Operator", "TranslatedOperator", "log_coordinate_scale", "transform", "translate_operator"]
```

Add this paragraph to the end of the module docstring:

```
A covariance :class:`Operator` lives on one full-record domain.  :func:`translate_operator`
gives a :class:`TranslatedOperator` that sees it from another domain lazily: every call
moves the dicts with :func:`transform` and applies the coordinate Jacobian
``Delta = log_coordinate_scale(target) - log_coordinate_scale(native)``.  The
real-coordinate convention (sqrt(2) Re/Im interiors, real endpoints, masks) stays inside
the native operator.
```

Append:

```python
@runtime_checkable
class Operator(Protocol):
    """A covariance on a full-record domain, acting on ``{channel: array}`` dicts in that
    domain's layout.  ``degrees_of_freedom`` counts the real coordinates it covers, summed
    over channels (masked-out coordinates excluded)."""

    @property
    def domain(self) -> Domain: ...

    @property
    def channel_names(self) -> tuple[str, ...]: ...

    @property
    def degrees_of_freedom(self) -> int: ...

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]: ...

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]: ...

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]: ...

    def quadratic_form(self, values: Mapping[str, Any]) -> float: ...

    def log_determinant(self) -> float: ...


@dataclass(frozen=True, eq=False)
class TranslatedOperator:
    """``native`` seen from ``target`` (module docstring).  It stores only these two: no
    dense matrix is ever built.

    * ``apply(x)`` = to_target(native.apply(to_native(x))) * exp(2 Delta)
    * ``solve(x)`` = to_target(native.solve(to_native(x))) * exp(-2 Delta)
    * ``project(x)`` = to_target(native.project(to_native(x)))
    * ``quadratic_form(x)`` = native.quadratic_form(to_native(x))
    * ``log_determinant()`` = native.log_determinant() + 2 * degrees_of_freedom * Delta
    """

    native: Operator
    target: Domain

    def __post_init__(self) -> None:
        if isinstance(self.native, TranslatedOperator) or not isinstance(self.native, Operator):
            raise TypeError(f"native must be a native domainia.translate.Operator (translate_operator "
                            f"re-targets a TranslatedOperator); got {type(self.native).__name__}")
        _check_pair(self.native.domain, self.target)

    @property
    def domain(self) -> Domain:
        return self.target

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.native.channel_names)

    @property
    def degrees_of_freedom(self) -> int:
        return int(self.native.degrees_of_freedom)

    def _delta(self) -> float:
        return log_coordinate_scale(self.target) - log_coordinate_scale(self.native.domain)

    def _to_native(self, values: Any) -> dict[str, np.ndarray]:
        if not isinstance(values, Mapping):
            raise TypeError(f"values must be a mapping {{channel: array}}; got {type(values).__name__}")
        if set(values) != set(self.channel_names):
            raise ValueError(f"values have channels {tuple(values)}; the operator has channels "
                             f"{self.channel_names}")
        return transform(values, self.target, self.native.domain)

    def _to_target(self, values: Mapping[str, Any], factor: float = 1.0) -> dict[str, np.ndarray]:
        out = transform(values, self.native.domain, self.target)
        return out if factor == 1.0 else {k: v * factor for k, v in out.items()}

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._to_target(self.native.apply(self._to_native(values)), np.exp(2.0 * self._delta()))

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._to_target(self.native.solve(self._to_native(values)), np.exp(-2.0 * self._delta()))

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._to_target(self.native.project(self._to_native(values)))

    def quadratic_form(self, values: Mapping[str, Any]) -> float:
        return float(self.native.quadratic_form(self._to_native(values)))

    def log_determinant(self) -> float:
        return float(self.native.log_determinant()) + 2.0 * self.degrees_of_freedom * self._delta()


def translate_operator(op: Operator, target: Domain) -> Operator:
    """``op`` seen from ``target``.  A :class:`TranslatedOperator` is re-targeted (its native
    operator is reused, never nested), and a target equal to the native domain returns the
    native operator itself."""
    if not isinstance(op, Operator):
        raise TypeError(f"op must implement domainia.translate.Operator; got {type(op).__name__}")
    native = op.native if isinstance(op, TranslatedOperator) else op
    _check_pair(native.domain, target)
    if target == native.domain:
        return native
    return TranslatedOperator(native, target)
```

`np.exp` of a Python float returns `np.float64`. Multiplying a `float64`/`complex128` array by it keeps the dtype, keeps the endpoints exactly real and keeps inactive cells exactly 0.

- [ ] **Step 4: Add `PointwiseOperator` to `src/domainia/testing.py`**

Add the imports `from collections.abc import Mapping`, `from dataclasses import dataclass` and `from domainia.domains import TimeDomain, WDMDomain` if they are missing. Add `"PointwiseOperator"` to `__all__`. Append:

```python
@dataclass(frozen=True, eq=False)
class PointwiseOperator:
    """A minimal native :class:`domainia.translate.Operator` for tests: independent samples
    (``TimeDomain``) or independent active cells (``WDMDomain.full``, dense layout,
    inactive cells masked out) with per-channel variances."""

    domain: TimeDomain | WDMDomain
    variances: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.domain, TimeDomain):
            mask = np.ones(self.domain.N, dtype=bool)
        elif isinstance(self.domain, WDMDomain):
            g = self.domain.cells.grid
            if self.domain.cells.size != g.N:
                raise ValueError("PointwiseOperator needs every active cell: use WDMDomain.full(grid)")
            mask = wilson.active_mask(g.Nf, g.Nt)
        else:
            raise TypeError(f"PointwiseOperator needs a TimeDomain or WDMDomain; got "
                            f"{type(self.domain).__name__}")
        if not isinstance(self.variances, Mapping) or not self.variances:
            raise ValueError("variances must be a non-empty {channel: array} mapping")
        var = {}
        for name, v in self.variances.items():
            a = np.array(v, dtype=np.float64)
            if a.shape != mask.shape:
                raise ValueError(f"channel {name!r}: variances have shape {a.shape}; need {mask.shape}")
            if not np.all(np.isfinite(a[mask])) or np.any(a[mask] <= 0.0):
                raise ValueError(f"channel {name!r}: variances must be finite and positive")
            a = np.where(mask, a, 1.0)
            a.setflags(write=False)
            var[name] = a
        object.__setattr__(self, "variances", var)
        object.__setattr__(self, "_mask", mask)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.variances)

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.variances) * int(self._mask.sum())

    def _values(self, values: Any) -> dict[str, Array]:
        if not isinstance(values, Mapping):
            raise TypeError(f"values must be a mapping; got {type(values).__name__}")
        if set(values) != set(self.variances):
            raise ValueError(f"values have channels {tuple(values)}; need {self.channel_names}")
        out = {}
        for name in self.variances:
            a = np.asarray(values[name], dtype=np.float64)
            if a.shape != self._mask.shape:
                raise ValueError(f"channel {name!r}: shape {a.shape}; need {self._mask.shape}")
            out[name] = a
        return out

    def apply(self, values: Any) -> dict[str, Array]:
        return {k: np.where(self._mask, self.variances[k] * v, 0.0) for k, v in self._values(values).items()}

    def solve(self, values: Any) -> dict[str, Array]:
        return {k: np.where(self._mask, v / self.variances[k], 0.0) for k, v in self._values(values).items()}

    def project(self, values: Any) -> dict[str, Array]:
        return {k: np.where(self._mask, v, 0.0) for k, v in self._values(values).items()}

    def quadratic_form(self, values: Any) -> float:
        return float(sum(np.sum(np.where(self._mask, v * v / self.variances[k], 0.0))
                         for k, v in self._values(values).items()))

    def log_determinant(self) -> float:
        return float(sum(np.sum(np.log(a[self._mask])) for a in self.variances.values()))
```

`testing.py` already defines `Array` and imports `numpy as np`, `Any` and `wilson`. Check this with `grep -n "^type Array\|^Array\|^from typing\|^from domainia import wilson\|^import numpy" src/domainia/testing.py`. Add any that is missing.

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest -q tests/test_translate_operator.py tests/test_translate.py tests/wilson/test_window_threading.py`
Expected: all pass.

- [ ] **Step 6: Run the whole suite and commit** (sandbox disabled)

Run: `$PY -m pytest -q` → all pass.

```bash
git add src/domainia/translate.py src/domainia/testing.py tests/test_translate_operator.py
git commit -m "Operator protocol, lazy TranslatedOperator and translate_operator

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: The 2.0 package surface, identity test and README

**Goal:** `domainia` 2.0.0 exports the general-library surface and never imports JAX. A new identity test pins the version, the public names and the JAX-free import, and the README describes the general library. The full verification command passes.

**Files:**
- Modify: `src/domainia/__init__.py`, `src/domainia/wilson/__init__.py` (docstring and `__all__` check), `README.md`
- Create: `tests/test_package.py`

**Acceptance Criteria:**
- [ ] `domainia.__version__ == "2.0.0"` matches `pyproject.toml`, and `set(domainia.__all__)` equals the 26 names in Step 1. Every name in `domainia.__all__` and `domainia.wilson.__all__` resolves.
- [ ] In a fresh subprocess, `import domainia, domainia.wilson, domainia.translate` resolves to `src/` and leaves `jax` out of `sys.modules`.
- [ ] `domainia.wilson` has no `paper`, `__release__` or `__wdm_api__`. `git ls-files wilson` prints nothing, and `grep -rn "import wdm\|from wdm" src tests` prints nothing.
- [ ] The README's quick-start code runs as written.
- [ ] `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q` passes with no collection errors.

**Verify:** `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q` → all pass, 0 errors.

**Steps:**

- [ ] **Step 1: Write the failing test `tests/test_package.py`**

```python
"""domainia 2.0 identity: version, public names, no JAX on import, no release leftovers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import domainia
from domainia import wilson

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

PUBLIC = {
    "CellSet", "ChannelAxis", "FourierBins", "FourierDomain", "Grid", "MEYER", "Meyer", "Operator",
    "RampWindow", "SampledWindow", "TimeDomain", "TimeSeries", "TranslatedOperator", "WDMCells",
    "WDMDomain", "Window", "WindowError", "__version__", "data", "domains", "linalg",
    "log_coordinate_scale", "transform", "translate", "translate_operator", "wilson",
}


def test_the_version_is_2_0_0_everywhere():
    assert domainia.__version__ == "2.0.0"
    assert tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"] == "2.0.0"


def test_the_public_names():
    assert set(domainia.__all__) == PUBLIC
    assert all(hasattr(domainia, name) for name in domainia.__all__)
    assert all(hasattr(wilson, name) for name in wilson.__all__)
    assert {"MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError", "windows"} <= set(wilson.__all__)
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid


def test_no_release_leftovers():
    for name in ("paper", "__release__", "__wdm_api__"):
        assert not hasattr(wilson, name) and name not in wilson.__all__


def test_import_resolves_to_src_and_does_not_import_jax():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC)
    code = ("import json, sys, domainia, domainia.wilson, domainia.translate\n"
            "print(json.dumps({'file': domainia.__file__, 'jax': 'jax' in sys.modules}))")
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(REPO),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert Path(info["file"]).resolve().is_relative_to(SRC.resolve())
    assert info["jax"] is False
```

Run: `$PY -m pytest -q tests/test_package.py`
Expected: FAIL in `test_the_public_names` (`translate`, `transform`, `MEYER` … are missing from `domainia.__all__`).

- [ ] **Step 2: Write the final `src/domainia/__init__.py`**

```python
"""domainia: data, linear maps and covariance across the time, Fourier and time-frequency
(Wilson / WDM) domains.

Importing ``domainia`` never imports JAX: the JAX backend is reached only through
:mod:`domainia.wilson.backend`.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`MEYER`); every kernel takes ``window=`` (any admissible
  :class:`Window`: ``Meyer(d)``, ``RampWindow``, ``SampledWindow``).
* :mod:`domainia.translate`: ``transform`` (``{channel: array}`` dicts) and
  ``translate_operator`` (lazy covariance change of basis) between the full-record
  ``TimeDomain``, ``FourierDomain.full`` and ``WDMDomain.full``.
* :mod:`domainia.domains`: ``TimeDomain``, ``FourierDomain``, ``CellSet``, ``WDMDomain``,
  ``ChannelAxis`` (frozen, hashable: static under ``jax.jit``).
* :mod:`domainia.data`: ``TimeSeries``, ``FourierBins``, ``WDMCells`` (pytrees).
* :mod:`domainia.linalg`: array kernels (block-tridiagonal, Hermitian block, PCG).
* Not imported here: :mod:`domainia.maps` (linear maps between partial domains),
  :mod:`domainia.covariance`, :mod:`domainia.variance`, :mod:`domainia.spectra`, and
  :mod:`domainia.testing` (dense oracles, test windows and operators).
"""

from __future__ import annotations

from . import data, domains, linalg, translate, wilson
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from .translate import Operator, TranslatedOperator, log_coordinate_scale, transform, translate_operator
from .wilson import MEYER, Grid, Meyer, RampWindow, SampledWindow, Window, WindowError

__version__ = "2.0.0"

__all__ = [
    "MEYER",
    "CellSet",
    "ChannelAxis",
    "FourierBins",
    "FourierDomain",
    "Grid",
    "Meyer",
    "Operator",
    "RampWindow",
    "SampledWindow",
    "TimeDomain",
    "TimeSeries",
    "TranslatedOperator",
    "WDMCells",
    "WDMDomain",
    "Window",
    "WindowError",
    "__version__",
    "data",
    "domains",
    "linalg",
    "log_coordinate_scale",
    "transform",
    "translate",
    "translate_operator",
    "wilson",
]
```

- [ ] **Step 3: Check `src/domainia/wilson/__init__.py`**

Run: `$PY -c "import domainia.wilson as w; missing=[n for n in w.__all__ if not hasattr(w, n)]; print(missing, [n for n in ('paper','__release__','__wdm_api__') if hasattr(w, n)])"`
Expected: `[] []`. If a name is missing, it was deleted in Task 1 or Task 2 (for example `"paper"`). Remove it from `__all__`.

Replace the first paragraph of the wilson module docstring with: "``domainia.wilson``: Wilson-basis kernels on the half-shifted lattice. With the default window :data:`MEYER` they are the WDM transform. Every kernel that samples the window takes ``window=`` (:mod:`domainia.wilson.windows`)." Then run `$PY -m pytest -q src/domainia/wilson/__init__.py`. Expected: the doctests pass, or "no tests ran" if the module has none.

- [ ] **Step 4: Rewrite `README.md`**

````markdown
# domainia

Move data and covariance between the time, Fourier and time-frequency domains.

* `domainia.wilson` holds the Wilson-basis kernels: forward and inverse transforms, the
  stationary and evolutionary noise models, inner products and dense oracles. With the
  default Meyer window (`MEYER`) the Wilson basis is the WDM transform. Every kernel takes
  `window=`, so the same code runs any *admissible* window: real, even, zero for
  |f| ≥ ΔF, and a partition of unity on the lattice. Such a window gives an orthonormal
  basis and block-tridiagonal stationary covariances.
  `Meyer(d)`, `RampWindow(ramp, name)` and `SampledWindow(fn, name)` are validated
  when sampled, and an inadmissible one raises `WindowError`.
* `domainia.translate` is the translation layer. `transform` moves `{channel: array}`
  dicts between the full-record domains, and `translate_operator` puts a covariance
  operator native to one domain into another lazily: no dense matrix, and the
  coordinate Jacobian is in `log_determinant`.
* `domainia.domains`, `domainia.data`, `domainia.maps`, `domainia.covariance`,
  `domainia.variance` and `domainia.spectra` cover partial domains (WDM cell sets, Fourier
  bands): maps between them, block-tridiagonal / dense / diagonal / evolutionary covariances
  and cell variances.

## Quick start

```python
import numpy as np
from domainia import FourierDomain, Grid, Meyer, TimeDomain, WDMDomain, transform

Nf, Nt, dt = 16, 64, 0.5
N = Nf * Nt
time = TimeDomain(N, dt)
freq = FourierDomain.full(N, dt)                        # dt * numpy.fft.rfft(x)
wdm = WDMDomain.full(Grid(Nf, Nt, dt))                  # (Nf + 1, Nt) cells, Meyer window
wdm8 = WDMDomain.full(Grid(Nf, Nt, dt), window=Meyer(8))

x = {"A": np.random.default_rng(0).standard_normal(N)}
w = transform(x, time, wdm)                             # {"A": (Nf + 1, Nt) float64}
X = transform(w, wdm, freq)                             # {"A": (N // 2 + 1,) complex128}
w8 = transform(w, wdm, wdm8)                            # another window, through the spectrum
assert np.allclose(transform(w8, wdm8, time)["A"], x["A"])
```

Coordinates scale as `exp(log_coordinate_scale(domain))` relative to the time
samples: 1 for time, `sqrt(dt)` for WDM (any window), `dt * sqrt(N)` for Fourier (in the
real coordinates `Re X_0`, `sqrt(2) Re/Im X_k`, `Re X_{N/2}`).

## Operators

An `Operator` lives on one full-record domain and acts on `{channel: array}` dicts:
`apply`, `solve`, `project`, `quadratic_form`, `log_determinant`, plus `domain`,
`channel_names` and `degrees_of_freedom`. `translate_operator(op, target)` returns a
`TranslatedOperator` that keeps only `op` and `target`. Translating again re-targets
`op`, and translating to `op.domain` returns `op` itself.

## Errors

Everything fails closed. `WindowError` (a `ValueError`) means an inadmissible window.
`ValueError` covers shape, finiteness, inactive-cell, endpoint, domain or window
mismatches. `TypeError` means the wrong kind of object.

## Install and test

```bash
pip install -e ".[test]"          # add ",jax" for the JAX backend
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu python -P -m pytest -q
```

`import domainia` never imports JAX. The JAX backend is reached only through
`domainia.wilson.backend`.
````

Run the quick start:

```bash
$PY - <<'EOF'
import numpy as np
from domainia import FourierDomain, Grid, Meyer, TimeDomain, WDMDomain, transform

Nf, Nt, dt = 16, 64, 0.5
N = Nf * Nt
time = TimeDomain(N, dt)
freq = FourierDomain.full(N, dt)
wdm = WDMDomain.full(Grid(Nf, Nt, dt))
wdm8 = WDMDomain.full(Grid(Nf, Nt, dt), window=Meyer(8))

x = {"A": np.random.default_rng(0).standard_normal(N)}
w = transform(x, time, wdm)
X = transform(w, wdm, freq)
w8 = transform(w, wdm, wdm8)
assert np.allclose(transform(w8, wdm8, time)["A"], x["A"])
print("ok", w["A"].shape, X["A"].shape, X["A"].dtype)
EOF
```
Expected: `ok (17, 64) (513,) complex128`.

- [ ] **Step 5: The leftover checks**

Run: `git ls-files wilson; grep -rn "import wdm\|from wdm" src tests; grep -rn "phi_centered(" src/domainia | grep -v "wilson/filter.py\|wilson/windows.py"`
Expected: no output from the first two. The third prints only docstring or comment lines.

If `wilson/` still exists on disk with only untracked files (`__pycache__`, `.claude`), delete it: `rm -rf wilson`.

- [ ] **Step 6: Run the full verification command**

Run: `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q`
Expected: all pass, no collection errors. JAX-marked tests run (not skipped) in `wdmrel`.

- [ ] **Step 7: Commit** (sandbox disabled)

```bash
git add -A src tests README.md
git commit -m "domainia 2.0.0: public surface, identity test, README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

## Execution errata

Deviations and additions agreed during execution (subagent-driven development: spec and code-quality review per task, then a final whole-branch review). The task text above is the plan as approved; these entries record what changed and why. Items marked FOR USER / FOLLOW-UP / OUT OF SCOPE were left for a user decision.

- Task 0: `energy_fraction(x ** 2, ...)` (was `w ** 2`); `np.nan_to_num(wrap_fractions(...), nan=-1.0)`; `np.savez_compressed`; no `# noqa` on `import wdm` in `_old_layout`; K0/B0 hoisted; added cases testing.wdm_congruence (non-identity Sigma), WDMToTime.adjoint, FourierToWDM.adjoint; argparse group + SystemExit.
- Task 7: `test_testing_oracles_take_the_window` uses a non-identity diagonal S for wdm_congruence (identity is window-blind).
- Task 1 review: restored oracle sha pins (tests/wilson/test_oracle_pins.py); meyer_cases.py made check-only (re-record = check out d991ac4 with the 24c4e55 recorder); backend/stationary/domains prose fixes. Task 10 gains Step 5b (py.typed to top level, dangling summary.md refs) and a word-bounded wdm grep.
- Task 2 review: stale prose (WDMDomain G4, Identity, covariance/base docstring, CholFactor.build, linalg) reworded; concrete_failure deleted; variance classes jittable=False and DiagonalCovariance honours it; EvolutionaryCovariance raises on layout+dense and unknown chol_impl. Task 3 gains Step 4b (tests/test_dom_restored_coverage.py: JAX map parity, evaluator paths, dense/cblock under jit, blocktri_solve_packed).
- Task 3 review: added FourierToWDM.after() coverage (FourierCrop + FourierZeroPad), cut-cell sel test, real padded t_max (1.25 T) + lazy "pair mid-times" refusal, stationary-limit tol 1e-12, independent variance asserts, fft+restrict refusal, push identity.
- Task 1: codemod import patterns use `[ 	]*$` not `\s*$` (splitlines keepends=True: \s* ate the newline).
- Task 3 re-review (approved): Task 7's test_band_averaged_variance_uses_its_window gains an independent reference (diag of build_C_wdm_active(window=ALT_WINDOW) at packed_positions).
- Task 4 review: SampledWindow u grid = (arange(Nt)-Nt//2)/(Nt//2) (exact at -1); require_window() helper (rejects classes), used by sample_window and (Task 7) WDMDomain/BandAveragedVariance; unhashable fn rejected at construction; read-only views of the cache; NaN/dtype/shape/third-party/evenness-per-bin/partition-j0 tests; pickle/deepcopy equality test; float() in messages. Task 5 note: float32 dt changes ~3e-8 (accepted).
- Task 5 review: added definition-level value tests per window (build_W vs impulses, inverse_many, atom spectra vs rfft of W rows, forward_bands/band_atoms, band adjoint); bad third-party window through every kernel raises WindowError; atom_spectrum calls require_window unconditionally and documents base-wins; AST scan flags .sample(, hard-coded windows, base= exemption only for atom_spectrum; THREADED moved to meyer_cases.py; docstring fixes + window doctest. Task 6 brief gains required per-window value tests and sample-then-discard mutant checks.
- Task 5 re-review (approved): forward notes to Task 6 (SCANNED comment excluding filter/windows; phi_fft_order rule), Task 7 (WDMDomain/EvolutionaryCovariance/BandAveragedVariance into CONSUMERS), Task 10 (matrix.py build_W docstring tolerance). OUT OF SCOPE, report to user: pre-existing float32 inverse path runs complex128 FFTs (transform.py _scatter_interior_bands/_scatter_edge_bands: float64 sign arrays promote), contradicting the 'entirely in single precision' docstring; fixing changes float32 bits — user decision.
- Task 6 review: scanner gains function-body hard-coded-window rule (rebinding); JAX legs for apply_lambda_from_psd + transform with window; GRIDS += (16,16,0.25); value tests for 1-D Y / bands= / keep= / 2-D w_d paths; zero-inside-slab window in psd_from_K round trip; docstring fixes. OUT OF SCOPE, report to user: stationary.py psd_from_K eps=1e-30 absolute (wrong at dt<=1e-31). Task 7 note: enforce spec rule 'covariances built for one window raise ValueError on another domain' where an API pairs them, or report none exists.
- Task 7 review: equal-but-distinct window tests at Restrict/Compose/PushedCovariance; Restrict meta checks moved above the pos guard (bypass closed); FourierToWDM adjoint/rows ALT tests; window read-only properties; Covariance.window base convention; ChannelStack common window; CONSUMERS comment; docstrings. OUT OF SCOPE, report to user: pre-existing CellSet pickle/deepcopy returns writable m_array/n_array that can desync from m/n tuples (fix: CellSet.__reduce__).
- Task 7 re-review (approved). Task 10 gains: DiagonalCovariance._window captured from raw variance; maps/base.py docstring warning about dataclasses.replace. FOLLOW-UP for user: constant-carrying maps (FourierToWDM, FourierZeroPad/FourierCrop, TimeToWDM/WDMToTime) trust built constants under dataclasses.replace(dst=...) — suggested guard: a built_for meta field compared on every construction route.
- Task 8 review: transform docstring + refusal messages explain canonical input and give remedies (exact rules kept); tests pin n-check, input finiteness, shape, odd-N scale, float64 promotion; t0 docstring; MaskedArray refused; JAX docstring; N_src/N_dst; log_coordinate_scale docstring. Task 9 brief: Operator output contract documented + test. FOR USER: spec line ~229 ('domainia's own outputs make the endpoints exactly real') holds only for transform outputs, not raw wilson kernel outputs — suggest rewording; optional public canonicaliser helper as follow-up.
- Task 8 re-review (approved). Task 9 brief gains ride-alongs: conditional relative-size remedy text, non-vacuous remedies test, (m, n) index for WDM non-finite, JAX/MaskedArray docstring wording; plus the Domain __all__ merge and Operator output contract notes.
- Task 9 review: rank-deficient + Fourier-native + Fourier-target independent dense references (port of review9q/dense_probe.py); non-circular log-det; Operator docstring semantics (projector, Moore-Penrose, pseudo-det, rank, view); _to_target re-raises contract violations naming the native+method; channel-set check on native output; same-target re-translate returns self; _check_pair roles=; PointwiseOperator TypeError split; stub instead of object.__setattr__ hack.
- Task 9 re-review (approved). Task 10 gains: _to_target also wraps TypeError; contract-wide wrapper wording; native.domain role.
- Task 10 review: _to_target re-raises a plain TypeError/ValueError by isinstance (type(err)(msg) broke JAX tracer errors, whose constructors take a tracer) + jax-marked regression test; README states canonical-input rules with remedies, the Operator contract (Protocol, real coordinates, projector/pseudo-inverse/pseudo-det, summed over channels, output contract), window construction screening and SampledWindow's dimensionless psi(u), JAX-on-use wording and that transform is not traceable; "every kernel that samples the window takes window="; test_package walks every module with pkgutil for the JAX-free check and checks translate.__all__ resolves; require_window/sample_window re-exported in wilson.__all__; external-file citations (wdm_toys/wdmtoysgpu/galactic_binary paths, file:line refs) dropped from src. FOLLOW-UP for user: WT/WG/GB/WDM II shorthand (~124 hits across 19 files) and "Ported from the GB toy-model oracle (frozen)" in maps/fourier.py (the GB tree does not ship).
- Final review (ready to integrate): Operator contract reworded for channel-correlated natives (stacked real coordinates) + correlated-channel dense-reference test; EvolutionaryCovariance/BandAveragedVariance wrong-kind args raise TypeError (CellSet check); PushedCovariance type/window checks at construction; stale "gap G9"/"gap G4" labels reworded; TabulatedSpectrum.on(domain=); README names wilson.grid pack/unpack (packed vs dense layout); JAX-missing ModuleNotFoundError gives the install hint; hatchling>=1.27; external-oracle tests gated on DOMAINIA_ORACLE_ROOT. Spec §5 wording deviation: pyproject testpaths = ["tests", "src/domainia/wilson"] (doctests collected from the package). FOR USER: FourierToWDM ignores src.units (cells differ from transform's on WDMDomain(scale=1) by sqrt(N/(2dt)) for rfft units); wilson Grid wrong-kind raises ValueError (vendored-test pinned); no bridge from domainia.covariance (WDM factor API) to translate.Operator; ~26 "spec §X" refs to the 1.x spec and 63 doctests aliasing `wilson as wdm`; covariance.evolutionary.keep_mask duplicates CellSet.keep_mask; EvolutionaryCovariance could accept a WDMDomain (silent-Meyer trap); sdist ships docs/superpowers.
