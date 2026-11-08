# domainia 2.0 Follow-ups Implementation Plan: units in place, covariance operators, fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved follow-up spec `docs/superpowers/specs/2026-11-04-domainia-units-operators-fixes-design.md` on the unreleased `domainia-2.0` branch. Every WDM value domainia produces becomes a Wilson coefficient and every Fourier value carries its declared units. domainia's covariances and explicit matrices become `Operator`s that `translate_operator` can move between domains. The review list is fixed: single precision, wrong-kind errors, smaller defects, dead code, and the text sweep.

**Architecture:** In-place changes to the 2.0 package, in the spec's Order. Units first (Tasks 1–4): `FourierDomain` gains `sigma0`/`rfft_scale`, the band kernels and `FourierToWDM` return Wilson coefficients, the pad/crop maps lose their scale, and `WDMDomain.scale` is removed. Operators next (Tasks 5–9): public real coordinates in `domainia.translate`, a new `domainia.operators` module (`DenseOperator`, `operator_matrix`, `CovarianceOperator`), `Covariance.multiply`, and `WDMDomain` accepted by the cell-set consumers. Then the fix list (Tasks 10–15). The Meyer bit-identity baseline recorded at d991ac4 is the regression gate throughout. Five case families change by a known factor (spec §1.7); everything else stays bit-exact.

**Tech Stack:** Python ≥3.12, NumPy ≥2.1, SciPy ≥1.14.1, optional JAX ≥0.9, pytest ≥8, hatchling. Test env: `~/miniconda3/envs/wdmrel` (Python 3.12.4, numpy 2.3.5, scipy 1.16.3, jax 0.9.0.1).

**User decisions (already made):**
- Spec, Decisions (2026-11-04): "For this package, I want to keep the units in their place." Approach A ("Go with A"): the maps convert, and a WDM value always means a Wilson coefficient. Approach B (keep the raw numbers and label them with `WDMDomain.scale`) is rejected.
- Spec, Decisions: the covariance bridge (§2) and the fix list (§3–§7) were approved together with A.
- Spec, Decisions: no push. The finishing menu (merge, PR, keep, discard) comes back after execution.
- 2026-11-07: "Do Nov 4 spec": review the spec, write this plan, and execute it on `domainia-2.0`. `main` stays untouched until the finishing menu.
- 2026-11-07: "ok, write the plan once it's approved". The spec review (two passes) approved it, with fixes folded in at 421b66f and df17ae4.

**Scope check:** one plan. The spec's three parts share the Gates section and the Meyer baseline, and the spec review did not flag scope. Tasks run in the spec's Order.

---

## Conventions (every task)

* **Repository.** Work only in the worktree `REPO=/Users/aaron/Documents/lisa/domainia/.claude/worktrees/domainia-2.0` (branch `domainia-2.0`). Never `cd` to `/Users/aaron/Documents/lisa/domainia`: that is the `main` checkout, with the old layout.
* **`$PY`** means `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P`, run from `$REPO`.
* **Full suite:** `$PY -m pytest -q`. **NumPy lane:** `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`. Both must pass with no collection errors at the end of every task. The README gate is `tests/test_readme.py` (Task 0), which is part of the suite.
* **Meyer baseline.** `tests/wilson/test_meyer_baseline.py` replays `tests/wilson/meyer_cases.py` against `tests/wilson/data/meyer_baseline.npz`. `meyer_cases.py` is check-only: never re-record the npz (spec Non-goals). From Task 2 on, the comparator allows exactly the `RESCALED` table of spec §1.7.
* **Edits are anchored to code, not line numbers.** Every step names the function, class or exact old text it changes. The spec cites lines at 73ec04a, and earlier tasks shift them.
* **Each task starts with "Assumes the tree after Task N".** The state is the end of the previous task, as defined by the Interface contract below.
* **Pin lists are a starting point.** Before editing, a task greps for every pattern it changes (spec, Test pin lists). Each grep step states the expected hits.
* **Git.** The Claude Code sandbox blocks writes inside `.git`, so every `git add`, `git mv`, `git rm` and `git commit` runs with the sandbox disabled. End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Nothing is pushed.
* **Scratch** scripts go in `$TMPDIR`, never in the repo.
* **Errata.** Deviations agreed during execution go in the "Execution errata" section at the end, as in the 2.0 plan.

## File structure

```
src/domainia/
  domains.py           FourierDomain: sigma0, rfft_scale (T1); WDMDomain(cells, window) (T4);
                       CellSet.__reduce__ (T12)
  translate.py         scale refusal gone (T4); to_coordinates / from_coordinates (T5); __all__ (T8)
  operators.py         NEW (T6): DenseOperator, operator_matrix; CovarianceOperator (T8)
  __init__.py          re-exports the operators and the coordinate functions (T8)
  maps/base.py         BinSelect replace + (units, sigma0) check (T1); Restrict / Identity scale gone (T4);
                       PushedCovariance message (T8); Restrict built_for (T13)
  maps/time.py         require_unit_scale gone (T4); built_for on TimeToWDM / WDMToTime / TimeToFourier (T13)
  maps/fourier.py      window replace + units checks (T1); FourierToWDM κ (T2); pad/crop no scale,
                       self-conjugate halving (T3); scale gone (T4); built_for (T13)
  covariance/base.py   Covariance.multiply (T7); batchable / complete_constants removed if unread (T14)
  covariance/{diagonal,dense,blocktri,evolutionary,stack}.py   multiply (T7); ChannelStack._matrix (T7)
  covariance/evolutionary.py   accepts a WDMDomain (T9); keep_mask removed (T14)
  covariance/__init__.py       keep_mask export removed (T14)
  variance.py          accepts a WDMDomain (T9); unused t leaves removed (T14)
  spectra.py           TabulatedSpectrum docstring (T1); GenericPSD.capabilities removed (T14)
  wilson/transform.py  band kernels in Wilson units (T2); single precision (T10)
  wilson/{grid,noise,filter}.py   wrong kinds raise TypeError (T11)
  wilson/stationary.py psd_from_K relative eps (T12)
  (every module)       text sweep (T15)
pyproject.toml         sdist excludes docs/superpowers (T12)
docs/superpowers/specs/2026-10-31-domainia-general-library-design.md   one wording fix (T12)
README.md              prose made true for the new API (T15)
tests/
  test_readme.py                 NEW (T0)
  test_translate_coordinates.py  NEW (T5)
  test_operators_dense.py        NEW (T6)
  test_dom_cov_multiply.py       NEW (T7)
  test_operators_covariance.py   NEW (T8)
  test_dom_built_for.py          NEW (T13)
  wilson/test_single_precision.py  NEW (T10)
  wilson/test_wrong_kinds.py       NEW (T11)
  (existing files)               updated per task (pin lists and greps)
```

## Interface contract (end state of each name; later tasks build on these exactly)

| Name | Final form | Task |
|---|---|---|
| `FourierDomain` | `FourierDomain(N, dt, k0, n, units, *, sigma0=None)`. `units="sigma0"` requires a positive finite `sigma0`; other units require `sigma0=None` (`ValueError` naming `sigma0`). `sigma0` is in `__eq__` and `__hash__`. | T1 |
| `FourierDomain.rfft_scale` | property `u` with `numpy.fft.rfft(x) = u · z`: `1.0` (rfft), `1/dt` (dt_rfft), `sigma0` (sigma0). | T1 |
| derived `FourierDomain`s | always via `dataclasses.replace`. `BinSelect`: `replace(src, k0=…, n=…)`. `FourierToWDM.window`: `replace(self.src, N=g.N, dt=g.dt, k0=geo["g0"], n=geo["G"])`. | T1 |
| same-units checks | `FourierZeroPad`, `FourierCrop`, `BinSelect` compare `(units, sigma0)`. | T1 |
| `wilson.transform.forward_bands(z, k0, bands, Nf, Nt, dt, *, be=None, window=MEYER)` | returns `forward(z)[bands]`: Wilson coefficients of rfft-unit bins (old output × `sqrt(2 dt/N)`), `N = Nf·Nt`. | T2 |
| `wilson.transform.inverse_bands(w, k0, n_bins, bands, Nf, Nt, dt, *, be=None, window=MEYER)` | returns `inverse(w_full)[k0:k0+n_bins]` (old output × `sqrt(N/(2 dt))`). Adjoint identity: `Σ forward_bands(z)·w = (2 dt/N)·Re⟨z, inverse_bands(w)⟩`. | T2 |
| `wilson.transform.band_atoms` | unchanged (unit-norm). Its docstring states the factor `sqrt(2 dt/N)`. | T2 |
| `FourierToWDM` | `κ = src.rfft_scale · sqrt(2 dt/N)`. Dense `C = κ·U`. fft: `u · forward_bands(z)`. `apply(z)` equals `TimeToWDM.apply(x)` on the same cells for every units value. `after(inner)`: `C = κ·(analysis of A and B)`. In T2 it still multiplies by `inner.scale`; that term goes in T3. | T2, T3 |
| `FourierZeroPad(src, dst, *, A=…, B=…)` | no `scale` field. `z′ = A z + B conj(z)` is `rfft_N′(x zero-padded)` in the same units. | T3 |
| `FourierCrop(src, dst, start=0, *, A=…, B=…)` | no `scale` attribute. | T3 |
| `pad_maps(N, Np, k_out, j_in)`, `crop_maps(N, start, Nc, k_out, j_in)` | the `A` and `B` columns of absolute source bin `j_in = 0`, and `j_in = N/2` for even `N`, are halved. | T3 |
| `WDMDomain` | `WDMDomain(cells, window=MEYER)`. No `scale`, no `require_unit_scale`. `WDMDomain.full(grid, window=MEYER)` unchanged. | T4 |
| `translate.to_coordinates(channel_data, domain, channel_names=None) -> ndarray (K, N) float64` | time: samples. Fourier (`X = dt·rfft(x)`, `h = (N−1)//2`): `[Re X_0, √2 Re X_1…√2 Re X_h, √2 Im X_1…√2 Im X_h]`, then `Re X_{N/2}` for even `N`. WDM: active cells in `wilson.grid.packed_cells` order (= `CellSet.all_active`). Rows follow `channel_names` when given; the dict's channel set must then equal it. Validation and errors as `transform`. | T5 |
| `translate.from_coordinates(R, domain, channel_names) -> dict` | canonical layout (inactive WDM cells exactly 0, Fourier end bins exactly real), in `channel_names` order. | T5 |
| `operators.DenseOperator(matrix, domain, channel_names, *, rtol=None)` | an `Operator` on stacked coordinates (channel-major in `channel_names` order). Symmetric to 1e-12 relative, stored `(C+Cᵀ)/2` read-only. `eigh`; `tol = max|λ|·K·N·eps` unless `rtol`; `λ < −tol` raises `ValueError`; keeps `λ > tol`. | T6 |
| `operators.operator_matrix(op) -> ndarray (K·N, K·N)` | column `j` = stacked `to_coordinates(op.apply(from_coordinates(e_j)), op.domain, op.channel_names)`. | T6 |
| `Covariance.multiply(nu, b, be=None)` | `Σ(nu) b`, with `b` in the factor `solve` layout: `(n,)`, or `(C, n_c)` for `ChannelStack`. Per class as in spec §2.5. The base class uses `_matrix @ b`; a class with neither raises `TypeError`. | T7 |
| `ChannelStack._matrix(C, nu, be)` | block-diagonal of the children's `_matrix` at `split(nu)`, in channel order. `multiply(nu, b).reshape(-1) == dense(nu) @ b.reshape(-1)`. | T7 |
| `operators.CovarianceOperator(cov, nu, support, channel_names)` | spec §2.6. `domain = WDMDomain.full(support.cells.grid, support.window)`. Imports `domainia.covariance` lazily inside the constructor. | T8 |
| package surface | `translate.__all__` += `to_coordinates`, `from_coordinates`. `operators.__all__ = ["CovarianceOperator", "DenseOperator", "operator_matrix"]`. `domainia.__all__` += `operators` and those five names. `import domainia` leaves `domainia.covariance` out of `sys.modules`. | T8 |
| `EvolutionaryCovariance(psd, cells, …, window=None)`, `BandAveragedVariance(psd, cells, *, window=None)`, `MidpointVariance(psd, cells)` | `cells` may be a `CellSet` or a `WDMDomain`. `window=None` means `MEYER` for a `CellSet` and the domain's window for a `WDMDomain`. A `WDMDomain` with a different explicit `window` raises `ValueError`. Call sites in `src/` keep passing `window=`. | T9 |
| float32 | a float32/complex64 input runs in single precision end to end (every constant cast to the working dtype first). float64 bits are unchanged. Output dtypes as in the spec §3 table. | T10 |
| `domainia.wilson` wrong kinds | `TypeError` at `grid._require_int`, `grid.validate_grid`, `grid.n_samples`, `noise.colored_noise`'s `N`, `filter.nu_d`'s `d` (kind; the range stays `ValueError`), and `validate_dt` (bool/str/None/complex). | T11 |
| `CellSet.__reduce__` | rebuilds from `(grid, m, n)`. | T12 |
| `built_for` | kw-only meta field (`repr=False`) on `TimeToWDM`, `WDMToTime`, `Restrict`, `FourierZeroPad` = `(src, dst)`; `FourierCrop` = `(src, dst, start)`; `FourierToWDM` = `(src, dst, impl, restrict)`; `TimeToFourier` = `(src, fmin)`. Routes as in spec §5.3. The guard runs after each map's existing validation. | T13 |

## Known collisions between tasks

Several tasks edit the same file. The later task works on the earlier task's result, and its steps name the earlier change.

* `wilson/transform.py`: T2 rescales `forward_bands`/`inverse_bands`; T10 changes their dtypes. T10 keeps T2's factors.
* `maps/fourier.py`: T1 (window replace, units checks), T2 (κ), T3 (pad/crop scale, `after()`), T4 (scale refusal), T13 (`built_for` on three maps), T15 (text).
* `maps/time.py`, `maps/base.py`: T4 removes the scale checks and T13 adds `built_for`. T8 edits the `PushedCovariance._matvec` message in `maps/base.py`.
* `covariance/evolutionary.py`: T7 (`multiply`), T9 (WDMDomain), T14 (`keep_mask`).
* `variance.py`: T9 (WDMDomain), T14 (`t` leaves).
* `domains.py`: T1 (`FourierDomain`), T4 (`WDMDomain`), T12 (`CellSet.__reduce__`).
* `translate.py`: T4 (scale refusal), T5 (coordinates), T8 (`__all__`).
* `tests/test_dom_maps_fourier_wdm.py`, `tests/test_dom_windows.py`, `tests/test_dom_maps_base.py`: T1–T4 and T13.

---

### Task 0: Run the README's python blocks as a test

**Goal:** A test executes every ```` ```python ```` block in `README.md` in order, in one namespace, so the spec's gate "every README python block runs" is part of the suite from the start.

**Files:**
- Create: `tests/test_readme.py`

**Acceptance Criteria:**
- [ ] `tests/test_readme.py::test_readme_python_blocks_run` runs the 3 python blocks of `README.md` and passes.
- [ ] A second test feeds the runner a synthetic README whose second block raises, and asserts that the error names the block's index.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_readme.py` → `2 passed`

---

### Task 1: `FourierDomain.sigma0` and `rfft_scale` (spec §1.2)

**Goal:** `FourierDomain` stores σ0 and exposes `rfft_scale`. Every derived `FourierDomain` keeps `units` and `sigma0` through `dataclasses.replace`. The same-units checks compare `(units, sigma0)`.

**Files:**
- Modify: `src/domainia/domains.py` (`FourierDomain`)
- Modify: `src/domainia/maps/base.py` (`BinSelect`)
- Modify: `src/domainia/maps/fourier.py` (`FourierToWDM.window`, the same-units checks of `FourierZeroPad` and `FourierCrop`)
- Modify: `src/domainia/spectra.py` (`TabulatedSpectrum.on` / `.at` docstrings: unit-blind)
- Test: `tests/test_dom_domains.py`, `tests/test_dom_spectra.py`, `tests/test_dom_maps_base.py`, `tests/test_dom_maps_fourier_wdm.py`

**Acceptance Criteria:**
- [ ] `FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2.0).rfft_scale == 2.0`; `"rfft"` gives `1.0`; `"dt_rfft"` gives `1/dt`.
- [ ] `"sigma0"` without `sigma0`, a non-positive or non-finite `sigma0`, and `sigma0` given with other units each raise `ValueError` whose message names `sigma0`.
- [ ] Two domains that differ only in `sigma0` are unequal, and equal domains hash equal.
- [ ] `BinSelect` and `FourierToWDM.window` carry `sigma0`. `FourierToWDM.window` keeps the grid record for an `after()` map, and the existing `tests/test_dom_maps_fourier_wdm.py` assertions on `Wc.window` still pass.
- [ ] `FourierZeroPad`, `FourierCrop` and `BinSelect` raise `ValueError` when `sigma0` differs between `src` and `dst`.
- [ ] `grep -n "FourierDomain(" src` lists only constructors that build a domain from scratch. Each derived site uses `replace`.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged (bit-exact).

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

---

### Task 2: The band kernels and `FourierToWDM` return Wilson coefficients (spec §1.3, §1.4, §1.7)

**Goal:** `forward_bands`/`inverse_bands` work in Wilson units, and `FourierToWDM` applies `κ = u·sqrt(2 dt/N)` on all its paths, so `FourierToWDM.apply(z)` equals `TimeToWDM.apply(x)`. The Meyer baseline comparator accepts exactly the five rescaled families of spec §1.7.

**Files:**
- Modify: `src/domainia/wilson/transform.py` (`forward_bands`, `inverse_bands`, `band_atoms` docstring)
- Modify: `src/domainia/maps/fourier.py` (`FourierToWDM`: `__post_init__` constants, `apply`, `adjoint`, `rows`, `push_rows`, `after`, class docstring)
- Modify: `tests/wilson/test_meyer_baseline.py`, `tests/wilson/meyer_cases.py` (the shared comparator and the `RESCALED` table)
- Test: `tests/wilson/test_grid_bands.py`, `tests/wilson/test_window_threading.py`, `tests/test_dom_windows.py`, `tests/test_dom_maps_fourier_wdm.py`

**Acceptance Criteria:**
- [ ] `forward_bands(rfft(x), 0, bands, Nf, Nt, dt)` equals `forward(rfft(x))[bands]` to 1e-12 relative on the three baseline grids and both `dt`.
- [ ] `inverse_bands(w, k0, n_bins, bands, …)` equals `inverse(w_full)[k0:k0+n_bins]` to 1e-12, and the adjoint identity holds with the factor `2 dt/N`.
- [ ] For each units value (`rfft`, `dt_rfft`, `sigma0`) and each `impl` (`dense`, `fft`), `FourierToWDM(src, wdm).apply(z)` equals `TimeToWDM(time, wdm).apply(x)` to 1e-12 relative.
- [ ] `restrict=True` gives `C = κ ·` the truncated atoms.
- [ ] White-noise push: `push_rows(N σ² I, all)` on the rfft bins equals `σ² dt I` on the cells to 1e-13, for bands 1..Nf−1 and 2..Nf−2.
- [ ] `test_meyer_baseline.py` and `meyer_cases.py --check` share one comparator with `RESCALED = {(layer, case): f(N, dt)}` for exactly the five families of spec §1.7: `inverse_bands` has `f = 1` and is compared by tolerance. Every other key is `np.array_equal` with the same dtype.
- [ ] `FourierToWDM.after()` keeps its `inner.scale` factor in this task, so `after()` still equals `Compose` for a `FourierZeroPad` inner.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors; `$PY tests/wilson/meyer_cases.py --check` → `baseline reproduces: <K> arrays`

---

### Task 3: `FourierZeroPad` and `FourierCrop` keep units; self-conjugate bins counted once (spec §1.5)

**Goal:** Pad and crop return the exact rfft of the padded or cropped real record in the source's units, with no `scale`. `pad_maps`/`crop_maps` halve the self-conjugate source columns, and `FourierToWDM.after()` drops its `inner.scale` term.

**Files:**
- Modify: `src/domainia/maps/fourier.py` (`pad_maps`, `crop_maps`, `_WidelyLinear` docstring, `FourierZeroPad`, `FourierCrop`, `FourierToWDM.after`)
- Test: `tests/test_dom_maps_dirichlet.py`, `tests/test_dom_maps_fourier_wdm.py`

**Acceptance Criteria:**
- [ ] Pad and crop applied to full-record bins with non-zero DC and Nyquist equal the rfft of the padded or cropped series to 1e-13 relative, for even and odd `N`.
- [ ] Headline test: `FourierToWDM(window′, wdm′).after(FourierZeroPad(store, window′)).apply(z)` and `Compose(FourierToWDM(…), FourierZeroPad(…)).apply(z)` both equal `TimeToWDM(time′, wdm′).apply(concat(x, zeros))` to 1e-12 relative.
- [ ] `FourierZeroPad` has no `scale` field, and `FourierCrop` has no `scale` attribute.
- [ ] Both maps raise `ValueError` for a `(units, sigma0)` mismatch.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline passes, with the five Task 2 families still the only rescaled keys.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

---

### Task 4: Remove `WDMDomain.scale` (spec §1.6)

**Goal:** `WDMDomain(cells, window=MEYER)` has no `scale`. `require_unit_scale` and every scale refusal and scale docstring are gone.

**Files:**
- Modify: `src/domainia/domains.py` (`WDMDomain`, the module docstring)
- Modify: `src/domainia/maps/time.py` (`TimeToWDM`, `WDMToTime`), `src/domainia/maps/fourier.py` (`FourierToWDM`), `src/domainia/maps/base.py` (`Restrict`, the `Identity` docstring)
- Modify: `src/domainia/translate.py` (the scale refusal)
- Test: `tests/test_translate.py`, `tests/wilson/test_window_threading.py`, `tests/test_dom_maps_base.py`, `tests/test_dom_domains.py`, `tests/wilson/test_windows.py`

**Acceptance Criteria:**
- [ ] `inspect.signature(WDMDomain)` lists exactly `cells, window`, and `WDMDomain(cells, 2.0)` raises `TypeError` (the window check).
- [ ] `grep -rn "require_unit_scale" src tests` prints nothing. `grep -rnE "\.scale\b" src` prints nothing, or only hits that are not on a `WDMDomain` (each listed in the step).
- [ ] The `"wdm-scale"` refusal case in `tests/test_translate.py` is removed.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged from Task 3.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

---

### Task 5: Public real coordinates `to_coordinates` / `from_coordinates` (spec §2.2)

**Goal:** `domainia.translate` exposes the real coordinates that the `Operator` contract already describes, with channel order pinned by `channel_names`.

**Files:**
- Modify: `src/domainia/translate.py`
- Create: `tests/test_translate_coordinates.py`
- Modify: `tests/test_translate_operator.py` (its private `_pack`/`_unpack` replaced by the public functions)

**Acceptance Criteria:**
- [ ] On each full-record domain (time, Fourier for even and odd `N`, WDM), `to_coordinates` returns `(K, N)` float64 in the layout of the Interface contract, and `from_coordinates` inverts it exactly to the canonical layout.
- [ ] WDM coordinates follow `CellSet.all_active(grid)` order (asserted).
- [ ] Each domain's coordinates are `exp(log_coordinate_scale(domain))` times an orthogonal map of the time samples (`QᵀQ = I` to 1e-12, measured on a basis).
- [ ] With `channel_names=("A", "E")`, a dict built as `{"E": …, "A": …}` gives rows in `A, E` order, and a mismatched channel set raises `ValueError`.
- [ ] Input refusals match `transform`'s (same exception type and message) for every refusal case `tests/test_translate.py` exercises.
- [ ] `tests/test_translate_operator.py` no longer defines `_pack` or `_unpack`, and passes.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_translate_coordinates.py tests/test_translate_operator.py` → all pass; then `$PY -m pytest -q` → all pass

---

### Task 6: `DenseOperator` and `operator_matrix` in the new `domainia.operators` (spec §2.3, §2.4)

**Goal:** An explicit covariance matrix becomes an `Operator`, and any `Operator` can be read back as its matrix in stacked coordinates, so a matrix can be translated between domains.

**Files:**
- Create: `src/domainia/operators.py`
- Create: `tests/test_operators_dense.py`
- Modify: `tests/wilson/test_window_threading.py` (`SCANNED` gains `operators.py`)

**Acceptance Criteria:**
- [ ] `DenseOperator` validation: a wrong shape, non-finite entries, asymmetry above 1e-12 relative, or an eigenvalue below `−tol` each raise `ValueError`. Wrong kinds raise `TypeError`. The stored matrix is `(C+Cᵀ)/2` and read-only.
- [ ] For a full-rank and a rank-deficient PSD matrix on each domain kind: `degrees_of_freedom`, `project = V Vᵀ`, `solve = V Λ⁻¹ Vᵀ`, `quadratic_form`, and `log_determinant = Σ log λ` match dense numpy references to 1e-10 relative. Outputs meet the `Operator` output contract (canonical layout).
- [ ] A dict with reversed channel order gives the same results as the ordered dict, on `DenseOperator` and on a `translate_operator` of it.
- [ ] `operator_matrix(translate_operator(DenseOperator(C, time, ("A",)), wdm))` equals `W C Wᵀ` with `W` from `wilson.matrix.active_W`, to 1e-12 relative.
- [ ] `operator_matrix(DenseOperator(C, …))` returns `(C + Cᵀ)/2`.
- [ ] `test_the_scan_covers_every_module_but_filter_and_windows` passes with `operators.py` in `SCANNED`.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_operators_dense.py tests/wilson/test_window_threading.py` → all pass; then `$PY -m pytest -q` → all pass

---

### Task 7: `Covariance.multiply` and `ChannelStack._matrix` (spec §2.5)

**Goal:** Every domainia covariance can multiply a vector by `Σ(nu)` in its factor layout, and `ChannelStack` gains a dense form.

**Files:**
- Modify: `src/domainia/covariance/base.py`, `src/domainia/covariance/diagonal.py`, `src/domainia/covariance/dense.py`, `src/domainia/covariance/blocktri.py`, `src/domainia/covariance/evolutionary.py`, `src/domainia/covariance/stack.py`
- Create: `tests/test_dom_cov_multiply.py`

**Acceptance Criteria:**
- [ ] `cov.multiply(nu, b)` equals `cov.dense(nu) @ b` to 1e-12 relative for `DiagonalCovariance`, `DenseCovariance`, `BlockTridiagonalCovariance` (each `layout`), `EvolutionaryCovariance` (blocktri and dense assembly), and a two-channel `ChannelStack` (flattened row-major).
- [ ] `BlockTridiagonalCovariance.multiply` never forms the dense matrix: the test makes `_matrix` and `blocks_to_dense` raise and `multiply` still succeeds.
- [ ] `ChannelStack(...).dense(nu)` is the block-diagonal of the children's `dense` at `split(nu)`.
- [ ] A `Covariance` subclass with neither `multiply` nor `_matrix` raises `TypeError` from `multiply`.
- [ ] `grep -rn "def matvec\|\.matvec(" src` hits are unchanged (`PushedCovariance` keeps its duck-typed `matvec(X, be)`).
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_dom_cov_multiply.py` → all pass; then `$PY -m pytest -q` → all pass

---

### Task 8: `CovarianceOperator` and the package surface (spec §2.6, §2.8)

**Goal:** A domainia covariance fixed at `nu` on a (possibly partial) WDM support becomes an `Operator`. The new names are exported, and `import domainia` still does not import `domainia.covariance`.

**Files:**
- Modify: `src/domainia/operators.py` (`CovarianceOperator`)
- Modify: `src/domainia/maps/base.py` (`PushedCovariance._matvec` error message)
- Modify: `src/domainia/translate.py` (`__all__`), `src/domainia/__init__.py` (`__all__`, imports, docstring)
- Create: `tests/test_operators_covariance.py`
- Modify: `tests/test_package.py` (`PUBLIC`, the lazy-import check)

**Acceptance Criteria:**
- [ ] Validation per spec §2.6: wrong kinds raise `TypeError`; a size, channel-count, window, cells or `n_u` mismatch raises `ValueError`; `factor(nu).ok` false raises `ValueError("… not positive definite at nu")` at construction.
- [ ] `operator_matrix(op) = Pᵀ Σ P` (with `cov.dense(nu)`) to 1e-12 relative for a diagonal, a block-tridiagonal, a small-grid `EvolutionaryCovariance` (blocktri and dense assembly), a `DenseCovariance`, a two-channel `ChannelStack`, a band-subset support and a `drop_columns` support.
- [ ] Translated to time, `operator_matrix` equals `Wᵀ Pᵀ Σ P W / dt²`. Translated to Fourier, it equals the same reference mapped by the Fourier coordinates. Both to 1e-10 relative.
- [ ] The translated `quadratic_form` and `log_determinant` match the dense references, including the pseudo-determinant for a partial support.
- [ ] A reversed-channel dict gives the same results as the ordered dict.
- [ ] `PushedCovariance`'s WDM → time refusal message names `CovarianceOperator`.
- [ ] `domainia.__all__` and `domainia.translate.__all__` contain the new names and `tests/test_package.py` `PUBLIC` matches. A fresh-subprocess `import domainia` leaves `domainia.covariance` out of `sys.modules`.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_operators_covariance.py tests/test_package.py` → all pass; then `$PY -m pytest -q` → all pass

---

### Task 9: The cell-set consumers accept a `WDMDomain` (spec §2.7)

**Goal:** `EvolutionaryCovariance`, `BandAveragedVariance` and `MidpointVariance` take a `WDMDomain` where they take `cells`, and read its window, so a non-Meyer domain can no longer fall back to Meyer silently.

**Files:**
- Modify: `src/domainia/covariance/evolutionary.py` (`EvolutionaryCovariance.__init__`)
- Modify: `src/domainia/variance.py` (`BandAveragedVariance`, `MidpointVariance`)
- Test: `tests/test_dom_evolutionary_cov.py`, the variance tests that `grep -ln "BandAveragedVariance\|MidpointVariance" tests` finds, `tests/test_dom_windows.py`

**Acceptance Criteria:**
- [ ] With `dom = WDMDomain(cells, window=ALT_WINDOW)`, each of the three built on `dom` equals the same class built on `(cells, window=ALT_WINDOW)` (`dense`/values equal, `window` attribute equal; `MidpointVariance` takes only the cells).
- [ ] `window=None` on a `CellSet` gives `MEYER`. A `WDMDomain` with a different explicit `window` raises `ValueError`. A non-`CellSet`, non-`WDMDomain` `cells` raises `TypeError`.
- [ ] The message "for a WDMDomain pass domain.cells" is gone (`grep -rn "pass domain.cells" src` prints nothing).
- [ ] The window scan (`tests/wilson/test_window_threading.py`) passes unchanged: `src/` call sites still pass `window=`.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged from Task 3.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

---

### Task 10: Single precision throughout (spec §3)

**Goal:** A float32/complex64 input to the transforms runs in single precision end to end, on NumPy and on JAX. float64 bits stay identical.

**Files:**
- Modify: `src/domainia/wilson/transform.py` (the forward post-FFT products and edge divides; the inverse family; `forward_bands`; `inverse_bands`; the "entirely in single precision" docstrings; the `_add_rounded`/`_set_rounded` comments)
- Create: `tests/wilson/test_single_precision.py`
- Modify: `tests/wilson/test_transform_shim.py` (compare dtypes against domainia's table, not the oracle's)

**Acceptance Criteria:**
- [ ] A recording backend that wraps the NumPy backend's `fft` namespace sees only float32/complex64 FFT inputs in a float32 run of every entry point in the spec §3 table. On the old code this test fails for the inverse family and the band kernels, and passes for the forward family.
- [ ] A test on the forward family's post-FFT product (the array handed to `_set_rounded` in a float32 run is single precision) fails on the old code and passes after.
- [ ] The output dtypes match the spec §3 table for every entry point (one parametrised test).
- [ ] The `jax`-marked variant of the recording test passes in the wdmrel env.
- [ ] `ORACLE_RTOL[float32] = 2e-6` still holds in `tests/wilson/test_transform_shim.py`.
- [ ] The Meyer baseline is unchanged from Task 3 (float64 is bit-exact).
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/wilson/test_single_precision.py tests/wilson/test_transform_shim.py tests/wilson/test_meyer_baseline.py` → all pass; then `$PY -m pytest -q` → all pass

---

### Task 11: Wrong kinds raise `TypeError` in `domainia.wilson` (spec §4)

**Goal:** `domainia.wilson` follows the rest of domainia: a wrong kind raises `TypeError`, and a wrong value raises `ValueError`.

**Files:**
- Modify: `src/domainia/wilson/grid.py` (`_require_int`, `validate_grid`, `n_samples`, `validate_dt`), `src/domainia/wilson/noise.py` (`colored_noise`), `src/domainia/wilson/filter.py` (`nu_d`)
- Create: `tests/wilson/test_wrong_kinds.py`
- Modify: `tests/wilson/test_noise_stationary.py`, `tests/wilson/vendored/test_filter.py`

**Acceptance Criteria:**
- [ ] Each site in the Interface contract raises `TypeError` for a non-integer or `bool` (or, for `validate_dt`, a bool/str/None/complex) with the same message text as before, and `ValueError` for an out-of-range value.
- [ ] `nu_d(…, d=2.5)` raises `TypeError("d must be an integer, got float")`. `d=0` and `d=-1` raise `ValueError` matching `"d must be >= 1"`.
- [ ] The frozen oracle `tests/wilson/oracle/vendored_2f7136c` is untouched (`git diff --stat` shows nothing under it), and its sha pins pass.
- [ ] The touched docstrings' "Raises" sections name the right exception.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/wilson` → all pass; then `$PY -m pytest -q` → all pass

---

### Task 12: The smaller fixes (spec §5.1, §5.2, §5.4, §5.5)

**Goal:** `psd_from_K` uses a relative threshold, `CellSet` pickles and copies to an equal read-only set, the 2.0 spec's endpoint sentence is accurate, and the sdist leaves out `docs/superpowers`. Each fix is its own commit.

**Files:**
- Modify: `src/domainia/wilson/stationary.py` (`psd_from_K`)
- Modify: `src/domainia/domains.py` (`CellSet.__reduce__`)
- Modify: `docs/superpowers/specs/2026-10-31-domainia-general-library-design.md` (the line-229 sentence)
- Modify: `pyproject.toml` (`[tool.hatch.build.targets.sdist]`)
- Test: `tests/wilson/test_noise_stationary.py` (or the module that `grep -ln psd_from_K tests/wilson` finds), `tests/test_dom_domains.py`, `tests/test_package.py`

**Acceptance Criteria:**
- [ ] At `dt = 1e-33` with a flat PSD, `psd_from_K(build_K_full_active(…))` recovers the PSD as it does at `dt = 1` (the test fails on the old code, where every bin is dropped). The Meyer entry `stationary.psd_from_K` stays bit-exact.
- [ ] `pickle`, `copy.copy` and `copy.deepcopy` of a `CellSet` give an equal set with an equal hash and read-only `m_array`/`n_array`, and the copy does not share the original's arrays.
- [ ] The 2.0 spec reads "`transform`'s outputs make the endpoints exactly real".
- [ ] A `tomllib` test asserts `docs/superpowers` is in the sdist `exclude`.
- [ ] Four commits, one per fix. The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

---

### Task 13: The `built_for` guard on constant-carrying maps (spec §5.3)

**Goal:** A map whose constants were built for one `src`/`dst` (and parameters) refuses to run with another, on every construction route, so `dataclasses.replace(map, dst=…)` can no longer reuse stale constants.

**Files:**
- Modify: `src/domainia/maps/time.py` (`TimeToWDM`, `WDMToTime` and its `adjoint`, `TimeToFourier`)
- Modify: `src/domainia/maps/base.py` (`Restrict`; the module docstring's `dataclasses.replace` warning)
- Modify: `src/domainia/maps/fourier.py` (`FourierZeroPad`, `FourierCrop`, `FourierToWDM`, `FourierToWDM.after`)
- Create: `tests/test_dom_built_for.py`

**Acceptance Criteria:**
- [ ] For each of the seven maps: a normal build sets `built_for` to the key in the Interface contract. `dataclasses.replace` that changes a key field raises `ValueError` ("built for another src/dst: construct a new map"). `replace` that changes nothing keeps working. Constants supplied without `built_for` raise `ValueError`.
- [ ] `after()` maps carry the key of the pair they were built for, and still equal `Compose`.
- [ ] `WDMToTime.adjoint` works (it passes `built_for`). `grep` over `src/` for constant-passing constructor calls finds no other site without `built_for`.
- [ ] Pytree flatten/unflatten round-trips each map, and `jax.jit` over `apply` works (`jax`-marked).
- [ ] `tests/test_dom_windows.py`'s `Restrict(..., pos=…)` window-mismatch test keeps its message.
- [ ] `built_for` is not in any map's `repr`.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_dom_built_for.py` → all pass; then `$PY -m pytest -q` → all pass

---

### Task 14: Dead code and duplicates (spec §6)

**Goal:** The unread items in spec §6 are removed, after a grep for each, and `_oracle_paths` is defined once.

**Files:**
- Modify: `src/domainia/covariance/base.py` (`batchable`, `complete_constants`), `src/domainia/spectra.py` (`GenericPSD.capabilities`), `src/domainia/variance.py` (`BandAveragedVariance.t`, `MidpointVariance.t`), `src/domainia/covariance/evolutionary.py` and `src/domainia/covariance/__init__.py` (`keep_mask`)
- Modify: `tests/test_dom_spectra.py`, `tests/test_dom_cov_api.py`, `tests/test_dom_evolutionary_cov.py`, `tests/wilson/conftest.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_grid_bands.py`

**Acceptance Criteria:**
- [ ] Before each removal, a recorded grep shows no reader in `src/`, `tests/` or `README.md` other than a test that only pins the item. Any item with a real reader is kept and listed in the errata with the reader.
- [ ] `EvolutionaryCovariance` calls `cells.keep_mask()`, and `grep -rn "keep_mask(cells)\|import.*keep_mask\|\"keep_mask\"" src tests` prints nothing.
- [ ] `BandAveragedVariance.on()` and `MidpointVariance.on()` still exist, and `as_evaluator` still recognises both classes.
- [ ] `_oracle_paths` is defined once, in `tests/wilson/conftest.py`, and both former definers use it. The `DOMAINIA_ORACLE_ROOT` skips still appear (3 skips).
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged from Task 3.

**Verify:** `$PY -m pytest -q` → all pass, 3 skipped, 0 errors

---

### Task 15: Text sweep (spec §7; last)

**Goal:** Comments, docstrings and README prose say what the code does in plain words. Shorthand, 1.x-spec references and the `wilson as wdm` doctest alias are gone, and README statements made false by Tasks 1–14 are corrected.

**Files:**
- Modify: every `src/domainia/**/*.py` with a hit, `README.md`, and the tests with hits outside `tests/wilson/oracle/` and `tests/wilson/vendored/`

**Acceptance Criteria:**
- [ ] The shorthand regex (WT/WG/GB/"WDM II", defined in the step) finds zero hits in `src/`, `README.md` and the non-frozen tests. Paper references are spelled out ("WDM Paper II, Eq. 36"), and `maps/fourier.py`'s "Ported from the GB toy-model oracle" line is reworded per spec §7.
- [ ] The 1.x spec-reference regex (defined in the step) finds zero hits.
- [ ] `grep -rn "wilson as wdm" src tests README.md` prints nothing outside the frozen oracle and the vendored tests, and the doctests pass.
- [ ] README prose mentions no `WDMDomain.scale`, and describes `FourierToWDM` in Wilson units and the `sigma0` unit.
- [ ] The full suite (doctests included), the NumPy lane and `tests/test_readme.py` pass.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors; `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q` → all pass

---

## Execution errata

(Filled in during execution.)
