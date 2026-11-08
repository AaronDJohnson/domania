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
* **`$PY`** is shorthand for the prefix `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P`, run from `$REPO`. The Bash tool runs zsh, which does not word-split a variable, so `PY="…"; $PY -m pytest` fails. Write the prefix out in full in each command, or set `PY=…` and use `${=PY}` in the same command. Quote globs too (`--include='*.py'`): an unmatched glob is an error in zsh.
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
| `Covariance.multiply(nu, b, be=None)` | `Σ(nu) b`, with `b` in the factor `solve` layout: `(n,)`, or `(C, n_c)` for `ChannelStack`. Defined once on the base class; the per-class hook is `_multiply(C, nu, b, be)`, as in spec §2.5. The base `_multiply` is `_matrix(C, nu, be) @ b`; a class with neither `_multiply` nor `_matrix` raises `TypeError`. | T7 |
| `ChannelStack._matrix(C, nu, be)` | block-diagonal of the children's `_matrix` at `split(nu)`, in channel order. `multiply(nu, b).reshape(-1) == dense(nu) @ b.reshape(-1)`. | T7 |
| `operators.CovarianceOperator(cov, nu, support, channel_names)` | spec §2.6. `domain = WDMDomain.full(support.cells.grid, support.window)`. Imports `domainia.covariance` lazily inside the constructor. | T8 |
| package surface | `translate.__all__` += `to_coordinates`, `from_coordinates`. `operators.__all__ = ["CovarianceOperator", "DenseOperator", "operator_matrix"]`. `domainia.__all__` += `operators` and those five names. `import domainia` leaves `domainia.covariance` out of `sys.modules`. | T8 |
| `EvolutionaryCovariance(psd, cells, …, window=None)`, `BandAveragedVariance(psd, cells, *, window=None)`, `MidpointVariance(psd, cells)` | `cells` may be a `CellSet` or a `WDMDomain`, resolved by the shared helper `domains._cells_and_window` (its `default=MEYER` is a parameter default, which the window scan accepts). `window=None` means `MEYER` for a `CellSet` and the domain's window for a `WDMDomain`. A `WDMDomain` with a different explicit `window` raises `ValueError`. Call sites in `src/` keep passing `window=`. | T9 |
| float32 | a float32/complex64 input runs in single precision end to end (every constant cast to the working dtype first). float64 bits are unchanged. Output dtypes as in the spec §3 table. | T10 |
| `domainia.wilson` wrong kinds | `TypeError` at `grid._require_int`, `grid.validate_grid`, `grid.n_samples`, `noise.colored_noise`'s `N`, `filter.nu_d`'s `d` (kind; the range stays `ValueError`), and `validate_dt` (bool/str/None/complex). | T11 |
| `CellSet.__reduce__` | rebuilds from `(grid, m, n)`. | T12 |
| `built_for` | kw-only meta field (`repr=False`), checked by the shared helper `maps.base._built_for(m, key, *, given)` on `TimeToWDM`, `WDMToTime`, `Restrict`, `FourierZeroPad` = `(src, dst)`; `FourierCrop` = `(src, dst, start)`; `FourierToWDM` = `(src, dst, impl, restrict)`; `TimeToFourier` = `(src, fmin)`. Routes as in spec §5.3. The guard runs after each map's existing validation. | T13 |

## Known collisions between tasks

Several tasks edit the same file. The later task works on the earlier task's result, and its steps name the earlier change.

* `wilson/transform.py`: T2 rescales `forward_bands`/`inverse_bands`; T10 changes their dtypes. T10 keeps T2's factors.
* `maps/fourier.py`: T1 (window replace, units checks), T2 (κ), T3 (pad/crop scale, `after()`), T4 (scale refusal), T13 (`built_for` on three maps), T15 (text).
* `maps/time.py`, `maps/base.py`: T4 removes the scale checks and T13 adds `built_for`. T8 edits the `PushedCovariance._matvec` message in `maps/base.py`.
* `covariance/evolutionary.py`: T7 (`multiply`), T9 (WDMDomain), T14 (`keep_mask`).
* `variance.py`: T9 (WDMDomain), T14 (`t` leaves).
* `domains.py`: T1 (`FourierDomain`), T4 (`WDMDomain`), T9 (`_cells_and_window`), T12 (`CellSet.__reduce__`).
* `translate.py`: T4 (scale refusal), T5 (coordinates), T8 (`__all__`).
* `tests/test_dom_maps_fourier_wdm.py`, `tests/test_dom_windows.py`, `tests/test_dom_maps_base.py`: T1–T4 and T13.

---

### Task 0: Run the README's python blocks as a test

**Goal:** A test executes every ```` ```python ```` block in `README.md` in order, in one namespace, so the spec's gate "every README python block runs" is part of the suite from the start.

**Files:**
- Create: `tests/test_readme.py`

**Acceptance Criteria:**
- [ ] `tests/test_readme.py::test_readme_python_blocks_run` runs every python block of `README.md` (3 today) and checks that the number it ran equals the number of ```` ```python ```` fences, so Task 15 can add a block without editing the test.
- [ ] A second test feeds the runner a synthetic README whose second block raises, and asserts that the error names the block's index.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_readme.py` → `2 passed`

**Steps:**

Assumes the tree at 75e3d9e (the plan skeleton; no earlier task).

- [ ] **Step 1: Look at the README's fences**

Run:

````bash
grep -n '^```' README.md
````

Expected hits:

````text
29:```python
45:```
69:```python
88:```
112:```python
126:```
136:```bash
139:```
````

Three ```` ```python ```` blocks (opening fences at lines 29, 69 and 112) and one ```` ```bash ```` block, which the test must skip. Each python block imports what it uses, and they run in one namespace in order.

- [ ] **Step 2: Write `tests/test_readme.py`**

The block runner is a module-level helper so that the second test can call it on a synthetic README. It reports a failing block as an `AssertionError` that names the block's 1-based index and the line of its opening fence, chained to the original exception. The first test does not pin the count 3: it checks that the fence pattern found every ```` ```python ```` fence, so a block added to the README later (Task 15) is run without editing the test.

````python
"""README.md's python blocks run, in order, in one namespace (the spec's README gate)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
#: A fenced ```python block: the opening fence, the body, the closing fence, each on its own line.
FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def run_python_blocks(text: str, name: str, namespace: dict | None = None) -> int:
    """Run every ```python block of the Markdown ``text`` in order, in one ``namespace``.

    Returns the number of blocks run.  A block that raises is reported as an
    ``AssertionError`` naming the block (1-based) and the line of its opening fence in
    ``name``, chained to the original exception.
    """
    ns = {"__name__": "__readme__"} if namespace is None else namespace
    blocks = list(FENCE.finditer(text))
    for i, block in enumerate(blocks, start=1):
        line = text.count("\n", 0, block.start()) + 1      # the opening fence
        where = f"{name} python block {i} of {len(blocks)} (line {line})"
        code = compile(block.group(1), f"<{where}>", "exec")
        try:
            exec(code, ns)
        except Exception as exc:
            raise AssertionError(f"{where} raised {type(exc).__name__}: {exc}") from exc
    return len(blocks)


def test_readme_python_blocks_run():
    text = README.read_text()
    n = run_python_blocks(text, "README.md")
    assert n >= 1 and n == text.count("```python")          # the fence pattern missed none


def test_a_failing_block_is_named_by_its_index():
    """Block 2 reads block 1's name (one namespace) and raises; block 3 never runs, and the
    ```bash block in between is not python."""
    text = ("# Title\n\n```python\na = 1\n```\n\nSome prose.\n\n```bash\nexit 1\n```\n\n"
            "```python\nb = 1 / (a - 1)\n```\n\n```python\nc = 3\n```\n")
    ns: dict = {}
    with pytest.raises(AssertionError, match=r"^SYNTHETIC\.md python block 2 of 3 \(line 13\) "
                                             r"raised ZeroDivisionError") as info:
        run_python_blocks(text, "SYNTHETIC.md", ns)
    assert isinstance(info.value.__cause__, ZeroDivisionError)
    assert ns["a"] == 1 and "b" not in ns and "c" not in ns
````

- [ ] **Step 3: Run the new tests**

Run: `$PY -m pytest -q tests/test_readme.py`
Expected: `2 passed`

- [ ] **Step 4: Check that the gate fails on a broken README, naming the block**

The runner is test code, so its red check is the gate catching a broken block. Save this scratch script as `$TMPDIR/readme_red.py` (do not commit it). It breaks the assertion in the third block in memory; `README.md` is not touched.

```python
"""Task 0 red check: the README gate fails, naming the block, when a README block breaks."""
import sys
from pathlib import Path

sys.path.insert(0, "tests")
from test_readme import README, run_python_blocks  # noqa: E402

text = README.read_text().replace("assert np.isclose(on_wdm.quadratic_form(w), op.quadratic_form(x))",
                                  "assert np.isclose(on_wdm.quadratic_form(w), 2 * op.quadratic_form(x))")
try:
    run_python_blocks(text, "README.md")
except AssertionError as exc:
    print(exc)
else:
    raise SystemExit("the gate did not fail")
```

Run: `PYTHONPATH=src $PY $TMPDIR/readme_red.py`
Expected: `README.md python block 3 of 3 (line 112) raised AssertionError: ` (the script prints the gate's message and exits 0).

Run: `git status --short README.md`
Expected: no output (README.md unchanged).

- [ ] **Step 5: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: `1462 passed, 3 skipped` (the 3 skips are the `DOMAINIA_ORACLE_ROOT` oracle tests).

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: `1373 passed, 92 skipped`

- [ ] **Step 6: Commit (sandbox disabled)**

```bash
git add tests/test_readme.py
git commit -m "Run the README's python blocks as a test

tests/test_readme.py runs every python block of README.md in order in one
namespace, so the spec gate (every README python block runs) is part of the
suite. A failing block is reported by its index and the line of its fence.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: `FourierDomain.sigma0` and `rfft_scale` (spec §1.2)

**Goal:** `FourierDomain` stores σ0 and exposes `rfft_scale`. Every derived `FourierDomain` keeps `units` and `sigma0` through `dataclasses.replace`. The same-units checks compare `(units, sigma0)`.

**Files:**
- Modify: `src/domainia/domains.py` (`FourierDomain`)
- Modify: `src/domainia/maps/base.py` (`BinSelect`)
- Modify: `src/domainia/maps/fourier.py` (`FourierToWDM.window`, the same-units checks of `FourierZeroPad` and `FourierCrop`)
- Modify: `src/domainia/spectra.py` (`TabulatedSpectrum.on` / `.at` docstrings: unit-blind)
- Modify: `src/domainia/maps/base.py` also gains the shared helper `_require_same_units`, used by `BinSelect`, `FourierZeroPad` and `FourierCrop`
- Test: `tests/test_dom_domains.py`, `tests/test_dom_spectra.py`, `tests/test_dom_maps_base.py`, `tests/test_dom_maps_fourier_wdm.py`, `tests/test_dom_maps_dirichlet.py` (the pad/crop `(units, sigma0)` mismatch test)

**Acceptance Criteria:**
- [ ] `FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2.0).rfft_scale == 2.0`; `"rfft"` gives `1.0`; `"dt_rfft"` gives `1/dt`.
- [ ] `"sigma0"` without `sigma0`, a non-positive or non-finite `sigma0`, and `sigma0` given with other units each raise `ValueError` whose message names `sigma0`.
- [ ] Two domains that differ only in `sigma0` are unequal, and equal domains hash equal.
- [ ] `BinSelect` and `FourierToWDM.window` carry `sigma0`. `FourierToWDM.window` keeps the grid record for an `after()` map, and the existing `tests/test_dom_maps_fourier_wdm.py` assertions on `Wc.window` still pass.
- [ ] `FourierZeroPad`, `FourierCrop` and `BinSelect` raise `ValueError` when `sigma0` differs between `src` and `dst`.
- [ ] `grep -n "FourierDomain(" src` lists only constructors that build a domain from scratch. Each derived site uses `replace`.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged (bit-exact).

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

**Steps:**

Assumes the tree after Task 0.

- [ ] **Step 1: Grep for the derived domains, the units checks and the `"sigma0"` pins**

Run:

```bash
grep -rn "FourierDomain(" src
```

Expected hits:

```text
src/domainia/maps/time.py:184:            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n, "rfft"))
src/domainia/maps/fourier.py:285:        return FourierDomain(g.N, g.dt, geo["g0"], geo["G"], self.src.units)
src/domainia/maps/base.py:403:            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n,
```

* `maps/time.py` (`TimeToFourier.__post_init__`): builds its `dst` from a `TimeDomain`, from scratch, in `"rfft"` units. Keep it.
* `maps/base.py` (`BinSelect.__post_init__`) and `maps/fourier.py` (`FourierToWDM.window`): derive a domain from `src`. Both become `dataclasses.replace` (Steps 5 and 6).

Run:

```bash
grep -rn "\.units\b" src
```

Expected hits:

```text
src/domainia/maps/fourier.py:103:            if d.N <= s.N or d.dt != s.dt or d.units != s.units:
src/domainia/maps/fourier.py:141:            if not (0 <= self.start and self.start + d.N <= s.N) or d.dt != s.dt or d.units != s.units:
src/domainia/maps/fourier.py:285:        return FourierDomain(g.N, g.dt, geo["g0"], geo["G"], self.src.units)
src/domainia/maps/base.py:404:                                                          self.src.units))
src/domainia/maps/base.py:406:        if (d.N, d.dt, d.units) != (s.N, s.dt, s.units) or d.k0 < s.k0 or d.k0 + d.n > s.k0 + s.n:
src/domainia/translate.py:53:        if domain.units != "dt_rfft" or domain.k0 != 0 or domain.n != domain.N // 2 + 1:
src/domainia/domains.py:107:        if self.units not in FOURIER_UNITS:
src/domainia/domains.py:108:            raise ValueError(f"units must be one of {FOURIER_UNITS}; got {self.units!r}")
```

* `translate.py` (`_record`): accepts only `FourierDomain.full` (`"dt_rfft"`, where `sigma0` is `None` by construction). Keep it.
* `domains.py`: the constructor's own check. Keep it.
* `maps/fourier.py` (`FourierZeroPad`, `FourierCrop`, `FourierToWDM.window`) and `maps/base.py` (`BinSelect`): become `(units, sigma0)` checks or `replace` (Steps 5 and 6).

Run:

```bash
grep -rn '"sigma0"' tests
```

Expected hits:

```text
tests/test_dom_domains.py:28:    d = FourierDomain(64, 0.5, 3, 10, "sigma0")
tests/test_dom_spectra.py:13:    dom = FourierDomain(4096, 5.0, 100, 50, "sigma0")
tests/test_dom_spectra.py:17:    assert np.array_equal(s.on(FourierDomain(4096, 5.0, 110, 5, "sigma0")), v[:, 10:15])
tests/test_dom_spectra.py:18:    assert np.array_equal(s.on(domain=FourierDomain(4096, 5.0, 110, 5, "sigma0")), v[:, 10:15])
tests/test_dom_spectra.py:25:        s.on(FourierDomain(4096, 5.0, 140, 20, "sigma0"))
tests/test_dom_spectra.py:29:        s.on(FourierDomain(8192, 5.0, 110, 5, "sigma0"))
tests/test_dom_maps_base.py:112:    store = FourierDomain(4096, 5.0, 100, 60, "sigma0")
tests/test_dom_maps_base.py:114:    assert S.dst == FourierDomain(4096, 5.0, 110, 30, "sigma0") and S.kind == "complex"
```

Every hit builds a `"sigma0"` domain without `sigma0`, which Step 4 makes a `ValueError`. Step 2 gives each one `sigma0=3.0` (`test_dom_spectra.py`, `test_dom_maps_base.py`) or `sigma0=2.0` (`test_dom_domains.py`).

- [ ] **Step 2: Write the failing tests**

Domains: the new field and property, and the `sigma0` the `"sigma0"` domains now need.

`tests/test_dom_domains.py`, `test_fourier_domain` (its first line):

```python
def test_fourier_domain():
    d = FourierDomain(64, 0.5, 3, 10, "sigma0")
```

→

```python
def test_fourier_domain():
    d = FourierDomain(64, 0.5, 3, 10, "sigma0", sigma0=2.0)
```

`tests/test_dom_domains.py`, after `test_fourier_domain` (add a test):

```python
        FourierDomain(64, 0.5, 0, 10, "hz")


```

→

```python
        FourierDomain(64, 0.5, 0, 10, "hz")


def test_fourier_units_carry_sigma0_and_the_rfft_scale():
    """``rfft_scale`` is ``u`` with ``numpy.fft.rfft(x) = u z`` (spec section 1.1), and
    ``sigma0`` is part of the domain's identity."""
    assert FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2.0).rfft_scale == 2.0
    assert FourierDomain(64, 0.5, 0, 33, "rfft").rfft_scale == 1.0
    assert FourierDomain(64, 0.5, 0, 33, "dt_rfft").rfft_scale == 1 / 0.5
    assert FourierDomain.full(64, 0.5).rfft_scale == 1 / 0.5
    x = np.random.default_rng(0).standard_normal(64)
    R = np.fft.rfft(x)
    for units, z, kw in (("rfft", R, {}), ("dt_rfft", 0.5 * R, {}),
                         ("sigma0", R / 2.0, {"sigma0": 2.0})):
        d = FourierDomain(64, 0.5, 0, 33, units, **kw)
        np.testing.assert_allclose(d.rfft_scale * z, R, rtol=1e-15, atol=0)
    for bad in (None, 0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="sigma0"):
            FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=bad)
    for units in ("rfft", "dt_rfft"):
        with pytest.raises(ValueError, match="sigma0"):
            FourierDomain(64, 0.5, 0, 33, units, sigma0=2.0)
    with pytest.raises(TypeError):
        FourierDomain(64, 0.5, 0, 33, "sigma0", 2.0)                    # sigma0 is keyword-only
    a = FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2)
    assert type(a.sigma0) is float and a.sigma0 == 2.0
    b = FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2.0)
    assert a == b and hash(a) == hash(b)
    assert a != FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=3.0)
    assert FourierDomain(64, 0.5, 0, 33, "rfft").sigma0 is None


```

Spectra: give the table domains a `sigma0`, and pin that `on` is unit-blind.

`tests/test_dom_spectra.py`, `test_tabulated_spectrum_slices_by_absolute_bin_and_refuses_outside`:

```python
    dom = FourierDomain(4096, 5.0, 100, 50, "sigma0")
```

→

```python
    dom = FourierDomain(4096, 5.0, 100, 50, "sigma0", sigma0=3.0)
```

`tests/test_dom_spectra.py`, the same test:

```python
    assert np.array_equal(s.on(FourierDomain(4096, 5.0, 110, 5, "sigma0")), v[:, 10:15])
    assert np.array_equal(s.on(domain=FourierDomain(4096, 5.0, 110, 5, "sigma0")), v[:, 10:15])
```

→

```python
    assert np.array_equal(s.on(FourierDomain(4096, 5.0, 110, 5, "sigma0", sigma0=3.0)), v[:, 10:15])
    assert np.array_equal(s.on(domain=FourierDomain(4096, 5.0, 110, 5, "sigma0", sigma0=3.0)),
                          v[:, 10:15])
    # unit-blind: a PSD does not depend on the data's Fourier normalisation
    for units, kw in (("rfft", {}), ("dt_rfft", {}), ("sigma0", {"sigma0": 7.0})):
        assert np.array_equal(s.on(FourierDomain(4096, 5.0, 110, 5, units, **kw)), v[:, 10:15])
```

`tests/test_dom_spectra.py`, the same test:

```python
        s.on(FourierDomain(4096, 5.0, 140, 20, "sigma0"))
```

→

```python
        s.on(FourierDomain(4096, 5.0, 140, 20, "sigma0", sigma0=3.0))
```

`tests/test_dom_spectra.py`, the same test:

```python
        s.on(FourierDomain(8192, 5.0, 110, 5, "sigma0"))
```

→

```python
        s.on(FourierDomain(8192, 5.0, 110, 5, "sigma0", sigma0=3.0))
```

`BinSelect` carries `sigma0` and refuses a `dst` in other units.

`tests/test_dom_maps_base.py`, `test_binselect_windows_and_complex_push`:

```python
    store = FourierDomain(4096, 5.0, 100, 60, "sigma0")
    S = BinSelect(store, (110, 30))
    assert S.dst == FourierDomain(4096, 5.0, 110, 30, "sigma0") and S.kind == "complex"
```

→

```python
    store = FourierDomain(4096, 5.0, 100, 60, "sigma0", sigma0=3.0)
    S = BinSelect(store, (110, 30))
    assert S.dst == FourierDomain(4096, 5.0, 110, 30, "sigma0", sigma0=3.0) and S.kind == "complex"
    assert BinSelect(store, S.dst).dst == S.dst
    for other in (FourierDomain(4096, 5.0, 110, 30, "sigma0", sigma0=4.0),
                  FourierDomain(4096, 5.0, 110, 30, "rfft")):
        with pytest.raises(ValueError, match="sigma0"):
            BinSelect(store, other)
```

`FourierToWDM.window` keeps `units` and `sigma0` (the existing `after()` assertions on `Wc.window` in `test_after_a_crop_is_the_composition_and_the_cropped_transform` stay as they are and must keep passing).

`tests/test_dom_maps_fourier_wdm.py`, before `test_restrict_truncates_the_atoms_to_a_sub_window` (add a test):

```python
def test_restrict_truncates_the_atoms_to_a_sub_window():
```

→

```python
def test_the_band_window_keeps_the_units_and_sigma0():
    """``window`` is ``dataclasses.replace`` of ``src``: the bins of the cells' grid record in
    ``src``'s units, ``sigma0`` included."""
    src = FourierDomain(N, DT, 0, N // 2 + 1, "sigma0", sigma0=2.5)
    win = FourierToWDM(src, WDMDomain(_cells())).window
    assert (win.N, win.dt, win.units, win.sigma0) == (N, DT, "sigma0", 2.5)
    assert win.k0 > 0 and win.k0 + win.n < N // 2 + 1
    assert FourierToWDM(win, WDMDomain(_cells())).window == win


def test_restrict_truncates_the_atoms_to_a_sub_window():
```

Pad and crop refuse a `dst` with another `sigma0` or other units.

`tests/test_dom_maps_dirichlet.py`, before `test_widely_linear_adjoint_and_no_push` (add a test):

```python
@pytest.mark.parametrize("cls", ["pad", "crop"])
def test_widely_linear_adjoint_and_no_push(cls):
```

→

```python
def test_pad_and_crop_keep_the_units_and_sigma0():
    """No map rescales silently: ``dst`` must have ``src``'s ``(units, sigma0)``."""
    src = FourierDomain(NS, 1.0, 20, 41, "sigma0", sigma0=2.0)
    for units, kw in (("rfft", {}), ("sigma0", {"sigma0": 3.0})):
        with pytest.raises(ValueError, match="sigma0"):
            FourierZeroPad(src, FourierDomain(320, 1.0, 5, 80, units, **kw))
        with pytest.raises(ValueError, match="sigma0"):
            FourierCrop(src, FourierDomain(200, 1.0, 10, 70, units, **kw), 24)
    pad = FourierZeroPad(src, FourierDomain(320, 1.0, 5, 80, "sigma0", sigma0=2.0))
    crop = FourierCrop(src, FourierDomain(200, 1.0, 10, 70, "sigma0", sigma0=2.0), 24)
    assert pad.dst.sigma0 == crop.dst.sigma0 == 2.0


@pytest.mark.parametrize("cls", ["pad", "crop"])
def test_widely_linear_adjoint_and_no_push(cls):
```

- [ ] **Step 3: Run them and see them fail**

Run: `$PY -m pytest -q tests/test_dom_domains.py tests/test_dom_spectra.py tests/test_dom_maps_base.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_dirichlet.py`
Expected: `6 failed, 39 passed`. Every failure is `TypeError: FourierDomain.__init__() got an unexpected keyword argument 'sigma0'`.

- [ ] **Step 4: `FourierDomain`: `sigma0` and `rfft_scale`**

`src/domainia/domains.py`, the module docstring's `FourierDomain` bullet:

```python
* :class:`FourierDomain`: the absolute rfft bins ``[k0, k0 + n)`` of an ``N``-sample
  record, in one of three unit conventions (``"rfft"``: ``numpy.fft.rfft(x)``;
  ``"dt_rfft"``: ``dt * rfft(x)``; ``"sigma0"``: ``rfft(x) / sigma0``, the GB
  normalisation).  ``.full(...)``: the full-record domain of :mod:`domainia.translate`.
```

→

```python
* :class:`FourierDomain`: the absolute rfft bins ``[k0, k0 + n)`` of an ``N``-sample
  record, in one of three unit conventions (``"rfft"``: ``numpy.fft.rfft(x)``;
  ``"dt_rfft"``: ``dt * rfft(x)``; ``"sigma0"``: ``rfft(x) / sigma0``, with the constant
  ``sigma0`` stored on the domain).  ``rfft_scale`` is ``u`` with ``rfft(x) = u z``.
  ``.full(...)``: the full-record domain of :mod:`domainia.translate`.
```

`src/domainia/domains.py`, the imports:

```python
import math
from dataclasses import dataclass
from typing import Literal
```

→

```python
import math
from dataclasses import dataclass, field
from typing import Literal
```

`src/domainia/domains.py`, `FourierDomain`: the class docstring, the fields and the start of `__post_init__`:

```python
@dataclass(frozen=True)
class FourierDomain:
    """The absolute rfft bins ``[k0, k0 + n)`` of an ``N``-sample record at spacing ``dt``."""

    N: int
    dt: float
    k0: int
    n: int
    units: Literal["rfft", "dt_rfft", "sigma0"]

    def __post_init__(self) -> None:
        N = _int("N", self.N, minimum=1)
        k0 = _int("k0", self.k0, minimum=0)
        n = _int("n", self.n, minimum=1)
        if k0 + n > N // 2 + 1:
            raise ValueError(f"bins [{k0}, {k0 + n}) exceed the {N // 2 + 1} rfft bins of N={N}")
        if self.units not in FOURIER_UNITS:
            raise ValueError(f"units must be one of {FOURIER_UNITS}; got {self.units!r}")
        object.__setattr__(self, "N", N)
```

→

```python
@dataclass(frozen=True)
class FourierDomain:
    """The absolute rfft bins ``[k0, k0 + n)`` of an ``N``-sample record at spacing ``dt``.

    A value ``z`` on the domain has the declared ``units``: ``"rfft"`` (``numpy.fft.rfft(x)``),
    ``"dt_rfft"`` (``dt * rfft(x)``) or ``"sigma0"`` (``rfft(x) / sigma0``).  ``sigma0`` is
    keyword-only, required (positive and finite) with ``units="sigma0"`` and ``None``
    otherwise (``ValueError``); it is part of equality and the hash.  Derive one domain from
    another with ``dataclasses.replace`` so that ``units`` and ``sigma0`` travel with it.
    """

    N: int
    dt: float
    k0: int
    n: int
    units: Literal["rfft", "dt_rfft", "sigma0"]
    sigma0: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        N = _int("N", self.N, minimum=1)
        k0 = _int("k0", self.k0, minimum=0)
        n = _int("n", self.n, minimum=1)
        if k0 + n > N // 2 + 1:
            raise ValueError(f"bins [{k0}, {k0 + n}) exceed the {N // 2 + 1} rfft bins of N={N}")
        if self.units not in FOURIER_UNITS:
            raise ValueError(f"units must be one of {FOURIER_UNITS}; got {self.units!r}")
        if self.units == "sigma0":
            if self.sigma0 is None:
                raise ValueError("units='sigma0' needs sigma0, the constant with "
                                 "rfft(x) = sigma0 * z")
            object.__setattr__(self, "sigma0", _positive_float("sigma0", self.sigma0))
        elif self.sigma0 is not None:
            raise ValueError(f"sigma0 is set only with units='sigma0'; got sigma0={self.sigma0!r} "
                             f"with units={self.units!r}")
        object.__setattr__(self, "N", N)
```

`src/domainia/domains.py`, `FourierDomain`: add the `rfft_scale` property after `size`:

```python
    @property
    def size(self) -> int:
        return self.n

    def bins(self) -> IntArray:
```

→

```python
    @property
    def size(self) -> int:
        return self.n

    @property
    def rfft_scale(self) -> float:
        """``u`` with ``numpy.fft.rfft(x) = u * z`` for the bins ``z`` of this domain: ``1.0``
        (``"rfft"``), ``1 / dt`` (``"dt_rfft"``) or ``sigma0`` (``"sigma0"``)."""
        if self.units == "rfft":
            return 1.0
        if self.units == "dt_rfft":
            return 1.0 / self.dt
        return self.sigma0  # type: ignore[return-value]

    def bins(self) -> IntArray:
```

- [ ] **Step 5: `maps/base.py`: the shared `(units, sigma0)` check and `BinSelect`**

`dataclasses` is already imported in `maps/base.py`.

`src/domainia/maps/base.py`, before `domain_size` (add the shared units check):

```python
def domain_size(dom: Any) -> int:
```

→

```python
def _require_same_units(what: str, src: FourierDomain, dst: FourierDomain) -> None:
    """Refuse a map between bins in different units: ``(units, sigma0)`` must match, because
    no map rescales silently."""
    if (src.units, src.sigma0) != (dst.units, dst.sigma0):
        raise ValueError(f"{what} keeps the units: src has units={src.units!r}, "
                         f"sigma0={src.sigma0!r} but dst has units={dst.units!r}, "
                         f"sigma0={dst.sigma0!r}")


def domain_size(dom: Any) -> int:
```

`src/domainia/maps/base.py`, `BinSelect.__post_init__`:

```python
    def __post_init__(self) -> None:
        if isinstance(self.dst, tuple):
            k0, n = (int(v) for v in self.dst)
            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n,
                                                          self.src.units))
        d, s = self.dst, self.src
        if (d.N, d.dt, d.units) != (s.N, s.dt, s.units) or d.k0 < s.k0 or d.k0 + d.n > s.k0 + s.n:
```

→

```python
    def __post_init__(self) -> None:
        if isinstance(self.dst, tuple):
            k0, n = (int(v) for v in self.dst)
            object.__setattr__(self, "dst", dataclasses.replace(self.src, k0=k0, n=n))
        d, s = self.dst, self.src
        _require_same_units("BinSelect", s, d)
        if (d.N, d.dt) != (s.N, s.dt) or d.k0 < s.k0 or d.k0 + d.n > s.k0 + s.n:
```

- [ ] **Step 6: `maps/fourier.py`: the pad/crop checks and `FourierToWDM.window`**

The record of `window` comes from `dst`'s grid, not from `src`: for a map returned by `after()`, `src` lies on the outer record (`src.N != g.N`), so `replace(self.src, k0=..., n=...)` alone would be wrong (spec section 1.2).

`src/domainia/maps/fourier.py`, the imports:

```python
import operator
from dataclasses import dataclass, field
from typing import Any
```

→

```python
import operator
from dataclasses import dataclass, field, replace
from typing import Any
```

`src/domainia/maps/fourier.py`, the `.base` import:

```python
from .base import _BUILD, LinearMap, _last_axis
```

→

```python
from .base import _BUILD, LinearMap, _last_axis, _require_same_units
```

`src/domainia/maps/fourier.py`, `FourierZeroPad.__post_init__`:

```python
            s, d = self.src, self.dst
            if d.N <= s.N or d.dt != s.dt or d.units != s.units:
                raise ValueError("FourierZeroPad needs a longer record on the same dt and units")
```

→

```python
            s, d = self.src, self.dst
            _require_same_units("FourierZeroPad", s, d)
            if d.N <= s.N or d.dt != s.dt:
                raise ValueError("FourierZeroPad needs a longer record on the same dt")
```

`src/domainia/maps/fourier.py`, `FourierCrop.__post_init__`:

```python
            s, d = self.src, self.dst
            if not (0 <= self.start and self.start + d.N <= s.N) or d.dt != s.dt or d.units != s.units:
                raise ValueError("FourierCrop needs [start, start + dst.N) inside the record")
```

→

```python
            s, d = self.src, self.dst
            _require_same_units("FourierCrop", s, d)
            if not (0 <= self.start and self.start + d.N <= s.N) or d.dt != s.dt:
                raise ValueError("FourierCrop needs [start, start + dst.N) inside the record")
```

`src/domainia/maps/fourier.py`, `FourierToWDM.window` (docstring end and body):

```python
        :meth:`after` reads the bins of the outer record.  (These are band-window *bins*.
        The Wilson window of the cells is ``dst.window``.)"""
        geo = _geometry(self.dst)
        g = self.dst.cells.grid
        return FourierDomain(g.N, g.dt, geo["g0"], geo["G"], self.src.units)
```

→

```python
        :meth:`after` reads the bins of the outer record.  It is ``dataclasses.replace`` of
        ``src``, so ``units`` and ``sigma0`` are ``src``'s.  (These are band-window *bins*.
        The Wilson window of the cells is ``dst.window``.)"""
        geo = _geometry(self.dst)
        g = self.dst.cells.grid
        return replace(self.src, N=g.N, dt=g.dt, k0=geo["g0"], n=geo["G"])
```

- [ ] **Step 7: `spectra.py`: say that `TabulatedSpectrum` is unit-blind**

`src/domainia/spectra.py`, the `TabulatedSpectrum` class docstring:

```python
    """``values[c, j]`` at absolute rfft bin ``domain.k0 + j`` of ``domain``'s record.

    ``values`` is a read-only float64 copy (the caller's array is never aliased), as
    D-A's ``CellSet`` keeps its indices.
    """
```

→

```python
    """``values[c, j]`` at absolute rfft bin ``domain.k0 + j`` of ``domain``'s record.

    ``values`` is a read-only float64 copy (the caller's array is never aliased), as
    D-A's ``CellSet`` keeps its indices.

    :meth:`on` and :meth:`at` are unit-blind: they read the record (``N``, ``dt``) and the bin
    range, never ``units`` or ``sigma0``, because a PSD does not depend on the data's Fourier
    normalisation.
    """
```

`src/domainia/spectra.py`, `TabulatedSpectrum.on` docstring:

```python
        """``(C, domain.n)``: the values on ``domain``'s bins; raises outside the table."""
```

→

```python
        """``(C, domain.n)``: the values on ``domain``'s bins; raises outside the table.
        ``domain.units`` and ``domain.sigma0`` are not read (a PSD has no Fourier units)."""
```

- [ ] **Step 8: Run the tests again**

Run: `$PY -m pytest -q tests/test_dom_domains.py tests/test_dom_spectra.py tests/test_dom_maps_base.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_dirichlet.py`
Expected: `45 passed`

Run:

```bash
grep -rn "FourierDomain(" src
```

Expected hits:

```text
src/domainia/maps/time.py:184:            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n, "rfft"))
```

Only `TimeToFourier`, which builds from scratch. Every derived site uses `replace`.

- [ ] **Step 9: Full suite, NumPy lane, Meyer baseline**

Run: `$PY -m pytest -q`
Expected: `1465 passed, 3 skipped`

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: `1376 passed, 92 skipped`

Run: `$PY tests/wilson/meyer_cases.py --check`
Expected: `baseline reproduces: 1504 arrays` (bit-exact: the baseline uses `"rfft"` units).

- [ ] **Step 10: Commit (sandbox disabled)**

```bash
git add src/domainia/domains.py src/domainia/maps/base.py src/domainia/maps/fourier.py src/domainia/spectra.py tests/test_dom_domains.py tests/test_dom_spectra.py tests/test_dom_maps_base.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_dirichlet.py
git commit -m "FourierDomain.sigma0 and rfft_scale; derived domains keep their units

FourierDomain takes a keyword-only sigma0 (required for units='sigma0', None
otherwise) and exposes rfft_scale, the u with rfft(x) = u z. BinSelect and
FourierToWDM.window derive their domains with dataclasses.replace, and
FourierZeroPad, FourierCrop and BinSelect require the same (units, sigma0).
TabulatedSpectrum stays unit-blind and says so.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

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

**Verify:** `$PY -m pytest -q` → all pass, 0 errors; `$PY tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1504 arrays` (30 of them RESCALED: 5 families × 3 grids × 2 `dt`)

**Steps:**

Assumes the tree after Task 1.

- [ ] **Step 1: Grep for the callers of the band kernels and for the old cell scale**

Run:

```bash
grep -rn "forward_bands\|inverse_bands" src --include='*.py' | grep -v "^src/domainia/wilson/transform.py:"
```

Expected hits:

```text
src/domainia/maps/fourier.py:230:    ``cells``, the golden default); ``impl="fft"`` evaluates ``wilson.forward_bands``
src/domainia/maps/fourier.py:298:        w = wilson.transform.forward_bands(z, self.src.k0, geo["bands"], g.Nf, g.Nt, g.dt, be=be,
```

The only library caller is `FourierToWDM` (`impl="fft"`): Step 7 multiplies its result by `src.rfft_scale`. The kernels' own tests and the baseline cases are in the Run steps below.

Run:

```bash
grep -rnE "sqrt\((N|AG\.N|g\.N) / \(2|sqrt\(2(\.0)? \* (dt|DT) / N|sqrt\(2\.0 / \(N|N / \(2 \* DT\)" src tests
```

Expected hits:

```text
src/domainia/maps/fourier.py:232:    For a record ``z = rfft(x)`` the cells are ``sqrt(N / (2 dt))`` times
tests/wilson/test_window_threading.py:467:    _close(fb, wilson.forward(z, Nf, Nt, dt, window=window)[bands] / np.sqrt(2.0 * dt / N), RTOL,
tests/wilson/test_grid_bands.py:245:    bands are the WDM coefficients times sqrt(N / (2 dt)) -- the GB comment's
tests/wilson/test_grid_bands.py:252:    scale = np.sqrt(N / (2.0 * dt))
tests/test_dom_maps_fourier_wdm.py:63:    np.testing.assert_allclose(got, np.sqrt(N / (2 * DT)) * want, rtol=0, atol=1e-12 * np.max(np.abs(got)))
tests/test_dom_maps_fourier_wdm.py:126:    cells ``sqrt(N / (2 dt)) W x`` have covariance ``(N / (2 dt)) dt I = (N / 2) I``: the push
tests/test_dom_maps_fourier_wdm.py:141:    ref = np.sqrt(AG.N / (2 * AG.dt)) * TimeToWDM(TimeDomain(AG.N, AG.dt), ACELLS).apply(xr[16:144])
tests/test_dom_maps_fourier_wdm.py:158:    units: ``W.after(zp)`` is ``sqrt(N / (2 dt))`` times the padded record's WDM cells."""
tests/test_dom_maps_fourier_wdm.py:165:    ref = np.sqrt(AG.N / (2 * AG.dt)) * TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), Wp.dst).apply(xpad)
tests/test_dom_windows.py:132:    want = np.sqrt(N / (2 * DT)) * TimeToWDM(TIME, dom).apply(X)
tests/test_dom_windows.py:154:    _close(got, np.sqrt(g.N / (2 * g.dt)) * TimeToWDM(TimeDomain(g.N, g.dt), dom).apply(xr[16:144]))
tests/test_dom_windows.py:163:    ``Re(rows @ rfft(x)) = sqrt(N / (2 dt)) W x``); and the push of a circulant time
tests/test_dom_windows.py:175:    R = np.sqrt(2.0 / (N * DT)) * np.conj(np.fft.rfft(W, axis=-1))
tests/test_dom_windows.py:185:    _close(got, (N / (2 * DT)) * W @ Sigma @ W.T)
```

Every hit is the old scale between unit-norm-atom cells and Wilson coefficients (`sqrt(N / (2 dt))`, or `N / (2 dt)` on a covariance, `sqrt(2 / (N dt))` on a row). After this task the cells are Wilson coefficients, so each one goes:

* `maps/fourier.py` (the `FourierToWDM` docstring): rewritten in Step 7.
* `test_dom_windows.py` (five hits in three tests), `test_dom_maps_fourier_wdm.py` (five hits in four tests), `test_window_threading.py` and `test_grid_bands.py`: the factor is dropped (the cells equal `TimeToWDM`, `forward`) in Step 4. The one exception is `test_after_a_zero_pad_is_the_padded_record_transform`, which keeps the pad's `sqrt(N / N')` until Task 3 removes `FourierZeroPad.scale`.

- [ ] **Step 2: The baseline comparator: the RESCALED families**

Spec section 1.7: five baseline families change units; every other array stays bit-exact. `RESCALED` lives in `tests/wilson/meyer_cases.py` and is read by both the pytest gate and `--check`, through one comparator `mismatches`. The npz is never re-recorded.

`tests/wilson/meyer_cases.py`, the module docstring:

```python
The gate is ``tests/wilson/test_meyer_baseline.py``: it recomputes every case through
``domainia.wilson`` and requires ``np.array_equal`` (same dtype) on every array.
``python tests/wilson/meyer_cases.py --check`` is the same replay as a script.  The
arrays are bit-exact only in the recording environment (the ``wdmrel`` env's NumPy
pocketfft and BLAS on the machine that recorded them).
```

→

```python
The gate is ``tests/wilson/test_meyer_baseline.py``: it recomputes every case through
``domainia.wilson`` and compares with :func:`mismatches`: ``np.array_equal`` (same dtype)
on every array except the :data:`RESCALED` families.  Those changed units on purpose (the
follow-up spec, section 1: every WDM value is a Wilson coefficient) and are compared as
``got = f * recorded`` to 1e-13.  ``python tests/wilson/meyer_cases.py --check`` is the
same replay as a script, with the same comparator.  The arrays are bit-exact only in the
recording environment (the ``wdmrel`` env's NumPy pocketfft and BLAS on the machine that
recorded them).
```

`tests/wilson/meyer_cases.py`, the imports:

```python
import argparse
import sys
```

→

```python
import argparse
import re
import sys
```

`tests/wilson/meyer_cases.py`, before `psd` (add the RESCALED table and the comparator):

```python
def psd(f: Any) -> np.ndarray:
```

→

```python
def _wilson_units(N: int, dt: float) -> float:
    """The unit-norm atoms' cells times this are Wilson coefficients (rfft-unit bins)."""
    return float(np.sqrt(2.0 * dt / N))


#: ``(layer, case) -> f(N, dt)`` with ``got = f * recorded``: the families whose units changed
#: in the follow-up spec (section 1.7).  The recording used ``"rfft"`` units (``u = 1``).
#: ``inverse_bands`` has ``f = 1``: its input ``wb = forward_bands(xr, ...)`` is recomputed on
#: every run, so the new ``sqrt(2 dt / N)`` of ``wb`` and ``sqrt(N / (2 dt))`` of
#: ``inverse_bands`` cancel, to rounding (compared by tolerance, not bits).
RESCALED: dict[tuple[str, str], Callable[[int, float], float]] = {
    ("domainia", "FourierToWDM.dense"): _wilson_units,
    ("domainia", "FourierToWDM.fft"): _wilson_units,
    ("domainia", "FourierToWDM.adjoint"): _wilson_units,           # kappa once
    ("wilson", "transform.forward_bands"): _wilson_units,
    ("wilson", "transform.inverse_bands"): lambda N, dt: 1.0,
}
#: The RESCALED comparison: ``max |got - f want| <= RESCALED_RTOL * max |f want|``.
RESCALED_RTOL = 1e-13
#: ``"{layer}::{case}::{Nf}x{Nt}::dt={dt!r}"`` plus :func:`flatten`'s ``[i]`` suffixes.
_KEY = re.compile(r"(?P<layer>[^:]+)::(?P<case>[^:]+)::(?P<Nf>\d+)x(?P<Nt>\d+)::dt=(?P<dt>[^\[]+)")


def rescale_factor(key: str) -> float | None:
    """``f(N, dt)`` of a baseline key in :data:`RESCALED`, else ``None`` (compared bit for bit)."""
    m = _KEY.match(key)
    if m is None:
        raise ValueError(f"not a baseline key: {key!r}")
    f = RESCALED.get((m["layer"], m["case"]))
    return None if f is None else f(int(m["Nf"]) * int(m["Nt"]), float(m["dt"]))


def mismatches(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> list[str]:
    """The keys of ``want`` that ``got`` does not reproduce (the keys must be the same).

    Same dtype everywhere; ``np.array_equal`` except for the :data:`RESCALED` keys, which
    need ``max |got - f want| <= RESCALED_RTOL * max |f want|``.
    """
    if sorted(got) != sorted(want):
        raise AssertionError(f"baseline keys differ: {sorted(set(got) ^ set(want))[:10]}")
    bad = []
    for k in want:
        g, w = got[k], want[k]
        f = rescale_factor(k)
        if g.dtype != w.dtype or g.shape != w.shape:
            bad.append(k)
        elif f is None:
            if not np.array_equal(g, w):
                bad.append(k)
        elif (np.max(np.abs(g - f * w), initial=0.0)
              > RESCALED_RTOL * np.max(np.abs(f * w), initial=0.0)):
            bad.append(k)
    return bad


def psd(f: Any) -> np.ndarray:
```

`tests/wilson/meyer_cases.py`, `main` (docstring and `--check` help):

```python
    ``tests/wilson/test_meyer_baseline.py``; this is the same replay as a script.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", required=True,
                   help="recompute and compare bit for bit")
```

→

```python
    ``tests/wilson/test_meyer_baseline.py``; this is the same replay as a script, with the
    same comparator (:func:`mismatches`).
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", required=True,
                   help="recompute and compare (bit for bit, RESCALED by tolerance)")
```

`tests/wilson/meyer_cases.py`, `main` (the comparison):

```python
    if sorted(stored) != sorted(out):
        raise SystemExit(f"baseline keys differ: {sorted(set(stored) ^ set(out))[:10]}")
    bad = [k for k in out if stored[k].dtype != out[k].dtype or not np.array_equal(stored[k], out[k])]
```

→

```python
    try:
        bad = mismatches(out, stored)
    except AssertionError as exc:
        raise SystemExit(str(exc)) from None
```

`tests/wilson/test_meyer_baseline.py`, the module docstring:

```python
"""The default window reproduces the Meyer baseline recorded at d991ac4 bit for bit."""
```

→

```python
"""The default window reproduces the Meyer baseline recorded at d991ac4 bit for bit.

Except the ``meyer_cases.RESCALED`` families: units (follow-up spec, section 1).  Their
values are now Wilson coefficients, ``got = f(N, dt) * recorded`` to 1e-13, with the same
dtype; every other array is still ``np.array_equal``.  ``meyer_cases.mismatches`` is the one
comparator, shared with ``meyer_cases.py --check``.
"""
```

`tests/wilson/test_meyer_baseline.py`, the `meyer_cases` import:

```python
from .meyer_cases import BASELINE, THREADED, compute
```

→

```python
from .meyer_cases import BASELINE, RESCALED, THREADED, compute, mismatches, rescale_factor
```

`tests/wilson/test_meyer_baseline.py`, `_mismatches` (replace it with a test of the table):

```python
def _mismatches(got: dict, want: dict) -> list[str]:
    assert sorted(got) == sorted(want), sorted(set(got) ^ set(want))[:10]
    return [k for k in want if got[k].dtype != want[k].dtype or not np.array_equal(got[k], want[k])]
```

→

```python
def test_the_rescaled_families_are_exactly_the_spec_table(baseline):
    """Five families (spec section 1.7), each present on every grid and dt of the baseline."""
    assert set(RESCALED) == {("domainia", "FourierToWDM.dense"), ("domainia", "FourierToWDM.fft"),
                             ("domainia", "FourierToWDM.adjoint"),
                             ("wilson", "transform.forward_bands"),
                             ("wilson", "transform.inverse_bands")}
    rescaled = [k for k in baseline if rescale_factor(k) is not None]
    assert len(rescaled) == 5 * 6                       # 3 grids x 2 dt each
    assert rescale_factor("domainia::FourierToWDM.fft::16x32::dt=0.25") == np.sqrt(2 * 0.25 / 512)
    assert rescale_factor("wilson::transform.inverse_bands::32x8::dt=1.0") == 1.0
    assert rescale_factor("wilson::transform.forward::8x16::dt=1.0") is None
```

`tests/wilson/test_meyer_baseline.py`, `test_default_window_reproduces_the_meyer_baseline_bit_for_bit`:

```python
    assert not _mismatches(compute(wilson, DM, {}, {}), baseline)
```

→

```python
    assert not mismatches(compute(wilson, DM, {}, {}), baseline)
```

`tests/wilson/test_meyer_baseline.py`, `test_explicit_meyer_reproduces_the_baseline_bit_for_bit`:

```python
    assert want and not _mismatches(got, want)
```

→

```python
    assert want and not mismatches(got, want)
```

`tests/wilson/test_meyer_baseline.py`, `test_explicit_meyer_on_both_layers_reproduces_the_whole_baseline`:

```python
    assert not _mismatches(got, baseline)
```

→

```python
    assert not mismatches(got, baseline)
```

- [ ] **Step 3: Run the baseline gate and see the replays fail on the RESCALED keys only**

Run: `$PY -m pytest -q tests/wilson/test_meyer_baseline.py`
Expected: `3 failed, 1 passed`. The three replays fail with `assert not ['wilson::transform.forward_bands::8x16::dt=1.0', ...]`: only the `FourierToWDM.dense/.fft/.adjoint` and `transform.forward_bands` keys (24 of the 30 RESCALED keys; the 6 `inverse_bands` keys, `f = 1`, already pass). The table test passes.

Run: `$PY tests/wilson/meyer_cases.py --check`
Expected: exit 1 with `24 arrays differ from the baseline: ['wilson::transform.forward_bands::8x16::dt=1.0', ...]`.

- [ ] **Step 4: Write the failing tests: Wilson units for the band kernels and `FourierToWDM`**

`tests/wilson/test_grid_bands.py`: the dense atom analysis carries `sqrt(2 dt / N)`, `forward_bands` is `forward(z)[bands]` and `inverse_bands` is `inverse` on its bins (on the Meyer baseline's grids), and the adjoint identity carries `2 dt / N`.

`tests/wilson/test_grid_bands.py`, the `domainia.wilson.transform` import:

```python
    base_atom_spectra,
    forward_bands,
    forward_time,
    inverse_bands,
)
```

→

```python
    base_atom_spectra,
    forward,
    forward_bands,
    forward_time,
    inverse,
    inverse_bands,
)
```

`tests/wilson/test_grid_bands.py`, `test_forward_bands_is_the_dense_atom_analysis`:

```python
    atoms = band_atoms(bands, Nf, Nt, dt)
    dense = np.stack([
```

→

```python
    atoms = band_atoms(bands, Nf, Nt, dt)
    # Wilson units: the unit-norm atoms times sqrt(2 dt / N) (band_atoms' docstring)
    dense = np.sqrt(2.0 * dt / (Nf * Nt)) * np.stack([
```

`tests/wilson/test_grid_bands.py`, `test_forward_bands_is_the_transform_times_sqrt_N_over_2dt` (replace it with two tests):

```python
@pytest.mark.parametrize("grid", [(8, 16, 0.5), (16, 32, 1.0), (8, 16, 2.0)])
def test_forward_bands_is_the_transform_times_sqrt_N_over_2dt(grid, rng):
    """On a whole record (z = rfft(x)) the unit-norm-atom cells of the interior
    bands are the WDM coefficients times sqrt(N / (2 dt)) -- the GB comment's
    "orthonormal coefficient times sqrt(N/2)" (orthonormal = w / sqrt(dt))."""
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = rng.standard_normal(N)
    w = forward_time(x, Nf, Nt, dt)
    cells = forward_bands(np.fft.rfft(x), 0, np.arange(1, Nf), Nf, Nt, dt)
    scale = np.sqrt(N / (2.0 * dt))
    assert np.max(np.abs(cells - scale * w[1:Nf])) <= 1e-12 * np.max(np.abs(cells))
```

→

```python
#: The Meyer baseline's grids and dt (tests/wilson/meyer_cases.py GRIDS x DTS).
BASELINE_GRIDS = [(Nf, Nt, dt) for Nf, Nt in ((8, 16), (16, 32), (32, 8)) for dt in (1.0, 0.25)]


@pytest.mark.parametrize("grid", BASELINE_GRIDS)
def test_forward_bands_is_the_transform_on_the_bands(grid, rng):
    """On a whole record (z = rfft(x)) the cells of any interior bands are the Wilson
    coefficients forward(z)[bands] (spec section 1.3: Wilson units)."""
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = rng.standard_normal(N)
    z = np.fft.rfft(x)
    for bands in (np.arange(1, Nf), np.array([1, Nf // 2, Nf - 1])):
        cells = forward_bands(z, 0, bands, Nf, Nt, dt)
        want = forward(z, Nf, Nt, dt)[bands]
        assert np.max(np.abs(cells - want)) <= 1e-12 * np.max(np.abs(want))
        np.testing.assert_allclose(want, forward_time(x, Nf, Nt, dt)[bands], rtol=0,
                                   atol=1e-13 * np.max(np.abs(want)))


@pytest.mark.parametrize("grid", BASELINE_GRIDS)
def test_inverse_bands_is_the_inverse_on_its_bins(grid, rng):
    """inverse_bands(w) is inverse(w_full) on the bins [k0, k0 + n_bins), w_full the dense grid
    holding w on the bands and 0 elsewhere."""
    Nf, Nt, dt = grid
    N = Nf * Nt
    for bands, k0, n_bins in ((np.arange(1, Nf), 0, N // 2 + 1),                 # every bin
                              (np.arange(2, Nf - 1), Nt // 2 - 1, N // 2 - Nt // 2)):  # a sub-range
        w = rng.standard_normal((bands.size, Nt))
        w_full = np.zeros((Nf + 1, Nt))
        w_full[bands] = w
        want = inverse(w_full, Nf, Nt, dt)
        got = inverse_bands(w, k0, n_bins, bands, Nf, Nt, dt)
        assert np.max(np.abs(got - want[k0:k0 + n_bins])) <= 1e-12 * np.max(np.abs(want))
```

`tests/wilson/test_grid_bands.py`, `test_inverse_bands_is_the_adjoint` (docstring and `N`):

```python
def test_inverse_bands_is_the_adjoint(backend, rng):
    Nf, Nt, dt = 16, 32, 0.5
    bands = np.array([3, 4, 5, 6])
```

→

```python
def test_inverse_bands_is_the_adjoint(backend, rng):
    """sum(forward_bands(z) w) = (2 dt / N) Re<z, inverse_bands(w)>: inverse_bands is
    N / (2 dt) times the adjoint (spec section 1.3)."""
    Nf, Nt, dt = 16, 32, 0.5
    N = Nf * Nt
    bands = np.array([3, 4, 5, 6])
```

`tests/wilson/test_grid_bands.py`, `test_inverse_bands_is_the_adjoint` (the right-hand side):

```python
    rhs = np.real(np.sum(np.conj(z) * np.asarray(
```

→

```python
    rhs = (2.0 * dt / N) * np.real(np.sum(np.conj(z) * np.asarray(
```

`tests/wilson/test_window_threading.py` (every window):

`tests/wilson/test_window_threading.py`, `test_forward_bands_matches_its_atoms_and_forward`:

```python
    """``forward_bands`` (one IFFT per band) == ``Re(band_atoms @ z[window])`` (dense) ==
    ``forward(z)[bands] / sqrt(2 dt / N)`` (the unit-norm-atom scale)."""
```

→

```python
    """``forward_bands`` (one IFFT per band) == ``sqrt(2 dt / N) Re(band_atoms @ z[window])``
    (dense, unit-norm atoms) == ``forward(z)[bands]`` (Wilson coefficients)."""
```

`tests/wilson/test_window_threading.py`, the same test (its two comparisons):

```python
    _close(fb, np.real(np.einsum("bns,bs->bn", atoms, z[ell])), RTOL, "Re(band_atoms @ z)")
    _close(fb, wilson.forward(z, Nf, Nt, dt, window=window)[bands] / np.sqrt(2.0 * dt / N), RTOL,
           "forward(z)[bands] / sqrt(2 dt / N)")
```

→

```python
    _close(fb, np.sqrt(2.0 * dt / N) * np.real(np.einsum("bns,bs->bn", atoms, z[ell])), RTOL,
           "sqrt(2 dt / N) Re(band_atoms @ z)")
    _close(fb, wilson.forward(z, Nf, Nt, dt, window=window)[bands], RTOL, "forward(z)[bands]")
```

`tests/wilson/test_window_threading.py`, `test_inverse_bands_is_the_adjoint_of_forward_bands` (docstring):

```python
    """``<forward_bands(z), w> == Re <z, inverse_bands(w)>``, to rounding of the
```

→

```python
    """``<forward_bands(z), w> == (2 dt / N) Re <z, inverse_bands(w)>``, to rounding of the
```

`tests/wilson/test_window_threading.py`, the same test (the right-hand side):

```python
    rhs = float(np.real(np.vdot(z, back)))
```

→

```python
    rhs = (2.0 * dt / N) * float(np.real(np.vdot(z, back)))
```

`tests/test_dom_windows.py`: the factors go.

`tests/test_dom_windows.py`, `test_fourier_cells_use_the_domain_window`:

```python
    want = np.sqrt(N / (2 * DT)) * TimeToWDM(TIME, dom).apply(X)
```

→

```python
    want = TimeToWDM(TIME, dom).apply(X)
```

`tests/test_dom_windows.py`, `test_after_a_crop_uses_the_domain_window`:

```python
    _close(got, np.sqrt(g.N / (2 * g.dt)) * TimeToWDM(TimeDomain(g.N, g.dt), dom).apply(xr[16:144]))
```

→

```python
    _close(got, TimeToWDM(TimeDomain(g.N, g.dt), dom).apply(xr[16:144]))
```

`tests/test_dom_windows.py`, `test_fourier_adjoint_rows_and_push_use_the_domain_window` (docstring):

```python
    ``sqrt(2 / (N dt)) conj(rfft(g_p))`` for the time atoms ``g_p`` of ``active_W`` (so that
    ``Re(rows @ rfft(x)) = sqrt(N / (2 dt)) W x``); and the push of a circulant time
    covariance's bin covariance ``M' = F Sigma F^H`` is ``(N / (2 dt)) W Sigma W^T``."""
```

→

```python
    ``(2 / N) conj(rfft(g_p))`` for the time atoms ``g_p`` of ``active_W`` (so that
    ``Re(rows @ rfft(x)) = W x``); and the push of a circulant time covariance's bin
    covariance ``M' = F Sigma F^H`` is ``W Sigma W^T``."""
```

`tests/test_dom_windows.py`, the same test (the rows):

```python
    R = np.sqrt(2.0 / (N * DT)) * np.conj(np.fft.rfft(W, axis=-1))
```

→

```python
    R = (2.0 / N) * np.conj(np.fft.rfft(W, axis=-1))
```

`tests/test_dom_windows.py`, the same test (the push):

```python
    _close(got, (N / (2 * DT)) * W @ Sigma @ W.T)
```

→

```python
    _close(got, W @ Sigma @ W.T)
```

`tests/test_dom_maps_fourier_wdm.py`: the cells are the time cells in every unit, `C = kappa U`, and white bins push to `sigma2 dt I`.

`tests/test_dom_maps_fourier_wdm.py`, the imports:

```python
from __future__ import annotations

import numpy as np
```

→

```python
from __future__ import annotations

from dataclasses import replace

import numpy as np
```

`tests/test_dom_maps_fourier_wdm.py`, the module constants:

```python
BANDS = (3, 4, 5)
```

→

```python
BANDS = (3, 4, 5)
SIGMA0 = 2.5
```

`tests/test_dom_maps_fourier_wdm.py`, before `_crop_case` (add two helpers):

```python
def _crop_case(rng):
```

→

```python
def _units_record(units: str) -> FourierDomain:
    """The whole record's bins in ``units`` (``sigma0 = SIGMA0`` for ``"sigma0"``)."""
    return FourierDomain(N, DT, 0, N // 2 + 1, units, sigma0=SIGMA0 if units == "sigma0" else None)


def _in_units(R: np.ndarray, units: str) -> np.ndarray:
    """``z`` in ``units`` from ``R = numpy.fft.rfft(x)`` (spec section 1.1's table)."""
    return {"rfft": R, "dt_rfft": DT * R, "sigma0": R / SIGMA0}[units]


def _crop_case(rng):
```

`tests/test_dom_maps_fourier_wdm.py`, `test_cells_are_the_record_scale_times_the_time_cells` (replace it with two tests):

```python
def test_cells_are_the_record_scale_times_the_time_cells():
    x = synthetic_record(N, 3)
    got = FourierToWDM(_record(), WDMDomain(_cells())).apply(np.fft.rfft(x))
    want = TimeToWDM(TimeDomain(N, DT), WDMDomain(_cells())).apply(x)
    np.testing.assert_allclose(got, np.sqrt(N / (2 * DT)) * want, rtol=0, atol=1e-12 * np.max(np.abs(got)))
```

→

```python
@pytest.mark.parametrize("impl", ["dense", "fft"])
@pytest.mark.parametrize("units", ["rfft", "dt_rfft", "sigma0"])
def test_cells_are_the_time_cells_in_every_unit(units, impl):
    """Spec section 1.4: the cells are the Wilson coefficients of the record, whatever units
    the bins are in."""
    x = synthetic_record(N, 3)
    got = FourierToWDM(_units_record(units), WDMDomain(_cells()), impl).apply(
        _in_units(np.fft.rfft(x), units))
    want = TimeToWDM(TimeDomain(N, DT), WDMDomain(_cells())).apply(x)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


@pytest.mark.parametrize("units", ["rfft", "dt_rfft", "sigma0"])
def test_the_cell_matrix_is_kappa_times_the_atoms(units):
    """``C = kappa U``, ``kappa = src.rfft_scale sqrt(2 dt / N)``, ``U`` the unit-norm atoms
    on the band windows; ``restrict=True`` truncates them to ``src``."""
    src = _units_record(units)
    full = FourierToWDM(src, WDMDomain(_cells()))
    win = full.window
    atoms = wilson.transform.band_atoms(np.array(BANDS), NF, NT, DT)
    U = np.zeros((len(BANDS) * NT, win.n), complex)
    for b in range(len(BANDS)):
        U[b * NT:(b + 1) * NT, (NT // 2) * b + np.arange(NT)] = atoms[b]
    kappa = src.rfft_scale * np.sqrt(2 * DT / N)
    assert np.max(np.abs(full.C - kappa * U)) <= 1e-15 * kappa * np.max(np.abs(U))
    inner = replace(win, k0=win.k0 + 2, n=win.n - 4)
    m = FourierToWDM(inner, WDMDomain(_cells()), restrict=True)
    assert np.max(np.abs(m.C - kappa * U[:, 2:win.n - 2])) <= 1e-15 * kappa * np.max(np.abs(U))
```

`tests/test_dom_maps_fourier_wdm.py`, `test_push_of_white_record_bins_is_the_identity` (replace it):

```python
def test_push_of_white_record_bins_is_the_identity():
    """For ``z = rfft(x)`` of unit white noise ``E[z z^H] = N I`` on the interior bins, and the
    cells ``sqrt(N / (2 dt)) W x`` have covariance ``(N / (2 dt)) dt I = (N / 2) I``: the push
    ``1/2 Re(C M' C^H)`` takes ``M' = 2 I`` to ``I``."""
    m = FourierToWDM(_record(), WDMDomain(_cells()))
    P = m.push(DenseCov(2 * np.eye(m.src.n))).dense()
    assert P.shape == (_cells().size, _cells().size)
    assert np.max(np.abs(P - np.eye(_cells().size))) <= 1e-12
```

→

```python
@pytest.mark.parametrize("bands", [tuple(range(1, NF)), tuple(range(2, NF - 1))])
def test_push_of_white_record_bins_is_the_white_cell_covariance(bands):
    """For ``z = rfft(x)`` of white noise of variance ``sigma2``, ``E[z z^H] = N sigma2 I`` on the
    bins, and the cells ``W x`` have covariance ``W sigma2 I W^T = sigma2 dt I``: the push
    ``1/2 Re(C M' C^H)`` takes ``M' = N sigma2 I`` to ``sigma2 dt I`` (spec section 1.4)."""
    sigma2 = 1.7
    cells = CellSet.bands(GRID, bands)
    m = FourierToWDM(_record(), WDMDomain(cells))
    P = m.push(DenseCov(N * sigma2 * np.eye(m.src.n))).dense()
    assert P.shape == (cells.size, cells.size)
    assert np.max(np.abs(P - sigma2 * DT * np.eye(cells.size))) <= 1e-13 * sigma2 * DT
```

`tests/test_dom_maps_fourier_wdm.py`, `test_after_a_crop_is_the_composition_and_the_cropped_transform`:

```python
    ref = np.sqrt(AG.N / (2 * AG.dt)) * TimeToWDM(TimeDomain(AG.N, AG.dt), ACELLS).apply(xr[16:144])
```

→

```python
    ref = TimeToWDM(TimeDomain(AG.N, AG.dt), ACELLS).apply(xr[16:144])
```

`tests/test_dom_maps_fourier_wdm.py`, `test_after_a_zero_pad_is_the_padded_record_transform` (docstring):

```python
    rfft, and ``FourierZeroPad.scale = sqrt(N / N')`` keeps the cells in the store record's
    units: ``W.after(zp)`` is ``sqrt(N / (2 dt))`` times the padded record's WDM cells."""
```

→

```python
    rfft, and ``W.after(zp)`` is the padded record's Wilson coefficients times
    ``FourierZeroPad.scale = sqrt(N / N')`` (spec section 1.5 removes that scale)."""
```

`tests/test_dom_maps_fourier_wdm.py`, the same test (the reference):

```python
    ref = np.sqrt(AG.N / (2 * AG.dt)) * TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), Wp.dst).apply(xpad)
```

→

```python
    ref = np.sqrt(AG.N / PAD_GRID.N) * TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), Wp.dst).apply(xpad)
```

- [ ] **Step 5: Run them and see them fail**

Run: `$PY -m pytest -q tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/test_dom_windows.py tests/test_dom_maps_fourier_wdm.py tests/wilson/test_meyer_baseline.py`
Expected: `79 failed, 587 passed, 1 skipped`. Per file: `test_grid_bands.py` 20, `test_window_threading.py` 40, `test_dom_windows.py` 3, `test_dom_maps_fourier_wdm.py` 13, `test_meyer_baseline.py` 3. Every failure is an `AssertionError` (none raises): the cells and kernel outputs are off by the old factor (`sqrt(2 dt / N)`, times `src.rfft_scale` for the other units; `2 dt / N` in the adjoint identities and the push), and the three baseline replays list the 24 RESCALED keys.

- [ ] **Step 6: `wilson/transform.py`: the band kernels return Wilson coefficients**

Each factor is a trailing multiply on the kernel's return line (later tasks anchor to these two lines). `n_samples` is already imported in `transform.py`; `phase = _band_phase(bands, Nt, phi)` is unchanged in both kernels, and `band_atoms` stays unit-norm.

`src/domainia/wilson/transform.py`, `band_atoms` (docstring):

```python
    Atom ``[b, n, s]`` multiplies the rfft bin ``(m_b - 1) Nt/2 + s`` of a
    record's spectrum; a cell is ``Re(atoms[b, n] . z[window])``.  Verbatim GB
    ``geometry.band_atoms`` with ``window``'s samples (Hz normalisation).
```

→

```python
    Atom ``[b, n, s]`` multiplies the rfft bin ``(m_b - 1) Nt/2 + s`` of a
    record's spectrum.  The atoms are unit-norm (the spectrum of a basis function, not a
    coefficient): for rfft bins ``z = numpy.fft.rfft(x)`` of an ``N = Nf Nt``-sample record,
    ``sqrt(2 dt / N) * Re(atoms[b, n] . z[window])`` is the Wilson coefficient
    ``forward(z)[m_b, n]``, which :func:`forward_bands` returns.  Verbatim GB
    ``geometry.band_atoms`` with ``window``'s samples (Hz normalisation).
```

`src/domainia/wilson/transform.py`, `forward_bands` (docstring, first paragraph):

```python
    """WDM cells of ``bands`` from band-limited rfft bins: ``(..., n) -> (..., nb, Nt)``.

    ``z[..., j]`` is the (unit-free) rfft bin ``k0 + j``.  Cell ``(b, n)`` is
    ``Re(band_atoms[b, n] . z[window_b])`` -- the unit-norm-atom convention of
    the GB toy -- computed with one length-``Nt`` IFFT per band (GB
    ``wdm_full.analyse``): ``Re(phase * ifft(phi * z[window]))``.  It agrees
    with the dense ``Re(band_atoms @ z)`` to ~1e-15 (not bitwise).  Interior
    bands only.  Cost ``O(nb Nt log Nt)`` against ``O(nb Nt^2)`` dense.
```

→

```python
    """Wilson coefficients of ``bands`` from band-limited rfft bins: ``(..., n) -> (..., nb, Nt)``.

    ``z[..., j]`` is the rfft bin ``k0 + j`` in ``numpy.fft.rfft`` units (no ``dt``).  On a
    whole record, ``z = rfft(x)``, the result is ``forward(z)[bands]`` (<= 1e-12 relative).
    Cell ``(b, n)`` is ``sqrt(2 dt / N) Re(band_atoms[b, n] . z[window_b])``, ``N = Nf Nt``,
    computed with one length-``Nt`` IFFT per band (GB ``wdm_full.analyse``):
    ``sqrt(2 dt / N) Re(phase * ifft(phi * z[window]))``.  It agrees with the dense
    ``sqrt(2 dt / N) Re(band_atoms @ z)`` to ~1e-15 (not bitwise).  Interior bands only.
    Cost ``O(nb Nt log Nt)`` against ``O(nb Nt^2)`` dense.
```

`src/domainia/wilson/transform.py`, `forward_bands` (docstring, parameter `z`):

```python
        Unit-free rfft bins ``k0 .. k0+n-1`` of a record; they must cover every band window.
```

→

```python
        rfft bins ``k0 .. k0+n-1`` of a record (``numpy.fft.rfft`` units); they must cover
        every band window.
```

`src/domainia/wilson/transform.py`, `forward_bands` (the return line):

```python
    Y = be.fft.ifft(phi * z[..., ell], axis=-1)
    return be.xp.real(phase * Y)
```

→

```python
    Y = be.fft.ifft(phi * z[..., ell], axis=-1)
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    return be.xp.real(phase * Y) * np.sqrt(2.0 * dt / n_samples(Nf, Nt))
```

`src/domainia/wilson/transform.py`, `inverse_bands` (docstring, first paragraph):

```python
    """The adjoint of :func:`forward_bands`: ``(..., nb, Nt) -> (..., n_bins)`` complex.

    For the real inner products ``<w, w'> = sum w w'`` and
    ``<z, z'> = Re sum conj(z) z'``,
    ``<forward_bands(z), w> == <z, inverse_bands(w)>`` exactly in exact
    arithmetic: bin ``k0 + j`` receives
    ``sum_n w[b, n] conj(atom[b, n, s])`` from every band window covering it
    (``phi * fft(conj(phase) * w) / Nt``, scattered band by band).  It is the
    map that pushes cell-space vectors back onto bins (GB congruences); it is
    not a left inverse of :func:`forward_bands`.  ``w`` must be real and
    finite (as in :func:`inverse`).
```

→

```python
    """rfft bins from the Wilson coefficients of ``bands``: ``(..., nb, Nt) -> (..., n_bins)``.

    It is ``inverse(w_full)[k0:k0 + n_bins]`` (<= 1e-12 relative), ``w_full`` the
    ``(Nf + 1, Nt)`` grid holding ``w`` on ``bands`` and zeros elsewhere: bin ``k0 + j``
    receives ``sqrt(N / (2 dt)) sum_n w[b, n] conj(atom[b, n, s])`` from every band window
    covering it (``sqrt(N / (2 dt)) phi * fft(conj(phase) * w) / Nt``, scattered band by
    band).  It is ``N / (2 dt)`` times the adjoint of :func:`forward_bands`: for the real
    inner products ``<w, w'> = sum w w'`` and ``<z, z'> = Re sum conj(z) z'``,
    ``<forward_bands(z), w> == (2 dt / N) <z, inverse_bands(w)>`` in exact arithmetic.
    ``inverse_bands(forward_bands(z))`` is ``z`` on the covered bins only when ``z``'s
    Wilson coefficients vanish outside ``bands``.  ``w`` must be real and finite (as in
    :func:`inverse`).
```

`src/domainia/wilson/transform.py`, `inverse_bands` (the return line):

```python
        out = be.add_at(out, (..., ell[b]), V[..., b, :], copy=False)
    return out
```

→

```python
        out = be.add_at(out, (..., ell[b]), V[..., b, :], copy=False)
    # rfft units: N / (2 dt) times the adjoint of forward_bands' sqrt(2 dt / N).
    return out * np.sqrt(n_samples(Nf, Nt) / (2.0 * dt))
```

- [ ] **Step 7: `maps/fourier.py`: `C = kappa U`**

`kappa = src.rfft_scale * sqrt(2 dt / N)` with `N`, `dt` of the cells' grid. The `fft` path gets `src.rfft_scale` only, because `forward_bands` now carries `sqrt(2 dt / N)`. `adjoint`, `rows` and `push_rows` read `C`, so they carry `kappa` (the push `kappa^2`) without a change. In `after()`, `g` is the cells' grid (already bound there).

`src/domainia/maps/fourier.py`, before `FourierToWDM` (add `_kappa`):

```python
@pytree_dataclass(data=("C",), meta=("src", "dst", "impl", "lo", "width", "restrict"))
```

→

```python
def _kappa(src: FourierDomain, g: wilson.Grid) -> float:
    """``src.rfft_scale * sqrt(2 dt / N)`` (``N``, ``dt`` of the cells' grid): the unit-norm-atom
    analysis of bins in ``src``'s units, times this, is the Wilson coefficients."""
    return src.rfft_scale * float(np.sqrt(2.0 * g.dt / g.N))


@pytree_dataclass(data=("C",), meta=("src", "dst", "impl", "lo", "width", "restrict"))
```

`src/domainia/maps/fourier.py`, `FourierToWDM` (class docstring, first line):

```python
    """Cells of interior bands from band-limited bins: ``cells = Re(C z)`` (kind ``"cells"``).
```

→

```python
    """Wilson coefficients of interior-band cells from band-limited bins: ``cells = Re(C z)``
    (kind ``"cells"``).
```

`src/domainia/maps/fourier.py`, `FourierToWDM` (class docstring, the paragraph on `C`):

```python
    ``C`` holds the unit-norm atoms in the Wilson basis of ``dst.window``
    (GB ``Variant.C``): ``impl="dense"`` evaluates ``Re(z[..., lo:lo+width] @ C.T)`` (GB
    ``cells``, the golden default); ``impl="fft"`` evaluates ``wilson.forward_bands``
    (opt-in, <= 1e-12 relative).
    For a record ``z = rfft(x)`` the cells are ``sqrt(N / (2 dt))`` times
    ``TimeToWDM.apply(x)`` on the same cells (no map applies ``WDMDomain.scale``, so
    ``dst.scale`` must be 1).
    """
```

→

```python
    ``C = kappa U``: ``U`` holds the unit-norm atoms in the Wilson basis of ``dst.window``
    (GB ``Variant.C``), and ``kappa = src.rfft_scale * sqrt(2 dt / N)`` (``N``, ``dt`` of the
    cells' grid) turns bins in ``src.units`` into Wilson coefficients.  ``impl="dense"``
    evaluates ``Re(z[..., lo:lo+width] @ C.T)`` (GB ``cells``, the golden default);
    ``impl="fft"`` evaluates ``src.rfft_scale * wilson.forward_bands`` (opt-in, <= 1e-12
    relative).  For a record ``x`` and its bins ``z`` in ``src.units``, the cells equal
    ``TimeToWDM.apply(x)`` on the same cells.
    """
```

`src/domainia/maps/fourier.py`, `FourierToWDM.__post_init__`:

```python
        U = _dense_atoms(geo, g.Nf, g.dt, window=self.dst.window)[geo["sel"]]  # (P, G), GB V.U
        if k0 <= g0 and g0 + G <= k0 + n:
            lo, width, C = g0 - k0, G, U
        elif self.restrict and g0 <= k0 and k0 + n <= g0 + G:
            if self.impl != "dense":
                raise ValueError("restrict=True truncates the atoms: impl='dense' only")
            lo, width, C = 0, n, U[:, k0 - g0:k0 - g0 + n]
```

→

```python
        U = _dense_atoms(geo, g.Nf, g.dt, window=self.dst.window)[geo["sel"]]  # (P, G), GB V.U
        kappa = _kappa(self.src, g)
        if k0 <= g0 and g0 + G <= k0 + n:
            lo, width, C = g0 - k0, G, kappa * U
        elif self.restrict and g0 <= k0 and k0 + n <= g0 + G:
            if self.impl != "dense":
                raise ValueError("restrict=True truncates the atoms: impl='dense' only")
            lo, width, C = 0, n, kappa * U[:, k0 - g0:k0 - g0 + n]
```

`src/domainia/maps/fourier.py`, `FourierToWDM.apply` (the `fft` branch's return):

```python
        return w.reshape(*w.shape[:-2], -1)[..., geo["sel"]]
```

→

```python
        return self.src.rfft_scale * w.reshape(*w.shape[:-2], -1)[..., geo["sel"]]
```

`src/domainia/maps/fourier.py`, `FourierToWDM.push_rows` (docstring):

```python
        ``M'`` is ``(src.n, src.n)``; the ``width`` bins ``C`` reads are sliced out of it."""
```

→

```python
        ``M'`` is ``(src.n, src.n)``, the bins' covariance in ``src.units``; the ``width`` bins
        ``C`` reads are sliced out of it.  ``C`` carries ``kappa``, so the push carries
        ``kappa^2``."""
```

`src/domainia/maps/fourier.py`, `FourierToWDM.after` (docstring):

```python
        ``C = (analyse(A[idx]) + conj(analyse(B[idx]))) * inner.scale``, rows of ``dst``.
```

→

```python
        ``C = kappa * inner.scale * (analyse(A[idx]) + conj(analyse(B[idx])))``, rows of
        ``dst``, with ``self``'s ``kappa``.
```

`src/domainia/maps/fourier.py`, `FourierToWDM.after` (body):

```python
        C = C * inner.scale
```

→

```python
        C = (_kappa(self.src, g) * inner.scale) * C
```

- [ ] **Step 8: Run the tests again**

Run: `$PY -m pytest -q tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/test_dom_windows.py tests/test_dom_maps_fourier_wdm.py tests/wilson/test_meyer_baseline.py`
Expected: `666 passed, 1 skipped`

- [ ] **Step 9: Full suite, NumPy lane, Meyer baseline**

Run: `$PY -m pytest -q`
Expected: `1484 passed, 3 skipped`

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: `1395 passed, 92 skipped`

Run: `$PY tests/wilson/meyer_cases.py --check`
Expected: `baseline reproduces: 1504 arrays` (the 30 RESCALED keys within `1e-13`, the other 1474 bit-exact).

- [ ] **Step 10: Commit (sandbox disabled)**

```bash
git add src/domainia/wilson/transform.py src/domainia/maps/fourier.py tests/wilson/meyer_cases.py tests/wilson/test_meyer_baseline.py tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/test_dom_windows.py tests/test_dom_maps_fourier_wdm.py
git commit -m "Band kernels and FourierToWDM in Wilson units

forward_bands returns Wilson coefficients (forward(z)[bands] on a whole record)
and inverse_bands the rfft bins (inverse on its bins), N / (2 dt) times the
adjoint; band_atoms stays unit-norm. FourierToWDM holds C = kappa U with
kappa = src.rfft_scale sqrt(2 dt / N), so its cells equal TimeToWDM in every
unit, and after() carries the same kappa. The Meyer baseline compares the five
rescaled families by tolerance (meyer_cases.RESCALED); every other array stays
bit-exact.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

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
- [ ] Both maps still raise `ValueError` for a `(units, sigma0)` mismatch (added in Task 1; unchanged here).
- [ ] The full suite and the NumPy lane pass. The Meyer baseline passes, with the five Task 2 families still the only rescaled keys.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

**Steps:**

Assumes the tree after Task 2.

- [ ] **Step 1: Grep for the pad scale and the pad/crop users**

Run:

```bash
grep -rnw "scale" src/domainia/maps
```

Expected hits:

```text
src/domainia/maps/fourier.py:74:    """``z -> scale (A z + B conj z)`` (kind ``"widely"``): there is no complex matrix
src/domainia/maps/fourier.py:76:    instead of returning ``conj(A^H e + B^T e)``; read ``.A``, ``.B`` (and ``.scale``).
src/domainia/maps/fourier.py:83:                        "complex row matrix; use its .A, .B and .scale")
src/domainia/maps/fourier.py:86:@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "scale"))
src/domainia/maps/fourier.py:90:    bins ``src``: ``z' = scale (A z + B conj z)``, ``scale = sqrt(N / N')`` (kind
src/domainia/maps/fourier.py:91:    ``"widely"``).  ``A, B`` are :func:`pad_maps`; ``scale`` keeps the padded record's
src/domainia/maps/fourier.py:98:    scale: float = field(default=_BUILD, kw_only=True)
src/domainia/maps/fourier.py:109:            object.__setattr__(self, "scale", float(np.sqrt(s.N / d.N)))
src/domainia/maps/fourier.py:114:        return self.scale * ((z @ self.A.T) + (be.xp.conj(z) @ self.B.T))
src/domainia/maps/fourier.py:119:        return self.scale * ((y @ be.xp.conj(self.A)) + (be.xp.conj(y) @ self.B))
src/domainia/maps/fourier.py:136:    scale = 1.0
src/domainia/maps/fourier.py:341:        ``C = kappa * inner.scale * (analyse(A[idx]) + conj(analyse(B[idx])))``, rows of
src/domainia/maps/fourier.py:361:        C = (_kappa(self.src, g) * inner.scale) * C
src/domainia/maps/base.py:283:    ``scale``."""
```

* `maps/fourier.py`, every hit: `FourierZeroPad.scale` (the field, its pytree meta entry, `__post_init__`, `apply`, `adjoint`), `FourierCrop`'s class attribute `scale = 1.0`, `_WidelyLinear`'s docstring and `rows` message, and `FourierToWDM.after` (`inner.scale`). All go in Step 5.
* `maps/base.py` (`Identity`'s docstring): that is `WDMDomain.scale`, Task 4's. Leave it.

Run:

```bash
grep -rnE "\b(Z|C|zp|pad|crop|inner)\.scale\b|FourierZeroPad\.scale" tests
```

Expected hits:

```text
tests/test_dom_maps_dirichlet.py:32:    assert Z.kind == "widely" and Z.scale == np.sqrt(NS / Np)
tests/test_dom_maps_dirichlet.py:33:    assert np.max(np.abs(Z.apply(X[20:61]) - Z.scale * Xp[5:85])) <= 1e-14 * np.max(np.abs(Xp))
tests/test_dom_maps_dirichlet.py:58:    assert np.allclose(Z.apply(z[j]) / Z.scale, want, rtol=0, atol=1e-13)
tests/test_dom_maps_fourier_wdm.py:197:    ``FourierZeroPad.scale = sqrt(N / N')`` (spec section 1.5 removes that scale)."""
```

The three `Z.scale` reads in `test_dom_maps_dirichlet.py` and the `FourierZeroPad.scale` docstring of `test_after_a_zero_pad_is_the_padded_record_transform` (whose reference also carries `np.sqrt(AG.N / PAD_GRID.N)`): all change in Step 2.

Run:

```bash
grep -rln "FourierZeroPad\|FourierCrop\|pad_maps\|crop_maps" src tests
```

Expected hits:

```text
src/domainia/maps/__init__.py
src/domainia/maps/fourier.py
tests/test_dom_maps_dirichlet.py
tests/test_dom_db_imports.py
tests/test_dom_maps_fourier_wdm.py
tests/test_dom_restored_coverage.py
tests/test_dom_windows.py
```

Besides the two test files of Step 2, `test_dom_windows.py` (`test_after_a_crop_uses_the_domain_window`) crops a full-record source (bins 0..80 of 160), but its record has no DC or Nyquist content (`Z[8:57]` only), so halving those columns does not change it. `test_dom_restored_coverage.py` (jit of the maps as arguments) and `test_dom_db_imports.py` (the export list) read no values. No change.

- [ ] **Step 2: Write the failing tests**

`tests/test_dom_maps_dirichlet.py`: no pad or crop scale, and every bin of a record with DC and Nyquist content pads and crops to the exact rfft (even and odd `N`, even and odd output lengths).

`tests/test_dom_maps_dirichlet.py`, `test_zero_pad_reproduces_the_padded_record_rfft`:

```python
    assert Z.kind == "widely" and Z.scale == np.sqrt(NS / Np)
    assert np.max(np.abs(Z.apply(X[20:61]) - Z.scale * Xp[5:85])) <= 1e-14 * np.max(np.abs(Xp))
```

→

```python
    assert Z.kind == "widely" and not hasattr(Z, "scale")
    assert np.max(np.abs(Z.apply(X[20:61]) - Xp[5:85])) <= 1e-14 * np.max(np.abs(Xp))
```

`tests/test_dom_maps_dirichlet.py`, `test_crop_reproduces_the_cropped_record_rfft`, and a new test after it:

```python
    C = FourierCrop(SRC, FourierDomain(Nc, 1.0, 10, 70, "rfft"), start)
    assert np.max(np.abs(C.apply(X[20:61]) - Xc[10:80])) <= 1e-14 * np.max(np.abs(Xc))


```

→

```python
    C = FourierCrop(SRC, FourierDomain(Nc, 1.0, 10, 70, "rfft"), start)
    assert not hasattr(C, "scale")
    assert np.max(np.abs(C.apply(X[20:61]) - Xc[10:80])) <= 1e-14 * np.max(np.abs(Xc))


@pytest.mark.parametrize("N", [256, 255])
def test_full_record_bins_pad_and_crop_to_the_exact_rfft(N):
    """Spec section 1.5: the self-conjugate source bins (DC, and Nyquist for even N) count
    once, so pad and crop of every bin of a record with non-zero DC and Nyquist content are
    the rfft of the padded or cropped series.  The output end bins are real to rounding."""
    x = np.random.default_rng(N).standard_normal(N) + 0.75             # non-zero DC
    x[::2] += 0.5                                                      # non-zero Nyquist (even N)
    X = np.fft.rfft(x)
    assert abs(X[0]) > 1 and (N % 2 or abs(X[-1]) > 1)
    src = FourierDomain(N, 1.0, 0, N // 2 + 1, "rfft")
    for Np in (N + 64, N + 65):
        want = np.fft.rfft(np.r_[x, np.zeros(Np - N)])
        got = FourierZeroPad(src, FourierDomain(Np, 1.0, 0, Np // 2 + 1, "rfft")).apply(X)
        assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))
        assert abs(got[0].imag) <= 1e-13 * np.max(np.abs(want))
    for start, Nc in ((24, 200), (7, 141)):
        want = np.fft.rfft(x[start:start + Nc])
        got = FourierCrop(src, FourierDomain(Nc, 1.0, 0, Nc // 2 + 1, "rfft"), start).apply(X)
        assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))
        assert abs(got[0].imag) <= 1e-13 * np.max(np.abs(want))


```

`tests/test_dom_maps_dirichlet.py`, `test_maps_match_direct_transforms`:

```python
    assert np.allclose(Z.apply(z[j]) / Z.scale, want, rtol=0, atol=1e-13)
```

→

```python
    assert np.allclose(Z.apply(z[j]), want, rtol=0, atol=1e-13)
```

`tests/test_dom_maps_fourier_wdm.py`: the pad reference loses `sqrt(N / N')`, the comments lose the old test name, and the headline test pads every bin in every unit.

`tests/test_dom_maps_fourier_wdm.py`, the comment on the `after()` cases:

```python
#: The after() cases (7211bd5 test_dom_maps_cells::test_adjoint_and_the_record_scale): the
#: interior bands of an (8, 16) grid at dt = 0.5, and the same bands padded by 4 columns.
```

→

```python
#: The after() cases (ported from 7211bd5's test_dom_maps_cells): the interior bands of an
#: (8, 16) grid at dt = 0.5, and the same bands padded by 4 columns.
```

`tests/test_dom_maps_fourier_wdm.py`, `test_after_a_crop_is_the_composition_and_the_cropped_transform` (docstring):

```python
    """7211bd5 test_dom_maps_cells::test_adjoint_and_the_record_scale, its golden-free half."""
```

→

```python
    """Ported from 7211bd5's test_dom_maps_cells (its golden-free half)."""
```

`tests/test_dom_maps_fourier_wdm.py`, `test_after_a_zero_pad_is_the_padded_record_transform` (docstring):

```python
    """With no DC or Nyquist bin in the store, ``A z + B conj z`` is the zero-padded record's
    rfft, and ``W.after(zp)`` is the padded record's Wilson coefficients times
    ``FourierZeroPad.scale = sqrt(N / N')`` (spec section 1.5 removes that scale)."""
```

→

```python
    """``A z + B conj z`` is the zero-padded record's rfft, so ``W.after(zp)`` is the padded
    record's Wilson coefficients on the cells (here the GB pad: no DC or Nyquist bin in the
    store, cells with ``n < 16``)."""
```

`tests/test_dom_maps_fourier_wdm.py`, the same test (the reference), and the headline test after it:

```python
    ref = np.sqrt(AG.N / PAD_GRID.N) * TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), Wp.dst).apply(xpad)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


```

→

```python
    ref = TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), Wp.dst).apply(xpad)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("units", ["rfft", "dt_rfft", "sigma0"])
def test_pad_then_wdm_is_the_padded_record_transform(units):
    """The headline units test (spec section 1.5): every bin of a white 128-sample record
    (non-zero DC and Nyquist), in any units, zero padded to 160 samples and analysed on the
    padded grid's cells, is the padded record's Wilson coefficients, through ``after`` and
    through ``Compose``."""
    x = synthetic_record(AG.N, 11)
    store = FourierDomain(AG.N, AG.dt, 0, AG.N // 2 + 1, units,
                          sigma0=SIGMA0 if units == "sigma0" else None)
    R = np.fft.rfft(x)
    z = {"rfft": R, "dt_rfft": AG.dt * R, "sigma0": R / SIGMA0}[units]
    cells = WDMDomain(CellSet.bands(PAD_GRID, tuple(range(1, 8))))
    win = FourierToWDM(replace(store, N=PAD_GRID.N, n=PAD_GRID.N // 2 + 1), cells).window
    W, pad = FourierToWDM(win, cells), FourierZeroPad(store, win)
    want = TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), cells).apply(
        np.concatenate([x, np.zeros(PAD_GRID.N - AG.N)]))
    for got in (W.after(pad).apply(z), Compose(W, pad).apply(z)):
        assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


```

- [ ] **Step 3: Run them and see them fail**

Run: `$PY -m pytest -q tests/test_dom_maps_dirichlet.py tests/test_dom_maps_fourier_wdm.py`
Expected: `9 failed, 26 passed`. The failures: the three tests with the `hasattr(..., "scale")` or `/ Z.scale` assertions, both cases of the full-record test (the DC and Nyquist columns count twice, and the pad still multiplies by `sqrt(N / N')`), the zero-pad `after()` test (its reference lost `sqrt(N / N')`) and the three headline cases.

- [ ] **Step 4: `maps/fourier.py`: count the self-conjugate bins once**

`pad_maps` and `crop_maps` return fresh arrays, so halving them in place is safe. `FourierCrop.__post_init__` and `FourierZeroPad.__post_init__` call them, so the maps (and `after()`, which reads `inner.A`, `inner.B`) pick up the fix.

`src/domainia/maps/fourier.py`, the module docstring:

```python
Ported from the GB toy-model oracle (frozen): ``_geometric``, ``pad_maps``, ``crop_maps``
verbatim, the cell matrices of ``Grid.dense_atoms`` and ``_cells`` (with
``wdm_full.analyse`` and ``wdm_bands._phase``), and the data map ``cells``.  The atoms are
``wilson.transform.band_atoms`` (GB ``geometry.band_atoms`` to rounding; bit-identical
until the Hz window of v0.12.0, S1 Task C6).
```

→

```python
Ported from the GB toy-model oracle (frozen): ``_geometric`` verbatim, ``pad_maps`` and
``crop_maps`` with the self-conjugate source bins counted once, the cell matrices of
``Grid.dense_atoms`` and ``_cells`` (with ``wdm_full.analyse`` and ``wdm_bands._phase``),
and the data map ``cells``.  The atoms are ``wilson.transform.band_atoms`` (GB
``geometry.band_atoms`` to rounding; bit-identical until the Hz window of v0.12.0, S1
Task C6).
```

`src/domainia/maps/fourier.py`, the section comment over the Dirichlet kernels:

```python
# Dirichlet kernels (GB, verbatim)
```

→

```python
# Dirichlet kernels (GB; the self-conjugate source bins counted once)
```

`src/domainia/maps/fourier.py`, before `pad_maps` (add the halving helper) and `pad_maps`'s docstring:

```python
def pad_maps(N: int, Np: int, k_out, j_in):
    r"""Bins ``k_out`` of the record zero-padded from ``N`` to ``Np`` samples, from the
    record's bins ``j_in``: ``z'_k = A z + B conj(z)``, both ``(len(k_out), len(j_in))``,
    ``A_kj = (1/N) sum_{s<N} e^{2 pi i s (j/N - k/Np)}`` and ``B`` the same at ``-j``."""
```

→

```python
def _count_self_conjugate_once(N: int, j_in, A, B):
    """Halve the ``A`` and ``B`` columns of the self-conjugate source bins (absolute bin 0, and
    ``N / 2`` for even ``N``): there ``z`` and ``conj(z)`` are the same DFT coefficient, which
    ``A z + B conj(z)`` would otherwise count twice."""
    j = np.asarray(j_in, np.int64)
    self_conj = (j == 0) | (2 * j == N)
    A[:, self_conj] *= 0.5
    B[:, self_conj] *= 0.5
    return A, B


def pad_maps(N: int, Np: int, k_out, j_in):
    r"""Bins ``k_out`` of the record zero-padded from ``N`` to ``Np`` samples, from the
    record's bins ``j_in``: ``z'_k = A z + B conj(z)``, both ``(len(k_out), len(j_in))``,
    ``A_kj = (1/N) sum_{s<N} e^{2 pi i s (j/N - k/Np)}`` and ``B`` the same at ``-j``, with
    the columns of the self-conjugate bins ``j = 0`` and ``j = N/2`` (even ``N``) halved, so
    ``z'`` is the exact rfft of the padded real record in ``z``'s units."""
```

`src/domainia/maps/fourier.py`, `pad_maps` (the return):

```python
        out.append(_geometric(num, N * Np, N, n_nu) / N)
    return out[0], out[1]
```

→

```python
        out.append(_geometric(num, N * Np, N, n_nu) / N)
    return _count_self_conjugate_once(N, j_in, out[0], out[1])
```

`src/domainia/maps/fourier.py`, `crop_maps` (docstring):

```python
    ``A_kj = (1/N) e^{2 pi i j start / N} sum_{s<Nc} e^{2 pi i s (j/N - k/Nc)}``."""
```

→

```python
    ``A_kj = (1/N) e^{2 pi i j start / N} sum_{s<Nc} e^{2 pi i s (j/N - k/Nc)}``, with the
    columns of the self-conjugate bins ``j = 0`` and ``j = N/2`` (even ``N``) halved, so
    ``z^c`` is the exact rfft of the cropped real record in ``z``'s units."""
```

`src/domainia/maps/fourier.py`, `crop_maps` (the return):

```python
        out.append(shift * _geometric(num, N * Nc, Nc, n_nu) / N)
    return out[0], out[1]
```

→

```python
        out.append(shift * _geometric(num, N * Nc, Nc, n_nu) / N)
    return _count_self_conjugate_once(N, j_in, out[0], out[1])
```

- [ ] **Step 5: `maps/fourier.py`: no pad scale**

`src/domainia/maps/fourier.py`, `_WidelyLinear` (docstring and `rows`):

```python
    """``z -> scale (A z + B conj z)`` (kind ``"widely"``): there is no complex matrix
    ``M`` with ``M z`` equal to it, so :meth:`rows` (and :meth:`LinearMap.dense`) refuse
    instead of returning ``conj(A^H e + B^T e)``; read ``.A``, ``.B`` (and ``.scale``).
    ``push`` raises in :class:`LinearMap`."""

    kind = "widely"

    def rows(self, idx):
        raise TypeError(f"{type(self).__name__} is widely linear (A z + B conj z): it has no "
                        "complex row matrix; use its .A, .B and .scale")
```

→

```python
    """``z -> A z + B conj z`` (kind ``"widely"``): there is no complex matrix ``M`` with
    ``M z`` equal to it, so :meth:`rows` (and :meth:`LinearMap.dense`) refuse instead of
    returning ``conj(A^H e + B^T e)``; read ``.A`` and ``.B``.  ``push`` raises in
    :class:`LinearMap`."""

    kind = "widely"

    def rows(self, idx):
        raise TypeError(f"{type(self).__name__} is widely linear (A z + B conj z): it has no "
                        "complex row matrix; use its .A and .B")
```

`src/domainia/maps/fourier.py`, `FourierZeroPad` (decorator, docstring, fields):

```python
@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "scale"))
@dataclass(frozen=True, eq=False)
class FourierZeroPad(_WidelyLinear):
    """Bins ``dst`` of the record zero-padded from ``src.N`` to ``dst.N`` samples, from the
    bins ``src``: ``z' = scale (A z + B conj z)``, ``scale = sqrt(N / N')`` (kind
    ``"widely"``).  ``A, B`` are :func:`pad_maps`; ``scale`` keeps the padded record's
    unit-norm cells in the record's cell units (GB ``variant``)."""

    src: FourierDomain
    dst: FourierDomain
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)
    scale: float = field(default=_BUILD, kw_only=True)
```

→

```python
@pytree_dataclass(data=("A", "B"), meta=("src", "dst"))
@dataclass(frozen=True, eq=False)
class FourierZeroPad(_WidelyLinear):
    """Bins ``dst`` of the record zero-padded from ``src.N`` to ``dst.N`` samples, from the
    bins ``src``: ``z' = A z + B conj z`` (kind ``"widely"``, ``A, B`` from :func:`pad_maps`),
    the rfft of the padded record in ``src``'s units (``dst`` must have the same
    ``(units, sigma0)``)."""

    src: FourierDomain
    dst: FourierDomain
    A: Any = field(default=_BUILD, kw_only=True)
    B: Any = field(default=_BUILD, kw_only=True)
```

`src/domainia/maps/fourier.py`, `FourierZeroPad.__post_init__`, `apply` and `adjoint`:

```python
            object.__setattr__(self, "B", B)
            object.__setattr__(self, "scale", float(np.sqrt(s.N / d.N)))

    def apply(self, z, be=None):
        be = self._be(z, be)
        z = be.asarray(z)
        return self.scale * ((z @ self.A.T) + (be.xp.conj(z) @ self.B.T))

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        return self.scale * ((y @ be.xp.conj(self.A)) + (be.xp.conj(y) @ self.B))
```

→

```python
            object.__setattr__(self, "B", B)

    def apply(self, z, be=None):
        be = self._be(z, be)
        z = be.asarray(z)
        return (z @ self.A.T) + (be.xp.conj(z) @ self.B.T)

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        return (y @ be.xp.conj(self.A)) + (be.xp.conj(y) @ self.B)
```

`src/domainia/maps/fourier.py`, `FourierCrop` (docstring and the class attribute `scale`):

```python
    bins ``src``: ``z^c = A z + B conj z`` (:func:`crop_maps`; GB ``whittle_map`` and
    ``mask_rows``).  ``apply`` is the oracle's template call site
    ``(h @ A.T) + (conj(h) @ B.T)``; :meth:`apply_data` its data call site."""
```

→

```python
    bins ``src``: ``z^c = A z + B conj z`` (:func:`crop_maps`; GB ``whittle_map`` and
    ``mask_rows``), the rfft of the cropped record in ``src``'s units (``dst`` must have the
    same ``(units, sigma0)``).  ``apply`` is the oracle's template call site
    ``(h @ A.T) + (conj(h) @ B.T)``; :meth:`apply_data` its data call site."""
```

`src/domainia/maps/fourier.py`, `FourierCrop` (remove `scale = 1.0`):

```python
    B: Any = field(default=_BUILD, kw_only=True)

    scale = 1.0

    def __post_init__(self) -> None:
```

→

```python
    B: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
```

`src/domainia/maps/fourier.py`, `FourierToWDM.after` (docstring):

```python
        ``C = kappa * inner.scale * (analyse(A[idx]) + conj(analyse(B[idx])))``, rows of
        ``dst``, with ``self``'s ``kappa``.
```

→

```python
        ``C = kappa * (analyse(A[idx]) + conj(analyse(B[idx])))``, rows of ``dst``, with
        ``self``'s ``kappa``: ``inner`` keeps the units, so the composite's cells are the
        Wilson coefficients of the padded or cropped record.
```

`src/domainia/maps/fourier.py`, `FourierToWDM.after` (body):

```python
        C = (_kappa(self.src, g) * inner.scale) * C
```

→

```python
        C = _kappa(self.src, g) * C
```

- [ ] **Step 6: Run the tests again**

Run: `$PY -m pytest -q tests/test_dom_maps_dirichlet.py tests/test_dom_maps_fourier_wdm.py`
Expected: `35 passed`

Run:

```bash
grep -rnw "scale" src/domainia/maps
```

Expected hits:

```text
src/domainia/maps/base.py:283:    ``scale``."""
```

Only `Identity`'s docstring in `maps/base.py` (Task 4).

- [ ] **Step 7: Full suite, NumPy lane, Meyer baseline**

Run: `$PY -m pytest -q`
Expected: `1489 passed, 3 skipped`

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: `1400 passed, 92 skipped`

Run: `$PY tests/wilson/meyer_cases.py --check`
Expected: `baseline reproduces: 1504 arrays` (the baseline has no pad or crop case).

- [ ] **Step 8: Commit (sandbox disabled)**

```bash
git add src/domainia/maps/fourier.py tests/test_dom_maps_dirichlet.py tests/test_dom_maps_fourier_wdm.py
git commit -m "Pad and crop keep the units; self-conjugate bins count once

pad_maps and crop_maps halve the A and B columns of the DC and (even N) Nyquist
source bins, so a full-record pad or crop is the exact rfft of the padded or
cropped series. FourierZeroPad loses its sqrt(N / N') scale and FourierCrop its
scale = 1.0: both keep src's units, and FourierToWDM.after(inner) is kappa times
the analysis of A and B, the Wilson coefficients of the padded or cropped record.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Remove `WDMDomain.scale` (spec §1.6)

**Goal:** `WDMDomain(cells, window=MEYER)` has no `scale`. `require_unit_scale` and every scale refusal and scale docstring are gone.

**Files:**
- Modify: `src/domainia/domains.py` (`WDMDomain`, the module docstring)
- Modify: `src/domainia/maps/time.py` (`TimeToWDM`, `WDMToTime`), `src/domainia/maps/fourier.py` (`FourierToWDM`), `src/domainia/maps/base.py` (`Restrict`, the `Identity` docstring)
- Modify: `src/domainia/translate.py` (the scale refusal)
- Test: `tests/test_translate.py`, `tests/wilson/test_window_threading.py`, `tests/test_dom_maps_base.py`, `tests/test_dom_domains.py` (`tests/wilson/test_windows.py` needs no change: its `.scale` is a window profile's own attribute)

**Acceptance Criteria:**
- [ ] `inspect.signature(WDMDomain)` lists exactly `cells, window`, and `WDMDomain(cells, 2.0)` raises `TypeError` (the window check).
- [ ] `grep -rn "require_unit_scale" src tests` prints nothing. `grep -rnE "\.scale\b" src` prints nothing, or only hits that are not on a `WDMDomain` (each listed in the step).
- [ ] The `"wdm-scale"` refusal case in `tests/test_translate.py` is removed.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged from Task 3.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

**Steps:**

Assumes the tree after Task 3.

- [ ] **Step 1: Grep for `WDMDomain.scale`: its readers, its checks and the calls that pass one**

Run:

```bash
grep -rn "require_unit_scale" src tests
```

Expected hits:

```text
src/domainia/maps/time.py:61:            self.dst.require_unit_scale("TimeToWDM")
src/domainia/maps/time.py:118:            self.src.require_unit_scale("WDMToTime")
src/domainia/maps/fourier.py:270:        self.dst.require_unit_scale("FourierToWDM")
src/domainia/maps/base.py:371:        self.src.require_unit_scale("Restrict")
src/domainia/maps/base.py:372:        self.dst.require_unit_scale("Restrict")
src/domainia/domains.py:362:    and ``Restrict`` refuse ``scale != 1`` (:meth:`require_unit_scale`) rather than treat it
src/domainia/domains.py:393:    def require_unit_scale(self, user: str) -> None:
```

The method and its mention in the `WDMDomain` docstring (`domains.py`), and its five calls (`TimeToWDM`, `WDMToTime`, `FourierToWDM`, `Restrict` twice). All go in Steps 4 and 5.

Run:

```bash
grep -rnw "scale" src/domainia/domains.py src/domainia/maps src/domainia/translate.py
```

Expected hits:

```text
src/domainia/domains.py:15:* :class:`WDMDomain`: a cell set with a constant ``scale`` (the GB cells are
src/domainia/domains.py:354:    """A cell set, the constant ``scale`` that relates its cells to the Wilson
src/domainia/domains.py:361:    No map applies ``scale``, so ``TimeToWDM``, ``WDMToTime``, ``FourierToWDM``
src/domainia/domains.py:362:    and ``Restrict`` refuse ``scale != 1`` (:meth:`require_unit_scale`) rather than treat it
src/domainia/domains.py:364:    that passes one through ``Identity`` must handle ``scale`` itself.
src/domainia/domains.py:368:    scale: float = 1.0
src/domainia/domains.py:374:        object.__setattr__(self, "scale", _positive_float("scale", self.scale))
src/domainia/domains.py:385:        """Every active cell of ``grid``, scale 1, Wilson ``window``: the WDM domain of
src/domainia/domains.py:394:        """Raise ``ValueError`` unless ``scale == 1.0``: ``user`` (a map) does not apply it."""
src/domainia/domains.py:395:        if self.scale != 1.0:
src/domainia/domains.py:396:            raise ValueError(f"WDMDomain.scale = {self.scale!r} is not honoured by {user}: no map "
src/domainia/domains.py:397:                             "applies a cell scale, so only scale = 1.0 is supported")
src/domainia/maps/base.py:283:    ``scale``."""
src/domainia/translate.py:61:                             f"active cell, scale 1); got {domain.cells.size} of {g.N} cells")
src/domainia/translate.py:62:        if domain.scale != 1.0:
src/domainia/translate.py:64:                             f"active cell, scale 1); got scale={float(domain.scale)!r}")
src/domainia/translate.py:235:    """``s``, the log of the scale per real coordinate: ``domain``'s real coordinates (module
```

* `domains.py`, every hit (the module bullet, the `WDMDomain` docstring, field and coercion, `full`'s docstring, `require_unit_scale`): Step 4.
* `maps/base.py` (`Identity`'s docstring): Step 5.
* `translate.py` (`_record`'s scale branch and both messages): Step 6. The hit in `log_coordinate_scale`'s first line is the coordinate scale it returns, not this field: keep it. (Its "or a scaled WDM one" further down, which `-w` does not match, also goes in Step 6.)

Run:

```bash
grep -rnE "WDMDomain\([A-Za-z_.]+, *[0-9]|scale=[0-9]|\b(d|WDM|domain)\.scale\b" src tests
```

Expected hits:

```text
src/domainia/translate.py:62:        if domain.scale != 1.0:
src/domainia/translate.py:64:                             f"active cell, scale 1); got scale={float(domain.scale)!r}")
tests/wilson/test_window_threading.py:61:    # (``WDMDomain(cells, 1.0, window)``) is flagged, a conservative false positive; a
tests/wilson/test_window_threading.py:263:    ("D.WDMDomain(cells, 2.0, window=self.dst.window)", False),
tests/wilson/test_window_threading.py:267:    ("domains.WDMDomain(cells, scale=2.0)", True),
tests/test_dom_domains.py:163:    assert (d.size, d.scale) == (128, 1.0)
tests/test_dom_domains.py:164:    assert WDMDomain(cells, 2.0) != d
tests/test_dom_domains.py:166:        WDMDomain(cells, 0.0)
tests/test_translate.py:64:    assert WDM == WDMDomain(CellSet.all_active(GRID)) and WDM.window is MEYER and WDM.scale == 1.0
tests/test_translate.py:173:    "wdm-scale": (X, TIME, WDMDomain(CellSet.all_active(GRID), scale=2.0), ValueError, "scale"),
tests/test_dom_maps_base.py:67:        Restrict(full, WDMDomain(cut.cells, 2.0), pos=R.pos)
tests/test_dom_maps_base.py:69:        dataclasses.replace(R, src=WDMDomain(full.cells, 2.0))
tests/test_dom_maps_base.py:199:    full2, cut2 = WDMDomain(full.cells, 2.0), WDMDomain(cut.cells, 2.0)
tests/test_dom_maps_base.py:209:        FourierToWDM(bins, WDMDomain(bands, 2.0))
```

Every call that passes a scale (positionally or as `scale=`) and every reader of `.scale` on a WDM domain:

* `test_dom_domains.py` (`test_wdm_domain_and_channels`), `test_translate.py` (`test_the_full_domains`, the `"wdm-scale"` refusal), `test_dom_maps_base.py` (the two `Restrict` scale cases, `test_a_scaled_wdm_domain_is_refused_by_every_map`) and `test_window_threading.py` (the `CONSUMERS` comment and two scan-case strings): Step 2.
* `translate.py` (`_record`): Step 6.

No other code passes a scale (`tests/wilson/test_windows.py`'s `self.scale` is a window profile's own attribute, not this field).

- [ ] **Step 2: Write the failing tests**

`tests/test_dom_domains.py`: the field is gone, and the second positional field is the window.

`tests/test_dom_domains.py`, the imports:

```python
from __future__ import annotations

import numpy as np
```

→

```python
from __future__ import annotations

import inspect

import numpy as np
```

`tests/test_dom_domains.py`, `test_wdm_domain_and_channels`:

```python
    assert (d.size, d.scale) == (128, 1.0)
    assert WDMDomain(cells, 2.0) != d
    with pytest.raises(ValueError):
        WDMDomain(cells, 0.0)
```

→

```python
    assert d.size == 128 and not hasattr(d, "scale")
    assert list(inspect.signature(WDMDomain).parameters) == ["cells", "window"]
    with pytest.raises(TypeError, match="Window"):
        WDMDomain(cells, 2.0)                       # the second field is the window now
    with pytest.raises(TypeError):
        WDMDomain(cells, scale=2.0)                 # the field is gone (spec section 1.6)
```

`tests/test_dom_maps_base.py`: the scale refusals go (they test a check that no longer exists).

`tests/test_dom_maps_base.py`, `test_restrict_checks_its_domains_even_when_pos_is_given`:

```python
    """The scale and grid checks read the domains only, so they run on every construction
    route (``pos`` given: ``on``, ``tree_unflatten``, ``dataclasses.replace``)."""
    full, cut = _cells()
    R = Restrict(full, cut)
    with pytest.raises(ValueError, match=r"scale = 2.0 is not honoured.*Restrict"):
        Restrict(full, WDMDomain(cut.cells, 2.0), pos=R.pos)
    with pytest.raises(ValueError, match=r"scale = 2.0 is not honoured.*Restrict"):
        dataclasses.replace(R, src=WDMDomain(full.cells, 2.0))
```

→

```python
    """The grid check reads the domains only, so it runs on every construction route
    (``pos`` given: ``on``, ``tree_unflatten``, ``dataclasses.replace``).  The window check
    is tests/test_dom_windows.py's."""
    full, cut = _cells()
    R = Restrict(full, cut)
```

`tests/test_dom_maps_base.py`, `test_a_scaled_wdm_domain_is_refused_by_every_map` (delete it):

```python
def test_a_scaled_wdm_domain_is_refused_by_every_map():
    """``WDMDomain.scale`` is applied by no map: a scale other than 1 is refused at
    construction instead of being silently treated as 1.  The domain itself (a label) and
    the identity on it stay valid."""
    from domainia.maps import FourierToWDM, TimeToWDM, WDMToTime

    full, cut = _cells()
    full2, cut2 = WDMDomain(full.cells, 2.0), WDMDomain(cut.cells, 2.0)
    time = TimeDomain(GRID.N, GRID.dt)
    bins = FourierDomain(GRID.N, GRID.dt, 0, GRID.N // 2 + 1, "rfft")
    bands = CellSet.bands(GRID, (1, 2, 3))
    msg = r"WDMDomain.scale = 2.0 is not honoured"
    with pytest.raises(ValueError, match=msg + r".*TimeToWDM"):
        TimeToWDM(time, full2)
    with pytest.raises(ValueError, match=msg + r".*WDMToTime"):
        WDMToTime(full2, time)
    with pytest.raises(ValueError, match=msg + r".*FourierToWDM"):
        FourierToWDM(bins, WDMDomain(bands, 2.0))
    for src, dst in ((full2, cut), (full, cut2), (full2, cut2)):
        with pytest.raises(ValueError, match=msg + r".*Restrict"):
            Restrict(src, dst)
    assert Identity(full2).dst == full2         # nothing to apply; the caller handles scale
    for m in (TimeToWDM(time, full), WDMToTime(full, time), FourierToWDM(bins, WDMDomain(bands)),
              Restrict(full, cut)):
        assert isinstance(m, LinearMap)                        # scale 1 is unchanged


```

→ delete it (nothing in its place).

`tests/test_translate.py`:

`tests/test_translate.py`, `test_the_full_domains`:

```python
    assert WDM == WDMDomain(CellSet.all_active(GRID)) and WDM.window is MEYER and WDM.scale == 1.0
```

→

```python
    assert WDM == WDMDomain(CellSet.all_active(GRID)) and WDM.window is MEYER
```

`tests/test_translate.py`, `REFUSALS` (drop the `"wdm-scale"` case):

```python
    "wdm-scale": (X, TIME, WDMDomain(CellSet.all_active(GRID), scale=2.0), ValueError, "scale"),
```

→ delete it (nothing in its place).

`tests/wilson/test_window_threading.py`: the scan cases no longer pass a scale. The positional-window case keeps the documented conservative flag.

`tests/wilson/test_window_threading.py`, the comment in `CONSUMERS`:

```python
    # (``WDMDomain(cells, 1.0, window)``) is flagged, a conservative false positive; a
```

→

```python
    # (``WDMDomain(cells, window)``) is flagged, a conservative false positive; a
```

`tests/wilson/test_window_threading.py`, the scan cases (the `WDMDomain` calls with a scale):

```python
    ("D.WDMDomain(cells, 2.0, window=self.dst.window)", False),
```

→

```python
    ("D.WDMDomain(cells, window=self.dst.window)", False),
```

`tests/wilson/test_window_threading.py`, the same list:

```python
    ("domains.WDMDomain(cells, scale=2.0)", True),
```

→

```python
    ("domains.WDMDomain(cells, ALT_WINDOW)", True),           # positional: flagged
```

- [ ] **Step 3: Run them and see them fail**

Run: `$PY -m pytest -q tests/test_dom_domains.py tests/test_dom_maps_base.py tests/test_translate.py tests/wilson/test_window_threading.py`
Expected: `1 failed, 657 passed`. The failure is `test_wdm_domain_and_channels`: `assert not hasattr(d, "scale")`. The deletions and the rewritten scan strings pass on both trees.

- [ ] **Step 4: `domains.py`: drop the field and `require_unit_scale`**

`_positive_float` stays (`FourierDomain` uses it for `dt` and `sigma0`).

`src/domainia/domains.py`, the module docstring's `WDMDomain` bullet:

```python
* :class:`WDMDomain`: a cell set with a constant ``scale`` (the GB cells are
  ``Re(C z)`` with unit-norm atoms on ``z = rfft / sigma0``; the sinusoid uses 1) and the
  Wilson ``window`` of its basis (default ``domainia.wilson.MEYER``, the WDM transform).
  ``.full(...)``: the full-record domain of :mod:`domainia.translate`.
```

→

```python
* :class:`WDMDomain`: a cell set and the Wilson ``window`` of its basis (default
  ``domainia.wilson.MEYER``, the WDM transform).  A value on it is the Wilson coefficient
  ``w = W x`` of the time record, whichever map produced it.  ``.full(...)``: the
  full-record domain of :mod:`domainia.translate`.
```

`src/domainia/domains.py`, `WDMDomain` (docstring, fields, `__post_init__`):

```python
    """A cell set, the constant ``scale`` that relates its cells to the Wilson
    coefficients, and the Wilson ``window`` of the basis (default :data:`MEYER`: WDM).

    The window is part of the domain's identity (equality and hashing), so two maps or
    covariances built for different windows never compose silently.  It must be a hashable
    :class:`~domainia.wilson.Window` instance (``TypeError`` otherwise).

    No map applies ``scale``, so ``TimeToWDM``, ``WDMToTime``, ``FourierToWDM``
    and ``Restrict`` refuse ``scale != 1`` (:meth:`require_unit_scale`) rather than treat it
    as 1 silently.  ``Identity`` accepts a scaled domain (it has nothing to apply): a caller
    that passes one through ``Identity`` must handle ``scale`` itself.
    """

    cells: CellSet
    scale: float = 1.0
    window: Window = MEYER

    def __post_init__(self) -> None:
        if not isinstance(self.cells, CellSet):
            raise TypeError(f"cells must be a CellSet; got {type(self.cells).__name__}")
        object.__setattr__(self, "scale", _positive_float("scale", self.scale))
        require_window(self.window)
```

→

```python
    """A cell set and the Wilson ``window`` of the basis (default :data:`MEYER`: WDM).

    A value on the domain is the Wilson coefficient ``w = W x`` of the time record in that
    window (``W W^T = dt I``), on every path that produces one: ``transform``,
    ``TimeToWDM`` and ``FourierToWDM`` (whatever the bins' units).

    The window is part of the domain's identity (equality and hashing), so two maps or
    covariances built for different windows never compose silently.  It must be a hashable
    :class:`~domainia.wilson.Window` instance (``TypeError`` otherwise).
    """

    cells: CellSet
    window: Window = MEYER

    def __post_init__(self) -> None:
        if not isinstance(self.cells, CellSet):
            raise TypeError(f"cells must be a CellSet; got {type(self.cells).__name__}")
        require_window(self.window)
```

`src/domainia/domains.py`, `WDMDomain.full` (docstring):

```python
        """Every active cell of ``grid``, scale 1, Wilson ``window``: the WDM domain of
```

→

```python
        """Every active cell of ``grid`` in the Wilson ``window``: the WDM domain of
```

`src/domainia/domains.py`, `WDMDomain.require_unit_scale` (delete it):

```python
    def require_unit_scale(self, user: str) -> None:
        """Raise ``ValueError`` unless ``scale == 1.0``: ``user`` (a map) does not apply it."""
        if self.scale != 1.0:
            raise ValueError(f"WDMDomain.scale = {self.scale!r} is not honoured by {user}: no map "
                             "applies a cell scale, so only scale = 1.0 is supported")

```

→ delete it (nothing in its place).

- [ ] **Step 5: The maps: drop the five `require_unit_scale` calls**

`src/domainia/maps/time.py`, `TimeToWDM.__post_init__`:

```python
            self.dst.require_unit_scale("TimeToWDM")
```

→ delete it (nothing in its place).

`src/domainia/maps/time.py`, `WDMToTime.__post_init__`:

```python
            self.src.require_unit_scale("WDMToTime")
```

→ delete it (nothing in its place).

`src/domainia/maps/fourier.py`, `FourierToWDM.__post_init__`:

```python
        self.dst.require_unit_scale("FourierToWDM")
```

→ delete it (nothing in its place).

`src/domainia/maps/base.py`, `Identity` (docstring):

```python
    """The identity on ``src`` (the sinusoid pad model's samples, spec §4.8).  A scaled
    ``WDMDomain`` passes through unchanged (nothing to apply): the caller handles
    ``scale``."""
```

→

```python
    """The identity on ``src`` (the sinusoid pad model's samples, spec §4.8)."""
```

`src/domainia/maps/base.py`, `Restrict.__post_init__`:

```python
        self.src.require_unit_scale("Restrict")
        self.dst.require_unit_scale("Restrict")
```

→ delete it (nothing in its place).

- [ ] **Step 6: `translate.py`: no scale branch**

`src/domainia/translate.py`, `_record` (the `WDMDomain` branch):

```python
            raise ValueError(f"{role}: translate handles only WDMDomain.full(grid, window) (every "
                             f"active cell, scale 1); got {domain.cells.size} of {g.N} cells")
        if domain.scale != 1.0:
            raise ValueError(f"{role}: translate handles only WDMDomain.full(grid, window) (every "
                             f"active cell, scale 1); got scale={float(domain.scale)!r}")
```

→

```python
            raise ValueError(f"{role}: translate handles only WDMDomain.full(grid, window) (every "
                             f"active cell); got {domain.cells.size} of {g.N} cells")
```

`src/domainia/translate.py`, `log_coordinate_scale` (docstring):

```python
    ``ValueError`` for a partial Fourier or WDM domain (or a scaled WDM one), ``TypeError``
    for anything else.
```

→

```python
    ``ValueError`` for a partial Fourier or WDM domain, ``TypeError`` for anything else.
```

- [ ] **Step 7: Run the tests again**

Run: `$PY -m pytest -q tests/test_dom_domains.py tests/test_dom_maps_base.py tests/test_translate.py tests/wilson/test_window_threading.py`
Expected: `658 passed`

Run:

```bash
grep -rn "require_unit_scale" src tests
```

Expected: no output.

Run:

```bash
grep -rnE "\.scale\b" src
```

Expected: no output.

No library code reads a `.scale` any more.

Run:

```bash
grep -rnE "WDMDomain\([A-Za-z_.]+, *[0-9]|scale=[0-9]|\b(d|WDM|domain)\.scale\b" src tests
```

Expected hits:

```text
tests/test_dom_domains.py:168:        WDMDomain(cells, 2.0)                       # the second field is the window now
tests/test_dom_domains.py:170:        WDMDomain(cells, scale=2.0)                 # the field is gone (spec section 1.6)
```

Only the two `TypeError` cases of `test_wdm_domain_and_channels`, which pass a scale on purpose.

- [ ] **Step 8: Full suite, NumPy lane, Meyer baseline**

Run: `$PY -m pytest -q`
Expected: `1487 passed, 3 skipped` (two fewer than after Task 3: `test_a_scaled_wdm_domain_is_refused_by_every_map` and the `"wdm-scale"` refusal case are gone).

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: `1398 passed, 92 skipped`

Run: `$PY tests/wilson/meyer_cases.py --check`
Expected: `baseline reproduces: 1504 arrays`

- [ ] **Step 9: Commit (sandbox disabled)**

```bash
git add src/domainia/domains.py src/domainia/maps/base.py src/domainia/maps/fourier.py src/domainia/maps/time.py src/domainia/translate.py tests/test_dom_domains.py tests/test_dom_maps_base.py tests/test_translate.py tests/wilson/test_window_threading.py
git commit -m "WDMDomain loses scale: every WDM value is a Wilson coefficient

With FourierToWDM in Wilson units, no domain needs a cell scale. WDMDomain is
(cells, window); require_unit_scale and its five callers go, and translate no
longer checks a scale. The scaled-domain refusal tests go with the check.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Public real coordinates `to_coordinates` / `from_coordinates` (spec §2.2)

**Goal:** `domainia.translate` exposes the real coordinates that the `Operator` contract already describes, with channel order pinned by `channel_names`.

**Files:**
- Modify: `src/domainia/translate.py`
- Create: `tests/test_translate_coordinates.py`
- Modify: `tests/test_translate_operator.py` (its private `_pack`/`_unpack` replaced by the public functions)

**Acceptance Criteria:**
- [ ] On each full-record domain (time, Fourier for even and odd `N`, WDM), `to_coordinates` returns `(K, N)` float64 in the layout of the Interface contract, and `from_coordinates` inverts it into the exact canonical layout (bit for bit on time and WDM, to 1e-15 on Fourier: the √2 scaling goes out and back).
- [ ] WDM coordinates follow `CellSet.all_active(grid)` order (asserted).
- [ ] Each domain's coordinates are `exp(log_coordinate_scale(domain))` times an orthogonal map of the time samples (`QᵀQ = I` to 1e-12, measured on a basis).
- [ ] With `channel_names=("A", "E")`, a dict built as `{"E": …, "A": …}` gives rows in `A, E` order, and a mismatched channel set raises `ValueError`.
- [ ] Input refusals match `transform`'s (same exception type and message) for every data refusal case `tests/test_translate.py` exercises. A refusal of the domain itself names its role (`domain` where `transform` says `source`); the test normalises that one word.
- [ ] `tests/test_translate_operator.py` no longer defines `_pack` or `_unpack`, and passes.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_translate_coordinates.py tests/test_translate_operator.py` → all pass; then `$PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 4.

Nothing here depends on the numbers Tasks 1–3 change: `FourierDomain.full(N, dt)` is still `"dt_rfft"` units with `sigma0=None`, and the WDM full-record domain is `WDMDomain.full(grid, window)`. `__all__` is not touched here; Task 8 exports the two functions.

- [ ] **Step 1: Check the private helpers' layout against the spec**

Run: `grep -n "_H = \|def _pack\|def _unpack\|def _cells" tests/test_translate_operator.py`
Expected: four hits: `_H = N // 2 - 1`, `def _cells(domain)` (`wilson.grid.packed_cells`), `def _pack(domain, a)`, `def _unpack(domain, r)`. The test record has even `N` (`assert N % 2 == 0` above `_H`), and for even `N` the spec's `h = (N − 1) // 2` equals `_H`. So `_pack` is the spec layout `[Re X_0, √2 Re X_1…X_h, √2 Im X_1…X_h, Re X_{N/2}]` on Fourier, `packed_cells` order on WDM and the samples on time, and `_unpack` is its inverse into the canonical layout. The public functions below add the odd-`N` case (no Nyquist coordinate) and the channel order.

- [ ] **Step 2: Write the failing tests `tests/test_translate_coordinates.py`**

The refusal test reuses the `REFUSALS` table of `tests/test_translate.py` (imported by module name, as the tests import `dom_db_inputs`). For each case it tries both of the case's domains: `to_coordinates(data, d)` must refuse exactly when `transform(data, d, d)` does, with the same type and message. A refusal of the domain itself names its role, `"source"` in `transform` and `"domain"` here, so `_refusal` reads that one word as `"domain"`. `t0` is the only case where neither domain refuses on its own (it is a rule of the pair).

```python
"""domainia.translate.to_coordinates / from_coordinates: the real coordinates of the Operator
contract, one row per channel, in channel_names order (spec section 2.2)."""

from __future__ import annotations

import numpy as np
import pytest
from test_translate import REFUSALS

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.testing import ALT_WINDOW
from domainia.translate import from_coordinates, log_coordinate_scale, to_coordinates, transform

NF, NT, DT = 8, 16, 0.25
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
TIME = TimeDomain(N, DT)
ODD_TIME = TimeDomain(15, 0.5)
DOMAINS = {
    "time": (TIME, TIME),
    "fourier": (TIME, FourierDomain.full(N, DT)),
    "fourier-odd": (ODD_TIME, FourierDomain.full(15, 0.5)),
    "wdm": (TIME, WDMDomain.full(GRID)),
    "wdm-alt": (TIME, WDMDomain.full(wilson.Grid(16, 8, DT), window=ALT_WINDOW)),
}
_rng = np.random.default_rng(47)


def _record(time):
    return {"A": _rng.standard_normal(time.N), "E": _rng.standard_normal(time.N)}


def _reference(values, domain):
    """The coordinates by their definition (numpy only, no to_coordinates), one row per channel."""
    rows = []
    for v in values.values():
        if isinstance(domain, TimeDomain):
            rows.append(v)
        elif isinstance(domain, FourierDomain):
            h = (domain.N - 1) // 2
            r = [v[:1].real, np.sqrt(2) * v[1:h + 1].real, np.sqrt(2) * v[1:h + 1].imag]
            rows.append(np.concatenate(r + ([v[-1:].real] if domain.N % 2 == 0 else [])))
        else:
            g = domain.cells.grid
            m, n = wilson.grid.packed_cells(g.Nf, g.Nt)
            rows.append(v[m, n])
    return np.stack(rows)


@pytest.mark.parametrize("name", DOMAINS)
def test_the_coordinates_are_the_documented_layout(name):
    time, d = DOMAINS[name]
    y = transform(_record(time), time, d)
    R = to_coordinates(y, d)
    assert R.dtype == np.float64 and R.shape == (2, time.N)
    np.testing.assert_array_equal(R, _reference(y, d))
    if isinstance(d, FourierDomain):
        X = d.dt * np.fft.rfft(transform(y, d, time)["A"])      # dt * rfft(x), independently
        h = (d.N - 1) // 2
        np.testing.assert_allclose(R[0, 1:h + 1], np.sqrt(2) * X[1:h + 1].real, rtol=0,
                                   atol=1e-13 * np.max(np.abs(X)))
        np.testing.assert_allclose(R[0, h + 1:2 * h + 1], np.sqrt(2) * X[1:h + 1].imag, rtol=0,
                                   atol=1e-13 * np.max(np.abs(X)))


def test_wdm_coordinates_follow_the_all_active_cell_set():
    d = DOMAINS["wdm"][1]
    y = transform(_record(TIME), TIME, d)
    cells = CellSet.all_active(GRID)
    m, n = wilson.grid.packed_cells(NF, NT)
    assert cells.m == tuple(m.tolist()) and cells.n == tuple(n.tolist())
    np.testing.assert_array_equal(to_coordinates(y, d)[1], y["E"][cells.m_array, cells.n_array])


@pytest.mark.parametrize("name", DOMAINS)
def test_from_coordinates_inverts_to_the_canonical_layout(name):
    time, d = DOMAINS[name]
    y = transform(_record(time), time, d)
    back = from_coordinates(to_coordinates(y, d), d, ("A", "E"))
    assert list(back) == ["A", "E"]
    for k in y:
        assert back[k].dtype == y[k].dtype and back[k].shape == y[k].shape
        if isinstance(d, FourierDomain):                          # sqrt(2) out and back: 1 ulp
            np.testing.assert_allclose(back[k], y[k], rtol=0, atol=1e-15 * np.max(np.abs(y[k])))
        else:
            np.testing.assert_array_equal(back[k], y[k])
    R = _rng.standard_normal((2, time.N))
    out = from_coordinates(R, d, ("A", "E"))
    if isinstance(d, FourierDomain):
        assert all(v[0].imag == 0.0 for v in out.values())
        if d.N % 2 == 0:
            assert all(v[-1].imag == 0.0 for v in out.values())
    if isinstance(d, WDMDomain):
        g = d.cells.grid
        assert all(np.all(v[~wilson.active_mask(g.Nf, g.Nt)] == 0.0) for v in out.values())
    np.testing.assert_allclose(to_coordinates(transform(out, d, d), d), R, rtol=0,
                               atol=1e-15 * np.max(np.abs(R)))


@pytest.mark.parametrize("name", DOMAINS)
def test_the_coordinates_are_an_orthogonal_map_of_the_time_samples_times_the_log_scale(name):
    """Column ``j`` of ``Q`` is the coordinates of the unit sample ``e_j``, scaled by
    ``exp(-log_coordinate_scale)``: ``Q^T Q = I``."""
    time, d = DOMAINS[name]
    Q = np.stack([to_coordinates(transform({"A": e}, time, d), d)[0] for e in np.eye(time.N)], axis=1)
    Q *= np.exp(-log_coordinate_scale(d))
    np.testing.assert_allclose(Q.T @ Q, np.eye(time.N), rtol=0, atol=1e-12)


def test_channel_names_pin_the_row_order():
    d = DOMAINS["fourier"][1]
    y = transform(_record(TIME), TIME, d)
    flipped = {"E": y["E"], "A": y["A"]}
    np.testing.assert_array_equal(to_coordinates(flipped, d, ("A", "E")), to_coordinates(y, d))
    np.testing.assert_array_equal(to_coordinates(flipped, d)[0], to_coordinates(y, d)[1])  # dict order
    out = from_coordinates(to_coordinates(y, d), d, ("E", "A"))
    assert list(out) == ["E", "A"]
    np.testing.assert_allclose(out["E"], y["A"], rtol=0, atol=1e-15 * np.max(np.abs(y["A"])))
    with pytest.raises(ValueError, match=r"^channel_data has channels \('A', 'E'\); channel_names are "
                                         r"\('A', 'T'\)$"):
        to_coordinates(y, d, ("A", "T"))
    with pytest.raises(ValueError, match="channel_names are"):
        to_coordinates({"A": y["A"]}, d, ("A", "E"))


def _refusal(fn, *args):
    """``(type, message)`` of the refusal ``fn(*args)`` raises, or ``None``; the role word that
    names a bad domain (``source`` in transform, ``domain`` here) is read as ``domain``."""
    try:
        fn(*args)
    except (TypeError, ValueError) as err:
        msg = str(err)
        return type(err), "domain" + msg[len("source"):] if msg.startswith("source") else msg
    return None


@pytest.mark.parametrize("case", REFUSALS)
def test_input_refusals_are_transforms(case):
    """Every refusal case of tests/test_translate.py, on each of its two domains in turn:
    ``to_coordinates(data, d)`` refuses exactly when ``transform(data, d, d)`` does, with the
    same type and message.  ``t0`` is a rule of the pair alone, so on that case both domains
    accept the data."""
    data, src, dst, _, _ = REFUSALS[case]
    seen = []
    for d in (src, dst):
        want = _refusal(transform, data, d, d)
        assert _refusal(to_coordinates, data, d) == want
        seen.append(want)
    assert any(seen) or case == "t0"


def test_argument_refusals():
    d = DOMAINS["wdm"][1]
    y = transform(_record(TIME), TIME, d)
    R = to_coordinates(y, d)
    with pytest.raises(TypeError, match="channel_names must be a tuple"):
        to_coordinates(y, d, "AE")
    with pytest.raises(TypeError, match="channel names must be non-empty strings"):
        from_coordinates(R, d, ("A", 1))
    with pytest.raises(ValueError, match="unique"):
        from_coordinates(R, d, ("A", "A"))
    with pytest.raises(ValueError, match="channel_names is empty"):
        from_coordinates(R[:0], d, ())
    with pytest.raises(ValueError, match=r"^R has shape \(1, 128\); 2 channel\(s\) on this WDMDomain need "
                                         r"shape \(2, 128\)$"):
        from_coordinates(R[:1], d, ("A", "E"))
    with pytest.raises(ValueError, match="complex"):
        from_coordinates(R + 0j, d, ("A", "E"))
    with pytest.raises(ValueError, match="not finite"):
        from_coordinates(np.where(np.arange(N) == 3, np.nan, R), d, ("A", "E"))
    with pytest.raises(TypeError, match="real coordinates"):
        from_coordinates(R.astype(object), d, ("A", "E"))
    with pytest.raises(TypeError, match="masked"):
        from_coordinates(np.ma.MaskedArray(R), d, ("A", "E"))
    with pytest.raises(ValueError, match="^domain: translate handles only WDMDomain.full"):
        from_coordinates(R, WDMDomain(CellSet.bands(GRID, (2, 3))), ("A", "E"))
    with pytest.raises(TypeError, match="^domain must be a full-record domain"):
        from_coordinates(R, GRID, ("A", "E"))
```

- [ ] **Step 3: Run the new tests and watch them fail**

Run: `$PY -m pytest -q tests/test_translate_coordinates.py`
Expected: `ImportError: cannot import name 'from_coordinates' from 'domainia.translate'`, then `Interrupted: 2 errors during collection` (`--doctest-modules` collects the file a second time).

- [ ] **Step 4: Implement the coordinates in `src/domainia/translate.py`**

(a) Module docstring, the paragraph on the real coordinates:

```python
In the real coordinates (time samples; WDM cells; Fourier ``Re X_0``, ``sqrt(2) Re/Im X_k``,
``Re X_{N/2}``) every domain is the time record scaled by ``exp(log_coordinate_scale(domain))``:
1 for time, ``sqrt(dt)`` for WDM (any admissible window), ``dt sqrt(N)`` for Fourier.
```
→
```python
In the real coordinates (time samples; WDM cells; Fourier ``Re X_0``, ``sqrt(2) Re/Im X_k``,
``Re X_{N/2}``) every domain is the time record scaled by ``exp(log_coordinate_scale(domain))``:
1 for time, ``sqrt(dt)`` for WDM (any admissible window), ``dt sqrt(N)`` for Fourier.
:func:`to_coordinates` turns a dict into these coordinates, one row per channel, and
:func:`from_coordinates` turns the rows back into a dict in the canonical layout.
```

(b) The domains import:

```python
from domainia.domains import FourierDomain, TimeDomain, WDMDomain
```
→
```python
from domainia.domains import ChannelAxis, FourierDomain, TimeDomain, WDMDomain
```

(c) After `_active`, add `_packed`:

```python
def _active(domain: WDMDomain) -> np.ndarray:
    g = domain.cells.grid
    return wilson.active_mask(g.Nf, g.Nt)
```
→
```python
def _active(domain: WDMDomain) -> np.ndarray:
    g = domain.cells.grid
    return wilson.active_mask(g.Nf, g.Nt)


def _packed(domain: WDMDomain) -> tuple[np.ndarray, np.ndarray]:
    """``(m, n)`` of every active cell in ``wilson.grid.packed_cells`` order."""
    g = domain.cells.grid
    return wilson.grid.packed_cells(g.Nf, g.Nt)
```

(d) Between `log_coordinate_scale` and `@runtime_checkable` / `class Operator(Protocol):`, insert these five functions. `to_coordinates` validates with the same `_record` and `_stack` that `transform` uses, so the data refusals are `transform`'s own. `_channel_names` is shared with `domainia.operators` (Tasks 6 and 8).

```python
def _channel_names(channel_names: Any) -> tuple[str, ...]:
    """``channel_names`` as a tuple, by the :class:`~domainia.domains.ChannelAxis` rules
    (non-empty strings, ``TypeError``; unique, ``ValueError``), and at least one name."""
    if isinstance(channel_names, str) or not isinstance(channel_names, (tuple, list)):
        raise TypeError(f"channel_names must be a tuple of channel names; got "
                        f"{type(channel_names).__name__}")
    names = ChannelAxis(tuple(channel_names)).names
    if not names:
        raise ValueError("channel_names is empty: name at least one channel")
    return names


def _real_coordinates(X: np.ndarray, domain: Domain) -> np.ndarray:
    """``(K, N)``: the real coordinates of the validated ``(K, *shape)`` stack ``X``."""
    if isinstance(domain, TimeDomain):
        return X
    if isinstance(domain, FourierDomain):
        h = (domain.N - 1) // 2
        root2 = np.sqrt(2.0)
        parts = [X[:, :1].real, root2 * X[:, 1:h + 1].real, root2 * X[:, 1:h + 1].imag]
        if domain.N % 2 == 0:
            parts.append(X[:, -1:].real)
        return np.concatenate(parts, axis=1)
    m, n = _packed(domain)
    return X[:, m, n]


def _canonical_layout(R: np.ndarray, domain: Domain) -> np.ndarray:
    """``(K, *shape)``: the canonical layout of the ``(K, N)`` float64 coordinates ``R``."""
    K = R.shape[0]
    if isinstance(domain, TimeDomain):
        return R.copy()
    if isinstance(domain, FourierDomain):
        N = domain.N
        h = (N - 1) // 2
        out = np.zeros((K, N // 2 + 1), dtype=np.complex128)
        out[:, 0] = R[:, 0]
        out[:, 1:h + 1] = (R[:, 1:h + 1] + 1j * R[:, h + 1:2 * h + 1]) / np.sqrt(2.0)
        if N % 2 == 0:
            out[:, -1] = R[:, -1]
        return out
    g = domain.cells.grid
    out = np.zeros((K, g.Nf + 1, g.Nt))
    m, n = _packed(domain)
    out[:, m, n] = R
    return out


def to_coordinates(channel_data: Mapping[str, Any], domain: Domain,
                   channel_names: tuple[str, ...] | None = None) -> np.ndarray:
    """``(K, N)`` float64: row ``i`` is channel ``i``'s real coordinates on ``domain``.

    The coordinates (module docstring): the time samples; the WDM active cells in
    ``wilson.grid.packed_cells`` order (the order of ``CellSet.all_active``); and, for
    Fourier with ``X = dt numpy.fft.rfft(x)`` and ``h = (N - 1) // 2``,
    ``[Re X_0, sqrt(2) Re X_1 .. sqrt(2) Re X_h, sqrt(2) Im X_1 .. sqrt(2) Im X_h]``, then
    ``Re X_{N/2}`` for even ``N`` only.

    The rows follow ``channel_names`` when it is given, whatever order the dict was built
    in, and the dict's channels must then be exactly those names (``ValueError``
    otherwise); without it they follow the dict's order.  ``channel_data`` is validated
    exactly as :func:`transform` validates its input, with the same errors; ``domain``
    must be a full-record domain (``TypeError`` for a non-domain, ``ValueError`` for a
    partial one, as :func:`log_coordinate_scale`).
    """
    _record(domain, "domain")
    want = None if channel_names is None else _channel_names(channel_names)
    names, X = _stack(channel_data, domain)
    if want is not None:
        if set(names) != set(want):
            raise ValueError(f"channel_data has channels {names}; channel_names are {want}")
        X = X[[names.index(k) for k in want]]
    return _real_coordinates(X, domain)


def from_coordinates(R: Any, domain: Domain, channel_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The ``(K, N)`` real coordinates ``R`` (:func:`to_coordinates`) as a
    ``{channel: array}`` dict on ``domain``, in ``channel_names`` order.

    The layout is canonical, exactly: inactive WDM cells are 0 and the Fourier DC bin (and
    the even-``N`` Nyquist bin) are real, so :func:`transform` takes the result back.
    Raises ``TypeError`` for a masked array, non-numeric values or bad ``channel_names``,
    and ``ValueError`` for complex or non-finite values, or a shape other than
    ``(len(channel_names), N)``; ``domain`` as :func:`to_coordinates`.
    """
    N, _ = _record(domain, "domain")
    names = _channel_names(channel_names)
    if isinstance(R, np.ma.MaskedArray):
        raise TypeError("R: masked arrays are refused (numpy.asarray drops the mask)")
    a = np.asarray(R)
    if a.dtype.kind not in "iufc":
        raise TypeError(f"R must hold real coordinates; got dtype {a.dtype}")
    if a.dtype.kind == "c":
        raise ValueError("R must hold real coordinates; got complex values")
    if a.shape != (len(names), N):
        raise ValueError(f"R has shape {a.shape}; {len(names)} channel(s) on this "
                         f"{type(domain).__name__} need shape ({len(names)}, {N})")
    wilson.validate_finite(a, "R")
    out = _canonical_layout(a.astype(np.float64), domain)
    return {name: out[i].copy() for i, name in enumerate(names)}
```

- [ ] **Step 5: Run the new tests and watch them pass**

Run: `$PY -m pytest -q tests/test_translate_coordinates.py`
Expected: `50 passed` (18 tests plus one per `REFUSALS` case: 32 cases once Task 4 removed `"wdm-scale"`).

- [ ] **Step 6: Replace `_pack`/`_unpack` in `tests/test_translate_operator.py` with a one-off codemod** (keep it in `$TMPDIR`, do not commit it)

Two places need more than a rename. `_native_covariance` packed `PointwiseOperator.variances`, which hold 1.0 at the inactive WDM cells (so `solve` never divides by 0); `to_coordinates` refuses a non-zero inactive cell, so those are zeroed first. The dense-reference loop now takes every channel's coordinates in one `to_coordinates(..., names)` call and indexes rows. Save as `$TMPDIR/t5_coordinates_in_tests.py`:

```python
"""One-off (Task 5): tests/test_translate_operator.py uses to_coordinates / from_coordinates."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "tests/test_translate_operator.py"
t = p.read_text()
EDITS = [
("""from domainia.translate import (
    Operator,
    TranslatedOperator,
    log_coordinate_scale,
    transform,
    translate_operator,
)
""", """from domainia.translate import (
    Operator,
    TranslatedOperator,
    from_coordinates,
    log_coordinate_scale,
    to_coordinates,
    transform,
    translate_operator,
)
"""),
("""# -- independent references: numpy.fft.rfft and wilson.matrix.active_W only, never transform --
assert N % 2 == 0                                    # the Fourier maps below have a Nyquist bin
_H = N // 2 - 1                                      # interior rfft bins


def _cells(domain):
    g = domain.cells.grid
    return wilson.grid.packed_cells(g.Nf, g.Nt)


def _coordinates(domain):""", """# -- independent references: numpy.fft.rfft and wilson.matrix.active_W only, never transform --
assert N % 2 == 0                                    # the Fourier maps below have a Nyquist bin


def _coordinates(domain):"""),
('''def _pack(domain, a):
    """Layout -> real coordinates."""
    a = np.asarray(a)
    if isinstance(domain, TimeDomain):
        return a.astype(float)
    if isinstance(domain, FourierDomain):
        return np.concatenate([a[:1].real, np.sqrt(2) * a[1:-1].real, np.sqrt(2) * a[1:-1].imag, a[-1:].real])
    m, n = _cells(domain)
    return a[m, n]


def _unpack(domain, r):
    """Real coordinates -> the canonical layout (inactive cells exactly 0, end bins exactly real)."""
    if isinstance(domain, TimeDomain):
        return np.array(r, dtype=float)
    if isinstance(domain, FourierDomain):
        out = np.empty(N // 2 + 1, dtype=complex)
        out[0], out[-1] = r[0], r[-1]
        out[1:-1] = (r[1:1 + _H] + 1j * r[1 + _H:-1]) / np.sqrt(2)
        return out
    g = domain.cells.grid
    out = np.zeros((g.Nf + 1, g.Nt))
    m, n = _cells(domain)
    out[m, n] = r
    return out


''', ""),
("""    def _z(self, values):
        return {k: self.U.T @ _pack(self.domain, values[k]) for k in self.channel_names}

    def apply(self, values):
        return {k: _unpack(self.domain, self.U @ (self.Cr[k] @ z)) for k, z in self._z(values).items()}

    def solve(self, values):
        return {k: _unpack(self.domain, self.U @ np.linalg.solve(self.Cr[k], z)) for k, z in self._z(values).items()}

    def project(self, values):
        return {k: _unpack(self.domain, self.U @ z) for k, z in self._z(values).items()}
""", """    def _z(self, values):
        R = to_coordinates(values, self.domain, self.channel_names)
        return {k: self.U.T @ R[i] for i, k in enumerate(self.channel_names)}

    def _out(self, rows):
        return from_coordinates(np.stack([rows[k] for k in self.channel_names]), self.domain,
                                self.channel_names)

    def apply(self, values):
        return self._out({k: self.U @ (self.Cr[k] @ z) for k, z in self._z(values).items()})

    def solve(self, values):
        return self._out({k: self.U @ np.linalg.solve(self.Cr[k], z) for k, z in self._z(values).items()})

    def project(self, values):
        return self._out({k: self.U @ z for k, z in self._z(values).items()})
"""),
('''    its definition, and the rank it was built with."""
    if isinstance(op, PointwiseOperator):
        return np.diag(_pack(op.domain, op.variances[k])), N
''', '''    its definition, and the rank it was built with."""
    if isinstance(op, PointwiseOperator):
        v = op.variances[k]
        if isinstance(op.domain, WDMDomain):                 # 1.0 at the inactive cells: not coordinates
            g = op.domain.cells.grid
            v = np.where(wilson.active_mask(g.Nf, g.Nt), v, 0.0)
        return np.diag(to_coordinates({k: v}, op.domain)[0]), N
'''),
('''    ``exp(s)`` times an orthogonal matrix, transform is ``M`` in real coordinates, and
    ``_unpack`` gives a canonical layout that transform takes back."""
    d = TARGETS[name]
    M = _maps(d)[0]
    s2 = np.exp(2.0 * log_coordinate_scale(d))
    np.testing.assert_allclose(M @ M.T, s2 * np.eye(N), rtol=0, atol=1e-12 * s2)
    for k, x in X.items():
        r = M @ x
        np.testing.assert_allclose(_pack(d, _y(d)[k]), r, rtol=0, atol=1e-12 * np.max(np.abs(r)))
        v = _unpack(d, r)
        np.testing.assert_allclose(_pack(d, v), r, rtol=0, atol=1e-15 * np.max(np.abs(r)))
        _close(transform({k: v}, d, TIME), {k: x}, rel=1e-12)
''', '''    ``exp(s)`` times an orthogonal matrix, transform is ``M`` in real coordinates
    (``to_coordinates``), and ``from_coordinates`` gives a canonical layout that transform
    takes back."""
    d = TARGETS[name]
    M = _maps(d)[0]
    s2 = np.exp(2.0 * log_coordinate_scale(d))
    np.testing.assert_allclose(M @ M.T, s2 * np.eye(N), rtol=0, atol=1e-12 * s2)
    for k, x in X.items():
        r = M @ x
        np.testing.assert_allclose(to_coordinates({k: _y(d)[k]}, d)[0], r, rtol=0,
                                   atol=1e-12 * np.max(np.abs(r)))
        v = from_coordinates(r[None], d, (k,))
        np.testing.assert_allclose(to_coordinates(v, d)[0], r, rtol=0, atol=1e-15 * np.max(np.abs(r)))
        _close(transform(v, d, TIME), {k: x}, rel=1e-12)
'''),
('''    y = _y(d)
    got = {name: getattr(t, name)(y) for name in ("apply", "solve", "project")}
    rank, logdet, qf = 0, 0.0, 0.0
    for k in op.channel_names:
        C, built = _native_covariance(op, k)
''', '''    y = _y(d)
    names = op.channel_names
    got = {name: to_coordinates(getattr(t, name)(y), d, names) for name in ("apply", "solve", "project")}
    Y = to_coordinates(y, d, names)
    rank, logdet, qf = 0, 0.0, 0.0
    for i, k in enumerate(names):
        C, built = _native_covariance(op, k)
'''),
('''        r = _pack(d, y[k])
        want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
        for name, v in want.items():
            np.testing.assert_allclose(_pack(d, got[name][k]), v, rtol=0, atol=1e-10 * np.max(np.abs(v)),
                                       err_msg=f"{name}[{k}]")
        if built < N:
            assert not np.allclose(got["project"][k], y[k])        # the projection is not vacuous
''', '''        r = Y[i]
        want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
        for name, v in want.items():
            np.testing.assert_allclose(got[name][i], v, rtol=0, atol=1e-10 * np.max(np.abs(v)),
                                       err_msg=f"{name}[{k}]")
        if built < N:
            assert not np.allclose(got["project"][i], r)           # the projection is not vacuous
'''),
("""    def _z(self, values):
        return np.stack([_pack(self.domain, values[k]) for k in self.channel_names], -1)

    def _out(self, Z):
        return {k: _unpack(self.domain, Z[:, i]) for i, k in enumerate(self.channel_names)}
""", """    def _z(self, values):
        return to_coordinates(values, self.domain, self.channel_names).T

    def _out(self, Z):
        return from_coordinates(Z.T, self.domain, self.channel_names)
"""),
("""    y = _y(d)
    r = np.concatenate([_pack(d, y["A"]), _pack(d, y["E"])])
    want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
    for name, v in want.items():
        out = getattr(t, name)(y)
        got = np.concatenate([_pack(d, out["A"]), _pack(d, out["E"])])
""", """    y = _y(d)
    r = to_coordinates(y, d, ("A", "E")).reshape(-1)
    want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
    for name, v in want.items():
        got = to_coordinates(getattr(t, name)(y), d, ("A", "E")).reshape(-1)
"""),
]
for old, new in EDITS:
    assert t.count(old) == 1, old[:200]
    t = t.replace(old, new)
p.write_text(t)
print("rewrote", p)
```

Run: `~/miniconda3/envs/wdmrel/bin/python -I $TMPDIR/t5_coordinates_in_tests.py .`
Expected: `rewrote tests/test_translate_operator.py`. An `AssertionError` naming an old text means an earlier task changed that passage: apply that edit by hand.

- [ ] **Step 7: Grep for leftovers**

Run: `grep -n "_pack\|_unpack\|_cells(\|_H\b" tests/test_translate_operator.py`
Expected: only the two `wilson.grid.packed_cells(NF, NT)` lines in `test_a_time_operator_seen_in_wdm_is_the_dense_congruence` and `test_a_wdm_operator_seen_in_time_is_the_dense_congruence` (independent references, kept).

- [ ] **Step 8: Run both files**

Run: `$PY -m pytest -q tests/test_translate_coordinates.py tests/test_translate_operator.py`
Expected: `143 passed` (50 + 93; `test_translate_operator.py` keeps its 93 tests).

- [ ] **Step 9: Full suite and NumPy lane**

Run: `$PY -m pytest -q` then `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: both pass with no errors and 50 more passed tests than after Task 4 (the full suite keeps its 3 `DOMAINIA_ORACLE_ROOT` skips).

- [ ] **Step 10: Commit** (sandbox disabled)

```bash
git add src/domainia/translate.py tests/test_translate_coordinates.py tests/test_translate_operator.py
git commit -m "translate: public real coordinates to_coordinates / from_coordinates

Rows follow channel_names; the operator tests use them instead of _pack/_unpack.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

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
- [ ] `operator_matrix(DenseOperator(C, …))` returns `(C + Cᵀ)/2` (bit for bit on time and WDM, to 1e-15 relative on Fourier), for Fourier `N = 32` and `N = 15` and a non-Meyer WDM domain.
- [ ] `test_the_scan_covers_every_module_but_filter_and_windows` passes with `operators.py` in `SCANNED`.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_operators_dense.py tests/wilson/test_window_threading.py` → all pass; then `$PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 5.

`domainia.operators` is new. In this task it imports only `domainia.wilson` and `domainia.translate` (Task 5's `to_coordinates`, `from_coordinates` and the private `_channel_names`, `_record`); Task 8 adds `CovarianceOperator` and the package exports. Nothing here is exported from `domainia` yet.

- [ ] **Step 1: Write the failing tests `tests/test_operators_dense.py`**

The references are numpy only: `numpy.linalg.eigh` of `C` for the methods, and `wilson.matrix.active_W` for the translated matrix. `_psd` correlates the two channel blocks, so a channel-order mistake changes every result. In the symmetrised-matrix test, the `1e-14` upper-triangle perturbation is below the 1e-12 symmetry gate, so `(C + Cᵀ)/2` differs from `C` and the test sees the symmetrisation.

```python
"""domainia.operators.DenseOperator and operator_matrix: an explicit covariance matrix as an
Operator, and any Operator as its matrix (spec sections 2.3, 2.4)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.operators import DenseOperator, operator_matrix
from domainia.testing import ALT_WINDOW
from domainia.translate import Operator, to_coordinates, transform, translate_operator

NF, NT, DT = 4, 8, 0.5
N = NF * NT
TIME = TimeDomain(N, DT)
WDM = WDMDomain.full(wilson.Grid(NF, NT, DT))
WDM_ALT = WDMDomain.full(wilson.Grid(8, 4, DT), window=ALT_WINDOW)
DOMAINS = {
    "time": TIME,
    "fourier": FourierDomain.full(N, DT),
    "fourier-odd": FourierDomain.full(15, DT),
    "wdm": WDM,
    "wdm-alt": WDM_ALT,
}
NAMES = ("A", "E")
_rng = np.random.default_rng(53)


def _n(domain):
    """The record length ``N`` of a full-record domain."""
    return domain.cells.grid.N if isinstance(domain, WDMDomain) else domain.N


def _y(domain):
    """A two-channel record on ``domain`` (canonical: it comes from transform)."""
    time = TimeDomain(_n(domain), DT)
    return transform({k: _rng.standard_normal(time.N) for k in NAMES}, time, domain)


def _psd(n, rank):
    """A random symmetric PSD ``(n, n)`` matrix of rank ``rank`` whose two channel blocks are
    correlated (no block structure)."""
    B = _rng.standard_normal((n, rank)) / np.sqrt(n)
    return B @ B.T + (0.3 * np.eye(n) if rank == n else 0.0)


def _close(got, want, rel=1e-10):
    np.testing.assert_allclose(got, want, rtol=0, atol=rel * float(np.max(np.abs(want))))


@pytest.mark.parametrize("rank", ["full", "deficient"])
@pytest.mark.parametrize("name", DOMAINS)
def test_every_method_matches_the_dense_reference(name, rank):
    d = DOMAINS[name]
    n = 2 * _n(d)
    built = n if rank == "full" else n - 7
    C = _psd(n, built)
    op = DenseOperator(C, d, NAMES)
    assert isinstance(op, Operator) and op.domain == d and op.channel_names == NAMES
    w, V = np.linalg.eigh(C)
    keep = w > 1e-9 * w.max()
    assert keep.sum() == built                                      # the reference itself
    w, V = w[keep], V[:, keep]
    y = _y(d)
    r = to_coordinates(y, d, NAMES).reshape(-1)
    want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
    for method, v in want.items():
        out = getattr(op, method)(y)
        assert list(out) == list(NAMES)
        transform(out, d, d)                                        # the Operator output contract
        _close(to_coordinates(out, d, NAMES).reshape(-1), v)
    assert op.degrees_of_freedom == built
    assert op.quadratic_form(y) == pytest.approx(float(r @ want["solve"]), rel=1e-10)
    assert op.log_determinant() == pytest.approx(float(np.sum(np.log(w))), rel=1e-10, abs=1e-9)


@pytest.mark.parametrize("target", ["time", "fourier", "wdm"])
def test_a_reversed_channel_dict_meets_the_same_blocks(target):
    """The rows follow ``channel_names``, not the dict's order: on the operator and on a
    translation of it (whose ``transform`` keeps the caller's order)."""
    C = _psd(2 * N, 2 * N)
    op = DenseOperator(C, WDM, NAMES)
    for o in (op, translate_operator(op, DOMAINS[target])):
        y = _y(o.domain)
        flipped = {"E": y["E"], "A": y["A"]}
        for method in ("apply", "solve", "project"):
            a, b = getattr(o, method)(y), getattr(o, method)(flipped)
            assert list(a) == list(b) == list(NAMES)
            for k in NAMES:
                np.testing.assert_array_equal(a[k], b[k])
        assert o.quadratic_form(flipped) == o.quadratic_form(y)


@pytest.mark.parametrize("name", DOMAINS)
def test_operator_matrix_of_a_dense_operator_is_its_symmetrised_matrix(name):
    d = DOMAINS[name]
    C = _psd(2 * _n(d), 2 * _n(d))
    C = C + 1e-14 * np.triu(np.ones_like(C), 1)                     # asymmetric below 1e-12
    S = DenseOperator(C, d, NAMES).matrix
    np.testing.assert_array_equal(S, (C + C.T) / 2)
    M = operator_matrix(DenseOperator(C, d, NAMES))
    if isinstance(d, FourierDomain):                              # sqrt(2) out and back
        _close(M, S, rel=1e-15)
    else:
        np.testing.assert_array_equal(M, S)


@pytest.mark.parametrize("wdm", [WDM, WDM_ALT], ids=["meyer", "alt"])
def test_operator_matrix_translates_a_time_matrix_to_w_c_w_transpose(wdm):
    """``TranslatedOperator.apply`` maps ``w`` to time (``W^T w / dt``), applies ``C``, maps
    back (``W``) and multiplies by ``exp(2 Delta) = dt``: the WDM matrix is ``W C W^T``."""
    C = _psd(N, N)
    g = wdm.cells.grid
    W = wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=wdm.window)
    got = operator_matrix(translate_operator(DenseOperator(C, TIME, ("A",)), wdm))
    _close(got, W @ C @ W.T, rel=1e-12)


def test_rtol_sets_the_kept_eigenvalues():
    w = np.array([1.0, 0.5, 1e-4, 1e-8] + [1.0] * (N - 4))
    Q = np.linalg.qr(_rng.standard_normal((N, N)))[0]
    C = (Q * w) @ Q.T
    assert DenseOperator(C, TIME, ("A",)).degrees_of_freedom == N
    assert DenseOperator(C, TIME, ("A",), rtol=1e-6).degrees_of_freedom == N - 1
    assert DenseOperator(C, TIME, ("A",), rtol=1e-3).degrees_of_freedom == N - 2


def test_the_matrix_is_validated():
    C = _psd(N, N)
    op = DenseOperator(C, TIME, ("A",))
    assert not op.matrix.flags.writeable and not np.shares_memory(op.matrix, C)
    with pytest.raises(ValueError, match=r"^matrix has shape \(32, 32\); 2 channel\(s\) of 32 coordinates "
                                         r"need shape \(64, 64\)$"):
        DenseOperator(C, TIME, NAMES)
    with pytest.raises(ValueError, match="not finite"):
        DenseOperator(np.where(np.eye(N) > 0, np.nan, C), TIME, ("A",))
    with pytest.raises(ValueError, match="not symmetric"):
        DenseOperator(C + 1e-9 * np.triu(np.ones_like(C), 1), TIME, ("A",))
    with pytest.raises(ValueError, match="not positive semi-definite"):
        DenseOperator(C - 2.0 * np.eye(N) * np.max(np.linalg.eigvalsh(C)), TIME, ("A",))
    with pytest.raises(TypeError, match="real numeric"):
        DenseOperator(C + 0j, TIME, ("A",))
    with pytest.raises(TypeError, match="real numeric"):
        DenseOperator(C.astype(object), TIME, ("A",))
    with pytest.raises(TypeError, match="masked"):
        DenseOperator(np.ma.MaskedArray(C), TIME, ("A",))
    with pytest.raises(TypeError, match="channel_names must be a tuple"):
        DenseOperator(C, TIME, "A")
    with pytest.raises(TypeError, match="^domain must be a full-record domain"):
        DenseOperator(C, "time", ("A",))
    with pytest.raises(ValueError, match="^domain: translate handles only WDMDomain.full"):
        DenseOperator(C, WDMDomain(CellSet.bands(WDM.cells.grid, (1, 2))), ("A",))
    with pytest.raises(TypeError, match="rtol"):
        DenseOperator(C, TIME, ("A",), rtol="1e-6")
    with pytest.raises(ValueError, match="rtol"):
        DenseOperator(C, TIME, ("A",), rtol=-1.0)
    y = {"A": np.zeros(N)}
    with pytest.raises(ValueError, match="channel_names are"):
        op.apply({"E": y["A"]})
    with pytest.raises(TypeError, match="mapping"):
        op.solve([y["A"]])
    with pytest.raises(TypeError, match="Operator"):
        operator_matrix(C)
```

- [ ] **Step 2: Run the new tests and watch them fail**

Run: `$PY -m pytest -q tests/test_operators_dense.py`
Expected: `ModuleNotFoundError: No module named 'domainia.operators'`, then `Interrupted: 2 errors during collection` (`--doctest-modules` collects the file a second time).

- [ ] **Step 3: Create `src/domainia/operators.py`**

`tol` is `numpy.linalg.matrix_rank`'s default, `max |λ| · K N · eps`. The symmetry gate runs on the float64 copy before the eigendecomposition, and the eigendecomposition is of the stored `(C + Cᵀ)/2`. `operator_matrix` builds each `e_j` through `from_coordinates`, so a translated operator receives a canonical input.

```python
"""Explicit covariance matrices as :class:`domainia.translate.Operator` s, and the matrix of
any operator (spec section 2).

* :class:`DenseOperator`: an explicit covariance matrix on the stacked real coordinates of
  a full-record domain.
* :func:`operator_matrix`: the matrix of any ``Operator``, native or translated, in the
  stacked real coordinates of its domain (``K N`` applies: for small ``N``).

The stacked real coordinates of ``K`` channels are the rows of
:func:`domainia.translate.to_coordinates` in ``channel_names`` order, concatenated
(channel-major): ``r = to_coordinates(values, domain, channel_names).reshape(-1)``, of
length ``K N``.  Every operator here passes its ``channel_names`` to ``to_coordinates``, so
a dict built in another channel order still meets the right blocks of the matrix.  NumPy
only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from domainia import wilson
from domainia.translate import (
    Domain,
    Operator,
    _channel_names,
    _record,
    from_coordinates,
    to_coordinates,
)

__all__ = ["DenseOperator", "operator_matrix"]


@dataclass(frozen=True, eq=False)
class DenseOperator:
    """An explicit covariance matrix ``C`` as an :class:`~domainia.translate.Operator`.

    ``matrix`` is ``(K N, K N)`` on the stacked real coordinates of ``domain`` (module
    docstring), ``K = len(channel_names)``, so channels may be correlated.  It must be
    finite and symmetric to 1e-12 relative (``max |C - C^T| <= 1e-12 max |C|``); the
    operator then stores ``(C + C^T) / 2`` as ``matrix``, a read-only float64 copy.

    ``numpy.linalg.eigh`` decomposes it once.  An eigenvalue below ``-tol`` raises
    ``ValueError`` (``C`` is not positive semi-definite); the eigenpairs with
    ``lambda > tol`` are kept (``V``, ``Lambda``).  ``tol = max |lambda| K N eps`` (the
    default of ``numpy.linalg.matrix_rank``), or ``rtol max |lambda|`` when ``rtol`` is
    given.

    * ``apply``: ``C r``; ``solve``: ``V Lambda^-1 V^T r``; ``project``: ``V V^T r``;
    * ``quadratic_form``: ``r^T solve(r)``; ``log_determinant``: ``sum log lambda``;
    * ``degrees_of_freedom``: the number of kept eigenpairs.

    Inputs go through :func:`~domainia.translate.to_coordinates` (``transform``'s input
    rules; channels other than ``channel_names`` raise ``ValueError``), and outputs come from
    :func:`~domainia.translate.from_coordinates`, so they meet the ``Operator`` output
    contract exactly.  Raises ``TypeError`` for a wrong kind (a masked, non-numeric or
    complex ``matrix``, a non-domain, bad ``channel_names``, a non-real ``rtol``) and
    ``ValueError`` for a wrong value.
    """

    matrix: Any
    domain: Domain
    channel_names: tuple[str, ...]
    rtol: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        N, _ = _record(self.domain, "domain")
        names = _channel_names(self.channel_names)
        n = len(names) * N
        if isinstance(self.matrix, np.ma.MaskedArray):
            raise TypeError("matrix: masked arrays are refused (numpy.asarray drops the mask)")
        C = np.asarray(self.matrix)
        if C.dtype.kind not in "iuf":
            raise TypeError(f"matrix must be a real numeric array; got dtype {C.dtype}")
        if C.shape != (n, n):
            raise ValueError(f"matrix has shape {C.shape}; {len(names)} channel(s) of {N} "
                             f"coordinates need shape ({n}, {n})")
        wilson.validate_finite(C, "matrix")
        C = C.astype(np.float64)
        peak = float(np.max(np.abs(C)))
        asym = float(np.max(np.abs(C - C.T)))
        if asym > 1e-12 * peak:
            raise ValueError(f"matrix is not symmetric: max |C - C^T| is {asym / peak:.2g} of "
                             f"max |C| (more than 1e-12)")
        rtol = self.rtol
        if rtol is not None:
            if isinstance(rtol, bool) or not isinstance(rtol, (int, float, np.integer, np.floating)):
                raise TypeError(f"rtol must be a real number or None; got {type(rtol).__name__}")
            rtol = float(rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError(f"rtol must be finite and >= 0; got {rtol!r}")
        S = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(S)
        top = float(np.max(np.abs(w)))
        tol = top * n * float(np.finfo(np.float64).eps) if rtol is None else rtol * top
        if w[0] < -tol:
            raise ValueError(f"matrix is not positive semi-definite: its smallest eigenvalue "
                             f"{float(w[0])!r} is below -tol = {-tol!r}")
        keep = w > tol
        V, w = V[:, keep], w[keep]
        for a in (S, V, w):
            a.setflags(write=False)
        object.__setattr__(self, "matrix", S)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "_V", V)
        object.__setattr__(self, "_w", w)

    @property
    def degrees_of_freedom(self) -> int:
        return int(self._w.size)  # type: ignore[attr-defined]

    def _r(self, values: Any) -> np.ndarray:
        return to_coordinates(values, self.domain, self.channel_names).reshape(-1)

    def _values(self, r: np.ndarray) -> dict[str, np.ndarray]:
        return from_coordinates(r.reshape(len(self.channel_names), -1), self.domain, self.channel_names)

    def _solve(self, r: np.ndarray) -> np.ndarray:
        V, w = self._V, self._w  # type: ignore[attr-defined]
        return V @ ((V.T @ r) / w)

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self.matrix @ self._r(values))

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self._solve(self._r(values)))

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        V = self._V  # type: ignore[attr-defined]
        return self._values(V @ (V.T @ self._r(values)))

    def quadratic_form(self, values: Mapping[str, Any]) -> float:
        r = self._r(values)
        return float(r @ self._solve(r))

    def log_determinant(self) -> float:
        return float(np.sum(np.log(self._w)))  # type: ignore[attr-defined]


def operator_matrix(op: Operator) -> np.ndarray:
    """``(K N, K N)``: the matrix of ``op.apply`` on the stacked real coordinates of
    ``op.domain`` (module docstring), for any :class:`~domainia.translate.Operator`, native
    or translated.

    Column ``j`` is the stacked ``to_coordinates(op.apply(from_coordinates(e_j)))``, with
    ``op.channel_names`` as the channel order.  It costs ``K N`` applies, so it is for small
    ``N`` (tests and dense references).  Translating a matrix:
    ``operator_matrix(translate_operator(DenseOperator(C, time, ("A",)), wdm))`` is
    ``W C W^T``.
    """
    if not isinstance(op, Operator):
        raise TypeError(f"op must implement domainia.translate.Operator; got {type(op).__name__}")
    names = _channel_names(tuple(op.channel_names))
    N, _ = _record(op.domain, "op.domain")
    K = len(names)
    out = np.empty((K * N, K * N))
    e = np.zeros(K * N)
    for j in range(K * N):
        e[j] = 1.0
        y = op.apply(from_coordinates(e.reshape(K, N), op.domain, names))
        out[:, j] = to_coordinates(y, op.domain, names).reshape(-1)
        e[j] = 0.0
    return out
```

- [ ] **Step 4: Run the new tests and watch them pass**

Run: `$PY -m pytest -q tests/test_operators_dense.py`
Expected: `22 passed`.

- [ ] **Step 5: Run the window scan and watch it fail**

The scan must cover every module of the package but `wilson/filter.py` and `wilson/windows.py`, and the new module is in neither list.

Run: `$PY -m pytest -q tests/wilson/test_window_threading.py`
Expected: `1 failed, 562 passed`. The failure is `test_the_scan_covers_every_module_but_filter_and_windows`, with `Extra items in the left set: 'operators.py'`.

- [ ] **Step 6: Add `operators.py` to `SCANNED` in `tests/wilson/test_window_threading.py`**

```python
           "covariance/__init__.py", "translate.py")
```
→
```python
           "covariance/__init__.py", "translate.py", "operators.py")
```

- [ ] **Step 7: Run both files**

Run: `$PY -m pytest -q tests/test_operators_dense.py tests/wilson/test_window_threading.py`
Expected: `586 passed`: 22, plus 564 in the scan file, which gains the case `test_no_kernel_call_drops_the_window[operators.py]` (the module calls no window consumer).

- [ ] **Step 8: Full suite and NumPy lane**

Run: `$PY -m pytest -q` then `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: both pass with no errors, and each has 23 more passed tests than after Task 5 (the 22 new tests and the new scan case).

- [ ] **Step 9: Commit** (sandbox disabled)

```bash
git add src/domainia/operators.py tests/test_operators_dense.py tests/wilson/test_window_threading.py
git commit -m "operators: DenseOperator and operator_matrix

An explicit matrix on the stacked real coordinates is an Operator; any Operator
reads back as its matrix. The window scan covers the new module.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `Covariance.multiply` and `ChannelStack._matrix` (spec §2.5)

**Goal:** Every domainia covariance can multiply a vector by `Σ(nu)` in its factor layout, and `ChannelStack` gains a dense form.

**Files:**
- Modify: `src/domainia/covariance/base.py`, `src/domainia/covariance/diagonal.py`, `src/domainia/covariance/blocktri.py`, `src/domainia/covariance/evolutionary.py`, `src/domainia/covariance/stack.py` (`covariance/dense.py` is unchanged: the base `_multiply` covers it)
- Create: `tests/test_dom_cov_multiply.py`

**Acceptance Criteria:**
- [ ] `cov.multiply(nu, b)` equals `cov.dense(nu) @ b` to 1e-12 relative for `DiagonalCovariance`, `DenseCovariance`, `BlockTridiagonalCovariance` (each `layout`, padded included), `EvolutionaryCovariance` (blocktri and dense assembly, `builder="direct"` included), and a two-channel `ChannelStack` (flattened row-major).
- [ ] `BlockTridiagonalCovariance.multiply` never forms the dense matrix: the test makes `_matrix` and `blocks_to_dense` raise and `multiply` still succeeds.
- [ ] `ChannelStack(...).dense(nu)` is the block-diagonal of the children's `dense` at `split(nu)`.
- [ ] A `Covariance` subclass with neither `_multiply` nor `_matrix` raises `TypeError` from `multiply`.
- [ ] `grep -rn "def matvec\|\.matvec(" src` hits are unchanged (`PushedCovariance` keeps its duck-typed `matvec(X, be)`).
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_dom_cov_multiply.py` → all pass; then `$PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 6.

The per-class hook is `_multiply(C, nu, b, be)`, pure in the constants `C` like `_factor` and `_matrix`. The public `multiply(nu, b, be=None)` fetches `self.constants(be)` and calls it. `DenseCovariance` needs no change: the base default `_matrix(C, nu, be) @ b` is its product. `PushedCovariance.matvec(X, be)` in `maps/base.py` is a different, duck-typed protocol and is not touched.

- [ ] **Step 1: Record the `matvec` hits before the change**

Run: `grep -rn "def matvec\|\.matvec(" src`
Expected: three hits, all in `src/domainia/maps/base.py`:

```
    def matvec(self, X: Any, be: Backend | None = None) -> Any: ...
    ``M (Sigma M[idx]^T)``, i.e. ``map.apply(cov.matvec(map.rows(idx)))`` (WT
    def matvec(self, X: Any, be: Backend | None = None) -> Any:
```

(the `SupportsMatvec` protocol, the `PushedCovariance` class docstring, and `PushedCovariance.matvec`). Step 8 checks that they are unchanged.

- [ ] **Step 2: Write the failing tests `tests/test_dom_cov_multiply.py`**

Every covariance class is compared with its own `dense(nu) @ b`. `SIZES` has an empty band, which the ragged product must skip. The never-dense test patches the module's `blocks_to_dense` and the class's `_matrix` to raise, checks that `dense` now raises (so the patch is live), and then runs `multiply`, both on a `BlockTridiagonalCovariance` and on an `EvolutionaryCovariance` whose `impl` is one.

```python
"""Covariance.multiply: ``Sigma(nu) b`` in the layout of the factor's ``solve``, for every
covariance class, and ChannelStack's dense form (spec section 2.5)."""

from __future__ import annotations

import numpy as np
import pytest

import domainia.covariance.blocktri as blocktri
from domainia import wilson
from domainia.covariance import (
    BlockTridiagonalCovariance,
    ChannelStack,
    Covariance,
    DenseCovariance,
    DiagFactor,
    DiagonalCovariance,
    EvolutionaryCovariance,
)
from domainia.domains import CellSet
from domainia.spectra import GenericPSD

NF, NT, DT = 8, 16, 1 / 16
GRID = wilson.Grid(NF, NT, DT)
CELLS = CellSet.all_active(GRID)
CUT = CELLS.drop_columns(3, 2)
NU = np.array([-1.3, 0.4])
_rng = np.random.default_rng(59)


def _drifting(f, t, nu):
    alpha = nu[0] + nu[1] * (np.asarray(t, float) / (GRID.N * DT) - 0.5)
    return np.exp(alpha * np.log(np.maximum(np.asarray(f, float), 0.5)))


PSD = GenericPSD(_drifting)
#: Band sizes with an empty band: the ragged product must skip it.
SIZES = (3, 4, 0, 2, 5)


def _spd(n):
    B = _rng.standard_normal((n, n))
    return B @ B.T / n + np.eye(n)


_DIAG = [_spd(s) for s in SIZES]
_UP = [_rng.standard_normal((SIZES[m], SIZES[m + 1])) / 4 for m in range(len(SIZES) - 1)]
_V = 0.5 + _rng.random(CELLS.size)
_S = _spd(20)


def _blocks(nu, be):
    """Ragged band blocks that depend on ``nu``."""
    s = be.xp.exp(nu[0])
    return [s * be.asarray(D) for D in _DIAG], [s * be.asarray(U) for U in _UP]


COVS = {
    "diagonal-array": lambda: DiagonalCovariance(_V, CELLS.size),
    "diagonal-callable": lambda: DiagonalCovariance(lambda nu, be: be.asarray(_V) * be.xp.exp(nu[0]),
                                                    CELLS.size),
    "dense": lambda: DenseCovariance(lambda nu, be: be.asarray(_S) * be.xp.exp(nu[0]), 20, chol_impl="numpy"),
    "blocktri-ragged": lambda: BlockTridiagonalCovariance(_blocks, SIZES),
    "blocktri-padded": lambda: BlockTridiagonalCovariance(_blocks, SIZES, layout="padded"),
    "evolutionary-blocktri": lambda: EvolutionaryCovariance(PSD, CELLS),
    "evolutionary-padded": lambda: EvolutionaryCovariance(PSD, CUT, layout="padded"),
    "evolutionary-dense": lambda: EvolutionaryCovariance(PSD, CUT, assembly="dense"),
    "evolutionary-direct": lambda: EvolutionaryCovariance(PSD, CUT, builder="direct", assembly="dense"),
}


@pytest.mark.parametrize("name", COVS)
def test_multiply_is_the_dense_matrix_times_b(name):
    cov = COVS[name]()
    b = _rng.standard_normal(cov.n)
    want = cov.dense(NU) @ b
    got = cov.multiply(NU, b)
    assert got.shape == (cov.n,)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


def test_block_tridiagonal_multiply_never_forms_the_dense_matrix(monkeypatch):
    """With the dense routes made to raise, the ragged product still runs."""
    covs = [BlockTridiagonalCovariance(_blocks, SIZES), EvolutionaryCovariance(PSD, CELLS)]
    bs = [_rng.standard_normal(c.n) for c in covs]
    want = [c.dense(NU) @ b for c, b in zip(covs, bs, strict=True)]

    def boom(*args, **kwargs):
        raise AssertionError("the dense matrix was formed")

    monkeypatch.setattr(blocktri, "blocks_to_dense", boom)
    monkeypatch.setattr(BlockTridiagonalCovariance, "_matrix", boom)
    for c, b, w in zip(covs, bs, want, strict=True):
        with pytest.raises(AssertionError, match="dense matrix was formed"):
            c.dense(NU)                                              # the patch is live
        np.testing.assert_allclose(c.multiply(NU, b), w, rtol=0, atol=1e-12 * np.max(np.abs(w)))


def test_a_channel_stack_multiplies_each_channel_at_its_slice_of_nu():
    """``b`` is ``(channels, n_c)``; ``dense`` is the block-diagonal of the channels' dense
    matrices at ``split(nu)``, so ``multiply(nu, b).reshape(-1) == dense(nu) @ b.reshape(-1)``."""
    c0 = COVS["diagonal-callable"]()
    c1 = EvolutionaryCovariance(PSD, CELLS)
    st = ChannelStack([c0, c1], 2)
    nu = np.array([0.3, 0.0, -1.1, 0.6])
    D = st.dense(nu)
    n = CELLS.size
    assert D.shape == (2 * n, 2 * n)
    np.testing.assert_array_equal(D[:n, :n], c0.dense(nu[:2]))
    np.testing.assert_array_equal(D[n:, n:], c1.dense(nu[2:]))
    assert not D[:n, n:].any() and not D[n:, :n].any()
    b = _rng.standard_normal((2, n))
    got = st.multiply(nu, b)
    assert got.shape == (2, n)
    want = D @ b.reshape(-1)
    assert np.max(np.abs(got.reshape(-1) - want)) <= 1e-12 * np.max(np.abs(want))
    np.testing.assert_allclose(st.factor(nu).solve(got), b, rtol=0, atol=1e-10 * np.max(np.abs(b)))


class _FactorOnly(Covariance):
    """A covariance with a factor and neither ``_multiply`` nor ``_matrix``."""

    n = 3

    def _factor(self, C, nu, be):
        return DiagFactor.build(np.ones(3), be)


def test_a_covariance_with_neither_multiply_nor_a_dense_form_raises_type_error():
    with pytest.raises(TypeError, match=r"^_FactorOnly has no multiply: it implements neither "
                                        r"_multiply nor _matrix"):
        _FactorOnly().multiply(np.zeros(1), np.ones(3))
```

- [ ] **Step 3: Run the new tests and watch them fail**

Run: `$PY -m pytest -q tests/test_dom_cov_multiply.py`
Expected: `12 failed`. Eleven fail with `AttributeError: '…Covariance' object has no attribute 'multiply'`. The `ChannelStack` test fails earlier, at `st.dense(nu)`, with `NotImplementedError: ChannelStack has no dense assembly`.

- [ ] **Step 4: Add `_multiply` and `multiply` to `Covariance` in `src/domainia/covariance/base.py`**

(a) The class docstring:

```python
    own caches.  Covariances with a dense form implement ``_matrix(C, nu, be)``;
    :meth:`dense` is its host value.  ``jittable = False`` marks a strategy that runs
    eagerly on both backends (the ragged layout, ``on_fail="raise"``).
    """
```
→
```python
    own caches.  Covariances with a dense form implement ``_matrix(C, nu, be)``;
    :meth:`dense` is its host value.  :meth:`multiply` is ``_multiply(C, nu, b, be)``,
    which defaults to ``_matrix(C, nu, be) @ b``; a class with a cheaper product overrides
    it.  ``jittable = False`` marks a strategy that runs eagerly on both backends (the
    ragged layout, ``on_fail="raise"``).
    """
```

(b) After `_matrix`:

```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no dense assembly")

    def constants(self, be: Backend | None = None) -> Any:
```
→
```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no dense assembly")

    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        """``Sigma(nu) b`` from the constants ``C``: ``_matrix(C, nu, be) @ b`` unless a
        subclass overrides it.  A class with neither raises ``TypeError``."""
        if type(self)._matrix is Covariance._matrix:
            raise TypeError(f"{type(self).__name__} has no multiply: it implements neither "
                            "_multiply nor _matrix (a dense assembly)")
        return self._matrix(C, nu, be) @ b

    def multiply(self, nu: Any, b: Any, be: Backend | None = None) -> Any:
        """``Sigma(nu) b``, eagerly.  ``b`` has the layout the factor's ``solve`` takes:
        ``(n,)``, or ``(channels, n_c)`` for :class:`~domainia.covariance.stack.ChannelStack`.

        NumPy is the tested backend; a JAX ``be`` is passed through.  The name is not
        ``matvec``: ``PushedCovariance`` duck-types ``matvec(X, be)``, which has a
        fixed-noise-state signature.
        """
        be = be or get_backend(nu, b)
        return self._multiply(self.constants(be), be.asarray(nu), be.asarray(b), be)

    def constants(self, be: Backend | None = None) -> Any:
```

- [ ] **Step 5: Override `_multiply` where the product is cheaper or delegated**

(a) `DiagonalCovariance` in `src/domainia/covariance/diagonal.py`, after `_matrix`:

```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return be.xp.diag(self.variances(C, nu, be))
```
→
```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return be.xp.diag(self.variances(C, nu, be))

    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        v = self.variances(C, nu, be)
        return v * b if b.ndim == 1 else v[:, None] * b
```

(b) `BlockTridiagonalCovariance` in `src/domainia/covariance/blocktri.py`, after `_matrix` (the last method of the file). It reads `band_blocks` in the ragged layout whatever `self.layout` is, so the padded layout's dummy blocks never enter:

```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = to_layout(*self.band_blocks(C, nu, be), self.sizes, "ragged", be)
        return blocks_to_dense(diag, up, be)
```
→
```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        diag, up = to_layout(*self.band_blocks(C, nu, be), self.sizes, "ragged", be)
        return blocks_to_dense(diag, up, be)

    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        """Band by band from the ragged blocks, ``D_m x_m + U_m x_{m+1} + U_{m-1}^T x_{m-1}``
        (``x_m`` the band-``m`` slice of ``b``): never the dense matrix."""
        diag, up = to_layout(*self.band_blocks(C, nu, be), self.sizes, "ragged", be)
        off = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        x = [b[off[m]:off[m + 1]] for m in range(len(self.sizes))]
        out = []
        for m, D in enumerate(diag):
            y = D @ x[m]
            if m + 1 < len(diag):
                y = y + up[m] @ x[m + 1]
            if m > 0:
                y = y + up[m - 1].T @ x[m - 1]
            out.append(y)
        return be.xp.concatenate(out)
```

(c) `EvolutionaryCovariance` in `src/domainia/covariance/evolutionary.py` delegates to its `impl`, as `_factor` and `_matrix` do (its constants are `impl.host_constants()`):

```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._matrix(C, nu, be)

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
```
→
```python
    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.impl._matrix(C, nu, be)

    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        return self.impl._multiply(C, nu, b, be)

    def band_blocks(self, C: Any, nu: Any, be: Backend) -> tuple[Any, Any]:
```

(d) `ChannelStack` in `src/domainia/covariance/stack.py` gains both, between `prepare` and `_factor`. Each child reads its own constants `C[str(c)]` at its own slice of `nu`, as in `_factor`:

```python
        return None if all(p is None for p in pre) else pre

    def _factor(self, C: Any, nu: Any, be: Backend, *, key: Any = None, pre: Any = None) -> StackFactor:
```
→
```python
        return None if all(p is None for p in pre) else pre

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        """The block-diagonal of the channels' dense matrices, each at its own slice of
        ``nu``, in channel order: ``dense(nu) @ b.reshape(-1)`` is
        ``multiply(nu, b).reshape(-1)`` (``b`` row-major, channel by channel)."""
        xp = be.xp
        blocks = [cov._matrix(C[str(c)], nu_c, be)
                  for c, (cov, nu_c) in enumerate(zip(self.covs, self.split(nu), strict=True))]
        sizes = [cov.n for cov in self.covs]
        return xp.block([[blocks[i] if i == j else xp.zeros((sizes[i], sizes[j]))
                          for j in range(len(blocks))] for i in range(len(blocks))])

    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        """Each channel's ``Sigma_c(nu_c) b[c]``, stacked: ``b`` is ``(channels, n_c)``, so the
        channels must share ``n_c`` (as :meth:`StackFactor.solve`)."""
        return be.xp.stack([cov._multiply(C[str(c)], nu_c, b[c], be)
                            for c, (cov, nu_c) in enumerate(zip(self.covs, self.split(nu), strict=True))])

    def _factor(self, C: Any, nu: Any, be: Backend, *, key: Any = None, pre: Any = None) -> StackFactor:
```

- [ ] **Step 6: Run the new tests and watch them pass**

Run: `$PY -m pytest -q tests/test_dom_cov_multiply.py`
Expected: `12 passed`.

- [ ] **Step 7: Check that no test pins `ChannelStack`'s missing dense form**

Run: `grep -rn "has no dense assembly" src tests`
Expected: one hit, the `raise NotImplementedError` in `Covariance._matrix` (`src/domainia/covariance/base.py`). No test asserted that `ChannelStack.dense` raises, so nothing else changes.

- [ ] **Step 8: Check the `matvec` hits are unchanged**

Run: `grep -rn "def matvec\|\.matvec(" src`
Expected: the same three hits as Step 1, all in `src/domainia/maps/base.py`.

- [ ] **Step 9: Full suite and NumPy lane**

Run: `$PY -m pytest -q` then `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: both pass with no errors, and each has 12 more passed tests than after Task 6.

- [ ] **Step 10: Commit** (sandbox disabled)

```bash
git add src/domainia/covariance/base.py src/domainia/covariance/diagonal.py src/domainia/covariance/blocktri.py src/domainia/covariance/evolutionary.py src/domainia/covariance/stack.py tests/test_dom_cov_multiply.py
git commit -m "Covariance.multiply and ChannelStack's dense form

multiply is Sigma(nu) b in the factor's solve layout; the block-tridiagonal
product runs band by band and never forms the dense matrix.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `CovarianceOperator` and the package surface (spec §2.6, §2.8)

**Goal:** A domainia covariance fixed at `nu` on a (possibly partial) WDM support becomes an `Operator`. The new names are exported, and `import domainia` still does not import `domainia.covariance`.

**Files:**
- Modify: `src/domainia/operators.py` (`CovarianceOperator`)
- Modify: `src/domainia/maps/base.py` (`PushedCovariance._matvec` error message, its docstring, and the module docstring that names the same route)
- Modify: `src/domainia/translate.py` (`__all__`), `src/domainia/__init__.py` (`__all__`, imports, docstring)
- Create: `tests/test_operators_covariance.py`
- Modify: `tests/test_package.py` (`PUBLIC`, the lazy-import check), `tests/test_dom_maps_base.py` (the regex that pins the old `_matvec` message)

**Acceptance Criteria:**
- [ ] Validation per spec §2.6: wrong kinds raise `TypeError`; a size, channel-count, window, cells or `n_u` mismatch raises `ValueError`; `factor(nu).ok` false raises `ValueError("… not positive definite at nu")` at construction.
- [ ] `operator_matrix(op) = Pᵀ Σ P` (with `cov.dense(nu)`) to 1e-12 relative for a diagonal, a block-tridiagonal, a small-grid `EvolutionaryCovariance` (blocktri and dense assembly), a `DenseCovariance`, a two-channel `ChannelStack`, a band-subset support and a `drop_columns` support.
- [ ] Translated to time, `operator_matrix` equals `Wᵀ Pᵀ Σ P W / dt²`. Translated to Fourier, it equals the same reference mapped by the Fourier coordinates. Both to 1e-10 relative.
- [ ] The translated `quadratic_form` and `log_determinant` match the dense references, including the pseudo-determinant for a partial support: `log det Σ + 2 K s Δ` with `Δ = log_coordinate_scale(target) − log_coordinate_scale(wdm)`, checked against `Σ log λ` of the reference's non-zero eigenvalues.
- [ ] A reversed-channel dict gives the same results as the ordered dict.
- [ ] `PushedCovariance`'s WDM → time refusal message names `CovarianceOperator`.
- [ ] `domainia.__all__` and `domainia.translate.__all__` contain the new names and `tests/test_package.py` `PUBLIC` matches. A fresh-subprocess `import domainia` leaves `domainia.covariance` out of `sys.modules`.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_operators_covariance.py tests/test_package.py` → all pass; then `$PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 7.

`CovarianceOperator` relies on Task 7's `Covariance.multiply` (for `apply`) and on Task 5's coordinates. `domainia.operators` keeps `domainia.covariance` out of its module-level imports. The class imports it inside `__post_init__`, and `Covariance` is imported only under `TYPE_CHECKING` for the annotation (the module has `from __future__ import annotations`). That keeps the package docstring's promise once `domainia/__init__.py` imports `operators`.

- [ ] **Step 1: Write the failing tests `tests/test_operators_covariance.py`**

The references come from `cov.dense(nu)` and numpy alone:
- WDM: `C = Pᵀ Σ P`, where `P = kron(I_K, eye(N)[support.cells.packed_positions()])`;
- time: `Wᵀ C W / dt²`, with `W` from `wilson.matrix.active_W` for the support's window;
- Fourier: that matrix mapped by the Fourier coordinates built from `numpy.fft.rfft`.

`CASES` covers a diagonal, a block-tridiagonal, an `EvolutionaryCovariance` in both assemblies, a two-channel `DenseCovariance` (channels correlated), a two-channel `ChannelStack`, a band-subset support, a `drop_columns` support, and a `ChannelStack` on the `drop_columns` support. The not-positive-definite case uses `DenseCovariance(-I)`: its factor reports `ok` false without a NumPy warning.

```python
"""domainia.operators.CovarianceOperator: a domainia covariance fixed at ``nu`` on a WDM
support, as an Operator that translate_operator moves to time and Fourier (spec section 2.6)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import (
    BlockTridiagonalCovariance,
    ChannelStack,
    DenseCovariance,
    DiagonalCovariance,
    EvolutionaryCovariance,
)
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.operators import CovarianceOperator, operator_matrix
from domainia.spectra import GenericPSD
from domainia.testing import ALT_WINDOW
from domainia.translate import Operator, to_coordinates, transform, translate_operator

NF, NT, DT = 4, 8, 0.5
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
ALL = CellSet.all_active(GRID)
TIME = TimeDomain(N, DT)
FOURIER = FourierDomain.full(N, DT)
WDM = WDMDomain.full(GRID)
NU = np.array([-1.3, 0.4])
_rng = np.random.default_rng(61)


def _drifting(f, t, nu):
    alpha = nu[0] + nu[1] * (np.asarray(t, float) / (N * DT) - 0.5)
    return np.exp(alpha * np.log(np.maximum(np.asarray(f, float), 0.5)))


PSD = GenericPSD(_drifting)


def _spd(n):
    B = _rng.standard_normal((n, n))
    return B @ B.T / n + 0.5 * np.eye(n)


def _blocktri(cells):
    """Random SPD band blocks on ``cells`` (``nu`` scales them)."""
    sizes = tuple(int(np.sum(cells.m_array == m)) for m in range(NF + 1))
    S = _spd(cells.size)
    off = np.concatenate([[0], np.cumsum(sizes)])
    diag = [S[off[m]:off[m + 1], off[m]:off[m + 1]] for m in range(len(sizes))]
    up = [S[off[m]:off[m + 1], off[m + 1]:off[m + 2]] for m in range(len(sizes) - 1)]

    def blocks(nu, be):
        s = be.xp.exp(nu[0])
        return [s * be.asarray(d) for d in diag], [s * be.asarray(u) for u in up]

    return BlockTridiagonalCovariance(blocks, sizes)


BANDS = CellSet.bands(GRID, (1, 2))
CUT = ALL.drop_columns(2, 1)
_V = 0.5 + _rng.random(2 * ALL.size)


def _diagonal(v):
    """A DiagonalCovariance of the variances ``v`` scaled by ``exp(nu[0])``."""
    return DiagonalCovariance(lambda nu, be: be.asarray(v) * be.xp.exp(nu[0]), v.size)


#: (covariance, support, channel_names): every kind of covariance, both kinds of partial support.
CASES = {
    "diagonal": lambda: (_diagonal(_V[:ALL.size]), WDM, ("A",)),
    "blocktri": lambda: (_blocktri(ALL), WDM, ("A",)),
    "evolutionary-blocktri": lambda: (EvolutionaryCovariance(PSD, ALL), WDM, ("A",)),
    "evolutionary-dense": lambda: (EvolutionaryCovariance(PSD, ALL, assembly="dense"), WDM, ("A",)),
    "dense-two-channels": lambda: (DenseCovariance(_spd(2 * ALL.size), 2 * ALL.size, chol_impl="numpy"),
                                   WDM, ("A", "E")),
    "stack": lambda: (ChannelStack([EvolutionaryCovariance(PSD, ALL), _diagonal(_V[ALL.size:])], 2), WDM,
                      ("A", "E")),
    "band-subset": lambda: (EvolutionaryCovariance(PSD, BANDS), WDMDomain(BANDS), ("A",)),
    "drop-columns": lambda: (EvolutionaryCovariance(PSD, CUT), WDMDomain(CUT), ("A",)),
    "drop-columns-stack": lambda: (ChannelStack([_blocktri(CUT), EvolutionaryCovariance(PSD, CUT)], 2),
                                   WDMDomain(CUT), ("A", "E")),
}


def _nu(cov):
    return np.concatenate([NU, NU[::-1]]) if isinstance(cov, ChannelStack) else NU


def _P(support, K):
    """``(K s, K N)``: row ``c s + i`` selects packed cell ``support.cells.packed_positions()[i]``
    of channel ``c``, so ``C = P^T Sigma P`` on the stacked coordinates (channel-major)."""
    return np.kron(np.eye(K), np.eye(N)[support.cells.packed_positions()])


def _fourier_coordinates():
    """``(N, N)``: time samples -> Fourier real coordinates, from ``numpy.fft.rfft`` alone."""
    F = DT * np.fft.rfft(np.eye(N), axis=0)
    return np.vstack([F[:1].real, np.sqrt(2) * F[1:-1].real, np.sqrt(2) * F[1:-1].imag, F[-1:].real])


def _references(cov, support, K):
    """The stacked matrices on WDM, time and Fourier from ``cov.dense(nu)`` alone."""
    C_wdm = _P(support, K).T @ cov.dense(_nu(cov)) @ _P(support, K)
    W = np.kron(np.eye(K), wilson.matrix.active_W(NF, NT, DT, window=support.window))
    C_time = W.T @ C_wdm @ W / DT**2
    M = np.kron(np.eye(K), _fourier_coordinates())
    return {"wdm": C_wdm, "time": C_time, "fourier": M @ C_time @ M.T}


def _close(got, want, rel):
    assert np.max(np.abs(got - want)) <= rel * np.max(np.abs(want))


@pytest.mark.parametrize("case", CASES)
def test_operator_matrix_is_the_selected_covariance_on_every_domain(case):
    """Native: ``P^T Sigma P``.  Seen from time: ``W^T P^T Sigma P W / dt^2``; from Fourier: the
    same mapped by the Fourier coordinates."""
    cov, support, names = CASES[case]()
    op = CovarianceOperator(cov, _nu(cov), support, names)
    assert isinstance(op, Operator) and op.domain == WDMDomain.full(GRID) and op.channel_names == names
    ref = _references(cov, support, len(names))
    _close(operator_matrix(op), ref["wdm"], 1e-12)
    for target, d in (("time", TIME), ("fourier", FOURIER)):
        _close(operator_matrix(translate_operator(op, d)), ref[target], 1e-10)


@pytest.mark.parametrize("case", CASES)
def test_every_method_matches_the_dense_moore_penrose_reference(case):
    """On WDM, time and Fourier: ``solve`` is the pseudo-inverse, ``project`` the projector onto
    the range, ``quadratic_form`` ``r^T C^+ r``, ``log_determinant`` the log pseudo-determinant
    and ``degrees_of_freedom`` the rank ``K s``.  The translated pseudo-determinant is
    ``log det Sigma + 2 K s Delta`` because ``P W W^T P^T = dt I_s``: the change of basis
    scales every non-zero eigenvalue by the same ``exp(2 Delta)``."""
    cov, support, names = CASES[case]()
    K = len(names)
    op = CovarianceOperator(cov, _nu(cov), support, names)
    assert op.degrees_of_freedom == K * support.size
    assert op.log_determinant() == pytest.approx(np.linalg.slogdet(cov.dense(_nu(cov)))[1], rel=1e-10)
    for target, C in _references(cov, support, K).items():
        d = {"wdm": WDM, "time": TIME, "fourier": FOURIER}[target]
        t = translate_operator(op, d)
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        keep = w > 1e-9 * w.max()
        assert keep.sum() == K * support.size
        w, V = w[keep], V[:, keep]
        y = transform({k: _rng.standard_normal(N) for k in names}, TIME, d)
        r = to_coordinates(y, d, names).reshape(-1)
        want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
        for method, v in want.items():
            out = getattr(t, method)(y)
            assert list(out) == list(names)
            _close(to_coordinates(out, d, names).reshape(-1), v, 1e-10)
        assert t.degrees_of_freedom == keep.sum()
        assert t.quadratic_form(y) == pytest.approx(float(r @ want["solve"]), rel=1e-10)
        assert t.log_determinant() == pytest.approx(float(np.sum(np.log(w))), rel=1e-10, abs=1e-9)


@pytest.mark.parametrize("case", ["dense-two-channels", "stack"])
@pytest.mark.parametrize("target", ["wdm", "time"])
def test_a_reversed_channel_dict_meets_the_same_blocks(case, target):
    cov, support, names = CASES[case]()
    op = CovarianceOperator(cov, _nu(cov), support, names)
    o = op if target == "wdm" else translate_operator(op, TIME)
    y = transform({k: _rng.standard_normal(N) for k in names}, TIME, o.domain)
    flipped = {"E": y["E"], "A": y["A"]}
    for method in ("apply", "solve", "project"):
        a, b = getattr(o, method)(y), getattr(o, method)(flipped)
        assert list(a) == list(b) == ["A", "E"]
        for k in names:
            np.testing.assert_array_equal(a[k], b[k])
    assert o.quadratic_form(flipped) == o.quadratic_form(y)


def test_projection_is_zero_off_the_support_and_outputs_are_canonical():
    cov, support, names = CASES["band-subset"]()
    op = CovarianceOperator(cov, NU, support, names)
    y = transform({"A": _rng.standard_normal(N)}, TIME, WDM)
    off = ~np.isin(np.arange(N), support.cells.packed_positions())
    for method in ("apply", "solve", "project"):
        out = getattr(op, method)(y)
        assert np.all(to_coordinates(out, WDM)[0, off] == 0.0)
        assert np.all(out["A"][~wilson.active_mask(NF, NT)] == 0.0)
    kept = to_coordinates(op.project(y), WDM)[0, ~off]
    np.testing.assert_array_equal(kept, to_coordinates(y, WDM)[0, ~off])


def test_the_push_refusal_names_the_route_and_the_route_works():
    """``PushedCovariance`` cannot push a cell covariance to time; the route its error names,
    a CovarianceOperator seen from time, gives ``W^T Lambda W / dt^2``."""
    from domainia.maps import WDMToTime

    cov = DiagonalCovariance(2.0 * np.ones(N), N)
    with pytest.raises(TypeError, match=r"domainia\.operators\.CovarianceOperator.*translate_operator"):
        WDMToTime(WDM, TIME).push(cov).rows([0])
    got = operator_matrix(translate_operator(CovarianceOperator(cov, np.zeros(0), WDM, ("A",)), TIME))
    _close(got, (2.0 / DT) * np.eye(N), 1e-12)                  # W^T W = dt I


class _WithNuisance(DiagonalCovariance):
    n_u = 1


def test_the_arguments_are_validated():
    ev = EvolutionaryCovariance(PSD, ALL)
    with pytest.raises(TypeError, match="cov must be a domainia.covariance.Covariance"):
        CovarianceOperator(np.eye(N), NU, WDM, ("A",))
    with pytest.raises(TypeError, match="support must be a WDMDomain"):
        CovarianceOperator(ev, NU, ALL, ("A",))
    with pytest.raises(TypeError, match="channel_names must be a tuple"):
        CovarianceOperator(ev, NU, WDM, "A")
    with pytest.raises(ValueError, match="unique"):
        CovarianceOperator(ChannelStack([ev, ev], 2), np.r_[NU, NU], WDM, ("A", "A"))
    with pytest.raises(TypeError, match="nu must be a real numeric array"):
        CovarianceOperator(ev, "nu", WDM, ("A",))
    with pytest.raises(ValueError, match=r"^cov\.n = 32, but 2 channel\(s\) on a support of 32 cells need "
                                         r"cov\.n = 64$"):
        CovarianceOperator(ev, NU, WDM, ("A", "E"))
    with pytest.raises(ValueError, match="the ChannelStack has 2 channels; channel_names has 1"):
        CovarianceOperator(ChannelStack([ev, ev], 2), np.r_[NU, NU], WDM, ("A",))
    with pytest.raises(ValueError, match=r"every ChannelStack channel needs n = support.size = 32"):
        CovarianceOperator(ChannelStack([ev, EvolutionaryCovariance(PSD, CUT)], 2), np.r_[NU, NU], WDM,
                           ("A", "E"))
    with pytest.raises(ValueError, match="window wide-smoothstep but the support uses meyer"):
        CovarianceOperator(EvolutionaryCovariance(PSD, ALL, window=ALT_WINDOW), NU, WDM, ("A",))
    left, right = ALL.drop_columns(2, 0), ALL.drop_columns(0, 2)
    assert left.size == right.size and left != right
    with pytest.raises(ValueError, match="are not support.cells"):
        CovarianceOperator(EvolutionaryCovariance(PSD, left), NU, WDMDomain(right), ("A",))
    with pytest.raises(ValueError, match=r"cov\.n_u = 1"):
        CovarianceOperator(_WithNuisance(np.ones(N), N), NU, WDM, ("A",))
    with pytest.raises(ValueError, match="^DenseCovariance is not positive definite at nu$"):
        CovarianceOperator(DenseCovariance(-np.eye(N), N, chol_impl="numpy"), NU, WDM, ("A",))
    op = CovarianceOperator(ev, NU, WDM, ("A",))
    assert not op.nu.flags.writeable
    with pytest.raises(ValueError, match="channel_names are"):
        op.apply({"E": np.zeros((NF + 1, NT))})
```

- [ ] **Step 2: Pin the package surface and the lazy import in `tests/test_package.py`**

(a) `PUBLIC`:

```python
PUBLIC = {
    "CellSet", "ChannelAxis", "FourierBins", "FourierDomain", "Grid", "MEYER", "Meyer", "Operator",
    "RampWindow", "SampledWindow", "TimeDomain", "TimeSeries", "TranslatedOperator", "WDMCells",
    "WDMDomain", "Window", "WindowError", "__version__", "data", "domains", "linalg",
    "log_coordinate_scale", "transform", "translate", "translate_operator", "wilson",
}
```
→
```python
PUBLIC = {
    "CellSet", "ChannelAxis", "CovarianceOperator", "DenseOperator", "FourierBins", "FourierDomain",
    "Grid", "MEYER", "Meyer", "Operator", "RampWindow", "SampledWindow", "TimeDomain", "TimeSeries",
    "TranslatedOperator", "WDMCells", "WDMDomain", "Window", "WindowError", "__version__", "data",
    "domains", "from_coordinates", "linalg", "log_coordinate_scale", "operator_matrix", "operators",
    "to_coordinates", "transform", "translate", "translate_operator", "wilson",
}
```

(b) The end of `test_the_public_names`:

```python
    assert wilson.pack is wilson.grid.pack and wilson.unpack is wilson.grid.unpack
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid
```
→
```python
    assert wilson.pack is wilson.grid.pack and wilson.unpack is wilson.grid.unpack
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid
    assert {"from_coordinates", "to_coordinates"} <= set(translate.__all__)
    assert domainia.operators.__all__ == ["CovarianceOperator", "DenseOperator", "operator_matrix"]
    assert domainia.to_coordinates is translate.to_coordinates
    assert domainia.CovarianceOperator is domainia.operators.CovarianceOperator
```

(c) Append to the end of the file, after `test_importing_every_module_resolves_to_src_and_does_not_import_jax`. The fresh interpreter is set up exactly as in that test:

```python
    assert info["jax"] == []
```
→
```python
    assert info["jax"] == []


def test_importing_domainia_leaves_the_covariance_layer_unimported():
    """The package docstring's promise, with domainia.operators imported: a fresh
    ``import domainia`` imports none of domainia.covariance, domainia.maps,
    domainia.variance, domainia.spectra and domainia.testing (CovarianceOperator imports
    domainia.covariance when it is built)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC)
    code = ("import json, sys, domainia\n"
            "print(json.dumps(sorted(k for k in sys.modules if k.startswith('domainia.'))))")
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(REPO),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert {"domainia.operators", "domainia.translate"} <= loaded
    assert not {m.split(".")[1] for m in loaded} & {"covariance", "maps", "spectra", "testing", "variance"}
```

- [ ] **Step 3: Pin the new route in the push refusal, in `tests/test_dom_maps_base.py`**

In `test_push_without_matvec_is_a_type_error_naming_translate_operator`:

```python
    from ``rows`` and ``matvec`` of a real or analysis push, naming the operator route."""
```
→
```python
    from ``rows`` and ``matvec`` of a real or analysis push, naming the operator route
    (``CovarianceOperator``, then ``translate_operator``)."""
```

and

```python
    msg = (r"DiagonalCovariance has no matvec: .*domainia\.translate\.Operator.*"
           r"domainia\.translate\.translate_operator")
```
→
```python
    msg = (r"DiagonalCovariance has no matvec: .*domainia\.operators\.CovarianceOperator.*"
           r"domainia\.translate\.translate_operator")
```

- [ ] **Step 4: Run the new and changed tests and watch them fail**

Run: `$PY -m pytest -q tests/test_operators_covariance.py`
Expected: `ImportError: cannot import name 'CovarianceOperator' from 'domainia.operators'`, then `Interrupted: 2 errors during collection`.

Run: `$PY -m pytest -q tests/test_package.py tests/test_dom_maps_base.py`
Expected: `3 failed`, the rest pass. The failures:
- `test_the_public_names`: `Extra items in the right set:` lists the six new names;
- `test_importing_domainia_leaves_the_covariance_layer_unimported`: `Extra items in the left set: 'domainia.operators'`;
- `test_push_without_matvec_is_a_type_error_naming_translate_operator`: `Regex pattern did not match`.

- [ ] **Step 5: Add `CovarianceOperator` to `src/domainia/operators.py`**

(a) The module docstring's first lines:

```python
"""Explicit covariance matrices as :class:`domainia.translate.Operator` s, and the matrix of
any operator (spec section 2).

* :class:`DenseOperator`: an explicit covariance matrix on the stacked real coordinates of
  a full-record domain.
* :func:`operator_matrix`: the matrix of any ``Operator``, native or translated, in the
```
→
```python
"""Covariances as :class:`domainia.translate.Operator` s, and the matrix of any operator
(spec section 2).

* :class:`DenseOperator`: an explicit covariance matrix on the stacked real coordinates of
  a full-record domain.
* :class:`CovarianceOperator`: a :mod:`domainia.covariance` covariance fixed at a noise
  state ``nu`` on a (possibly partial) WDM support, so that
  :func:`~domainia.translate.translate_operator` can see it from time or Fourier.
* :func:`operator_matrix`: the matrix of any ``Operator``, native or translated, in the
```

(b) The docstring's last lines:

```python
a dict built in another channel order still meets the right blocks of the matrix.  NumPy
only.
"""
```
→
```python
a dict built in another channel order still meets the right blocks of the matrix.  NumPy
only.  Importing this module does not import :mod:`domainia.covariance`:
:class:`CovarianceOperator` imports it when it is constructed.
"""
```

(c) The imports and `__all__`:

```python
from typing import Any

import numpy as np

from domainia import wilson
from domainia.translate import (
    Domain,
    Operator,
    _channel_names,
    _record,
    from_coordinates,
    to_coordinates,
)

__all__ = ["DenseOperator", "operator_matrix"]
```
→
```python
from typing import TYPE_CHECKING, Any

import numpy as np

from domainia import wilson
from domainia.domains import CellSet, WDMDomain
from domainia.translate import (
    Domain,
    Operator,
    _channel_names,
    _record,
    from_coordinates,
    to_coordinates,
)
from domainia.wilson.backend import NUMPY

if TYPE_CHECKING:
    from domainia.covariance import Covariance

__all__ = ["CovarianceOperator", "DenseOperator", "operator_matrix"]
```

(d) Insert the class between `DenseOperator` and `operator_matrix`:

```python
def operator_matrix(op: Operator) -> np.ndarray:
```
→
```python
@dataclass(frozen=True, eq=False)
class CovarianceOperator:
    """A :mod:`domainia.covariance` covariance ``cov`` fixed at the noise state ``nu`` on the
    WDM ``support``, as an :class:`~domainia.translate.Operator`.

    ``support`` is the :class:`~domainia.domains.WDMDomain` (cells and window) the
    covariance lives on; it may be partial.  The operator's ``domain`` is
    ``WDMDomain.full(support.cells.grid, window=support.window)``.  Let ``K`` be the number
    of channels, ``s = support.size`` and ``P`` the ``(K s, K N)`` selection of the
    support's cells in every channel (channel-major).  The operator is ``C = P^T Sigma P``
    with ``Sigma = cov`` at ``nu``, and its methods are the Moore-Penrose structure of that
    ``C`` (``F = cov.factor(nu)``):

    * ``apply``: ``cov.multiply``; ``solve``: ``F.solve``; ``project``: zeros off the
      support (the identity for a full support);
    * ``quadratic_form``: ``F.quad``; ``log_determinant``: ``F.logdet``;
      ``degrees_of_freedom``: ``K s``.

    Layout: the values go through :func:`~domainia.translate.to_coordinates` (``(K, N)``,
    ``channel_names`` order) and are cut to the support, ``R[:, support.cells.packed_positions()]``
    (``(K, s)``).  That array is flattened channel-major for a single covariance
    (``cov.n == K s``, so the covariance may correlate the channels) and kept as rows for a
    :class:`~domainia.covariance.ChannelStack` (one child per channel, each with
    ``n == s``).  Results are put back with exact zeros off the support and returned
    through :func:`~domainia.translate.from_coordinates`.

    Validation: ``TypeError`` for a wrong kind (``cov`` not a ``Covariance``, ``support``
    not a ``WDMDomain``, ``channel_names`` against the ``ChannelAxis`` rules, a non-numeric
    ``nu``); ``ValueError`` for a size or channel-count mismatch, a covariance ``window``
    (when not ``None``) or ``cells`` (when it has them) other than the support's, a
    non-zero ``n_u``, or ``F.ok`` false (the factor is computed once, eagerly on NumPy, at
    construction).  NumPy only.
    """

    cov: Covariance
    nu: Any
    support: WDMDomain
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        # Imported here, not at module level: ``import domainia`` imports this module and
        # must not import domainia.covariance (the package docstring).
        from domainia.covariance import ChannelStack, Covariance

        cov, support = self.cov, self.support
        if not isinstance(cov, Covariance):
            raise TypeError(f"cov must be a domainia.covariance.Covariance; got {type(cov).__name__}")
        if not isinstance(support, WDMDomain):
            raise TypeError(f"support must be a WDMDomain; got {type(support).__name__}")
        names = _channel_names(self.channel_names)
        K, s = len(names), support.size
        stack = isinstance(cov, ChannelStack)
        parts = cov.covs if stack else (cov,)
        if stack:
            if len(parts) != K:
                raise ValueError(f"the ChannelStack has {len(parts)} channels; channel_names has {K}")
            if any(c.n != s for c in parts):
                raise ValueError(f"every ChannelStack channel needs n = support.size = {s}; got "
                                 f"n = {tuple(c.n for c in parts)}")
        elif cov.n != K * s:
            raise ValueError(f"cov.n = {cov.n}, but {K} channel(s) on a support of {s} cells need "
                             f"cov.n = {K * s}")
        for c in parts:
            window = getattr(c, "window", None)
            if window is not None and window != support.window:
                raise ValueError(f"the covariance is built for the window {window.name} but the "
                                 f"support uses {support.window.name}")
            cells = getattr(c, "cells", None)
            if isinstance(cells, CellSet) and cells != support.cells:
                raise ValueError(f"the covariance's cells ({cells!r}) are not support.cells "
                                 f"({support.cells!r})")
        if cov.n_u != 0:
            raise ValueError(f"cov.n_u = {cov.n_u}: an Operator has no nuisance directions")
        nu = np.asarray(self.nu)
        if nu.dtype.kind not in "iuf":
            raise TypeError(f"nu must be a real numeric array; got dtype {nu.dtype}")
        nu = np.array(nu, dtype=np.float64)
        nu.setflags(write=False)
        F = cov.factor(nu, NUMPY)
        if not bool(F.ok):
            raise ValueError(f"{type(cov).__name__} is not positive definite at nu")
        pos = support.cells.packed_positions()
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "_domain", WDMDomain.full(support.cells.grid, window=support.window))
        object.__setattr__(self, "_factor", F)
        object.__setattr__(self, "_pos", pos)
        object.__setattr__(self, "_stack", stack)

    @property
    def domain(self) -> WDMDomain:
        return self._domain  # type: ignore[attr-defined]

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.channel_names) * self.support.size

    def _b(self, values: Any) -> np.ndarray:
        """``values`` on the support: ``(K, s)`` rows for a ChannelStack, else ``(K s,)``."""
        pos, stack = self._pos, self._stack  # type: ignore[attr-defined]
        R = to_coordinates(values, self.domain, self.channel_names)[:, pos]
        return R if stack else R.reshape(-1)

    def _values(self, y: Any) -> dict[str, np.ndarray]:
        """The support vector ``y`` back on the full domain: exact zeros off the support."""
        K = len(self.channel_names)
        R = np.zeros((K, self.domain.cells.grid.N))
        R[:, self._pos] = np.asarray(y).reshape(K, -1)  # type: ignore[attr-defined]
        return from_coordinates(R, self.domain, self.channel_names)

    def apply(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self.cov.multiply(self.nu, self._b(values), be=NUMPY))

    def solve(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self._factor.solve(self._b(values)))  # type: ignore[attr-defined]

    def project(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._values(self._b(values))

    def quadratic_form(self, values: Mapping[str, Any]) -> float:
        return float(self._factor.quad(self._b(values)))  # type: ignore[attr-defined]

    def log_determinant(self) -> float:
        return float(self._factor.logdet)  # type: ignore[attr-defined]


def operator_matrix(op: Operator) -> np.ndarray:
```

How the class works:
- The support coordinates are `to_coordinates(...)[:, support.cells.packed_positions()]`. Task 5's WDM coordinates are in `packed_cells` order, so the positions index them directly.
- A single covariance takes them flattened channel-major (`(K s,)`). A `ChannelStack` takes them as rows (`(K, s)`), the layout of `StackFactor.solve` and of Task 7's `ChannelStack.multiply`.
- The window and cells checks read `window`/`cells` with `getattr`. `EvolutionaryCovariance` has both, while `DiagonalCovariance`, `DenseCovariance` and `BlockTridiagonalCovariance` have neither, so for those only the size check applies.

- [ ] **Step 6: Export the coordinates from `src/domainia/translate.py`**

```python
__all__ = ["Domain", "Operator", "TranslatedOperator", "log_coordinate_scale", "transform",
           "translate_operator"]
```
→
```python
__all__ = ["Domain", "Operator", "TranslatedOperator", "from_coordinates", "log_coordinate_scale",
           "to_coordinates", "transform", "translate_operator"]
```

- [ ] **Step 7: Export the new names from `src/domainia/__init__.py`**

(a) The docstring's `domainia.translate` bullet:

```python
* :mod:`domainia.translate`: ``transform`` (``{channel: array}`` dicts) and
  ``translate_operator`` (lazy covariance change of basis) between the full-record
  ``TimeDomain``, ``FourierDomain.full`` and ``WDMDomain.full``.
```
→
```python
* :mod:`domainia.translate`: ``transform`` (``{channel: array}`` dicts) and
  ``translate_operator`` (lazy covariance change of basis) between the full-record
  ``TimeDomain``, ``FourierDomain.full`` and ``WDMDomain.full``; ``to_coordinates`` and
  ``from_coordinates`` (the real coordinates of the ``Operator`` contract).
* :mod:`domainia.operators`: ``DenseOperator`` (an explicit covariance matrix),
  ``CovarianceOperator`` (a :mod:`domainia.covariance` covariance at a fixed noise state)
  and ``operator_matrix`` (the matrix of any ``Operator``).
```

(b) The imports:

```python
from . import data, domains, linalg, translate, wilson
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from .translate import Operator, TranslatedOperator, log_coordinate_scale, transform, translate_operator
```
→
```python
from . import data, domains, linalg, operators, translate, wilson
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from .operators import CovarianceOperator, DenseOperator, operator_matrix
from .translate import (Operator, TranslatedOperator, from_coordinates, log_coordinate_scale,
                        to_coordinates, transform, translate_operator)
```

(c) `__all__`, in its sorted order:

```python
    "ChannelAxis",
    "FourierBins",
```
→
```python
    "ChannelAxis",
    "CovarianceOperator",
    "DenseOperator",
    "FourierBins",
```

and

```python
    "domains",
    "linalg",
    "log_coordinate_scale",
    "transform",
```
→
```python
    "domains",
    "from_coordinates",
    "linalg",
    "log_coordinate_scale",
    "operator_matrix",
    "operators",
    "to_coordinates",
    "transform",
```

- [ ] **Step 8: Name the route in `src/domainia/maps/base.py`**

(a) The module docstring:

```python
section 4.2's WDM -> time push ``W^T Lambda W / dt^2`` is not available).  A full-record WDM
covariance wrapped as a :class:`domainia.translate.Operator` is seen from time through
:func:`domainia.translate.translate_operator` instead.
"""
```
→
```python
section 4.2's WDM -> time push ``W^T Lambda W / dt^2`` is not available).  Fix the
covariance at a noise state as a :class:`domainia.operators.CovarianceOperator` and see it
from time through :func:`domainia.translate.translate_operator` instead.
"""
```

(b) `PushedCovariance._matvec`, its docstring and message:

```python
        4.2's WDM -> time push ``W^T Lambda W / dt^2`` is not available; the error names the
        :func:`domainia.translate.translate_operator` route."""
        fn = getattr(self.cov, "matvec", None)
        if fn is None:
            raise TypeError(f"{type(self.cov).__name__} has no matvec: a push through a "
                            f"{self.map.kind!r} map needs a fixed-noise-state SupportsMatvec "
                            "(X -> X Sigma), so the WDM -> time push is not available; wrap a "
                            "full-record WDM covariance as a domainia.translate.Operator and see "
                            "it from time with domainia.translate.translate_operator")
```
→
```python
        4.2's WDM -> time push ``W^T Lambda W / dt^2`` is not available; the error names the
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
```

- [ ] **Step 9: Run the tests and watch them pass**

Run: `$PY -m pytest -q tests/test_operators_covariance.py tests/test_package.py`
Expected: `30 passed` (25 + 5).

Run: `$PY -m pytest -q tests/test_dom_maps_base.py tests/test_operators_dense.py tests/wilson/test_window_threading.py`
Expected: all pass. The window scan finds no consumer call in `CovarianceOperator`.

- [ ] **Step 10: Grep for the old route**

Run: `grep -rn 'domainia.translate.Operator and\|wrapped as a :class:\|domainia\\\.translate\\\.Operator' src tests`
Expected: no output. Before this task it found three lines: the `maps/base.py` module docstring, the `_matvec` message and the `test_dom_maps_base.py` regex, all changed above.

- [ ] **Step 11: Full suite and NumPy lane**

Run: `$PY -m pytest -q` then `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: both pass with no errors, and each has 26 more passed tests than after Task 7 (25 new tests and the lazy-import test).

- [ ] **Step 12: Commit** (sandbox disabled)

```bash
git add src/domainia/operators.py src/domainia/maps/base.py src/domainia/translate.py src/domainia/__init__.py tests/test_operators_covariance.py tests/test_package.py tests/test_dom_maps_base.py
git commit -m "CovarianceOperator; export the operators and the coordinates

A covariance fixed at nu on a WDM support is an Operator that translate_operator
moves to time and Fourier. import domainia still leaves domainia.covariance out.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The cell-set consumers accept a `WDMDomain` (spec §2.7)

**Goal:** `EvolutionaryCovariance`, `BandAveragedVariance` and `MidpointVariance` take a `WDMDomain` where they take `cells`, and read its window, so a non-Meyer domain can no longer fall back to Meyer silently.

**Files:**
- Modify: `src/domainia/domains.py` (new helper `_cells_and_window`)
- Modify: `src/domainia/covariance/evolutionary.py` (`EvolutionaryCovariance.__init__`)
- Modify: `src/domainia/variance.py` (`BandAveragedVariance`, `MidpointVariance`)
- Test: `tests/test_dom_evolutionary_cov.py`, `tests/test_dom_variance_generic.py`, `tests/test_dom_windows.py`

**Acceptance Criteria:**
- [ ] With `dom = WDMDomain(cells, window=ALT_WINDOW)`, each of the three built on `dom` equals the same class built on `(cells, window=ALT_WINDOW)` (`dense`/values equal, `window` attribute equal; `MidpointVariance` takes only the cells).
- [ ] `window=None` on a `CellSet` gives `MEYER`. A `WDMDomain` with a different explicit `window` raises `ValueError`. A non-`CellSet`, non-`WDMDomain` `cells` raises `TypeError`.
- [ ] The message "for a WDMDomain pass domain.cells" is gone (`grep -rn "pass domain.cells" src` prints nothing).
- [ ] The window scan (`tests/wilson/test_window_threading.py`) passes unchanged. `src/` has no call site of the three classes (grep in Step 1), and the helper takes `MEYER` as a parameter default, not in its body.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged from Task 3.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

**Steps:**

Assumes the tree after Task 8.

The three consumers share one private helper, `domains._cells_and_window(cells, window=None, *, default=MEYER)`. The window scan (`tests/wilson/test_window_threading.py`, `_body_hard_codes`) flags any Meyer name in a function body except in parameter defaults and comparison operands. So the fallback is the parameter default `default=MEYER`, not `MEYER if window is None else window` in a body, which the scan reports as "hard-codes a window in a function body". `src/` has no call site of the three classes (Step 1), so no `window=` argument changes.

- [ ] **Step 1: Find the pins and the call sites**

Run: `grep -rn 'pass domain.cells\|domain\\\.cells\|cells must be a CellSet' src tests`
Expected:

```
src/domainia/variance.py:        raise TypeError(f"cells must be a CellSet, got {type(cells).__name__}; for a "
src/domainia/variance.py:                        "WDMDomain pass domain.cells")
src/domainia/domains.py:            raise TypeError(f"cells must be a CellSet; got {type(self.cells).__name__}")
src/domainia/covariance/evolutionary.py:            raise TypeError(f"cells must be a CellSet, got {type(cells).__name__}; for a "
src/domainia/covariance/evolutionary.py:                            "WDMDomain pass domain.cells")
tests/test_dom_evolutionary_cov.py:    with pytest.raises(TypeError, match=r"cells must be a CellSet.*domain\.cells"):
tests/test_dom_variance_generic.py:    with pytest.raises(TypeError, match=r"cells must be a CellSet.*domain\.cells"):
```

(line numbers omitted). Here is what happens to each hit:
- the two `variance.py` lines (`_require_cells`) and the two `evolutionary.py` lines are replaced in Steps 4–6;
- the `domains.py` hit is `WDMDomain.__post_init__`, which stays;
- the two test hits are `test_cells_must_be_a_cell_set`, rewritten in Step 2.

Run: `grep -rn "EvolutionaryCovariance(\|BandAveragedVariance(\|MidpointVariance(" src`
Expected: two hits, neither of them a call: the `variance.py` module docstring (``BandAveragedVariance(..., window=)`` …) and `class EvolutionaryCovariance(Covariance):`.

Run: `grep -rln "BandAveragedVariance\|MidpointVariance" tests | sort`
Expected: `tests/test_dom_cov_base.py`, `tests/test_dom_db_imports.py`, `tests/test_dom_variance_generic.py`, `tests/test_dom_windows.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_window_threading.py`. Only `test_dom_variance_generic.py` (the `domain.cells` pin) and `test_dom_windows.py` (new tests) change. The others build the classes on a `CellSet`, with or without `window=`, which behaves as before. `test_dom_windows.py` keeps `assert BandAveragedVariance(psd, CELLS).window is MEYER`, which is the `window=None`-on-a-`CellSet` default.

- [ ] **Step 2: Rewrite the `CellSet`-only tests**

(a) `tests/test_dom_evolutionary_cov.py`:

```python
@pytest.mark.parametrize("wrong", ["domain", "grid"])
def test_cells_must_be_a_cell_set(wrong):
    cells = _cells()
    arg = WDMDomain(cells) if wrong == "domain" else cells.grid
    with pytest.raises(TypeError, match=r"cells must be a CellSet.*domain\.cells"):
        EvolutionaryCovariance(PSD, arg)
```
→
```python
def test_cells_must_be_a_cell_set_or_a_wdm_domain():
    cells = _cells()
    ev = EvolutionaryCovariance(PSD, WDMDomain(cells))
    assert ev.cells == cells and isinstance(ev.cells, CellSet)
    with pytest.raises(TypeError, match=r"^cells must be a CellSet or a WDMDomain; got Grid$"):
        EvolutionaryCovariance(PSD, cells.grid)
```

(b) `tests/test_dom_variance_generic.py`:

```python
@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
@pytest.mark.parametrize("wrong", ["domain", "grid"])
def test_cells_must_be_a_cell_set(cls, wrong):
    arg = WDMDomain(CELLS) if wrong == "domain" else GRID
    with pytest.raises(TypeError, match=r"cells must be a CellSet.*domain\.cells"):
        cls(PSD, arg)
```
→
```python
@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
def test_cells_must_be_a_cell_set_or_a_wdm_domain(cls):
    v = cls(PSD, WDMDomain(CUT))
    assert v.cells == CUT and isinstance(v.cells, CellSet)
    np.testing.assert_array_equal(v(NU), cls(PSD, CUT)(NU))
    with pytest.raises(TypeError, match=r"^cells must be a CellSet or a WDMDomain; got Grid$"):
        cls(PSD, GRID)
```

(c) `tests/test_dom_windows.py`: two new tests, before `test_evolutionary_covariance_stationary_limit_with_another_window`. The file already imports everything they use (`ALT_WINDOW`, `MEYER`, `WDMDomain`, the three classes, `DiagonalCovariance`, `_stationary_psd`, `_roundtrip`):

```python
def test_evolutionary_covariance_stationary_limit_with_another_window():
```
→
```python
def test_the_cell_set_consumers_take_a_wdm_domain_and_its_window():
    """Spec section 2.7: built on a ``WDMDomain`` they use its cells and its window, so a
    non-Meyer domain cannot fall back to Meyer; ``MidpointVariance`` reads the cells only."""
    psd, nu = _stationary_psd()
    cut = CELLS.drop_columns(2, 1)
    dom = WDMDomain(cut, window=ALT_WINDOW)
    ev = EvolutionaryCovariance(psd, dom)
    assert ev.window is ALT_WINDOW and ev.cells == cut
    np.testing.assert_array_equal(ev.dense(nu), EvolutionaryCovariance(psd, cut, window=ALT_WINDOW).dense(nu))
    bv = BandAveragedVariance(psd, dom)
    assert bv.window is ALT_WINDOW and bv.cells == cut
    np.testing.assert_array_equal(bv(nu), BandAveragedVariance(psd, cut, window=ALT_WINDOW)(nu))
    mv = MidpointVariance(psd, dom)
    assert mv.cells == cut
    np.testing.assert_array_equal(mv(nu), MidpointVariance(psd, cut)(nu))
    dg = DiagonalCovariance(bv, cut.size)
    assert dg.window is ALT_WINDOW and dg.cells == cut
    # the domain's own window passed again (equal, not identical) is accepted
    assert EvolutionaryCovariance(psd, dom, window=_roundtrip(ALT_WINDOW)).window is ALT_WINDOW
    assert BandAveragedVariance(psd, dom, window=ALT_WINDOW).window is ALT_WINDOW


def test_a_wdm_domain_with_another_explicit_window_is_refused():
    psd, _ = _stationary_psd()
    alt = WDMDomain(CELLS, window=ALT_WINDOW)
    for build in (EvolutionaryCovariance, BandAveragedVariance):
        with pytest.raises(ValueError, match=r"^window meyer\(d=4\) is not the WDMDomain's window "
                                             r"wide-smoothstep: pass window=None"):
            build(psd, alt, window=MEYER)
        with pytest.raises(ValueError, match=r"window wide-smoothstep is not the WDMDomain's window meyer"):
            build(psd, WDMDomain(CELLS), window=ALT_WINDOW)
        with pytest.raises(TypeError, match="Window"):
            build(psd, alt, window="meyer")
    assert EvolutionaryCovariance(psd, CELLS, window=None).window is MEYER     # None on a CellSet
    assert BandAveragedVariance(psd, CELLS, window=None).window is MEYER


def test_evolutionary_covariance_stationary_limit_with_another_window():
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `$PY -m pytest -q tests/test_dom_evolutionary_cov.py tests/test_dom_variance_generic.py tests/test_dom_windows.py`
Expected: `5 failed`, the rest pass. All five fail with `TypeError: cells must be a CellSet, got WDMDomain; for a WDMDomain pass domain.cells`: the rewritten test in `test_dom_evolutionary_cov.py`, its two cases in `test_dom_variance_generic.py`, and the two new tests in `test_dom_windows.py`.

- [ ] **Step 4: Add `_cells_and_window` to `src/domainia/domains.py`**

It goes after `WDMDomain`, so both classes are defined when it runs. The module already imports `MEYER`, `Window` and `require_window`.

```python
@dataclass(frozen=True)
class ChannelAxis:
```
→
```python
def _cells_and_window(cells: object, window: object = None, *,
                      default: Window = MEYER) -> tuple[CellSet, Window]:
    """``(cell set, window)`` for a consumer that takes ``cells`` and ``window=``
    (``EvolutionaryCovariance``, ``BandAveragedVariance``; ``MidpointVariance`` reads the
    cells only).

    A :class:`WDMDomain` gives its cells and its window; an explicit ``window`` must equal
    the domain's (``ValueError`` otherwise).  A :class:`CellSet` takes ``window``, where
    ``None`` means ``default`` (:data:`MEYER`, the WDM transform).  Any other ``cells``
    raises ``TypeError``, and so does a ``window`` that is not a Window instance.
    ``default`` is a parameter, not a name in the body, so the window scan
    (``tests/wilson/test_window_threading.py``) sees no hard-coded window.
    """
    if isinstance(cells, WDMDomain):
        if window is not None:
            w = require_window(window)
            if w != cells.window:
                raise ValueError(f"window {w.name} is not the WDMDomain's window "
                                 f"{cells.window.name}: pass window=None to use the domain's")
        return cells.cells, cells.window
    if not isinstance(cells, CellSet):
        raise TypeError(f"cells must be a CellSet or a WDMDomain; got {type(cells).__name__}")
    return cells, require_window(default if window is None else window)


@dataclass(frozen=True)
class ChannelAxis:
```

- [ ] **Step 5: `EvolutionaryCovariance.__init__` in `src/domainia/covariance/evolutionary.py`**

(a) The imports (`MEYER`, `require_window` and `CellSet` have no other use in the module):

```python
from domainia.wilson.windows import MEYER, Window, require_window

from ..domains import CellSet
```
→
```python
from domainia.wilson.windows import Window

from ..domains import _cells_and_window
```

(b) The class docstring, `cells`:

```python
    cells
        A ``CellSet`` in band order.
```
→
```python
    cells
        A ``CellSet`` in band order, or a ``WDMDomain`` (its cells and its window).
```

and `window`:

```python
    window
        The Wilson window of the cells' basis (default :data:`MEYER`).
    """
```
→
```python
    window
        The Wilson window of the cells' basis.  ``None`` (default) means the
        ``WDMDomain``'s window, or :data:`~domainia.wilson.MEYER` for a ``CellSet``; a
        ``WDMDomain`` with another ``window`` raises ``ValueError``.
    """
```

(c) The signature's last line:

```python
                 window: Window = MEYER) -> None:
```
→
```python
                 window: Window | None = None) -> None:
```

(d) The `CellSet` check:

```python
        if not isinstance(cells, CellSet):
            raise TypeError(f"cells must be a CellSet, got {type(cells).__name__}; for a "
                            "WDMDomain pass domain.cells")
        if assembly not in ("blocktri", "dense"):
```
→
```python
        cells, window = _cells_and_window(cells, window)
        if assembly not in ("blocktri", "dense"):
```

(e) The window check, now done by the helper:

```python
        require_window(window)
        self.psd, self.cells, self.builder, self.assembly = psd, cells, builder, assembly
```
→
```python
        self.psd, self.cells, self.builder, self.assembly = psd, cells, builder, assembly
```

- [ ] **Step 6: `BandAveragedVariance` and `MidpointVariance` in `src/domainia/variance.py`**

Both are `pytree_dataclass`es whose `__post_init__` returns early when `t` is not `_BUILD`. A JAX unflatten passes the stored `CellSet` and window back, and they are not re-checked. On a build, `__post_init__` stores the domain's cells, so `cells` (static metadata) is always a `CellSet`.

(a) The module docstring:

```python
``BandAveragedVariance(..., window=)`` averages over that window (default Meyer); the
midpoint variance is window-free.
```
→
```python
Both take a ``CellSet`` or a ``WDMDomain``.  ``BandAveragedVariance`` averages over the
window of the cells' basis: the ``WDMDomain``'s window, or ``window=`` (default Meyer) for a
``CellSet``.  The midpoint variance is window-free.
```

(b) The imports:

```python
from domainia.wilson.windows import MEYER, Window, require_window

from .domains import CellSet
```
→
```python
from domainia.wilson.windows import Window

from .domains import CellSet, _cells_and_window
```

(c) Delete `_require_cells` (its two callers change below):

```python
def _require_cells(cells: Any) -> None:
    if not isinstance(cells, CellSet):
        raise TypeError(f"cells must be a CellSet, got {type(cells).__name__}; for a "
                        "WDMDomain pass domain.cells")


@pytree_dataclass(data=("t",), meta=("psd", "cells", "window"))
```
→
```python
@pytree_dataclass(data=("t",), meta=("psd", "cells", "window"))
```

(d) `BandAveragedVariance`'s docstring:

```python
    ``window`` is the Wilson window of the cells' basis (default :data:`MEYER`).
    """
```
→
```python
    ``cells`` is a ``CellSet`` or a ``WDMDomain`` (stored as its cells).  ``window`` is the
    Wilson window of the cells' basis: ``None`` (default) means the ``WDMDomain``'s window,
    or :data:`~domainia.wilson.MEYER` for a ``CellSet``; a ``WDMDomain`` with another
    ``window`` raises ``ValueError``.
    """
```

(e) Its `window` field and `__post_init__`:

```python
    window: Window = field(default=MEYER, kw_only=True)

    def __post_init__(self) -> None:
        if self.t is not _BUILD:
            return
        _require_generic(self.psd, "band-averaged")
        _require_cells(self.cells)
        require_window(self.window)
        object.__setattr__(self, "t", _cell_arrays(self.cells)[2])
```
→
```python
    window: Window | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.t is not _BUILD:
            return
        _require_generic(self.psd, "band-averaged")
        cells, window = _cells_and_window(self.cells, self.window)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "t", _cell_arrays(cells)[2])
```

(f) `MidpointVariance`'s docstring:

```python
    """Eq. 39: ``0.5 S(f_m, t_n)``."""
```
→
```python
    """Eq. 39: ``0.5 S(f_m, t_n)``.  ``cells`` is a ``CellSet`` or a ``WDMDomain`` (stored as
    its cells; the midpoint variance does not depend on the window)."""
```

(g) Its `__post_init__`:

```python
        _require_generic(self.psd, "midpoint")
        _require_cells(self.cells)
        object.__setattr__(self, "t", _cell_arrays(self.cells)[2])
```
→
```python
        _require_generic(self.psd, "midpoint")
        cells, _ = _cells_and_window(self.cells)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "t", _cell_arrays(cells)[2])
```

- [ ] **Step 7: Run the tests and watch them pass**

Run: `$PY -m pytest -q tests/test_dom_evolutionary_cov.py tests/test_dom_variance_generic.py tests/test_dom_windows.py`
Expected: all pass, 0 failed.

- [ ] **Step 8: The window scan and the Meyer baseline**

Run: `$PY -m pytest -q tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py`
Expected: `567 passed` (564 + 3), with neither file edited. The scan sees no hard-coded window in `_cells_and_window`, because `MEYER` appears only as the parameter default. The baseline is bit for bit, because a `CellSet` with `window` omitted still means `MEYER`.

- [ ] **Step 9: Grep the old message away**

Run: `grep -rn "pass domain.cells" src`
Expected: no output.

Run: `grep -rn 'cells must be a CellSet' src tests`
Expected: `WDMDomain.__post_init__` (`src/domainia/domains.py`, unchanged), `_cells_and_window`'s `cells must be a CellSet or a WDMDomain; got …`, and the two rewritten test regexes.

- [ ] **Step 10: Full suite and NumPy lane**

Run: `$PY -m pytest -q` then `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: both pass with no errors, and each has 1 fewer passed test than after Task 8. The six parametrised `test_cells_must_be_a_cell_set` cases (`domain`/`grid` × one or two classes) are replaced by three `test_cells_must_be_a_cell_set_or_a_wdm_domain` cases and two new window tests.

- [ ] **Step 11: Commit** (sandbox disabled)

```bash
git add src/domainia/domains.py src/domainia/covariance/evolutionary.py src/domainia/variance.py tests/test_dom_evolutionary_cov.py tests/test_dom_variance_generic.py tests/test_dom_windows.py
git commit -m "The cell-set consumers take a WDMDomain and read its window

EvolutionaryCovariance, BandAveragedVariance and MidpointVariance accept a
WDMDomain; window=None means the domain's window (Meyer for a CellSet), and a
different explicit window is refused.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Single precision throughout (spec §3)

**Goal:** A float32/complex64 input to the transforms runs in single precision end to end, on NumPy and on JAX. float64 bits stay identical.

**Files:**
- Modify: `src/domainia/wilson/transform.py` (the forward post-FFT products and edge divides; the inverse family; `forward_bands`; `inverse_bands`; the "entirely in single precision" docstrings; the `_add_rounded`/`_set_rounded` comments)
- Create: `tests/wilson/test_single_precision.py` (also the home of domainia's dtype table, `OUTPUT_DTYPE` / `output_dtype`)
- Modify: `tests/wilson/test_transform_shim.py` (imports that table and compares dtypes against it, not the oracle's)

**Acceptance Criteria:**
- [ ] A recording backend that wraps the NumPy backend's `fft` namespace sees only float32/complex64 FFT inputs in a float32 run of all ten entry points of the spec §3 table (`_many` forms included). On the old code this test fails for the six inverse-family and band-kernel entry points, and passes for the four forward ones.
- [ ] A test on the forward family's post-FFT product (the array handed to `_set_rounded` in a float32 run is single precision) fails on the old code and passes after.
- [ ] The output dtypes match the spec §3 table for every entry point (one parametrised test).
- [ ] The `jax`-marked variant of the recording test passes in the wdmrel env.
- [ ] `ORACLE_RTOL[float32] = 2e-6` still holds in `tests/wilson/test_transform_shim.py`.
- [ ] float64 is bit-exact: a record/check script run before and after the task reports `0 of 1504 arrays differ` (the baseline test alone cannot show it for the two band kernels, which it compares by tolerance after Task 2), and the Meyer baseline passes.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/wilson/test_single_precision.py tests/wilson/test_transform_shim.py tests/wilson/test_meyer_baseline.py` → all pass; `$PY $TMPDIR/t10_bits.py check $TMPDIR/t10_pre.npz` → `0 of 1504 arrays differ`; then `$PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 9. In particular Task 2 has rescaled the band kernels: `forward_bands` ends in `return be.xp.real(phase * Y) * np.sqrt(2.0 * dt / n_samples(Nf, Nt))` and `inverse_bands` in `return out * np.sqrt(n_samples(Nf, Nt) / (2.0 * dt))` (Interface contract: old output × `sqrt(2 dt/N)` and × `sqrt(N/(2 dt))`). Step 7 keeps those factors and only moves them into the working dtype.

- [ ] **Step 1: Record the float64 outputs before any edit**

After Task 2 the baseline compares the five rescaled families (`FourierToWDM.*`, `forward_bands`, `inverse_bands`) by tolerance, so it cannot see a float64 bit change in the band kernels. This record/check pair compares every Meyer case bit for bit, rescaled ones included, before and after this task.

Save as `$TMPDIR/t10_bits.py` (scratch, never committed):

```python
"""Task 10 float64 gate: every Meyer case is bit-identical (same dtype, np.array_equal)
before and after the single-precision casts.  Run from the repo root:
``record <npz>`` before editing, ``check <npz>`` after."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path[:0] = ["src", "tests/wilson"]
import domainia.covariance  # noqa: E402
import domainia.domains  # noqa: E402
import domainia.maps  # noqa: E402
import domainia.spectra  # noqa: E402
import domainia.testing  # noqa: E402
import domainia.variance  # noqa: E402
from domainia import wilson  # noqa: E402
from meyer_cases import compute  # noqa: E402

DM = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                     variance=domainia.variance, covariance=domainia.covariance,
                     testing=domainia.testing)
mode, path = sys.argv[1], Path(sys.argv[2])
out = compute(wilson, DM, {}, {})
if mode == "record":
    np.savez(path, **out)
    print(f"recorded {len(out)} arrays")
else:
    with np.load(path) as z:
        old = {k: z[k] for k in z.files}
    assert sorted(old) == sorted(out), sorted(set(old) ^ set(out))[:10]
    bad = [k for k in out if old[k].dtype != out[k].dtype or not np.array_equal(old[k], out[k])]
    print(f"{len(bad)} of {len(out)} arrays differ" + (f": {bad[:10]}" if bad else ""))
    raise SystemExit(1 if bad else 0)
```

Run: `cd $REPO && $PY $TMPDIR/t10_bits.py record $TMPDIR/t10_pre.npz`
Expected: `recorded 1504 arrays`

- [ ] **Step 2: Grep the patterns this task changes**

Run: `grep -rn "_set_rounded\|_add_rounded\|_working_dtypes\|entirely in single" src tests README.md | grep -v "tests/wilson/oracle"`
Expected (live tree; Tasks 0–9 do not touch these lines): only `src/domainia/wilson/transform.py`: the two "entirely in single precision" docstrings (module docstring, `_working_dtypes`), the two helper definitions and their call sites in `_prepare_edges_for_inverse`, `_scatter_interior_bands`, `_scatter_edge_bands` and `_gather_bands`, and the three `_working_dtypes` calls. No test references them, so nothing outside `transform.py` needs editing for them.

Run: `grep -rn "float32\|complex64" tests | grep -v "tests/wilson/oracle"`
Expected: `tests/test_translate.py` (the float32/complex64 input tests of `transform`, which computes in float64 whatever the input dtype; unaffected), `tests/wilson/test_backend.py` (the `add_at` cast-rule test; unaffected), `tests/wilson/test_inner_product_fixes.py` and `tests/wilson/test_noise_stationary.py` (`_psd32` and the float32 PSD gate: PSDs and coefficient grids, not transforms; unaffected), `tests/wilson/test_windows.py` (a float32 `dt`; unaffected), and `tests/wilson/test_transform_shim.py` (updated in Step 10). No vendored test pins float32 bits.

- [ ] **Step 3: Write the failing tests `tests/wilson/test_single_precision.py`**

```python
"""Single precision end to end in ``domainia.wilson.transform``.

A float32 / complex64 input runs in single precision: every FFT receives a single-precision
array, and every product stored into the output is formed in the working dtype.  float64 runs
are bit for bit unchanged (``tests/wilson/test_meyer_baseline.py``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import domainia.wilson.transform as T
from domainia.wilson.backend import NUMPY

NF, NT, DT = 8, 16, 0.5
N = NF * NT
BANDS = np.array([2, 3])
SINGLE = (np.dtype(np.float32), np.dtype(np.complex64))

#: domainia's dtype table: each entry point's output dtype for a (float32, float64) run.
OUTPUT_DTYPE = {
    "forward_time": (np.float32, np.float64),
    "forward_time_many": (np.float32, np.float64),
    "forward": (np.float32, np.float64),
    "forward_many": (np.float32, np.float64),
    "inverse": (np.complex64, np.complex128),
    "inverse_many": (np.complex64, np.complex128),
    "inverse_time": (np.float32, np.float64),
    "inverse_time_many": (np.float32, np.float64),
    "forward_bands": (np.float32, np.float64),
    "inverse_bands": (np.complex64, np.complex128),
}
FORWARD = ("forward_time", "forward_time_many", "forward", "forward_many")
INVERSE = ("inverse", "inverse_many", "inverse_time", "inverse_time_many")


def output_dtype(name: str, precision) -> np.dtype:
    """``OUTPUT_DTYPE[name]`` for a run at ``precision`` (``float32`` or ``float64``)."""
    single, double = OUTPUT_DTYPE[name]
    return np.dtype(single if np.dtype(precision) == np.float32 else double)


def _cases(precision, rng, to=np.asarray) -> dict:
    """``{entry point: call(be)}`` for every row of the table, on inputs of ``precision``
    (real inputs in it, complex ones in the matching complex dtype), converted by ``to``."""
    real = np.dtype(precision)
    cplx = np.dtype(np.complex64 if real == np.float32 else np.complex128)
    x, X = rng.standard_normal(N), rng.standard_normal((3, N))
    z, Z = np.fft.rfft(x), np.fft.rfft(X, axis=-1)
    w, W = T.forward_time(x, NF, NT, DT), T.forward_time_many(X, NF, NT, DT)
    wb = T.forward_bands(z, 0, BANDS, NF, NT, DT)
    x, X, w, W, wb = (to(a.astype(real)) for a in (x, X, w, W, wb))
    z, Z = (to(a.astype(cplx)) for a in (z, Z))
    return {
        "forward_time": lambda be: T.forward_time(x, NF, NT, DT, be=be),
        "forward_time_many": lambda be: T.forward_time_many(X, NF, NT, DT, be=be),
        "forward": lambda be: T.forward(z, NF, NT, DT, be=be),
        "forward_many": lambda be: T.forward_many(Z, NF, NT, DT, be=be),
        "inverse": lambda be: T.inverse(w, NF, NT, DT, be=be),
        "inverse_many": lambda be: T.inverse_many(W, NF, NT, DT, be=be),
        "inverse_time": lambda be: T.inverse_time(w, NF, NT, DT, be=be),
        "inverse_time_many": lambda be: T.inverse_time_many(W, NF, NT, DT, be=be),
        "forward_bands": lambda be: T.forward_bands(z, 0, BANDS, NF, NT, DT, be=be),
        "inverse_bands": lambda be: T.inverse_bands(wb, 0, N // 2 + 1, BANDS, NF, NT, DT, be=be),
    }


class _RecordingFFT:
    """An ``fft`` namespace that records ``(function, input dtype)`` for every call, then
    calls the wrapped namespace (``numpy.fft`` or ``jax.numpy.fft``)."""

    def __init__(self, fft):
        self._fft = fft
        self.calls: list[tuple[str, np.dtype]] = []

    def __getattr__(self, name):
        fn = getattr(self._fft, name)

        def recorded(a, *args, **kw):
            self.calls.append((name, np.dtype(a.dtype)))
            return fn(a, *args, **kw)

        return recorded


def _recording(be):
    """``be`` with its ``fft`` namespace wrapped in a :class:`_RecordingFFT` (and the recorder)."""
    rec = _RecordingFFT(be.fft)
    return dataclasses.replace(be, fft=rec), rec


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_every_fft_of_a_float32_run_is_single_precision(name, rng):
    """The inverse family and both band kernels used to run complex128 FFTs on float32 input
    (float64 sign patterns, ``lam`` and window samples promoted the FFT input)."""
    be, rec = _recording(NUMPY)
    _cases(np.float32, rng)[name](be)
    assert rec.calls, f"{name} ran no FFT through be.fft"
    assert all(dt in SINGLE for _, dt in rec.calls), rec.calls


@pytest.mark.parametrize("name", FORWARD + INVERSE)
def test_every_product_stored_in_a_float32_run_is_single_precision(name, rng, monkeypatch):
    """The value handed to ``_set_rounded`` / ``_add_rounded`` is already in the working dtype.

    The forward family's FFTs were single precision already; its leak was after the FFT, where
    float64 ``kappa`` and sign arrays (and the edge ``np.sqrt(2)``) made the stored product
    float64, rounded once on store."""
    run = _cases(np.float32, rng)[name]          # its float64 inputs are built before the spies
    seen: list[np.dtype] = []
    for helper in ("_set_rounded", "_add_rounded"):
        def spy(a, idx, v, be, _helper=getattr(T, helper)):
            seen.append(np.dtype(v.dtype))
            return _helper(a, idx, v, be)

        monkeypatch.setattr(T, helper, spy)
    run(NUMPY)
    assert seen and all(dt in SINGLE for dt in seen), seen


@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_output_dtypes_follow_the_table(name, precision, rng):
    assert _cases(precision, rng)[name](NUMPY).dtype == output_dtype(name, precision)


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_float32_runs_match_float64_to_single_precision_rounding(name):
    """Same inputs (rounded to float32 once), two precisions: the casts change rounding only."""
    lo = np.asarray(_cases(np.float32, np.random.default_rng(5))[name](NUMPY))
    hi = np.asarray(_cases(np.float64, np.random.default_rng(5))[name](NUMPY))
    assert np.max(np.abs(lo - hi)) <= 2e-6 * np.max(np.abs(hi))


@pytest.mark.jax
@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_jax_float32_runs_are_single_precision(name, jax_be, rng):
    """The same recording on the JAX backend: single-precision FFT inputs and the table's
    output dtype, returned as a ``jax.Array``."""
    import jax

    be, rec = _recording(jax_be)
    out = _cases(np.float32, rng, to=jax_be.asarray)[name](be)
    assert isinstance(out, jax.Array), type(out)
    assert out.dtype == output_dtype(name, np.float32)
    assert rec.calls and all(dt in SINGLE for _, dt in rec.calls), rec.calls
```

Note on the spy test: `run` must be built before the spies go in, because `_cases` computes its float64 inputs with `forward_time` and friends, which also call `_set_rounded`.

- [ ] **Step 4: Run the new tests to see them fail**

Run: `$PY -m pytest -q tests/wilson/test_single_precision.py`
Expected: `22 failed, 36 passed`. The failures:
- `test_every_fft_of_a_float32_run_is_single_precision[...]`: red for `inverse`, `inverse_many`, `inverse_time`, `inverse_time_many`, `forward_bands`, `inverse_bands` (the recorder sees `complex128`). **Green** for the four forward entry points, whose FFTs are already single precision.
- `test_every_product_stored_in_a_float32_run_is_single_precision[...]`: red for all eight (the spy sees `float64` / `complex128`).
- `test_output_dtypes_follow_the_table[forward_bands-float32]` (`float64`) and `[inverse_bands-float32]` (`complex128`).
- `test_jax_float32_runs_are_single_precision[...]`: red for the same six as the NumPy recording test.

`test_float32_runs_match_float64_to_single_precision_rounding` passes before and after: it guards the casts, it does not drive them.

- [ ] **Step 5: The forward family: cast the post-FFT constants (`_gather_bands`)**

In `src/domainia/wilson/transform.py`, `_gather_bands`, four edits.

`kappa`:

```python
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)          # host constant
    kappa = np.sqrt(2.0) / Nf
```
→
```python
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)          # host constant
    # Every constant below is in the working dtype, so a float32 run stays float32
    # after the FFT (for float64 the casts are no-ops).
    kappa = rdtype.type(np.sqrt(2.0) / Nf)
```

The interior sign pattern (this `sign_int` line is the one followed by the `out[..., 1:Nf, :]` comment; `_scatter_interior_bands` has another, edited in Step 6):

```python
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
    # ``out[..., 1:Nf, :] = (kappa * sign_int) * re_or_im_int``: same grouping,
    # computed in float64 and rounded once into the working dtype, exactly as before.
```
→
```python
    sign_int = (1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])).astype(rdtype)
    # ``out[..., 1:Nf, :] = (kappa * sign_int) * re_or_im_int``, in the working dtype.
```

The edge-band sign:

```python
                           kappa * sign * xp.where(even, y_m.real, y_m.imag), be)
```
→
```python
                           kappa * sign.astype(rdtype) * xp.where(even, y_m.real, y_m.imag), be)
```

The edge-band divide:

```python
    # ``out[..., 0, :] /= np.sqrt(2)``: the in-place divide computes in float64
    # (np.sqrt(2) is a float64 scalar) and rounds once on store -- as here.
    root2 = np.sqrt(2)
```
→
```python
    # ``out[..., 0, :] /= sqrt(2)``, with sqrt(2) in the working dtype.
    root2 = rdtype.type(np.sqrt(2))
```

Why these are float64-neutral: for a float64 run `rdtype.type(...)` is `np.float64` of the same value and `.astype(np.float64)` copies the same values, and every product keeps its grouping (`kappa * sign_int * re_or_im_int` is still `(kappa * sign_int) * re_or_im_int`). For a float32 run, a NumPy `float64` scalar or array is a strong type under NumPy 2 (NEP 50) and under JAX, so the old code promoted the product to float64.

- [ ] **Step 6: The inverse family: cast the pre-FFT constants**

`_prepare_edges_for_inverse`:

```python
    mask = active_mask(Nf, Nt)
    root2 = np.sqrt(2)
    # ``work[..., 0, :] *= mask[0] * root2`` as a functional update; on NumPy
    # it is the same in-place multiply (float64 product, rounded once into the
    # working dtype), so the rounding is unchanged.
```
→
```python
    mask = active_mask(Nf, Nt)
    root2 = rdtype.type(np.sqrt(2))   # mask[m] * root2 is then in the working dtype too
    # ``work[..., 0, :] *= mask[0] * root2`` as a functional update.
```
(The two `work = _set_rounded(...)` lines below stay as they are. A bool array times an `np.float32` scalar is `float32`.)

`_scatter_interior_bands` (this `sign_int` line is followed by `xp = be.xp`):

```python
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
    xp = be.xp
    scaled_int = wmn_work[..., 1:Nf, :] * sign_int
```
→
```python
    # In the working dtype, so a float32 G (and so the FFT) stays complex64;
    # the callers pass ``phi_c`` and ``lam`` in that dtype too.
    sign_int = (1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])).astype(wmn_work.dtype)
    xp = be.xp
    scaled_int = wmn_work[..., 1:Nf, :] * sign_int
```

`_scatter_edge_bands`:

```python
        even, sign = _parity(m, n_arr)
        scaled = wmn_work[..., m, :] * sign
```
→
```python
        even, sign = _parity(m, n_arr)
        scaled = wmn_work[..., m, :] * sign.astype(wmn_work.dtype)
```

`inverse`:

```python
    lam = 1.0 / (np.sqrt(2.0) * dt)
    X = be.xp.zeros(N // 2 + 1, dtype=cdtype)
```
→
```python
    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = be.xp.zeros(N // 2 + 1, dtype=cdtype)
```

`inverse_many`:

```python
    lam = 1.0 / (np.sqrt(2.0) * dt)
    X = be.xp.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
```
→
```python
    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = be.xp.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
```

`G = xp.where(even_int, scaled_int, 0.0) + 1j * xp.where(...)` needs no edit: the Python scalars `0.0` and `1j` are weak, so a float32 `scaled_int` gives a complex64 `G`. `inverse_time` and `inverse_time_many` need no edit: `irfft` of a complex64 spectrum returns float32.

- [ ] **Step 7: The band kernels: work in the input's precision**

`forward_bands`, the four lines after `_check_finite(z, "forward_bands: z")` (Task 2's factor on the return line):

```python
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi)
    Y = be.fft.ifft(phi * z[..., ell], axis=-1)
    return be.xp.real(phase * Y) * np.sqrt(2.0 * dt / n_samples(Nf, Nt))
```
→
```python
    rdtype, cdtype = _working_dtypes(z)
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi).astype(cdtype)
    Y = be.fft.ifft(phi.astype(rdtype) * z[..., ell], axis=-1)
    return be.xp.real(phase * Y) * rdtype.type(np.sqrt(2.0 * dt / n_samples(Nf, Nt)))
```

`inverse_bands`, the seven lines after `ell = _band_windows(bands, k0, n_bins, Nt)`:

```python
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi)
    V = phi * be.fft.fft(np.conj(phase) * w, axis=-1) / Nt
    out = be.xp.zeros((*w.shape[:-2], n_bins), dtype=complex)
    for b in range(bands.size):
        out = be.add_at(out, (..., ell[b]), V[..., b, :], copy=False)
    return out * np.sqrt(n_samples(Nf, Nt) / (2.0 * dt))
```
→
```python
    rdtype, cdtype = _working_dtypes(w)
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi)
    V = phi.astype(rdtype) * be.fft.fft(np.conj(phase).astype(cdtype) * w, axis=-1) / Nt
    out = be.xp.zeros((*w.shape[:-2], n_bins), dtype=cdtype)
    for b in range(bands.size):
        out = be.add_at(out, (..., ell[b]), V[..., b, :], copy=False)
    return out * rdtype.type(np.sqrt(n_samples(Nf, Nt) / (2.0 * dt)))
```

`_band_phase` keeps computing from the float64 `phi` (its `np.linalg.norm(phi)` is a float64 constant); only its result is cast. If Task 2 placed either factor somewhere other than the return line (for example folded into `_band_phase`'s result), keep Task 2's placement and apply only the casts shown here: `.astype(cdtype)` on the phase, `phi.astype(rdtype)` in the FFT input, `dtype=cdtype` for `out`, and `rdtype.type(...)` around a NumPy scalar factor. Step 11 catches any float64 regrouping.

- [ ] **Step 8: Make the precision text true**

Module docstring, the first paragraph under "Precision":

```
Single precision propagates: a ``float32`` series (or ``complex64`` spectrum)
is transformed entirely in single precision and returns ``float32``.  Anything
else, integers included, promotes to double.  It is opt-in by dtype rather
than by flag, so the precision of a result is readable off its own ``.dtype``.
```
→
```
Single precision propagates: a ``float32`` series (or ``complex64`` spectrum)
is transformed in single precision and returns ``float32``.  Every constant
that multiplies a working array (the sign patterns, ``kappa``, ``lam``, the
window samples, the edge-band ``sqrt(2)``, the band phases) is cast to the
working dtype first, so every FFT receives a single-precision array, on NumPy
and on JAX.  Anything else, integers included, promotes to double.  It is
opt-in by dtype rather than by flag, so the precision of a result is readable
off its own ``.dtype``.

float64 runs are bit for bit what they were before the casts (a cast to
float64 is a no-op, and no float64 expression is regrouped; the Meyer baseline,
``tests/wilson/test_meyer_baseline.py``, is the gate).  float32 runs differ
from the frozen oracle at single-precision rounding, because the oracle formed
some products in double before storing them.
```

Module docstring, the first bullet of "Where it reaches, and where it stops":

```
* **The transform** -- ``forward``/``inverse`` and their ``_time`` and
  ``_many`` forms -- follows the input dtype.
```
→
```
* **The transform** -- ``forward``/``inverse`` and their ``_time`` and
  ``_many`` forms, and the band kernels ``forward_bands``/``inverse_bands`` --
  follows the input dtype.
```

`_working_dtypes` docstring:

```
    Single precision propagates: a ``float32`` series or ``complex64`` spectrum
    is transformed entirely in single precision and comes back ``float32``.
    Anything else -- including integer input -- promotes to double.
```
→
```
    Single precision propagates: a ``float32`` series or ``complex64`` spectrum
    is transformed in single precision (every constant is cast to the working
    dtype before it multiplies a working array) and comes back ``float32``.
    Anything else -- including integer input -- promotes to double.
```

`_set_rounded` docstring (the whole docstring):

```
    """``a[idx] = v`` with ``v`` first rounded to ``a.dtype`` (``copy=False``).

    NumPy rounds a float64 ``v`` once on store into a float32 ``a``; casting
    first is that same rounding, so the result is bit-identical.  On JAX it
    keeps the scatter dtype-safe: a float32 transform would otherwise store
    float64 values, which jax 0.9 warns about (``FutureWarning``: "scatter
    inputs have incompatible types") and a later JAX will reject.
    """
```
→
```
    """``a[idx] = v`` with ``v`` cast to ``a.dtype`` (``copy=False``).

    The transforms form ``v`` in the working dtype (their constants are cast
    first), so the cast is a no-op for them.  It stays as the guard that keeps
    the scatter dtype-safe on JAX: storing float64 values into a float32 array
    warns in jax 0.9 (``FutureWarning``: "scatter inputs have incompatible
    types"), and a later JAX will reject it.
    """
```

`_add_rounded` docstring (the whole docstring):

```
    """``a[idx] += v`` for a duplicate-free ``idx``, rounded once into ``a.dtype``.

    NumPy's in-place ``+=`` adds in the promoted dtype and rounds once on
    store; this spells it out -- gather, add, round, set -- so it is
    bit-identical on NumPy for float32 and float64 alike, and dtype-safe on
    JAX (see :func:`_set_rounded`).  Rounding ``v`` to ``a.dtype`` *before* the
    add would round twice and change the float32 bits (120 of 160 random
    float32 inverse cases differed from the oracle).
    """
```
→
```
    """``a[idx] += v`` for a duplicate-free ``idx``, stored in ``a.dtype``.

    Gather, add, cast, set: the functional form of NumPy's in-place ``+=``,
    so float64 bits are the in-place statement's.  The inverse transforms hand
    it a ``v`` already in ``a``'s dtype, so a float32 run adds in single
    precision and the cast is a no-op; the cast keeps the scatter dtype-safe
    on JAX (see :func:`_set_rounded`).
    """
```

The module docstring is not a raw string: keep `sqrt(2)` in plain text (a `\s` would be an invalid escape).

Run: `grep -n "entirely in single\|computed in float64\|rounds once on store" src/domainia/wilson/transform.py`
Expected: no output.

- [ ] **Step 9: Run the new tests to see them pass**

Run: `$PY -m pytest -q tests/wilson/test_single_precision.py`
Expected: `58 passed`

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q tests/wilson/test_single_precision.py`
Expected: `48 passed, 10 skipped` (the ten JAX legs skip).

- [ ] **Step 10: The shim tests compare dtypes against domainia's table**

In `tests/wilson/test_transform_shim.py`, six edits.

Module docstring:

```
product rounded differently, so the gate is ORACLE_RTOL[dtype] * max|ref|.
```
→
```
product rounded differently, so the gate is ORACLE_RTOL[dtype] * max|ref|.
float32 runs in single precision end to end, so its bits differ from the
oracle's at rounding; output dtypes are checked against domainia's table
(``test_single_precision.OUTPUT_DTYPE``), not the oracle's.
```

Imports:

```python
import domainia.wilson.transform as T

GRIDS
```
→
```python
import domainia.wilson.transform as T

from .test_single_precision import output_dtype

GRIDS
```

The float32 tolerance comment (the value stays `2e-6`; measured over 20 seeds on the eight `GRIDS` in the scratch prototype):

```python
    np.dtype(np.float32): 2e-6,      # Hz window (v0.12.0): measured 2.1e-7 fwd, 2.5e-7 inv
```
→
```python
    np.dtype(np.float32): 2e-6,      # single precision throughout: measured 3.2e-7 fwd, 3.1e-7 inv
```

The helper:

```python
def _assert_is_the_oracle_to_rounding(got, ref, dtype):
    """Same dtype as the oracle, and the same values to the working precision's rounding."""
    assert got.dtype == ref.dtype
```
→
```python
def _assert_is_the_oracle_to_rounding(name, got, ref, dtype):
    """The dtype of domainia's table (``test_single_precision.OUTPUT_DTYPE``), and the
    oracle's values to the working precision's rounding."""
    assert got.dtype == output_dtype(name, dtype)
```

`test_forward_family_is_the_oracle_to_rounding`:

```python
    pairs = [
        (T.forward_time(x, Nf, Nt, dt), oracle.forward_time(x, Nf, Nt, dt)),
        (T.forward(np.fft.rfft(x), Nf, Nt, dt), oracle.forward(np.fft.rfft(x), Nf, Nt, dt)),
        (T.forward_time_many(X, Nf, Nt, dt), oracle.forward_time_many(X, Nf, Nt, dt)),
        (T.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt),
         oracle.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref, dtype)
```
→
```python
    pairs = [
        ("forward_time", T.forward_time(x, Nf, Nt, dt), oracle.forward_time(x, Nf, Nt, dt)),
        ("forward", T.forward(np.fft.rfft(x), Nf, Nt, dt),
         oracle.forward(np.fft.rfft(x), Nf, Nt, dt)),
        ("forward_time_many", T.forward_time_many(X, Nf, Nt, dt),
         oracle.forward_time_many(X, Nf, Nt, dt)),
        ("forward_many", T.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt),
         oracle.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt)),
    ]
    for name, got, ref in pairs:
        _assert_is_the_oracle_to_rounding(name, got, ref, dtype)
```

`test_inverse_family_is_the_oracle_to_rounding`:

```python
    pairs = [
        (T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
        (T.inverse_many(W, Nf, Nt, dt), oracle.inverse_many(W, Nf, Nt, dt)),
        (T.inverse_time_many(W, Nf, Nt, dt), oracle.inverse_time_many(W, Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref, dtype)
```
→
```python
    pairs = [
        ("inverse", T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
        ("inverse_time", T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
        ("inverse_many", T.inverse_many(W, Nf, Nt, dt), oracle.inverse_many(W, Nf, Nt, dt)),
        ("inverse_time_many", T.inverse_time_many(W, Nf, Nt, dt),
         oracle.inverse_time_many(W, Nf, Nt, dt)),
    ]
    for name, got, ref in pairs:
        _assert_is_the_oracle_to_rounding(name, got, ref, dtype)
```

`tests/wilson` is a package (`__init__.py`), so the relative import works under pytest, including the NumPy-lane subprocess of `tests/wilson/test_lanes.py`.

- [ ] **Step 11: Verify, and check float64 bit for bit**

Run: `$PY -m pytest -q tests/wilson/test_single_precision.py tests/wilson/test_transform_shim.py tests/wilson/test_meyer_baseline.py`
Expected: all pass, 0 errors (58 in `test_single_precision.py`, 40 in `test_transform_shim.py`, and the baseline tests; 101 in the scratch prototype, where the baseline file had its 3 tests).

Run: `$PY $TMPDIR/t10_bits.py check $TMPDIR/t10_pre.npz`
Expected: `0 of 1504 arrays differ`. Any difference is a float64 regrouping in Steps 5–7: undo it (the casts must be the only change).

Run: `$PY tests/wilson/meyer_cases.py --check`
Expected: `baseline reproduces: 1504 arrays`

- [ ] **Step 12: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 0 errors (3 skipped: the `DOMAINIA_ORACLE_ROOT` skips).

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass, 0 errors.

- [ ] **Step 13: Commit** (sandbox disabled)

```bash
git add src/domainia/wilson/transform.py tests/wilson/test_single_precision.py tests/wilson/test_transform_shim.py
git commit -m "Single precision end to end in the transforms and band kernels

Cast every constant to the working dtype before it multiplies a working array:
kappa, the sign patterns and the edge sqrt(2) after the forward FFT, lam and the
signs before the inverse FFTs, and the window samples, phases and Wilson factors
of forward_bands / inverse_bands. float32 FFT inputs are now single precision on
NumPy and JAX; float64 bits are unchanged. The shim compares dtypes against
domainia's table.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Wrong kinds raise `TypeError` in `domainia.wilson` (spec §4)

**Goal:** `domainia.wilson` follows the rest of domainia: a wrong kind raises `TypeError`, and a wrong value raises `ValueError`.

**Files:**
- Modify: `src/domainia/wilson/grid.py` (`_require_int`, `validate_grid`, `n_samples`, `validate_dt`, the `delta_F` Raises section), `src/domainia/wilson/noise.py` (`colored_noise`), `src/domainia/wilson/filter.py` (`nu_d`, the `phi_centered` Raises section), `src/domainia/wilson/transform.py` (the Raises sections of `forward`, `inverse`, `forward_time`)
- Create: `tests/wilson/test_wrong_kinds.py`
- Modify: `tests/wilson/test_noise_stationary.py`, `tests/wilson/vendored/test_filter.py`

**Acceptance Criteria:**
- [ ] Each integer site in the Interface contract raises `TypeError` for a non-integer or `bool` with the same message text as before, and `ValueError` for an out-of-range value. `validate_dt` raises `TypeError("dt must be a real number, got <type>")` for a bool/str/None/complex: new text, since it had no message for these kinds.
- [ ] `nu_d(…, d=2.5)` raises `TypeError("d must be an integer, got float")`, and `d=True` (which used to pass as `d = 1`) raises `TypeError`, matching `Meyer(d)`. `d=0` and `d=-1` raise `ValueError` matching `"d must be >= 1"`.
- [ ] The frozen oracle `tests/wilson/oracle/vendored_2f7136c` is untouched (`git diff --stat` shows nothing under it), and its sha pins pass.
- [ ] The touched docstrings' "Raises" sections name the right exception.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/wilson` → all pass; `git diff --stat -- tests/wilson/oracle` → no output; `$PY -m pytest -q tests/wilson/test_oracle_pins.py` → `6 passed`; then `$PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 10. Tasks 0–10 do not touch `wilson/grid.py`, `wilson/noise.py` or `wilson/filter.py`. Task 10 edited `wilson/transform.py`, but not the three "Raises" sections Step 6 edits.

- [ ] **Step 1: Grep the sites and the tests that pin them**

Run: `grep -rn "must be an integer" src/domainia/wilson`
Expected: five raise sites, all `ValueError` today:
- `grid.py`: `_require_int`, `validate_grid`, `n_samples` (`{name} must be an integer, got {type}`);
- `noise.py`: `colored_noise` (`N must be an integer, got {type}`);
- `filter.py`: `nu_d` (`d must be an integer >= 1, got {d!r}`, kind and range in one check).

`validate_dt` has no kind check: it calls `float(dt)`, which takes `True` as 1.0, `"0.5"` as 0.5 and `np.complex64(1)` as 1.0 (with a `ComplexWarning`), and raises Python's own `TypeError` for `None` and `complex`.

Run: `grep -rn "must be an integer\|positive finite number\|d must be\|must be even\|must be positive" tests | grep -v "tests/wilson/oracle"`
Expected hits, and what to do with each:
- `tests/wilson/test_noise_stationary.py:100-101` (`True` and `8.0` for `N`, expecting `ValueError`): move to a `TypeError` test (Step 3).
- `tests/wilson/test_noise_stationary.py:95-99, 102-105` (range of `N`, range of `dt`): stay `ValueError`, unchanged.
- `tests/wilson/vendored/test_filter.py:106` (`d` in `0, -1, 2.5`, `match="d must be an integer"`): `2.5` moves to a `TypeError` test; `0` and `-1` stay `ValueError` with the new range message `"d must be >= 1"` (Step 3).
- `tests/wilson/test_inner_product_fixes.py:156-167` (`Nf must be even`, `N must be even`): range checks, unchanged.

A bare `pytest.raises(ValueError)` around a wrong kind would not show in this grep. In the scratch prototype, the full suite with Steps 5–6 applied and no test edits failed exactly 5 tests: the two `test_noise_stationary` cases above and the three `test_invalid_smoothness_order_raises` cases. So these two files are the whole pin list. Step 8 confirms it on the real tree.

`tests/wilson/vendored/test_grid.py:114-116` passes only ints, and the frozen oracle `tests/wilson/oracle/vendored_2f7136c` is never edited (its sha pins are in `tests/wilson/test_oracle_pins.py`). `tests/wilson/vendored/` is not pinned.

- [ ] **Step 2: Write the failing tests `tests/wilson/test_wrong_kinds.py`**

```python
"""``domainia.wilson`` refuses a wrong kind with ``TypeError`` and a wrong value with ``ValueError``,
as the rest of domainia does."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.wilson.filter import nu_d, phi_centered
from domainia.wilson.grid import _require_int, n_samples, validate_dt, validate_grid


@pytest.mark.parametrize(("args", "match"), [
    ((8.0, 16), "Nf must be an integer, got float"),
    ((8, 16.0), "Nt must be an integer, got float"),
    ((True, 16), "Nf must be an integer, got bool"),
    ((np.float64(8), 16), "Nf must be an integer, got float64"),
    (("8", 16), "Nf must be an integer, got str"),
])
@pytest.mark.parametrize("check", [validate_grid, n_samples], ids=["validate_grid", "n_samples"])
def test_grid_sizes_of_the_wrong_kind_raise_type_error(check, args, match):
    with pytest.raises(TypeError, match=match):
        check(*args)


@pytest.mark.parametrize(("args", "match"), [
    ((9, 16), "Nf must be even, got 9"),
    ((0, 16), "Nf must be positive, got 0"),
    ((8, -2), "Nt must be positive, got -2"),
])
def test_grid_sizes_out_of_range_stay_value_errors(args, match):
    with pytest.raises(ValueError, match=match):
        validate_grid(*args)
    assert n_samples(np.int64(8), 16) == 128 and type(n_samples(np.int64(8), 16)) is int


def test_require_int_and_its_callers_raise_type_error():
    with pytest.raises(TypeError, match="N must be an integer, got float"):
        _require_int("N", 32.0)
    with pytest.raises(TypeError, match="N must be an integer, got bool"):
        _require_int("N", False)
    assert _require_int("N", np.int32(32)) == 32
    with pytest.raises(TypeError, match="N must be an integer, got float"):
        wilson.matrix.rfft_to_real(np.zeros(17, complex), 32.0)
    with pytest.raises(TypeError, match="Nf must be an integer, got float"):
        wilson.delta_F(8.0, 1.0)
    with pytest.raises(ValueError, match="N must be even, got 5"):
        wilson.matrix.rfft_to_real(np.zeros(3, complex), 5)
    with pytest.raises(TypeError, match="Nf must be an integer, got float"):
        wilson.Grid(8.0, 16, 1.0)


@pytest.mark.parametrize(("dt", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (None, "NoneType"),
    (1.0 + 0.0j, "complex"), (np.complex64(1.0), "complex64"), (np.array(1.0 + 0.0j), "ndarray"),
])
def test_dt_of_the_wrong_kind_raises_type_error(dt, kind):
    """``float()`` used to accept ``True`` (as 1.0), ``"0.5"`` and a NumPy complex scalar."""
    with pytest.raises(TypeError, match=f"dt must be a real number, got {kind}"):
        validate_dt(dt)
    with pytest.raises(TypeError, match="dt must be a real number"):
        wilson.forward_time(np.zeros(128), 8, 16, dt)


@pytest.mark.parametrize("dt", [0.5, 1, np.float32(0.1), np.float64(0.25), np.int64(2), np.array(0.5)])
def test_real_dt_kinds_still_pass(dt):
    validate_dt(dt)


@pytest.mark.parametrize("dt", [0.0, -1.0, np.inf, np.nan])
def test_dt_out_of_range_stays_a_value_error(dt):
    with pytest.raises(ValueError, match="dt must be a positive finite number"):
        validate_dt(dt)


def test_nu_d_splits_the_kind_and_range_checks():
    x = np.array([0.5])
    with pytest.raises(TypeError, match="d must be an integer, got float"):
        nu_d(x, d=2.5)
    with pytest.raises(TypeError, match="d must be an integer, got bool"):
        nu_d(x, d=True)
    with pytest.raises(ValueError, match="d must be >= 1, got 0"):
        nu_d(x, d=0)
    with pytest.raises(ValueError, match="d must be >= 1, got -1"):
        nu_d(x, d=-1)
    assert nu_d(x, d=np.int64(4)).item() == 0.5
    with pytest.raises(TypeError, match="d must be an integer, got float"):
        phi_centered(8, 16, 1.0, d=2.5)
```

`nu_d` rejects `bool` like `_require_int` and `Meyer(d)` do. Today `nu_d(x, d=True)` passes as `d = 1`.

- [ ] **Step 3: Move the pinned cases to `TypeError`**

`tests/wilson/test_noise_stationary.py`, in the parametrisation of `test_colored_noise_rejects_bad_N_and_dt`:

```python
    (-8, 1.0, "N must be positive, got -8"),
    (True, 1.0, "N must be an integer, got bool"),
    (8.0, 1.0, "N must be an integer, got float"),
    (8, 0.0, "dt must be a positive finite number"),
```
→
```python
    (-8, 1.0, "N must be positive, got -8"),
    (8, 0.0, "dt must be a positive finite number"),
```

and after the end of `test_colored_noise_rejects_bad_N_and_dt` (its last four lines are shown, kept as they are):

```python
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match=match):
        noise.colored_noise(N, dt, psd, rng)
    assert rng.bit_generator.state == state
```
→
```python
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match=match):
        noise.colored_noise(N, dt, psd, rng)
    assert rng.bit_generator.state == state


@pytest.mark.parametrize(("N", "dt", "match"), [
    (True, 1.0, "N must be an integer, got bool"),
    (8.0, 1.0, "N must be an integer, got float"),
    (8, True, "dt must be a real number, got bool"),
    (8, "0.5", "dt must be a real number, got str"),
    (8, None, "dt must be a real number, got NoneType"),
])
def test_colored_noise_rejects_wrong_kinds_with_type_error(N, dt, match):
    """A wrong kind of ``N`` or ``dt`` is a ``TypeError`` (``dt=True`` used to pass as 1.0),
    raised before the PSD is evaluated or the generator is touched."""
    def psd(f):
        raise AssertionError("PSD evaluated before the input checks")

    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(TypeError, match=match):
        noise.colored_noise(N, dt, psd, rng)
    assert rng.bit_generator.state == state
```

`tests/wilson/vendored/test_filter.py`:

```python
@pytest.mark.parametrize("bad_d", [0, -1, 2.5])
def test_invalid_smoothness_order_raises(bad_d):
    """``betainc(0, 0, x)`` is NaN, so a bad ``d`` must not slip through."""
    with pytest.raises(ValueError, match="d must be an integer"):
        nu_d(np.array([0.5]), d=bad_d)
```
→
```python
@pytest.mark.parametrize("bad_d", [0, -1])
def test_invalid_smoothness_order_raises(bad_d):
    """``betainc(0, 0, x)`` is NaN, so a bad ``d`` must not slip through."""
    with pytest.raises(ValueError, match="d must be >= 1"):
        nu_d(np.array([0.5]), d=bad_d)


def test_non_integer_smoothness_order_raises_type_error():
    with pytest.raises(TypeError, match="d must be an integer, got float"):
        nu_d(np.array([0.5]), d=2.5)
```

- [ ] **Step 4: Run the tests to see them fail**

Run: `$PY -m pytest -q tests/wilson/test_wrong_kinds.py tests/wilson/test_noise_stationary.py tests/wilson/vendored/test_filter.py`
Expected: `27 failed, 61 passed` (and one `ComplexWarning`, from `float(np.complex64(1.0))` in the old `validate_dt`):
- `test_wrong_kinds.py`: 19 failed. Every `TypeError` case gets `ValueError` (grid sizes, `_require_int`, `nu_d`), or no error at all (`validate_dt(True)`, `validate_dt("0.5")`), or Python's message (`validate_dt(None)`). The 13 range and pass-through cases pass.
- `test_noise_stationary.py`: the 5 `test_colored_noise_rejects_wrong_kinds_with_type_error` cases.
- `vendored/test_filter.py`: `test_invalid_smoothness_order_raises[0]` and `[-1]` (the old message `d must be an integer >= 1, got 0` does not match `d must be >= 1`), and `test_non_integer_smoothness_order_raises_type_error`.

- [ ] **Step 5: `grid.py`: wrong kinds raise `TypeError`**

`_require_int`:

```python
def _require_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``, raising ``ValueError`` if it is not one
    (``bool`` is rejected too).  Verbatim from lisa/wdm b5973b7."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
```
→
```python
def _require_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``, raising ``TypeError`` if it is not one
    (``bool`` is rejected too)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
```

`_require_even_positive_int` docstring:

```python
    """Return ``value`` as an ``int``, requiring it to be positive and even.
    Verbatim from lisa/wdm b5973b7 (same messages as :func:`validate_grid`)."""
```
→
```python
    """Return ``value`` as an ``int``, requiring it to be positive and even.
    Same messages and exceptions as :func:`validate_grid`."""
```

`validate_grid`, the docstring's first line:

```python
    """Raise ``ValueError`` if ``Nf`` or ``Nt`` are not positive even integers.
```
→
```python
    """Raise unless ``Nf`` and ``Nt`` are positive even integers.
```

its "Raises" section and examples:

```python
    Raises
    ------
    ValueError
        If either argument is not a positive even integer.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson as wdm
    >>> wdm.validate_grid(8, 16)          # even and positive: fine
    >>> wdm.validate_grid(9, 16)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    """
```
→
```python
    Raises
    ------
    TypeError
        If either argument is not an integer (``bool`` included).
    ValueError
        If either argument is not positive and even.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson as wdm
    >>> wdm.validate_grid(8, 16)          # even and positive: fine
    >>> wdm.validate_grid(9, 16)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    >>> wdm.validate_grid(8.0, 16)
    Traceback (most recent call last):
    TypeError: Nf must be an integer, got float
    """
```

and its kind check:

```python
            raise ValueError(f"{name} must be an integer, got {type(val).__name__}")
        if val <= 0:
```
→
```python
            raise TypeError(f"{name} must be an integer, got {type(val).__name__}")
        if val <= 0:
```

`validate_dt`, the docstring's first line:

```python
    """Raise ``ValueError`` unless ``dt`` is a positive finite number.
```
→
```python
    """Raise unless ``dt`` is a positive finite real number.
```

and from its "Raises" section to the `float` call:

```python
    Raises
    ------
    ValueError
        If ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson as wdm
    >>> wdm.validate_dt(0.5)
    >>> wdm.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    """
    value = float(dt)
```
→
```python
    Raises
    ------
    TypeError
        If ``dt`` is ``None``, a ``bool``, a string or complex (scalar or array):
        ``float()`` would take ``True`` as 1.0 and ``"0.5"`` as 0.5 silently, and
        drop the imaginary part of a NumPy complex scalar.
    ValueError
        If ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson as wdm
    >>> wdm.validate_dt(0.5)
    >>> wdm.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    >>> wdm.validate_dt(True)
    Traceback (most recent call last):
    TypeError: dt must be a real number, got bool
    """
    kind = getattr(getattr(dt, "dtype", None), "kind", None)
    if dt is None or isinstance(dt, (bool, str, bytes, complex)) or kind in ("b", "c", "S", "U"):
        raise TypeError(f"dt must be a real number, got {type(dt).__name__}")
    value = float(dt)
```

The check refuses only those kinds and then calls `float(dt)` as before, so Python ints, NumPy integer and float scalars (`np.float32(0.1)`, used by `tests/wilson/test_windows.py:345`), 0-d real arrays and concrete JAX scalars still pass. (`np.True_` and NumPy complex scalars carry a `dtype` of kind `b` / `c`.)

`n_samples`, a "Raises" section between "Returns" and "Examples":

```python
    int
        $N = N_f \\cdot N_t$.

    Examples
```
→
```python
    int
        $N = N_f \\cdot N_t$.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer (``bool`` included).

    Examples
```

and its raise:

```python
            # bool is an int subclass, so n_samples(True, 16) would give 16.
            raise ValueError(f"{name} must be an integer, got {type(val).__name__}")
```
→
```python
            # bool is an int subclass, so n_samples(True, 16) would give 16.
            raise TypeError(f"{name} must be an integer, got {type(val).__name__}")
```

`delta_F` reaches `_require_even_positive_int`, so its "Raises" section changes too:

```python
    Raises
    ------
    ValueError
        If ``Nf`` is not a positive even integer, if ``dt`` is not positive
        and finite, or if
```
→
```python
    Raises
    ------
    TypeError
        If ``Nf`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` is not positive and even, if ``dt`` is not positive
        and finite, or if
```

(The `\\cdot` is as it stands in the non-raw docstring; copy it unchanged.)

- [ ] **Step 6: `noise.py`, `filter.py` and the transform "Raises" sections**

`noise.colored_noise`, the "Raises" section:

```python
    Raises
    ------
    ValueError
        If ``N`` is not a positive even integer, or ``dt`` is not positive and
        finite.  Both are checked before the PSD is evaluated or ``rng`` is
        drawn from.
```
→
```python
    Raises
    ------
    TypeError
        If ``N`` is not an integer (``bool`` included), or ``dt`` is not a real
        number.
    ValueError
        If ``N`` is not positive and even, or ``dt`` is not positive and
        finite.  Both are checked before the PSD is evaluated or ``rng`` is
        drawn from.
```
(the rest of that paragraph, from "An odd ``N`` has no Nyquist bin" on, stays), and the check:

```python
        raise ValueError(f"N must be an integer, got {type(N).__name__}")
```
→
```python
        raise TypeError(f"N must be an integer, got {type(N).__name__}")
```

`filter.nu_d`, from "Returns" to the end of the check:

```python
    Returns
    -------
    ndarray
        $I_x(d, d)$ evaluated element-wise.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson as wdm
    >>> wdm.filter.nu_d(np.array([0.0, 0.5, 1.0]))
    array([0. , 0.5, 1. ])
    """
    if not isinstance(d, (int, np.integer)) or d < 1:
        # betainc(0, 0, x) is NaN, so a bad d silently yields a NaN filter.
        raise ValueError(f"d must be an integer >= 1, got {d!r}")
```
→
```python
    Returns
    -------
    ndarray
        $I_x(d, d)$ evaluated element-wise.

    Raises
    ------
    TypeError
        If ``d`` is not an integer (``bool`` included).
    ValueError
        If ``d < 1``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson as wdm
    >>> wdm.filter.nu_d(np.array([0.0, 0.5, 1.0]))
    array([0. , 0.5, 1. ])
    """
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise TypeError(f"d must be an integer, got {type(d).__name__}")
    if d < 1:
        # betainc(0, 0, x) is NaN, so a bad d silently yields a NaN filter.
        raise ValueError(f"d must be >= 1, got {d}")
```

`filter.phi_centered`, the "Raises" section:

```python
    Raises
    ------
    ValueError
        If the grid or ``dt`` is invalid, or if
```
→
```python
    Raises
    ------
    TypeError
        If ``Nf``, ``Nt`` or ``d`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid or ``dt`` is invalid, or if
```

`wilson/transform.py`: three "Raises" sections name the grid check. Each gains the same `TypeError` entry.

`forward`:

```python
    Raises
    ------
    ValueError
        If ``Nf`` or ``Nt`` is not a positive even integer, if ``x_rfft``
```
→
```python
    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` or ``Nt`` is not positive and even, if ``x_rfft``
```

`inverse`:

```python
    Raises
    ------
    ValueError
        If the grid is invalid, if ``wmn`` is complex or wrongly shaped, or
```
→
```python
    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if ``wmn`` is complex or wrongly shaped, or
```

`forward_time`:

```python
    Raises
    ------
    ValueError
        If the grid is invalid, if ``x`` does not have ``Nf*Nt`` samples, or
```
→
```python
    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if ``x`` does not have ``Nf*Nt`` samples, or
```

Run: `grep -rn "raise ValueError(f\"{name} must be an integer\|raise ValueError(f\"N must be an integer\|must be an integer >= 1" src/domainia/wilson`
Expected: no output.

- [ ] **Step 7: Run the tests to see them pass**

Run: `$PY -m pytest -q tests/wilson/test_wrong_kinds.py tests/wilson/test_noise_stationary.py tests/wilson/vendored/test_filter.py`
Expected: `88 passed`, no warnings.

- [ ] **Step 8: The wilson suite, its doctests, and the frozen oracle**

Run: `$PY -m pytest -q tests/wilson src/domainia/wilson`
Expected: all pass, 3 skipped (the `DOMAINIA_ORACLE_ROOT` skips), 0 errors. This includes the new `validate_grid` and `validate_dt` doctests. A failure here is a wrong-kind pin Step 1 did not find: move it to `TypeError` as in Step 3.

Run: `git diff --stat -- tests/wilson/oracle && $PY -m pytest -q tests/wilson/test_oracle_pins.py`
Expected: no diff lines, then `6 passed`.

- [ ] **Step 9: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 0 errors.

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass, 0 errors.

- [ ] **Step 10: Commit** (sandbox disabled)

```bash
git add src/domainia/wilson/grid.py src/domainia/wilson/noise.py src/domainia/wilson/filter.py src/domainia/wilson/transform.py tests/wilson/test_wrong_kinds.py tests/wilson/test_noise_stationary.py tests/wilson/vendored/test_filter.py
git commit -m "domainia.wilson: wrong kinds raise TypeError, wrong values ValueError

_require_int, validate_grid, n_samples and colored_noise's N raise TypeError
for a non-integer or bool, with the same message. nu_d splits its kind check
(TypeError) from its range check (ValueError: d must be >= 1). validate_dt
refuses None, bool, str and complex dt instead of letting float() accept
True and \"0.5\". The Raises sections say which is which.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: The smaller fixes (spec §5.1, §5.2, §5.4, §5.5)

**Goal:** `psd_from_K` uses a relative threshold, `CellSet` pickles and copies to an equal read-only set, the 2.0 spec's endpoint sentence is accurate, and the sdist leaves out `docs/superpowers`. Each fix is its own commit.

**Files:**
- Modify: `src/domainia/wilson/stationary.py` (`psd_from_K`)
- Modify: `src/domainia/domains.py` (`CellSet.__reduce__`)
- Modify: `docs/superpowers/specs/2026-10-31-domainia-general-library-design.md` (the line-229 sentence)
- Modify: `pyproject.toml` (`[tool.hatch.build.targets.sdist]`)
- Test: `tests/wilson/test_noise_stationary.py` (next to `test_psd_round_trips`), `tests/test_dom_domains.py`, `tests/test_package.py`

**Acceptance Criteria:**
- [ ] At `dt = 1e-33` with a flat PSD, `psd_from_K(build_K_full_active(…))` recovers the PSD as it does at `dt = 1` (the test fails on the old code, where every bin is dropped). The Meyer entry `stationary.psd_from_K` stays bit-exact.
- [ ] `pickle`, `copy.copy` and `copy.deepcopy` of a `CellSet` give an equal set with an equal hash and read-only `m_array`/`n_array`, and the copy does not share the original's arrays.
- [ ] The 2.0 spec reads "`transform`'s outputs make the endpoints exactly real" (the sentence wraps over lines 229–230; only the line-229 fragment changes). This doc-only commit is checked with `git diff --stat`.
- [ ] A `tomllib` test asserts `docs/superpowers` is in the sdist `exclude`.
- [ ] Four commits, one per fix. The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors

**Steps:**

Assumes the tree after Task 11. Earlier tasks that touch these files: Tasks 1 and 4 edit `src/domainia/domains.py` (`FourierDomain`, `WDMDomain`) and `tests/test_dom_domains.py` (`test_fourier_domain`, `test_wdm_domain_and_channels`), and Task 8 edits `tests/test_package.py` (`PUBLIC`, the lazy-import check). The anchors below sit outside those edits. Four fixes, four commits.

#### Fix 1: `psd_from_K` uses a relative threshold (spec §5.1)

- [ ] **Step 1: Grep the threshold and its tests**

Run: `grep -n "eps" src/domainia/wilson/stationary.py`
Expected: only inside `psd_from_K`: the `eps = 1e-30` line and its five uses (three `np.where(phi2[...] > eps, phi2[...], 1.0)` divisions and `usable = wgt > eps`). No other function in the module reads `eps`.

Run: `grep -rln "psd_from_K" tests`
Expected: `tests/wilson/meyer_cases.py` (the bit-exact baseline case `stationary.psd_from_K`), `tests/wilson/test_noise_stationary.py` (the oracle comparison and `test_psd_round_trips`, at `dt` 0.5 to 2^-10), `tests/wilson/test_window_threading.py` (round trip with other windows). All of them run at `dt` where `max |phi|^2 = 2 dt Nf` is far above `1e-30 · max |phi|^2`, so none of them should move.

- [ ] **Step 2: Write the failing test**

In `tests/wilson/test_noise_stationary.py`, after `test_psd_round_trips` (its last two lines are shown, kept as they are):

```python
    assert np.max(np.abs(stat.psd_from_K(K, Nf, Nt, dt) / S - 1)) < 1e-12
    assert np.max(np.abs(stat.psd_from_lambda_blocks(d, o, n, Nf, Nt, dt) / S - 1)) < 1e-12
```
→
```python
    assert np.max(np.abs(stat.psd_from_K(K, Nf, Nt, dt) / S - 1)) < 1e-12
    assert np.max(np.abs(stat.psd_from_lambda_blocks(d, o, n, Nf, Nt, dt) / S - 1)) < 1e-12


@pytest.mark.parametrize("dt", [1.0, 1e-33])
def test_psd_from_K_drops_bins_relative_to_the_window_peak(dt):
    """|phi|^2 peaks at 1/dF = 2 dt Nf, which is 1.6e-32 at dt = 1e-33: below the old
    absolute threshold 1e-30, so every bin was dropped and psd_from_K returned zeros."""
    Nf, Nt = 8, 16
    K, _ = stat.build_K_full_active(Nf, Nt, dt, lambda f: 1e-3 * np.ones_like(f))
    got = stat.psd_from_K(K, Nf, Nt, dt)
    assert got.shape == (Nf * Nt // 2 + 1,)
    assert np.max(np.abs(got / 1e-3 - 1)) < 1e-12
```

- [ ] **Step 3: Run it to see it fail**

Run: `$PY -m pytest -q tests/wilson/test_noise_stationary.py -k drops_bins`
Expected: `1 failed, 1 passed`. `[1.0]` passes (error 2.2e-16). `[1e-33]` fails with `AssertionError: assert np.float64(1.0) < 1e-12`: `got` is all zeros.

- [ ] **Step 4: Make the threshold relative**

In `src/domainia/wilson/stationary.py`, `psd_from_K` (`eps` moves below `phi2`, which it now needs):

```python
    phi_c = sample_window(window, Nf, Nt, dt)
    eps = 1e-30

    # One batched FFT for every band, then a single vectorised
    # "largest window value wins" reduction over all (bin, band) candidates.
    V_all = (np.fft.fft(np.asarray(K_diag), axis=-1) / Nt).real  # (Nf+1, Nt)
    phi2 = phi_c * phi_c
```
→
```python
    phi_c = sample_window(window, Nf, Nt, dt)

    # One batched FFT for every band, then a single vectorised
    # "largest window value wins" reduction over all (bin, band) candidates.
    V_all = (np.fft.fft(np.asarray(K_diag), axis=-1) / Nt).real  # (Nf+1, Nt)
    phi2 = phi_c * phi_c
    # Window values at or below eps count as zero.  Relative to the peak, because
    # |phi|^2 scales with 1/dF = 2 dt Nf: an absolute 1e-30 dropped every bin once
    # 2 dt Nf fell below it (dt = 1e-33 gave an all-zero PSD).
    eps = 1e-30 * np.max(phi2)
```

- [ ] **Step 5: Run it to see it pass; the baseline entry stays bit-exact**

Run: `$PY -m pytest -q tests/wilson/test_noise_stationary.py -k drops_bins`
Expected: `2 passed`

Run: `$PY -m pytest -q tests/wilson/test_meyer_baseline.py && $PY tests/wilson/meyer_cases.py --check`
Expected: all pass, then `baseline reproduces: 1504 arrays`. `stationary.psd_from_K` is not in `RESCALED`, so it is compared with `np.array_equal`: the Meyer window's `|phi|^2` is either exactly 0 (outside `3 dF/4`) or far above `1e-30 · max`, so the set of kept bins is unchanged.

- [ ] **Step 6: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 0 errors.

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass, 0 errors.

- [ ] **Step 7: Commit** (sandbox disabled)

```bash
git add src/domainia/wilson/stationary.py tests/wilson/test_noise_stationary.py
git commit -m "psd_from_K: drop window bins relative to the window peak

eps = 1e-30 * max(phi^2) instead of an absolute 1e-30. |phi|^2 peaks at
2 dt Nf, so at dt = 1e-33 every bin was dropped and the recovered PSD was all
zeros. The Meyer baseline entry stays bit-exact.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

#### Fix 2: `CellSet.__reduce__` (spec §5.2)

- [ ] **Step 8: Grep the copy and pickle users**

Run: `grep -rn "pickle\|deepcopy\|copy\.copy\|__reduce" src/domainia tests | grep -v "tests/wilson/oracle"`
Expected:
- `tests/test_dom_windows.py:7, 46-47, 85-90, 208`: `WDMDomain` deepcopy/pickle round trips. A `WDMDomain` holds a `CellSet`, so these go through the new `__reduce__` and must keep passing.
- `tests/wilson/test_windows.py:6, 336-338` and `src/domainia/wilson/windows.py:198, 244`: windows, unaffected.

No `__reduce__` is defined in `src/` yet.

- [ ] **Step 9: Write the failing test**

In `tests/test_dom_domains.py`, the imports:

```python
from __future__ import annotations

import numpy as np
import pytest
```
→
```python
from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
```
(If an earlier task already added a standard-library import there, insert `copy` and `pickle` in alphabetical order among them.)

After `test_identity_is_value_based_and_cheap` (its last two lines are shown, kept as they are):

```python
    assert m_in.flags.writeable and n_in.flags.writeable              # caller's arrays untouched
    assert not c.m_array.flags.writeable and c.m == (1, 1, 2, 2)
```
→
```python
    assert m_in.flags.writeable and n_in.flags.writeable              # caller's arrays untouched
    assert not c.m_array.flags.writeable and c.m == (1, 1, 2, 2)


@pytest.mark.parametrize("roundtrip", [lambda c: pickle.loads(pickle.dumps(c)), copy.copy,
                                       copy.deepcopy], ids=["pickle", "copy", "deepcopy"])
def test_a_cell_set_survives_pickle_and_copy_as_an_equal_read_only_set(roundtrip):
    """``__reduce__`` rebuilds from ``(grid, m, n)``.  Before it, pickle and deepcopy gave
    writeable index arrays and ``copy.copy`` shared the original's."""
    cells = CellSet.all_active(wdm.Grid(8, 16, 0.5)).drop_columns(2, 1)
    back = roundtrip(cells)
    assert back is not cells and back == cells and hash(back) == hash(cells)
    for got, orig in ((back.m_array, cells.m_array), (back.n_array, cells.n_array)):
        assert not got.flags.writeable
        assert np.array_equal(got, orig) and not np.shares_memory(got, orig)
    assert np.array_equal(back.packed_positions(), cells.packed_positions())
```

- [ ] **Step 10: Run it to see it fail**

Run: `$PY -m pytest -q tests/test_dom_domains.py -k survives`
Expected: `3 failed`. `[pickle]` and `[deepcopy]` fail on `assert not got.flags.writeable` (the default dataclass pickling restores `_m`/`_n` as writeable arrays). `[copy]` fails on `not np.shares_memory(got, orig)` (the shallow copy shares the original's arrays).

- [ ] **Step 11: Add `__reduce__`**

In `src/domainia/domains.py`, `CellSet`:

```python
    # -- identity -------------------------------------------------------------
    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]
```
→
```python
    # -- identity -------------------------------------------------------------
    def __reduce__(self) -> tuple:
        """Rebuild from ``(grid, m, n)``: a pickled or copied set runs ``__post_init__`` again,
        so it gets its own read-only ``m_array``/``n_array`` and recomputes its hash."""
        return type(self), (self.grid, self.m, self.n)

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]
```

`copy.copy` and `copy.deepcopy` use `__reduce_ex__`, which calls an overridden `__reduce__`, so one method serves all three.

- [ ] **Step 12: Run the tests to see them pass**

Run: `$PY -m pytest -q tests/test_dom_domains.py tests/test_dom_windows.py`
Expected: all pass, including the three new cases and `test_the_domain_survives_copy_and_pickle_with_its_window[deepcopy]` / `[pickle]`.

- [ ] **Step 13: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 0 errors.

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass, 0 errors.

- [ ] **Step 14: Commit** (sandbox disabled)

```bash
git add src/domainia/domains.py tests/test_dom_domains.py
git commit -m "CellSet.__reduce__: pickle and copy rebuild from (grid, m, n)

A round trip runs __post_init__ again, so pickle, copy.copy and copy.deepcopy
give an equal set with an equal hash and its own read-only m_array/n_array.
Before, pickle and deepcopy returned writeable arrays and copy.copy shared
the original's.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

#### Fix 3: the 2.0 spec's endpoint sentence (spec §5.4)

- [ ] **Step 15: Grep and edit**

Run: `grep -rn "own outputs make the endpoints" docs/superpowers/specs`
Expected: `docs/superpowers/specs/2026-10-31-domainia-general-library-design.md:229`, and the 2026-11-04 spec's §5.4 item, which quotes it and stays as it is.

In `docs/superpowers/specs/2026-10-31-domainia-general-library-design.md` (the sentence runs onto the next line, "real, so its round trips always pass."; only this fragment changes):

```
domainia's own outputs make the endpoints exactly
```
→
```
`transform`'s outputs make the endpoints exactly
```

Run: `grep -n "outputs make the endpoints exactly" docs/superpowers/specs/2026-10-31-domainia-general-library-design.md`
Expected: ``229:  `block_state.check_rfft_endpoints`. `transform`'s outputs make the endpoints exactly``

Run: `git diff --stat`
Expected: the spec's line, ending with `1 file changed, 1 insertion(+), 1 deletion(-)`. Nothing in `src/` or `tests/` changed, so the suites from Step 13 still hold.

- [ ] **Step 16: Commit** (sandbox disabled)

```bash
git add docs/superpowers/specs/2026-10-31-domainia-general-library-design.md
git commit -m "2.0 spec: transform's outputs, not all of domainia's, have exactly real endpoints

translate.transform is what canonicalises its outputs (exactly real endpoint
bins, exactly zero inactive cells); the sentence now names it instead of
claiming it for every domainia output.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

#### Fix 4: the sdist leaves out `docs/superpowers` (spec §5.5)

- [ ] **Step 17: Grep the build config**

Run: `grep -n "tool.hatch" pyproject.toml`
Expected: one hit, `[tool.hatch.build.targets.wheel]`. There is no sdist table yet, so hatchling's default sdist includes `docs/superpowers`.

- [ ] **Step 18: Write the failing test**

In `tests/test_package.py`, after `test_the_version_is_2_0_0_everywhere` (`tomllib` and `REPO` are already imported and defined there):

```python
def test_the_version_is_2_0_0_everywhere():
    assert domainia.__version__ == "2.0.0"
    assert tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"] == "2.0.0"
```
→
```python
def test_the_version_is_2_0_0_everywhere():
    assert domainia.__version__ == "2.0.0"
    assert tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"] == "2.0.0"


def test_the_sdist_leaves_out_the_working_documents():
    """The specs and plans under docs/superpowers stay out of the sdist.  hatchling is not
    installed in the test env, so this reads the build config rather than building."""
    hatch = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["hatch"]
    assert "docs/superpowers" in hatch["build"]["targets"]["sdist"]["exclude"]
```

- [ ] **Step 19: Run it to see it fail**

Run: `$PY -m pytest -q tests/test_package.py -k sdist`
Expected: `1 failed`, `KeyError: 'sdist'`

- [ ] **Step 20: Add the sdist table**

In `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/domainia"]
```
→
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/domainia"]

[tool.hatch.build.targets.sdist]
# The design specs and implementation plans are working documents, not part of the release.
exclude = ["docs/superpowers"]
```

- [ ] **Step 21: Run the tests to see them pass**

Run: `$PY -m pytest -q tests/test_package.py`
Expected: all pass, 0 errors.

- [ ] **Step 22: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 0 errors (3 skipped: the `DOMAINIA_ORACLE_ROOT` skips).

Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass, 0 errors.

- [ ] **Step 23: Commit** (sandbox disabled)

```bash
git add pyproject.toml tests/test_package.py
git commit -m "Leave docs/superpowers out of the sdist

[tool.hatch.build.targets.sdist] exclude = [\"docs/superpowers\"]: the specs and
plans are working documents. hatchling is not in the test env, so a tomllib
test pins the exclude.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: The `built_for` guard on constant-carrying maps (spec §5.3)

**Goal:** A map whose constants were built for one `src`/`dst` (and parameters) refuses to run with another, on every construction route, so `dataclasses.replace(map, dst=…)` can no longer reuse stale constants.

**Files:**
- Modify: `src/domainia/maps/time.py` (`TimeToWDM`, `WDMToTime` and its `adjoint`, `TimeToFourier`)
- Modify: `src/domainia/maps/base.py` (`Restrict`; the module docstring's `dataclasses.replace` warning)
- Modify: `src/domainia/maps/fourier.py` (`FourierZeroPad`, `FourierCrop`, `FourierToWDM`, `FourierToWDM.after`)
- Modify: `src/domainia/maps/base.py` also gains the shared helper `_built_for(m, key, *, given)`, which `maps/time.py` and `maps/fourier.py` import
- Modify (only if Step 1 finds it needed): `tests/test_dom_maps_base.py`, if Task 4's version of `test_restrict_checks_its_domains_even_when_pos_is_given` expects a same-grid, same-window `dataclasses.replace(R, src/dst=…)` to succeed
- Create: `tests/test_dom_built_for.py`

**Acceptance Criteria:**
- [ ] For each of the seven maps: a normal build sets `built_for` to the key in the Interface contract. `dataclasses.replace` that changes a key field raises `ValueError("<Map>: constants built for another src/dst: construct a new map")`. `replace` that changes nothing keeps working. Constants supplied without `built_for` raise `ValueError("<Map>: constants given without built_for: construct the map from its fields")`.
- [ ] `after()` maps carry the key of the pair they were built for, and still equal `Compose`.
- [ ] `WDMToTime.adjoint` works (it passes `built_for`). `grep` over `src/` for constant-passing constructor calls finds no other site without `built_for`.
- [ ] Pytree flatten/unflatten round-trips each map, and `jax.jit` over `apply` works (`jax`-marked).
- [ ] `tests/test_dom_windows.py`'s `Restrict(..., pos=…)` window-mismatch test keeps its message.
- [ ] `built_for` is not in any map's `repr`.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `$PY -m pytest -q tests/test_dom_built_for.py` → all pass; then `$PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 12.

The maps are in their post-Task-4 form. `FourierZeroPad` has the fields `(src, dst, *, A, B)` and the pytree `meta=("src", "dst")`. `FourierCrop` has no `scale = 1.0` class attribute. `require_unit_scale` is gone everywhere, and `FourierToWDM.after()` multiplies by no `inner.scale`. **Do not reintroduce `scale` anywhere.** If an old text below does not match exactly, Tasks 1–4 have reworded a neighbouring line. Make the same insertion at the same place: the `built_for` field goes after the last constant field, and the `_built_for(...)` call goes directly before the map's `if <constant> is _BUILD:` test.

- [ ] **Step 1: Grep the construction routes**

Run: `grep -rnE "\b(TimeToWDM|WDMToTime|Restrict|FourierZeroPad|FourierCrop|FourierToWDM|TimeToFourier)\(" src | grep -v ":class "`
Expected (live tree, unchanged by Tasks 1–12 unless Task 5 or 8 added a map construction):
- `src/domainia/maps/time.py`, in `WDMToTime.adjoint`: `fwd = TimeToWDM(self.dst, self.src, m_idx=self.m_idx, n_idx=self.n_idx)`. It passes constants, so Step 6 adds `built_for`.
- `src/domainia/maps/fourier.py`, at the end of `FourierToWDM.after`: `return FourierToWDM(inner.src, self.dst, "dense", lo=0, width=inner.src.n, C=C[geo["sel"]])` (live text). It passes constants, so Step 7 adds `built_for`.
- Any other hit that passes no constant keyword (`m_idx=`, `n_idx=`, `pos=`, `A=`, `B=`, `lo=`, `width=`, `C=`, or `dst=` on `TimeToFourier`) builds its constants and needs nothing. A hit that passes one needs `built_for=` as in Steps 6–7. Step 9's checker lists them all.

Run: `grep -rn "dataclasses\.replace(" src tests`
Expected hits and what each does under the guard:
- `src/domainia/maps/base.py` `on_backend`: replaces the constants only, and `built_for` travels unchanged. It keeps working, and every `LinearMap.on` uses it.
- `src/domainia/data.py` (`_Data.replace`) and `src/domainia/covariance/blocktri.py`: these are not maps, so the guard does not apply.
- `tests/test_dom_windows.py` `test_restrict_checks_the_windows_on_every_construction_route`: `replace(R, dst=alt_cut)` and `replace(R, src=WDMDomain(CELLS, window=ALT_WINDOW))` still raise the window message, because the window check runs before the guard. `replace(R)` keeps the key and still works. No edit.
- `tests/test_dom_maps_base.py` `test_restrict_checks_its_domains_even_when_pos_is_given` (as Task 4 left it): `replace(R)` still works. A `replace(R, src=…)` or `replace(R, dst=…)` that keeps the grid and the window now raises `"built for another src/dst"`, where it used to pass. If Task 4 wrote such a call expecting success, change it to `pytest.raises(ValueError, match="built for another src/dst")`. A call whose window or grid differs keeps its message.

Run: `grep -rnE "Restrict\(.*pos=|\"pos\": R\.pos" tests`
Expected: `tests/test_dom_windows.py` (`Restrict(meyer, alt_cut, pos=…)`: the window mismatch raises first, unchanged), and `tests/test_dom_maps_base.py` (`for kw in ({}, {"pos": R.pos})` with another grid: the grid check raises first, unchanged). No edit.

- [ ] **Step 2: Write the failing test `tests/test_dom_built_for.py`**

```python
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
```

- [ ] **Step 3: Run it and see it fail**

Run: `$PY -m pytest -q tests/test_dom_built_for.py`
Expected: `46 failed, 9 passed`. The failures are `AttributeError: … has no attribute 'built_for'`, `TypeError` from `dataclasses.replace` (no field `built_for`), and `Failed: DID NOT RAISE <class 'ValueError'>` for the 17 key changes, which pass silently today with stale constants. The 9 passes are already true and guard against regressions: the 7 repr cases, the Restrict validation order and `WDMToTime.adjoint`.

- [ ] **Step 4: The guard helper and the module docstring (`src/domainia/maps/base.py`)**

Directly after `_BUILD: Any = _Build()`, add:

```python


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
```

In the module docstring, replace:

```
(the ``_BUILD`` sentinel); ``jax.tree_util`` rebuilds a map from its leaves without
recomputing anything.  ``dataclasses.replace`` of a map's domain keeps the constants built
for the old domain; build a new map instead.
```
with:
```
(the ``_BUILD`` sentinel), and records the fields they derive from in ``built_for`` (static
meta, left out of the ``repr``).  ``jax.tree_util`` rebuilds a map from its leaves without
recomputing anything.  Constants given to a constructor (``dataclasses.replace``, ``on``,
``tree_unflatten``) must come with a ``built_for`` equal to the map's own fields:
``dataclasses.replace(map, dst=...)`` raises ``ValueError`` instead of reusing constants
built for the old domain.
```

- [ ] **Step 5: `Restrict` (`src/domainia/maps/base.py`)**

Decorator: `@pytree_dataclass(data=("pos",), meta=("src", "dst"))` → `@pytree_dataclass(data=("pos",), meta=("src", "dst", "built_for"))`.

The end of the docstring and the fields:

```python
    ``dataclasses.replace``); only the subset positions ``pos`` are built once."""

    src: WDMDomain
    dst: WDMDomain
    pos: Any = field(default=_BUILD, kw_only=True)
```
→
```python
    ``dataclasses.replace``); only the subset positions ``pos`` are built once.  The
    ``built_for`` guard runs after these checks, so a window or grid mismatch keeps its
    message."""

    src: WDMDomain
    dst: WDMDomain
    pos: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)
```

In `__post_init__`, the guard goes after the window and grid checks:

```python
                             f"({self.dst.cells.grid} vs {self.src.cells.grid})")
        if self.pos is _BUILD:
```
→
```python
                             f"({self.dst.cells.grid} vs {self.src.cells.grid})")
        _built_for(self, (self.src, self.dst), given=self.pos is not _BUILD)
        if self.pos is _BUILD:
```

- [ ] **Step 6: `TimeToWDM`, `WDMToTime`, `TimeToFourier` (`src/domainia/maps/time.py`)**

Import: `from .base import _BUILD, LinearMap, _last_axis` → `from .base import _BUILD, LinearMap, _built_for, _last_axis`.

Both decorators of `TimeToWDM` and `WDMToTime` (replace both occurrences):
`@pytree_dataclass(data=("m_idx", "n_idx"), meta=("src", "dst"))` → `@pytree_dataclass(data=("m_idx", "n_idx"), meta=("src", "dst", "built_for"))`.

The field and the guard. The old text occurs twice, in `TimeToWDM` and in `WDMToTime`. Replace both: the key is `(self.src, self.dst)` for each.

```python
    n_idx: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.m_idx is _BUILD:
```
→
```python
    n_idx: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.dst), given=self.m_idx is not _BUILD)
        if self.m_idx is _BUILD:
```

`WDMToTime.adjoint`:

```python
        fwd = TimeToWDM(self.dst, self.src, m_idx=self.m_idx, n_idx=self.n_idx)
```
→
```python
        fwd = TimeToWDM(self.dst, self.src, m_idx=self.m_idx, n_idx=self.n_idx,
                        built_for=(self.dst, self.src))
```

`TimeToFourier` (its derived `dst` is the constant, built from `(src, fmin)`):
`@pytree_dataclass(data=(), meta=("src", "dst", "fmin"))` → `@pytree_dataclass(data=(), meta=("src", "dst", "fmin", "built_for"))`, and

```python
    dst: Any = field(default=_BUILD, kw_only=True)

    kind = "analysis"

    def __post_init__(self) -> None:
        if self.dst is _BUILD:
```
→
```python
    dst: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    kind = "analysis"

    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.fmin), given=self.dst is not _BUILD)
        if self.dst is _BUILD:
```

- [ ] **Step 7: `FourierZeroPad`, `FourierCrop`, `FourierToWDM` (`src/domainia/maps/fourier.py`)**

Import: `from .base import _BUILD, LinearMap, _last_axis` → `from .base import _BUILD, LinearMap, _built_for, _last_axis`.

`FourierZeroPad` (Task 3 removed its `scale` field: there is no `scale` here):
`@pytree_dataclass(data=("A", "B"), meta=("src", "dst"))` → `@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "built_for"))`, and

```python
    B: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.A is _BUILD:
```
→
```python
    B: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.dst), given=self.A is not _BUILD)
        if self.A is _BUILD:
```

`FourierCrop` (Task 3 removed its `scale = 1.0` class attribute):
`@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "start"))` → `@pytree_dataclass(data=("A", "B"), meta=("src", "dst", "start", "built_for"))`, and

```python
    B: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", operator.index(self.start))
        if self.A is _BUILD:
```
→
```python
    B: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", operator.index(self.start))
        _built_for(self, (self.src, self.dst, self.start), given=self.A is not _BUILD)
        if self.A is _BUILD:
```
(`start` is normalised first, so the key holds the int.)

`FourierToWDM`:
`@pytree_dataclass(data=("C",), meta=("src", "dst", "impl", "lo", "width", "restrict"))` →
```python
@pytree_dataclass(data=("C",), meta=("src", "dst", "impl", "lo", "width", "restrict",
                                     "built_for"))
```
and

```python
    C: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.C is not _BUILD:
            return
```
→
```python
    C: Any = field(default=_BUILD, kw_only=True)
    built_for: Any = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.dst, self.impl, self.restrict),
                   given=self.C is not _BUILD)
        if self.C is not _BUILD:
            return
```

`FourierToWDM.after`: its final `return FourierToWDM(inner.src, self.dst, "dense", lo=0, width=inner.src.n, C=C[geo["sel"]])` (live text) gains `built_for`, the key of the pair it built. Keep the `C=` argument exactly as Tasks 2–3 left it. On the live tree:

```python
        return FourierToWDM(inner.src, self.dst, "dense", lo=0, width=inner.src.n, C=C[geo["sel"]])
```
→
```python
        return FourierToWDM(inner.src, self.dst, "dense", lo=0, width=inner.src.n, C=C[geo["sel"]],
                            built_for=(inner.src, self.dst, "dense", False))
```

- [ ] **Step 8: Run the new test**

Run: `$PY -m pytest -q tests/test_dom_built_for.py`
Expected: `55 passed`. In the NumPy lane (`WDM_TESTS_HIDE_JAX=1`), `48 passed, 7 skipped`.

- [ ] **Step 9: Check every constant-passing construction in `src/`, and the map pins**

Save as `$TMPDIR/check_built_for.py`:

```python
"""List every call in src/ to a guarded map that passes constants, and whether it passes built_for."""
import ast
import pathlib

CONSTANTS = {"TimeToWDM": {"m_idx", "n_idx"}, "WDMToTime": {"m_idx", "n_idx"}, "Restrict": {"pos"},
             "FourierZeroPad": {"A", "B"}, "FourierCrop": {"A", "B"},
             "FourierToWDM": {"lo", "width", "C"}, "TimeToFourier": {"dst"}}
for path in sorted(pathlib.Path("src").rglob("*.py")):
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        kws = {k.arg for k in node.keywords}
        if name in CONSTANTS and kws & CONSTANTS[name]:
            print(f"{path}:{node.lineno}: {name}(...) passes {sorted(kws & CONSTANTS[name])}, "
                  f"built_for {'yes' if 'built_for' in kws else 'MISSING'}")
```

Run: `$PY $TMPDIR/check_built_for.py`
Expected: two lines, both `built_for yes`: `src/domainia/maps/fourier.py:…: FourierToWDM(...) passes ['C', 'lo', 'width']` and `src/domainia/maps/time.py:…: TimeToWDM(...) passes ['m_idx', 'n_idx']`. Before Steps 6–7 both said `MISSING`. Any other `MISSING` line is a site Step 1 found: pass it the key of the map whose constants it reuses.

Run: `$PY -m pytest -q tests/test_dom_windows.py tests/test_dom_maps_base.py tests/test_dom_maps_wdm.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_fourier_time.py tests/test_dom_maps_dirichlet.py`
Expected: all pass. `tests/test_dom_windows.py`'s `Restrict(..., pos=…)` window-mismatch test still passes unchanged, and so does its jitted `test_the_window_is_static_under_jit` (maps rebuilt from leaves).

- [ ] **Step 10: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 3 skipped. The count is Task 12's plus 55 (on the pre-Task-0 tree the same change took 1460 to 1515 passed).
Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass.

- [ ] **Step 11: Commit** (sandbox disabled)

```bash
git add src/domainia/maps/base.py src/domainia/maps/time.py src/domainia/maps/fourier.py tests/test_dom_built_for.py
git commit -m "built_for guard: maps refuse constants built for another src/dst

TimeToWDM, WDMToTime, Restrict, FourierZeroPad, FourierCrop, FourierToWDM and
TimeToFourier record the fields their constants derive from (static meta, not in
the repr). dataclasses.replace changing one raises ValueError; constants without
built_for raise; after(), WDMToTime.adjoint, on() and tree_unflatten pass it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
If Step 1 required an edit to `tests/test_dom_maps_base.py`, add it to the `git add`.

---

### Task 14: Dead code and duplicates (spec §6)

**Goal:** The unread items in spec §6 are removed, after a grep for each, and `_oracle_paths` is defined once.

**Files:**
- Modify: `src/domainia/covariance/base.py` (`batchable`, `complete_constants`), `src/domainia/spectra.py` (`GenericPSD.capabilities`), `src/domainia/variance.py` (`BandAveragedVariance.t`, `MidpointVariance.t`), `src/domainia/covariance/evolutionary.py` and `src/domainia/covariance/__init__.py` (`keep_mask`)
- Modify: `tests/test_dom_spectra.py`, `tests/test_dom_cov_api.py`, `tests/test_dom_evolutionary_cov.py`, `tests/test_dom_windows.py` (`test_the_window_is_static_under_jit` pins the variance's `t` leaf), `tests/wilson/conftest.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_grid_bands.py`

**Acceptance Criteria:**
- [ ] Before each removal, a recorded grep shows no reader in `src/`, `tests/` or `README.md` other than a test that only pins the item. Any item with a real reader is kept and listed in the errata with the reader.
- [ ] `EvolutionaryCovariance` calls `cells.keep_mask()`, and `grep -rn "keep_mask(cells)\|import.*keep_mask\|\"keep_mask\"" src tests` prints nothing.
- [ ] `BandAveragedVariance.on()` and `MidpointVariance.on()` still exist, and `as_evaluator` still recognises both classes.
- [ ] `_oracle_paths` (with `ORACLE_ROOT_ENV`) is defined once, in `tests/wilson/conftest.py`, and both former definers use it. The `DOMAINIA_ORACLE_ROOT` skips still appear (3 skips, now reported from `tests/wilson/conftest.py`, same messages). Imports left unused by the removals (`ClassVar` in `spectra.py`, `_BUILD` in `variance.py`) go too.
- [ ] The full suite and the NumPy lane pass. The Meyer baseline is unchanged from Task 3.

**Verify:** `$PY -m pytest -q` → all pass, 3 skipped, 0 errors

**Steps:**

Assumes the tree after Task 13.

`variance.py` and `EvolutionaryCovariance.__init__` are in their Task 9 form (they accept a `CellSet` or a `WDMDomain`, `window=None`). The edits below touch only the `t` leaf, `on()`, the decorators and the `keep_mask` call. Task 9 does not change any of them. If Task 9 reworded a neighbouring line, make the same deletion in place.

- [ ] **Step 1: Grep every item for readers** (spec §6: remove only what has no reader outside a pin)

Run each command and compare with the expected hits (live tree; Tasks 1–13 add no reader):

1. `grep -rn "batchable" src tests README.md`
   Expected: `src/domainia/covariance/base.py`, the `Covariance.batchable` property and its own recursion `all(c.batchable for c in self.children().values())`, nothing else. No class overrides it and nothing reads it. **Remove.**
2. `grep -rn "complete_constants" src tests README.md`
   Expected: `src/domainia/covariance/base.py` only: the sentence in the `constants` docstring, the call `own = self.complete_constants(own, be)`, and the definition. `grep -rn "def complete_constants" src` finds only the base class, so no subclass overrides the hook. **Remove all three.**
3. `grep -rn "capabilities" src tests README.md`
   Expected: `src/domainia/spectra.py`, the `GenericPSD.capabilities` ClassVar, and `tests/test_dom_spectra.py`, `assert g.capabilities == {"interp", "direct"}` in `test_generic_psd_binds`, which is a pin and goes with it. **Remove both.** `ClassVar` has no other use in `spectra.py` (`grep -n ClassVar src/domainia/spectra.py` shows only the import and this line), so drop it from the `typing` import.
4. `grep -rnE '"t"|\.t\b' src tests README.md | grep -v "tests/wilson/oracle/\|tests/wilson/vendored/"`
   Expected: `src/domainia/variance.py` (both decorators' `data=("t",)`, the two `if self.t is not _BUILD` lines, the two `object.__setattr__(self, "t", ...)` lines and the two `on_backend(self, be, ("t",))` calls). Also `src/domainia/wilson/evolutionary.py` (`data=(... "t" ...)` of `Eq45Plan`, `plan.t[p]`, `g.t[...]`): those are the plan's own fields and unrelated. `src/domainia/wilson/matrix.py` matches only `s.t.` in a comment. Nothing reads a variance's `t`.
   Run: `grep -rnE "Variance\(.*\)\.on\(" tests`
   Expected: `tests/test_dom_windows.py`, `v = BandAveragedVariance(psd, CELLS, window=ALT_WINDOW).on(JB)` in `test_the_window_is_static_under_jit`. The next line pins the `t` leaf (`assert len(leaves) == 1 and ...`), so it is a pin that changes with the item (Step 6). **Remove the `t` leaves and keep both `on()` methods:** `as_evaluator` (`covariance/base.py`) recognises a variance by `callable(obj) and hasattr(obj, "on")`.
5. `grep -rn "keep_mask" src tests README.md`
   Expected:
   - `src/domainia/covariance/evolutionary.py`: `__all__`, `def keep_mask(cells)`, and the call `keep = keep_mask(cells)` in `EvolutionaryCovariance.__init__`;
   - `src/domainia/covariance/__init__.py`: the import and the `__all__` entry;
   - `tests/test_dom_cov_api.py`: `"keep_mask"` in `EXPORTS`;
   - `tests/test_dom_evolutionary_cov.py`: the import and `test_keep_mask_selects_the_cells_in_band_order`;
   - `src/domainia/domains.py` `CellSet.keep_mask`, and its callers `src/domainia/testing.py`, `tests/test_dom_linalg_blocktri.py`, `tests/test_dom_linalg_jax.py`, `tests/test_dom_testing.py` and `tests/test_dom_domains.py` (`cells.keep_mask()`): this is the replacement, and it stays.

   **Remove the function** and switch the call to `cells.keep_mask()`. Its extra checks (active cells, band order) are already guaranteed by `CellSet.__post_init__`, and `EvolutionaryCovariance` only reaches the call with a `CellSet`.
6. `grep -rn "_oracle_paths\|ORACLE_ROOT_ENV" tests`
   Expected: the same `ORACLE_ROOT_ENV = "DOMAINIA_ORACLE_ROOT"` constant and the same `_oracle_paths` function defined in both `tests/wilson/test_grid_bands.py` and `tests/wilson/test_evolutionary_kernels.py`. The callers are `_gb_reference` (test_grid_bands), and `test_bessel_blocks_are_the_wt_toy_to_rounding` and `test_wdm_dev_ev_oracle_is_the_seed_with_imports_rewritten` (test_evolutionary_kernels). **Move both definitions to `tests/wilson/conftest.py`.**

Any item with a reader beyond these hits is kept and recorded in the plan's Execution errata, with the reader.

- [ ] **Step 2: Write the failing tests** (`tests/test_dom_cov_api.py`)

Imports. The old text is:

```python
from __future__ import annotations

import os
```
→
```python
from __future__ import annotations

import dataclasses
import os
```
and
```python
from domainia.spectra import GenericPSD
```
→
```python
from domainia.spectra import GenericPSD
from domainia.variance import BandAveragedVariance, MidpointVariance
from domainia.wilson.backend import NUMPY
```

`EXPORTS` loses `keep_mask`:

```python
           "QuadForm", "StackFactor", "as_evaluator", "dense_factor", "keep_mask", "quad_form"}
```
→
```python
           "QuadForm", "StackFactor", "as_evaluator", "dense_factor", "quad_form"}
```

Append at the end of the file:

```python


def test_the_unread_hooks_are_gone():
    """``Covariance.batchable``, ``Covariance.complete_constants``, ``GenericPSD.capabilities``
    and the variances' ``t`` leaves had no reader, so they are removed."""
    assert not hasattr(cov.Covariance, "batchable")
    assert not hasattr(cov.Covariance, "complete_constants")
    assert not hasattr(GenericPSD, "capabilities")
    for cls in (BandAveragedVariance, MidpointVariance):
        assert "t" not in {f.name for f in dataclasses.fields(cls)}


@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance], ids=lambda c: c.__name__)
def test_as_evaluator_still_takes_the_variances(cls):
    """``as_evaluator`` recognises a variance by its ``on`` method, which stays."""
    v = cls(FLAT, CELLS)
    ev = cov.as_evaluator(v, channel=0)
    assert ev.host_constants() is v
    nu = np.zeros(1)
    np.testing.assert_array_equal(ev.evaluate(ev.host_constants(), nu, NUMPY), v(nu))
```

(`FLAT` and `CELLS` are the module's existing constants above `test_evolutionary_covariance_refuses_a_layout_with_dense_assembly`.)

- [ ] **Step 3: Run them and see them fail**

Run: `$PY -m pytest -q tests/test_dom_cov_api.py`
Expected: `2 failed, 11 passed`. `test_exports_are_the_documented_names` fails with `AssertionError` (`keep_mask` is still exported), and `test_the_unread_hooks_are_gone` fails with `assert not True` (`batchable`). The two `as_evaluator` cases pass already, and they guard Step 6.

- [ ] **Step 4: `Covariance.batchable` and `complete_constants` (`src/domainia/covariance/base.py`)**

Delete the property (and the blank line after it):

```python
    @property
    def batchable(self) -> bool:
        """Whether a sampler may stack this covariance's noise states into one ``vmap``:
        every child's, by default."""
        return all(c.batchable for c in self.children().values())

```

In `Covariance.constants`, the docstring:

```python
        """The constants on ``be``: built once in host NumPy, cached per backend (JAX:
        ``require_x64()``, then one ``jax.device_put``), then completed on ``be`` by
        :meth:`complete_constants`.  They are cached for the object's lifetime: build a new
        covariance for a new configuration, never mutate one."""
```
→
```python
        """The constants on ``be``: built once in host NumPy, cached per backend (JAX:
        ``require_x64()``, then one ``jax.device_put``).  They are cached for the object's
        lifetime: build a new covariance for a new configuration, never mutate one."""
```
and the body:
```python
                own = jax.device_put(own)
            own = self.complete_constants(own, be)
```
→
```python
                own = jax.device_put(own)
```

Delete the hook (and the blank line after it):

```python
    def complete_constants(self, C: Any, be: Backend) -> Any:
        """Constants derived ON ``be`` from the device-put host constants ``C`` (default:
        none)."""
        return C

```

- [ ] **Step 5: `GenericPSD.capabilities` (`src/domainia/spectra.py`, `tests/test_dom_spectra.py`)**

`src/domainia/spectra.py`:

```python
    fn: Callable[[Any, Any, Any], Any]

    capabilities: ClassVar[frozenset[str]] = frozenset({"interp", "direct"})

```
→
```python
    fn: Callable[[Any, Any, Any], Any]

```
and `from typing import Any, Callable, ClassVar` → `from typing import Any, Callable`.

`tests/test_dom_spectra.py`, in `test_generic_psd_binds`, delete the pin line:

```python
    assert g.capabilities == {"interp", "direct"}
```

- [ ] **Step 6: The variances' `t` leaves (`src/domainia/variance.py`, `tests/test_dom_windows.py`)**

Decorators:
`@pytree_dataclass(data=("t",), meta=("psd", "cells", "window"))` → `@pytree_dataclass(data=(), meta=("psd", "cells", "window"))`, and
`@pytree_dataclass(data=("t",), meta=("psd", "cells"))` → `@pytree_dataclass(data=(), meta=("psd", "cells"))`.

Delete the field line in both classes (two occurrences):

```python
    t: Any = field(default=_BUILD, kw_only=True)
```

In both `__post_init__` methods, delete the early exit at the top (two occurrences):

```python
        if self.t is not _BUILD:
            return
```
and delete the last statement (two occurrences):
```python
        object.__setattr__(self, "t", _cell_arrays(self.cells)[2])
```
The validation in `__post_init__` (Task 9's `CellSet`/`WDMDomain` resolution, `_require_generic`, `_require_cells` or its Task 9 successor, and `require_window`) now runs on every construction, including `dataclasses.replace`, `jax.tree_util` unflatten and `jax.device_put` of a `DiagonalCovariance`'s constants. It reads only the static fields, and it is idempotent on an already resolved `cells`/`window`.

Both `on` methods. Replace the docstring and the body; the signature stays:

```python
        """The constants moved to ``be``."""
        return on_backend(self, be, ("t",))
```
→ (both occurrences)
```python
        """The variance for ``be``: it has no array leaves (the PSD runs on the host), so this
        only checks ``be`` (``require_x64`` for JAX).  ``as_evaluator`` recognises a variance
        by this method."""
        return on_backend(self, be, ())
```

Imports: `from .maps.base import _BUILD, on_backend` → `from .maps.base import on_backend`, provided `grep -n "_BUILD" src/domainia/variance.py` now prints nothing. `field` stays: `BandAveragedVariance.window` uses it.

`tests/test_dom_windows.py`, at the end of `test_the_window_is_static_under_jit`:

```python
    assert len(leaves) == 1 and jax.tree_util.tree_unflatten(tree, leaves).window is ALT_WINDOW
```
→
```python
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves).window is ALT_WINDOW
```

- [ ] **Step 7: `covariance.evolutionary.keep_mask` (`src/domainia/covariance/evolutionary.py`, `__init__.py`, two tests)**

`src/domainia/covariance/evolutionary.py`:
- `__all__ = ["BUILDERS", "EvolutionaryCovariance", "keep_mask"]` → `__all__ = ["BUILDERS", "EvolutionaryCovariance"]`;
- delete the whole function `def keep_mask(cells: Any) -> np.ndarray:` (its docstring and body, through `return keep`) and the two blank lines after it;
- in `EvolutionaryCovariance.__init__`: `keep = keep_mask(cells)` → `keep = cells.keep_mask()`. If Task 9 renamed the resolved cell set, use that name: the receiver is the `CellSet` the covariance stores as `self.cells`.

`packed_cells` stays imported, because `__init__` still calls `packed_cells(cells.grid.Nf, cells.grid.Nt)` for the band sizes. `np` and `Any` stay too, because the module uses them elsewhere.

`src/domainia/covariance/__init__.py`:
- `from .evolutionary import EvolutionaryCovariance, keep_mask` → `from .evolutionary import EvolutionaryCovariance`;
- in `__all__`, delete the line `    "keep_mask",`.

`tests/test_dom_evolutionary_cov.py`:
- `from domainia.covariance import EvolutionaryCovariance, keep_mask` → `from domainia.covariance import EvolutionaryCovariance`;
- delete the pin, with the two blank lines before it:

```python
def test_keep_mask_selects_the_cells_in_band_order():
    cells = _cells().drop_columns(2, 2)
    keep = keep_mask(cells)
    assert int(keep.sum()) == cells.size
    np.testing.assert_array_equal(np.flatnonzero(keep), cells.packed_positions())
```

`CellSet.keep_mask()` stays covered by `tests/test_dom_domains.py` (`cells.keep_mask().all()`, `np.flatnonzero(sub.keep_mask()) == pos`, `CellSet.from_mask(grid, sub.keep_mask()) == sub`).

Run: `grep -rn "keep_mask(cells)\|import.*keep_mask\|\"keep_mask\"" src tests`
Expected: no output.

- [ ] **Step 8: `_oracle_paths` defined once (`tests/wilson/conftest.py`, two test modules)**

In both `tests/wilson/test_grid_bands.py` and `tests/wilson/test_evolutionary_kernels.py`, delete this block. It is identical in both, and starts after the module's constants with two blank lines:

```python


#: The out-of-repo oracle trees live under the directory this variable names; unset, or
#: without the expected subdirectory, the tests that need them skip.
ORACLE_ROOT_ENV = "DOMAINIA_ORACLE_ROOT"


def _oracle_paths(*subpaths: str) -> list[Path]:
    """``$DOMAINIA_ORACLE_ROOT/<subpath>`` for each ``subpath``; skips the test unless the
    variable is set and every path exists."""
    root = os.environ.get(ORACLE_ROOT_ENV)
    if not root:
        pytest.skip(f"{ORACLE_ROOT_ENV} is not set: point it at the directory holding "
                    + " and ".join(subpaths))
    paths = [Path(root).expanduser().resolve() / s for s in subpaths]
    missing = [s for s, p in zip(subpaths, paths, strict=True) if not p.exists()]
    if missing:
        pytest.skip(f"{ORACLE_ROOT_ENV}={root} has no " + " and no ".join(missing))
    return paths
```

Import it instead (`tests/wilson` is a package; `test_meyer_baseline.py` already uses `from .meyer_cases import …`). In `tests/wilson/test_grid_bands.py`, after the `from domainia.wilson.transform import (...)` block:

```python
    forward_time,
    inverse_bands,
)
```
→
```python
    forward_time,
    inverse_bands,
)

from .conftest import _oracle_paths
```

In `tests/wilson/test_evolutionary_kernels.py`:

```python
from domainia.wilson.noise import evolutionary_band_variance
```
→
```python
from domainia.wilson.noise import evolutionary_band_variance

from .conftest import _oracle_paths
```

Both modules keep `import os` and `from pathlib import Path`. `test_grid_bands` still uses them in `_gb_reference`, and `test_evolutionary_kernels` in the lineage-A subprocess env and the `wdm_dev` seed test.

`tests/wilson/conftest.py` already imports `os`, `Path` and `pytest`. Append at the end:

```python


#: The out-of-repo oracle trees live under the directory this variable names; unset, or
#: without the expected subdirectory, the tests that need them skip.
ORACLE_ROOT_ENV = "DOMAINIA_ORACLE_ROOT"


def _oracle_paths(*subpaths: str) -> list[Path]:
    """``$DOMAINIA_ORACLE_ROOT/<subpath>`` for each ``subpath``; skips the test unless the
    variable is set and every path exists.  ``test_grid_bands`` and
    ``test_evolutionary_kernels`` import it (``from .conftest import _oracle_paths``)."""
    root = os.environ.get(ORACLE_ROOT_ENV)
    if not root:
        pytest.skip(f"{ORACLE_ROOT_ENV} is not set: point it at the directory holding "
                    + " and ".join(subpaths))
    paths = [Path(root).expanduser().resolve() / s for s in subpaths]
    missing = [s for s, p in zip(subpaths, paths, strict=True) if not p.exists()]
    if missing:
        pytest.skip(f"{ORACLE_ROOT_ENV}={root} has no " + " and no ".join(missing))
    return paths
```

Run: `grep -rn "def _oracle_paths\|ORACLE_ROOT_ENV = " tests`
Expected: one line each, both in `tests/wilson/conftest.py`.

- [ ] **Step 9: Run the touched tests**

Run: `$PY -m pytest -q tests/test_dom_cov_api.py tests/test_dom_spectra.py tests/test_dom_evolutionary_cov.py tests/test_dom_windows.py tests/test_dom_variance_generic.py tests/wilson/test_grid_bands.py tests/wilson/test_evolutionary_kernels.py`
Expected: all pass, `3 skipped`. The three skips now report `tests/wilson/conftest.py:<line>: DOMAINIA_ORACLE_ROOT is not set: …`, with the same subdirectory lists as before. `tests/test_dom_cov_api.py` alone gives `13 passed`.

Run: `$PY -m pytest -q tests/wilson/test_meyer_baseline.py`
Expected: all pass, with the baseline unchanged from Task 3 (`EvolutionaryCovariance` now gets the same mask from `CellSet.keep_mask()`).

- [ ] **Step 10: Full suite and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 3 skipped, 0 errors. The count is Task 13's plus 2 (one pin removed, three tests added). On the pre-Task-0 tree the same change took 1460 to 1462.
Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass (pre-Task-0 tree: 1371 to 1373 passed, 92 skipped).

- [ ] **Step 11: Commit** (sandbox disabled)

```bash
git add src/domainia/covariance/base.py src/domainia/spectra.py src/domainia/variance.py src/domainia/covariance/evolutionary.py src/domainia/covariance/__init__.py tests/test_dom_cov_api.py tests/test_dom_spectra.py tests/test_dom_windows.py tests/test_dom_evolutionary_cov.py tests/wilson/conftest.py tests/wilson/test_grid_bands.py tests/wilson/test_evolutionary_kernels.py
git commit -m "Drop unread hooks and duplicates: batchable, complete_constants, capabilities, variance t, keep_mask

Each had no reader outside its own pin (greps in the plan). EvolutionaryCovariance
uses CellSet.keep_mask(); the variances keep on() for as_evaluator; _oracle_paths
lives once in tests/wilson/conftest.py.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Text sweep (spec §7; last)

**Goal:** Comments, docstrings and README prose say what the code does in plain words. Shorthand, 1.x-spec and 1.x-plan references and the `wilson as wdm` alias are gone, and the README documents the units rule and the operator surface.

**Files:**
- Modify: every `src/domainia/**/*.py` with a hit, `README.md`, and the tests with hits outside `tests/wilson/oracle/` and `tests/wilson/vendored/`

**Acceptance Criteria:**
- [ ] The shorthand regex (WT/WG/GB/"WDM II", defined in the step) finds zero hits in `src/`, `README.md` and the non-frozen tests. Paper references are spelled out ("WDM Paper II, Eq. 36"), and `maps/fourier.py`'s "Ported from the GB toy-model oracle" line is reworded per spec §7.
- [ ] The 1.x reference regex (defined in the step: "spec §X", "spec section", and the 1.x plan labels such as `S1 Task C6`, `D-A`, `D-C`) finds zero hits in `src/` and the non-frozen tests. The 6 references to the domainia 2.0 spec stay, written "domainia 2.0 spec, section N".
- [ ] `grep -rn "wilson as wdm" src tests README.md` prints nothing outside the frozen oracle and the vendored tests (the 63 doctests and the 12 test modules that import the alias), and the doctests pass.
- [ ] README gains a "Units" section (a WDM value is a Wilson coefficient on every path; the Fourier units table with `sigma0`) and a paragraph on `domainia.operators` and `to_coordinates`/`from_coordinates`. Every README python block still runs.
- [ ] The full suite (doctests included), the NumPy lane and `tests/test_readme.py` pass.

**Verify:** `$PY -m pytest -q` → all pass, 0 errors; `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q` → all pass

**Steps:**

Assumes the tree after Task 14.

The sweep has a mechanical half (a codemod) and a prose half (by hand). The live tree before Task 0 has the following hits, outside the frozen `tests/wilson/oracle/` and `tests/wilson/vendored/`. Tasks 1–14 move a few of them, for example by rewriting the `FourierZeroPad`, `FourierToWDM` and `WDMDomain` docstrings. Work from what the greps print at the time, not from these numbers.

| regex | live hits |
|---|---|
| shorthand `\b(WT\|WG\|GB)\b\|WDM II` | 167 lines in 32 files: `src` 123 lines in 18 files (the spec's "about 124 in 19"), `tests` 44 lines in 14 files, `README.md` 0. By name: GB 69, WT 61, WG 35, "WDM II" 13 (all in `src`). |
| spec references (below) | 51 lines in 27 files. 45 lines (33 in `src`) cite the 1.x spec or the 1.x plan. 6 cite the domainia 2.0 spec: `translate.py:1`, `test_translate.py:1`, `test_translate_operator.py:1`, `tests/wilson/test_windows.py:1` and `tests/test_dom_windows.py` "Spec section 2:" (2.0 sections 2 Windows and 3 translate), plus `tests/wilson/meyer_cases.py:1`, which already says "domainia 2.0 spec". |
| `wilson as wdm` | 63 doctest lines in 9 `src/domainia/wilson` modules, plus 12 test modules that import the alias (`tests/wilson/test_backend.py` has it inside a subprocess code string). |

The shorthand names stand for three toy-model trees that do not ship:
- **WT**: the WDM Paper II toy models (`wdm_toys`, the sinusoid model);
- **WG**: the GPU port of those toy models (`wdmtoysgpu`);
- **GB**: the galactic-binary toy model.

- [ ] **Step 1: Measure**

```bash
FROZEN='tests/wilson/oracle/\|tests/wilson/vendored/'
SHORT='\b(WT|WG|GB)\b|WDM II'
SPEC="[Ss]pec,? (§|sections?\b)|§ ?[0-9]|^[[:space:]]*section [0-9]|\bS[1-4] (Task|deferral|had)|\(D-[A-C]\)|\bD-[A-C]'s"
grep -rnE "$SHORT" src tests README.md | grep -v "$FROZEN" | wc -l
grep -rnE "$SPEC" src tests README.md | grep -v "$FROZEN" | grep -v "2\.0 spec" | wc -l
grep -rn "wilson as wdm" src tests README.md | grep -v "$FROZEN" | wc -l
```
Expected: about `167`, `50` and `75` (see the table). Record the three numbers in the Execution errata. The `SPEC` regex covers `spec §X`, `spec section(s) X`, a wrapped `spec`/`section X` line, `Spec section`, and the 1.x plan references in `src` (`S1 Task`, `S2 had`, `S4 deferral`, `(D-A)`, `(D-B)`, `D-A's`, `D-C's`). Any `spec §` reference that Tasks 1–14 added (to the 2026-11-04 spec) is a hit too, and is treated the same way. The tests' leading plan tags (`"""D-A2: …`, `"""C8: …`) are not in scope.

- [ ] **Step 2: The codemod** (keep it in the scratchpad, do not commit it)

Save as `$TMPDIR/sweep_text.py`:

```python
"""One-off: the mechanical half of the text sweep (doctest alias, WDM II, 1.x spec refs).

Usage: python -I sweep_text.py REPO
Prints every rewritten file, and MISSING for an exact replacement whose old text is not
in the file (an earlier task rewrote it: handle that hit by hand from the gate grep).
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
FROZEN = ("tests/wilson/oracle/", "tests/wilson/vendored/")

ALIAS = "from domainia import wilson as wdm"
REF = re.compile(r"(?<![A-Za-z0-9_/.-])wdm\.(?=[A-Za-z_])")

# "WDM II Eq. 36" -> "WDM Paper II, Eq. 36"; every other "WDM II" -> "WDM Paper II".
WDM_II_EQ = re.compile(r"\bWDM II (?=(Eq\.|eq:))")
WDM_II = re.compile(r"\bWDM II\b")

# A parenthetical that only cites a 1.x spec section: "(spec §4.2)", "(spec section 4.1)",
# "(spec sections 3.4 and 4)", "(spec §3.2)".  Removed with the space before it.
SPEC_PAREN = re.compile(r" ?\(spec (?:§|sections? )[0-9][0-9.]*(?:(?: and |-)[0-9][0-9.]*)?\)")

# The references to the domainia 2.0 spec (its section 2, windows, and 3, translate) stay,
# made explicit.  (tests/wilson/meyer_cases.py already says "domainia 2.0 spec, section 5".)
KEEP_2_0 = {
    "src/domainia/translate.py": ("(spec section 3)", "(domainia 2.0 spec, section 3)"),
    "tests/test_translate.py": ("(spec section 3)", "(domainia 2.0 spec, section 3)"),
    "tests/test_translate_operator.py": ("(spec section 3)", "(domainia 2.0 spec, section 3)"),
    "tests/wilson/test_windows.py": ("(spec section 2)", "(domainia 2.0 spec, section 2)"),
    "tests/test_dom_windows.py": ('"""Spec section 2: a covariance built',
                                  '"""domainia 2.0 spec, section 2: a covariance built'),
}

# The 1.x references inside prose: each replaced by what it cites, or removed.
EXACT = {
    "src/domainia/testing.py": [
        ("small grids (spec section 4; NumPy only,", "small grids (NumPy only,"),
    ],
    "src/domainia/linalg.py": [
        ("library call at each call site (spec section 0, rule 1).",
         "library call at each call site."),
        ("**One factor family (S4 deferral (a)).**", "**One factor family.**"),
        ("this module keeps the array kernels they call.  The duplicate\n"
         "free-function copies S2 had here (``diag_factor`` ... ``factor_solve``) are gone.\n",
         "this module keeps the array kernels they call.\n"),
        ("callers map that to ``-inf`` (spec section 4.7, PD policy).",
         "callers map that to ``-inf``."),
    ],
    "src/domainia/maps/fourier.py": [
        ('"""Maps out of band-limited Fourier bins: zero padding, cropping, WDM cells (GB, spec §4.2).\n'
         "\n"
         "Ported from the GB toy-model oracle (frozen): ``_geometric``, ``pad_maps``, ``crop_maps``\n"
         "verbatim, the cell matrices of ``Grid.dense_atoms`` and ``_cells`` (with\n"
         "``wdm_full.analyse`` and ``wdm_bands._phase``), and the data map ``cells``.  The atoms are\n"
         "``wilson.transform.band_atoms`` (GB ``geometry.band_atoms`` to rounding; bit-identical\n"
         "until the Hz window of v0.12.0, S1 Task C6).\n",
         '"""Maps out of band-limited Fourier bins: zero padding, cropping, WDM cells.\n'
         "\n"
         "The Dirichlet kernels (:func:`pad_maps`, :func:`crop_maps`) and the cell matrices of\n"
         ":class:`FourierToWDM` (its dense atoms and :meth:`FourierToWDM.after`) are ported from\n"
         "the galactic-binary toy model, which does not ship.  The atoms are\n"
         "``wilson.transform.band_atoms``.\n"),
    ],
    "src/domainia/maps/base.py": [
        ("to the device (spec §3.3, the host-constant rule §0.4).", "to the device."),
        ("``be.asarray`` -- the host-constant rule (spec §0.4): build once on",
         "``be.asarray``: build once on"),
        ("raise a ``TypeError`` (spec\nsection 4.2's WDM -> time push",
         "raise a ``TypeError`` (the\nWDM -> time push"),
        ("``matvec``, so spec section\n        4.2's WDM -> time push",
         "``matvec``, so the\n        WDM -> time push"),
        ("(the sinusoid pad model's samples, spec §4.8)", "(the sinusoid pad model's samples)"),
        ("(store -> window -> nominal, spec §4.9)", "(store -> window -> nominal)"),
    ],
    "src/domainia/covariance/base.py": [
        ("**A factor holds per-state arrays only** (spec section 3.3, S4 deferral (b)).",
         "**A factor holds per-state arrays only**."),
        ("host NumPy float64 (spec §0 rule 4) and", "host NumPy float64 and"),
        ("    classes (D-B).  The object", "    classes.  The object"),
        ("reference path).  S4 deferral (c): before, a keyed combinator ran eagerly on JAX.",
         "reference path)."),
    ],
    "src/domainia/covariance/blocktri.py": [
        ("# domainia.linalg (D-A) takes", "# domainia.linalg takes"),
    ],
    "src/domainia/spectra.py": [
        ("aliased), as\n    D-A's ``CellSet`` keeps its indices.",
         "aliased), as\n    ``CellSet`` keeps its indices."),
    ],
    "tests/test_dom_domains.py": [
        (" (spec section 4.1; counts from spec sections 4.8-4.9)", ""),
        ('"""Spec section 4.8: 16384 cells,', '"""The sinusoid grid: 16384 cells,'),
        ('"""Spec section 4.9: 3 bands x NT cells', '"""The galactic-binary grids: 3 bands x NT cells'),
    ],
    "tests/wilson/test_lanes.py": [
        (' (spec section 1, "CI lanes" 1)', ""),
    ],
    "tests/test_dom_maps_dirichlet.py": [
        ("pad_maps/crop_maps keep the oracle signature (D-C's mask_rows calls crop_maps on chunks).",
         "pad_maps/crop_maps take the output and input bins as arrays, so a caller can build the\n"
         "    rows in chunks."),
    ],
    "tests/test_dom_maps_base.py": [
        ('"""Spec section 4.2\'s WDM -> time push', '"""The WDM -> time push'),
    ],
}


def sweep(rel: str, text: str) -> str:
    if rel in KEEP_2_0:
        old, new = KEEP_2_0[rel]
        if old in text:
            text = text.replace(old, new)
        else:
            print("MISSING", rel, repr(old))
    for old, new in EXACT.get(rel, []):
        if text.count(old) == 1:
            text = text.replace(old, new)
        else:
            print("MISSING", rel, repr(old[:70]))
    if ALIAS in text:
        text = text.replace(ALIAS, "from domainia import wilson")
        text = REF.sub("wilson.", text)
    text = WDM_II_EQ.sub("WDM Paper II, ", text)
    text = WDM_II.sub("WDM Paper II", text)
    return SPEC_PAREN.sub("", text)


paths = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "tests").rglob("*.py")),
         ROOT / "README.md"]
for path in paths:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(FROZEN) or "__pycache__" in rel:
        continue
    text = path.read_text()
    new = sweep(rel, text)
    if new != text:
        path.write_text(new)
        print("rewrote", rel)
```

The alias rewrite runs only in files that import the alias, so the README's local variable `wdm` (quick start) is untouched. The `(?<![A-Za-z0-9_/.-])` guard leaves `lisa/wdm b5973b7` and `wdm_dev` alone.

- [ ] **Step 3: Run it once**

Run: `cd $REPO && ~/miniconda3/envs/wdmrel/bin/python -I $TMPDIR/sweep_text.py .`
Expected: about 46 `rewrote …` lines: 25 `src` files and 21 test files on the live tree. The script prints a `MISSING …` line for each exact replacement whose old text an earlier task changed. Likely candidates are the `maps/base.py` entries for `PushedCovariance` (Task 8 rewrote that message and may have rewritten its docstring) and for `Identity` (Task 4). Fix each `MISSING` hit by hand with the rule in Step 5. Run the codemod only once: a second run prints `MISSING` for every exact entry, harmlessly.

- [ ] **Step 4: Doctests and the alias gate**

Run: `grep -rn "wilson as wdm" src tests README.md | grep -v "$FROZEN"`
Expected: no output.
Run: `$PY -m pytest -q src/domainia/wilson`
Expected: all pass (64 on the live tree, plus any doctests Tasks 10–12 added).
Run: `$PY -m pytest -q tests/wilson/test_backend.py tests/test_dom_domains.py tests/test_dom_maps_base.py tests/test_dom_maps_wdm.py tests/test_dom_testing.py tests/wilson/test_grid_bands.py`
Expected: all pass. These modules lost the `wdm` alias, and `test_backend.py` runs its rewritten code string in a subprocess.

- [ ] **Step 5: The spec-reference gate**

Run: `grep -rnE "$SPEC" src tests README.md | grep -v "$FROZEN" | grep -v "2\.0 spec"`
Expected: no output. For each remaining line (a `MISSING` entry, or a reference Tasks 1–14 added), replace the reference by what it cites, in plain words, or delete it. Before/after:
- 1.x reference, removed: `"""Linear maps between domains (spec §4.2)."""` → `"""Linear maps between domains."""`
- 1.x reference that carries content, kept as the content: `callers map that to ``-inf`` (spec section 4.7, PD policy).` → `callers map that to ``-inf``.`, and `**One factor family (S4 deferral (a)).**` → `**One factor family.**`
- a reference to the 2026-11-04 follow-up spec (for example `(spec §2.6)` in `operators.py`): delete it. The docstring already says what the code does.
- a reference to the domainia 2.0 spec: keep it, in the explicit form `(domainia 2.0 spec, section 3)`.

Run: `grep -rn "2\.0 spec" src tests`
Expected: the six kept references: `src/domainia/translate.py`, `tests/test_translate.py`, `tests/test_translate_operator.py`, `tests/wilson/test_windows.py`, `tests/test_dom_windows.py` and `tests/wilson/meyer_cases.py`.

- [ ] **Step 6: The shorthand in plain words** (by hand)

Run: `grep -rnE "$SHORT" src tests README.md | grep -v "$FROZEN"`
After the codemod, the live tree has 149 lines in 27 files, and none of them is "WDM II":
- `src` (107): `linalg.py` 31, `wilson/evolutionary.py` 18, `maps/fourier.py` 14, `maps/time.py` 7, `covariance/base.py` 6, `testing.py` 5, `covariance/blocktri.py` 5, `wilson/transform.py` 4, `wilson/special.py` 4, `covariance/dense.py` 4, `domains.py` 3, `wilson/backend.py` 2, `maps/base.py` 2, `covariance/stack.py` 2;
- `tests` (42): `wilson/test_evolutionary_kernels.py` 6, `test_dom_linalg_blocktri.py` 6, `wilson/test_grid_bands.py` 5, `test_dom_testing.py` 5, `test_dom_linalg_factors.py` 5, `test_dom_linalg_pcg.py` 4, `test_dom_maps_fourier_wdm.py` 2, `test_dom_maps_fourier_time.py` 2, `test_dom_domains.py` 2, `test_dom_cov_base.py` 2, `test_dom_maps_wdm.py` 1, `test_dom_maps_dirichlet.py` 1, `test_dom_cov_blocktri.py` 1.

**Rule** (spec §7):
1. Say what the code does.
2. A module that ports code from a tree that does not ship says so **once**, in its module docstring, with the tree described in words.
3. Every other mention in that module is either dropped (a provenance tag) or replaced by what it means (an operation order, a model's numbers).
4. File:line references into those trees go too, as the 2.0 Task 10 review did for `src`. `GB_SUBDIR` and function names such as `test_bessel_blocks_are_the_wt_toy_to_rounding` do not match the regex. Leave them.
5. Rewrap any edited line that goes past 100 characters.

**The one provenance sentence per `src` module** (the codemod already wrote `maps/fourier.py`'s):

| module | module-docstring sentence |
|---|---|
| `linalg.py` | Replace "Every routine is a line-for-line port of one oracle code path and keeps that oracle's library call at each call site." with "Every routine is a line-for-line port of one code path of the WDM Paper II toy models, their GPU port or the galactic-binary toy model (none of them ships), and keeps that code's library call at each call site." In the table, rename the `oracle` column `ported from`, and spell out its cells: `blocktri_factor`: "WDM Paper II toy models (ragged); GPU port (padded)"; `cblock_factor`: "galactic-binary toy model (12-month exact)"; `pcg`: "WDM Paper II toy models; GPU port". |
| `wilson/evolutionary.py` | In the "Numerics" paragraph, "and the WT, WG and GB toy-model oracles" → "and the WDM Paper II toy models, their GPU port and the galactic-binary toy model, none of which ships". |
| `maps/time.py` | "Every evaluator here is the library call the WT oracle made (``Grid.forward``/ ``forward_many``/``inverse_many``/``atom``, ``Variant.extend``, ``MatrixWhittle``, ``FourierBand``), so the NumPy path is bitwise:" → "Every evaluator here makes the library call of the WDM Paper II toy models it was ported from (they do not ship), so the NumPy path is bitwise:". |
| `wilson/special.py` | Add to the module docstring: "Both are ported from the WDM Paper II toy models and their GPU port, which do not ship." |
| `testing.py` | "(GB ``wdm_full.hhat_at`` / ``dense_lambda_generic``, in the dense form that the GB test helper ``generic_M`` pins it to)" → "(ported from the galactic-binary toy model, which does not ship)". |
| `wilson/transform.py` | In the `band_atoms` docstring, "Verbatim GB …" → "Ported verbatim from the galactic-binary toy model (not shipped)". This is its only provenance mention. |
| `covariance/base.py`, `covariance/dense.py`, `covariance/blocktri.py`, `covariance/stack.py`, `domains.py`, `maps/base.py`, `wilson/backend.py` | none: these describe what the code does, so the names go. |

**One before/after per kind of hit** (live text):

(a) A provenance tag on a docstring or comment: drop it.

`maps/time.py`, `TimeToWDM`:
```
    ``irfft(atom_spectrum)`` (WT ``Grid.atom``).
```
→
```
    ``irfft(atom_spectrum)``.
```

`maps/fourier.py`, `_band_phase`:
```
    """``(nb, Nt)`` carrier sign x quadrature x ``(-1)^n Nt / |phi|`` (GB ``wdm_bands._phase``)."""
```
→
```
    """``(nb, Nt)`` carrier sign x quadrature x ``(-1)^n Nt / |phi|``."""
```

(b) "Verbatim" or "line for line" notes: say what the order is for.

`linalg.py`, `_ragged_factor`:
```
    # WT BlockTri.__init__, line for line.
```
→
```
    # Ported line for line: keep the operation order (it fixes the float64 bits).
```
`# WG _btri_factor, line for line.` and `# WT BlockTri.logdet: Python sum of per-block float sums.` become `# Ported line for line: keep the operation order (it fixes the float64 bits).` and `# A Python sum of per-block float sums (the ported order).`

(c) The name of a model in domain prose: spell it out, or say what is general.

`domains.py`:
```
  ``"dt_rfft"``: ``dt * rfft(x)``; ``"sigma0"``: ``rfft(x) / sigma0``, the GB
  normalisation).  ``.full(...)``: the full-record domain of :mod:`domainia.translate`.
```
→
```
  ``"dt_rfft"``: ``dt * rfft(x)``; ``"sigma0"``: ``rfft(x) / sigma0``, the
  galactic-binary toy model's normalisation).  ``.full(...)``: the full-record domain of
  :mod:`domainia.translate`.
```

`covariance/stack.py`:
```
"""``ChannelStack``: independent channels (GB A and E), one covariance each."""
```
→
```
"""``ChannelStack``: independent channels (for example A and E), one covariance each."""
```
and `(GB: ``n_env`` envelope coefficients per channel)` → `(for example ``n_env`` envelope coefficients per channel)`.

(d) An oracle named as the reason for a choice: give the choice.

`covariance/base.py`, the quadratic-form orders:
```
Quadratic-form orders (each is the operation order of an oracle, kept per call site):

* ``"solve"``: ``b @ solve(b)`` -- WT ``marginal_lnlike`` (``float(y @ Ciy)``);
* ``"whiten"``: ``w @ w`` with ``w = whiten(b)`` -- GB ``_dense_quad_logdet``, WG forward-only;
* ``"elementwise"``: ``sum(b**2 / v)`` -- GB diagonal models (:class:`DiagFactor` only).
```
→
```
Quadratic-form orders (each fixes one operation order, kept per call site):

* ``"solve"``: ``b @ solve(b)``, one dot product with the solve;
* ``"whiten"``: ``w @ w`` with ``w = whiten(b)``, the forward solve only;
* ``"elementwise"``: ``sum(b**2 / v)`` (:class:`DiagFactor` only).
```

`covariance/dense.py`, `DenseCovariance`:
```
    callable ``fn(nu, be)`` or a fixed array.  ``chol_impl`` pins the Cholesky of the
    oracle (``"scipy"``: WT; ``"numpy"``: GB) and ``symmetrize`` whether ``0.5 (S + S^T)``
    is factorised (GB) or ``S`` itself (WT).
```
→
```
    callable ``fn(nu, be)`` or a fixed array.  ``chol_impl`` pins the Cholesky
    (``"scipy"``: ``cho_factor``; ``"numpy"``: ``np.linalg.cholesky``) and ``symmetrize``
    whether ``0.5 (S + S^T)`` is factorised or ``S`` itself.
```

(e) Tests: say once, in words, what a test reproduces.

`tests/test_dom_linalg_pcg.py`:
```
"""D-A6: preconditioned CG -- one frozen-row body (WG) whose single-row run is the WT
early-exit loop bit for bit, with the WT true-residual check (NumPy)."""
```
→
```
"""D-A6: preconditioned CG -- one frozen-row body (the GPU toy models' semantics) whose
single-row run is the WDM Paper II toy models' early-exit loop bit for bit, with their
true-residual check (NumPy)."""
```

`tests/test_dom_maps_wdm.py`:
```
    """Ported from WT tests/test_matrix.py:58-64 on its grid (Nf, Nt, dt) = (8, 16, 1/16)."""
```
→
```
    """Ported from the WDM Paper II toy models' test_matrix.py, on its grid
    (Nf, Nt, dt) = (8, 16, 1/16)."""
```

Run: `grep -rnE "$SHORT" src tests README.md | grep -v "$FROZEN"`
Expected: no output.

- [ ] **Step 7: README prose for Tasks 1–14**

The live README says nothing about `WDMDomain.scale`, `FourierToWDM` or σ0, so nothing in it is false. The acceptance criterion asks it to describe the units. Insert this section directly before `## Operators`:

```markdown
### Units

A WDM value is always a Wilson coefficient `w = W x` of the time record (`W W^T = dt I`),
whichever path produced it: `transform`, the maps `TimeToWDM` and `FourierToWDM` (both
implementations, and `after()`) and the `domainia.wilson` band kernels; the WDM variances
and covariances describe these coefficients. A Fourier value `z` carries the units its
`FourierDomain` declares, and `FourierDomain.rfft_scale` is the `u` with
`numpy.fft.rfft(x) = u * z`:

| `units`     | `z`                 | `u`      |
|-------------|---------------------|----------|
| `"rfft"`    | `numpy.fft.rfft(x)` | 1        |
| `"dt_rfft"` | `dt * rfft(x)`      | `1 / dt` |
| `"sigma0"`  | `rfft(x) / sigma0`  | `sigma0` |

`FourierDomain(N, dt, k0, n, "sigma0", sigma0=s)` stores the value `s` (positive and
finite; it is part of the domain's equality and hash), and the other units take no
`sigma0`. Every map honours the label: `FourierToWDM(src, wdm).apply(z)` equals
`TimeToWDM(time, wdm).apply(x)` on the same cells for every `units` value, and
`FourierZeroPad` and `FourierCrop` return the rfft of the zero-padded or cropped record in
the source's units (`sigma0` included).

```

The Tasks 5–8 surface is not in the README yet. At the end of the `## Operators` section, directly after the code block that ends with `assert np.isclose(on_wdm.log_determinant(), op.log_determinant() + N * np.log(dt))` and before `## Errors`, insert:

```markdown

`domainia.operators` makes operators out of explicit matrices and domainia covariances.
`DenseOperator(matrix, domain, channel_names)` is a symmetric positive semi-definite matrix
on the stacked real coordinates (channel-major, in `channel_names` order).
`CovarianceOperator(cov, nu, support, channel_names)` fixes a `domainia.covariance`
covariance at the noise state `nu` on its WDM `support` (a `WDMDomain`, possibly partial);
its `domain` is the full-record WDM domain. `operator_matrix(op)` reads any operator back as
its matrix in those coordinates, with one `apply` per column (small grids only).
`domainia.translate.to_coordinates` and `from_coordinates` convert between `{channel: array}`
dicts and the `(K, N)` coordinate rows, in `channel_names` order. Each of these works with
`translate_operator`, so a covariance matrix or a domainia covariance can be seen from time
or Fourier.
```

Run: `grep -nE "scale|sigma0|FourierToWDM" README.md`
Expected: `log_coordinate_scale` (coordinates paragraph) and `rfft_scale`, `sigma0` and `FourierToWDM` (the new Units section) only; no `WDMDomain.scale` and no `scale=`.
Run: `$PY -m pytest -q tests/test_readme.py`
Expected: `2 passed`. The new text is prose and adds no python block.
This runs on the post-Task-14 tree, so it also checks the existing blocks against Tasks 1–14. If a block fails or prints something different, the README text is wrong: fix it in this step. The README names none of the Task 14 removals: `grep -nE "batchable|complete_constants|capabilities|keep_mask" README.md` gives no hits.

- [ ] **Step 8: The three gates**

```bash
grep -rnE "$SHORT" src tests README.md | grep -v "$FROZEN"
grep -rnE "$SPEC" src tests README.md | grep -v "$FROZEN" | grep -v "2\.0 spec"
grep -rn "wilson as wdm" src tests README.md | grep -v "$FROZEN"
git diff --stat -- tests/wilson/oracle tests/wilson/vendored
```
Expected: no output from any of the four. The frozen trees are untouched.

- [ ] **Step 9: Full suite (doctests included) and NumPy lane**

Run: `$PY -m pytest -q`
Expected: all pass, 3 skipped, 0 errors. The count is the same as Task 14's: the sweep adds and removes no test.
Run: `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`
Expected: all pass.
Run: `$PY -m pytest -q tests/wilson/test_meyer_baseline.py tests/test_readme.py`
Expected: all pass.

- [ ] **Step 10: Commit** (sandbox disabled)

Two commits: the codemod's output first, then the hand-written prose.

Before Step 6 (right after Step 5 passes):

```bash
git add -A src tests
git commit -m "Text sweep: wilson doctest alias, WDM Paper II, 1.x spec references

from domainia import wilson (wilson. calls) in the doctests and tests; WDM II
spelled out; 1.x spec and plan references replaced by what they cite or removed;
the 2.0 spec references made explicit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

After Step 9:

```bash
git add -A src tests README.md
git commit -m "Text sweep: toy-model shorthand in plain words; README units and operators

WT/WG/GB become the WDM Paper II toy models, their GPU port and the
galactic-binary toy model, named once per module that ports from them. The README
describes the Fourier units, sigma0 and FourierToWDM in Wilson units, and the
domainia.operators surface.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Run `git status --short` before each `git add -A`. It must list only files the sweep touched: no scratch files, nothing under `tests/wilson/oracle` or `tests/wilson/vendored`.

---

## Known open points (FOR USER, found while drafting; not in this plan's scope)

* After Task 10, `FourierToWDM(impl="fft").apply` on complex64 bins returns float32 (through `forward_bands`), while `impl="dense"` returns float64 (`z @ C.T` with a complex128 `C`). No test covers complex64 maps.
* Task 15 leaves the 1.x plan tags at the start of test docstrings (`"""D-A2: …`, `"""C8: …`, 22 files) and the stale `wdm.` module names in three test docstrings (`tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_inner_product_fixes.py`, `tests/wilson/test_noise_stationary.py`, first line each). Neither of its regexes catches them.
* Task 12's sdist exclude is checked only by reading `pyproject.toml` (hatchling is not installed in the test env), so a built sdist has not been inspected.

## Execution errata

Executed 2026-11-08, subagent-driven: one implementer per task, then a review of that task's diff, fixes, and a re-review. The commits from 81a8597 to dd59860 hold the work; the counts below come from the final tree.

### Final state

- Full suite: 1768 passed, 3 skipped. The skips are the three `DOMAINIA_ORACLE_ROOT` ones.
- NumPy lane: 1661 passed, 110 skipped.
- `meyer_cases.py --check`: baseline reproduces, 1504 arrays. Task 10's float64 bit check (scratch `t10_bits.py`, every Meyer case including the RESCALED families): `0 of 1504 arrays differ`.
- The suite counts run above the plan's figures by +15 (full suite) and +14 passed / +1 skipped (NumPy lane). The extra tests are the review-fix tests listed under "Fixes beyond the plan text".

### Deviations from the plan text

- **Commit trailers.** Each commit names the model that wrote it. Sonnet implementers wrote `Claude Sonnet 5.5` where the plan says `Claude Opus 5.5`.
- **Task 7.** The plan says all 12 tests go red with `AttributeError`. The `ChannelStack` test goes red with `NotImplementedError`, because `st.dense(nu)` runs first.
- **Task 8.** In `test_a_reversed_channel_dict_meets_the_same_blocks`, `solve` and `quadratic_form` are compared to 1e-12 relative, not bit for bit; `apply` and `project` stay exact.
  - The plan's exact form failed in up to 8 of 30 pytest runs on this machine, always on `solve` for `dense-two-channels`, by at most 1.5e-15 relative.
  - The cause is the two-step triangular solve (`covariance/base.py`, scipy `solve_triangular` on Accelerate). Its last bit does not reproduce under pytest; the mechanism was not pinned down.
  - The 1e-12 tolerance still catches a channel-order bug, which changes `solve` by O(1).
- **Task 10.**
  - Task 2's comment lines inside the `forward_bands`/`inverse_bands` blocks were kept, and the casts applied line by line.
  - The Step 11 count is 103, not 101: the baseline file has two more tests (d026bdc, 0bab5d1).
- **Task 11.** Step 1's grep (`must be an integer`) missed two sites:
  - `build_C_time`'s `N`;
  - `eq45_plan`'s `P`: `P=2.7` became 2, `True` became 1, and `P=0` raised `IndexError`.

  Both now go through `_require_int`, and `eq45_plan` gains `P >= 1` (87a57c7).
- **Task 14.** Step 8's import anchor in `tests/wilson/test_grid_bands.py` misses an `inverse,` line between `forward_time,` and `inverse_bands,`. The import goes after the block's closing `)`.
- **Task 15.**
  - Step 1 measured 164 / 67 / 75, against the plan's "about" 167 / 50 / 75. The +17 spec hits are the 2026-11-04 references Tasks 1–14 added.
  - The codemod's `maps/fourier.py` module-docstring replacement came out MISSING, because Task 3 had reworded it. It was done by hand, keeping Task 3's self-conjugate clause.
  - Eight 2026-11-04 references in forms the codemod misses were fixed by hand.
  - The Meyer baseline docstrings drop the follow-up-spec citation, because the zero-hit gate overrides spec §1.7's request to cite it.
  - Two further commits:
    - e8d2471 holds wording items routed from earlier reviews.
    - dd59860 holds leftovers no gate regex catches: 1.x labels (`S1`, `S4 merge`, `E-C2`), `wdm.` alias prose, two follow-up-spec references, stale test paths, "oracle" with no referent, and README fixes. In the README, `## Units` is now a top-level section, and the interior-bands condition on `FourierToWDM` is stated.

### Fixes beyond the plan text (from the task reviews)

- **Task 2 (0bab5d1).** The RESCALED comparator in `meyer_cases.mismatches` accepted NaN, because `max|g − f·w| > tol` is False for NaN. It is now NaN-safe, with a red-first test. The fix also adds a per-family count assertion and a `dt_rfft` test of `after()` against `Compose`.
- **Task 6.**
  - 8af18b8: `DenseOperator` takes `eps` from a float32 input's dtype; before, a float32 rank-deficient PSD matrix was refused. `operator_matrix` validates `channel_names` before converting it. The reversed-dict test gets a real WDM→WDM translation target.
  - 4d2c7f6: the narrow `eps` is for float32 only. float16 had silently dropped directions.
- **Task 8 (3ffaaff).** Tests only: an alt-window case pins `CovarianceOperator.domain`'s window, and the `ChannelStack` children are checked one by one for window and cells.
- **Task 10 (6396bb0).** An `add_at` spy pins `inverse_bands`' post-FFT `phi` cast. Without it, `np.add.at` rounded complex128 into the complex64 output and no test failed. The fix also tightens three texts.

### Review notes left open

These are minor; each review judged them not worth a fix in this plan.

- Test strength:
  - `pytest.raises(TypeError)` without `match` in `test_dom_domains.py` (Task 4).
  - `to_coordinates`'s role word is not pinned (Task 5).
  - No PSD-boundary test for `DenseOperator` (Task 6).
  - The never-dense test covers only the ragged layout (Task 7).
  - No `jax.jit` test of `WDMToTime.adjoint` or an `after()` map (Task 13; checked by hand to 4e-16).
  - No JAX test of `MidpointVariance.on()` (Task 14).
- Input checks:
  - `multiply` does not check `b`'s shape, as `factor(nu).solve` does not (Task 7).
  - `empirical_variance`'s `n_realizations` has no kind check (Task 11).
  - `to_coordinates`' √2 multiply can overflow above ~1.27e308 (Task 5).
- Structure:
  - `from .conftest import _oracle_paths` imports a conftest (Task 14).
  - `test_transform_shim.py` imports from `test_single_precision.py` (Task 10).
  - The integer-kind check is written out five times in `domainia.wilson` (Task 11).
  - The window scan's `CONSUMERS` does not list `_cells_and_window`, the one remaining Meyer fallback (Task 9).
- The coordinator's untracked `.superpowers/` folder is not in the root `.gitignore`, so hatchling would pack it into an sdist built from this tree (Task 12 review). At the end of execution the folder was moved out of the repo, into the session scratchpad.

### Final whole-branch review

- **Verdict:** ready to merge once this section is filled and the tasks file is committed. No code changes were needed.
- **Units cross-check:** every route agrees with the dense `W_a x` reference to within 1.6e-14. The routes covered are `TimeToWDM`/`WDMToTime`, `FourierToWDM` (dense, fft, `restrict`, `after()` over pad and crop, `Compose`), `translate.transform`, `translate_operator`, `to_coordinates` and `CovarianceOperator`. Each was run on four grids, two windows and the three units, with odd-N pad/crop sources.
- **Correction to an earlier review note:** `FourierToWDM(impl="fft")` cannot return float64 for complex64 input through an `np.float64` `rfft_scale`. `FourierDomain` coerces `dt` and `sigma0` to Python floats, so the fft path is always float32 for complex64 input.
- **Minor notes, left open:**
  - `domains._positive_float` takes `dt=True` or `dt="2"` (bare `float()`), while `wilson.validate_dt` now refuses both with `TypeError`.
  - A wrong-kind domain passed to `FourierToWDM`, `BinSelect` or `Restrict` raises `AttributeError`.
  - Silent `int()` truncation remains in `ChannelStack`'s `params_per_channel`, `DiagonalCovariance`'s `n` and `BinSelect`'s bins.
  - `testing.dense_lambda_generic` takes unit-norm atoms, so its Λ is at the old record scale. For Wilson units pass `FourierToWDM.C` (= κU).
  - The toy-model ports are mentioned more than once in a few modules.
