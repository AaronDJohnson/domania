# domainia Implementation Plan: the folded WDM layout

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved spec `docs/superpowers/specs/2026-11-13-domainia-folded-wdm-design.md` on branch `folded-wdm`. Every WDM array in domainia, in the public API and in the `domainia.wilson` kernels, changes from the padded `(..., Nf+1, Nt)` layout to the folded `(..., Nf, Nt)` layout:
* rows 1 to Nf−1 are interior band m at time n, as before;
* row 0's even slot j holds band 0 (DC) at time j;
* row 0's odd slot j holds band Nf (Nyquist) at time j−1.

The flat form is `.reshape(..., N)`. `wilson.grid` gains the public `cells`, `fold` and `unfold`.

**Architecture:** Fold at the boundary.
* The kernels keep computing on a private padded grid. The private twins `_forward`, `_inverse` and `_inverse_time` are the padded kernels, and the other modules call them.
* Each public function gathers its WDM output into the folded slots, and scatters its WDM input back, through `wilson.grid.cells`. The values do not change; only the slots move.
* The two recorded fixtures are converted as they load and never re-recorded:
  * `tests/wilson/meyer_cases.load_baseline()` converts the Meyer replay;
  * `tests/test_transform.py` folds `translate_reference.npz`.
* The paper oracles stay padded. Each comparison gathers their output into folded slots.

**Tech Stack:** Python ≥3.12, NumPy ≥2.1, SciPy ≥1.14.1, pytest ≥8; no JAX. There are four test environments:
* `wdmrel`: Python 3.12, numpy 2.3.5;
* `floor312`: numpy 2.1.0, scipy 1.14.1;
* `cur313` and `cur314`: current numpy and scipy.

**User decisions (already made):** all on 2026-11-13, quoted from the spec.
1. "We should be using the packed representation though."
2. Which packed form: "Both should be supported. Why don't we use these?"
3. Scope: "Kernels too".
4. Flat order: "Rectangle, raveled". Flat is `rect.reshape(..., N)`, the one cell order for vectors, masks and N×N matrices.
5. API: "Rect in and out". Every entry point takes and returns `(..., Nf, Nt)`, with no layout keyword.
6. Padded arrays: "Public fold/unfold".
7. Approach: "Fold at the boundary".
8. Design sections 1–5: "yes, looks right" / "yes, next section".
9. Delivery: "Stacked branch". `folded-wdm` sits on `translation-core` at 8931b99, and its PR targets `translation-core`.

Also already decided, earlier in the project:
* keep code and tests small, and flag growth;
* fp64 only, no jit;
* never push, tag or publish unless asked;
* Enchilada is read-only.

**This plan departs from the spec in these places.** Each was found while prototyping. The user can overrule any of them.
1. **`_forward_time` is a twin only until Task 5.** Spec §3 lists four padded twins.
   - Tasks 4–5 move `apply_lambda_from_psd`, `apply_time_mask` and `exact_covariance` to the public kernels.
   - That is the same two boundary conversions with no round trip, and it gives them the padded/flat refusal with the named fix that §3 requires of every public kernel.
   - After that, nothing but `forward_time` calls `_forward_time`, so Task 5 deletes it and `forward_time` is again its e6d1e3f body.
   - The twins at the end are `_forward`, `_inverse` and `_inverse_time`.
2. **`_inverse` keeps only its finiteness check** (Task 7). Its callers build or check the padded float grid, and `sample_window` checks the grid and dt.
   - A 274-case probe of single bad inputs over every public caller gave identical exception kinds and messages.
   - `wrap_fractions` gains a `validate_dt` to keep its dt-before-window order.
3. **The replay does not rebuild each subset case's selection** (spec §5). Every recorded subset case selects every cell: `eq45_plan`'s `keep=None` and the all-True `testing` mask.
   - So one band-order → folded permutation per grid serves `eq45_direct`, the three `testing.*` cases and `EvolutionaryCovariance.direct`.
   - A future subset case would fail on shape, not silently.
   - `BandAveragedVariance` is converted at load, with the other band-order vectors.
4. **`RESIDUE` keeps its name.** Spec §2 says "one private module constant". It lives in `wilson.grid` outside `__all__`, rather than being renamed `_RESIDUE`, which is the smaller diff.
5. **`testing._keep` is replaced by a smaller `_mask`**, not deleted. The converter and the inactive-cell refusal go; the kind, grid and shape checks stay.
   - `apply_time_mask` no longer promotes a 1-D input: one `(Nf, Nt)` grid in gives one grid out.
6. **`translate_reference.npz` is folded at load with the checking `wilson.grid.fold`**, not a raw gather. Its inactive cells are exactly 0, so `fold` is a plain gather there and checks that too.
7. **No separate norm test.** Σw² = dt·Σx² is checked on the folded output by three existing tests (Task 2).
8. **Tests removed beyond spec §5's list**, each implied by a stronger test (prototype review):
   - vendored `test_W_complete`, `test_W_inverse_via_transpose`, `test_C_wdm_block_tridiagonal` and `test_full_active_blocks_match_dense_active` (Task 3);
   - the `bands=` row comparisons in `test_evolutionary_kernels` and `test_window_threading` (Task 4). Spec §3 said to keep comparing rows, but with `bands=` gone they repeat the full-grid comparison.
9. **Padded wording stays in private padded code** (the twins, `_block_tridiag_matvec`). `eq45_plan`'s `TypeError` keeps "a boolean mask over the packed cells", which a test pins.

**Scope check:** one plan. The spec covers one package, with one Gates section. Enchilada's side (spec §6) is a hand-off and is not a task here.

---

## Conventions (every task)

* **Repository.** `REPO=/Users/aaron/Documents/lisa/domainia`, the main checkout, on branch `folded-wdm`. There is no worktree. Work only in `$REPO`, and never `git checkout` another branch.
* **Starting point.** Task 1 starts from the commit that adds this plan, whose code is e6d1e3f's (= 8931b99's). Each later task starts from the commit of the task before it.
* **`$PY`** is shorthand for the prefix `PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P`, run from `$REPO`.
  * The Bash tool runs zsh, which does not word-split a variable. So `PY="…"; $PY -m pytest` fails. Write the prefix out in full in each command, as every task's commands do.
  * A probe script outside pytest needs `PYTHONPATH=src`, because the package is not installed.
* **`$FLOOR`** is the same prefix with `$TMPDIR/dom3-envs/floor312/bin/python`.
* **`$CUR313` and `$CUR314`** use `$TMPDIR/dom3-envs/cur313/bin/python` and `$TMPDIR/dom3-envs/cur314/bin/python`.
* **Lanes.** Each must pass with `-W error`, no collection errors and no warnings.
  * At the end of every task: `$PY -m pytest -q -W error` and `$FLOOR -m pytest -q -W error`.
  * Task 7 adds `$CUR313` and `$CUR314`.
  * pytest's config (`pyproject.toml`) also collects the doctests in `src/domainia/wilson`. The 3 skips in every lane are the oracle-root skips: `DOMAINIA_ORACLE_ROOT` is not set.
* **The replay gate.** `tests/wilson/test_meyer_baseline.py` replays `tests/wilson/meyer_cases.py` against `tests/wilson/data/meyer_baseline.npz` at 1e-12.
  * It passes with 0 mismatches at the end of every task.
  * `$PY tests/wilson/meyer_cases.py --check` prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)` after Tasks 1–2, and `1404 arrays (100 DROPPED keys set aside)` from Task 3 on.
  * Never edit or re-record `tests/wilson/data/meyer_baseline.npz` or `tests/data/translate_reference.npz`. Never edit `tests/wilson/oracle/**`.
  * Key names never change. A task converts a recorded family in `load_baseline()` in the same patch that changes its kernel.
* **Size.** `$PY $TMPDIR/dom3-size.py` prints src lines and code lines, and the test lines with the oracle copies shown apart. Every task ends by recording it, and the table below gives the prototype's numbers.
* **Git.** The Claude Code sandbox blocks writes inside `.git`, so every `git add` and `git commit` runs with the sandbox disabled (`dangerouslyDisableSandbox: true`).
  * One commit per task, with the message the task gives. It ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  * Stage named paths only. Nothing is pushed. Never amend, and never `git stash`.
* **Scratch** scripts go in `$TMPDIR`, never in the repo.
* **Errata.** Deviations agreed during execution go in an "Execution errata" section at the end of this plan.

## How a task's code is given: two patches

Every task was prototyped in order in a scratch clone of this branch, and its code is given as two unified diffs. The **tests patch** changes only `tests/`. The **implementation patch** changes src, the README and, in Task 5, `tests/wilson/meyer_cases.py`. Each patch sits in this file after an HTML comment `<!-- patch: T<n>-tests -->` or `<!-- patch: T<n>-impl -->`, inside a ` ````diff ` fence of four backticks. The four backticks keep a Markdown fence inside the diff from closing it.

Extract and apply a patch with this command, changing only `ID`:

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T1-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

* `git apply --check` refuses a patch whose context does not match. If it refuses, the tree differs from what the task expects: stop and report, and do not apply the patch by hand.
* Read each patch before applying it. The task's prose says what it changes and why; the patch is the exact text.
* Applying the patches of Tasks 1–7 in order, on a fresh clone at e6d1e3f, reproduces each prototype task's tree byte for byte. This was checked when the plan was written.

## Files edited by several tasks

The tasks run strictly in order (each is `blockedBy` the one before), because most files are edited by several tasks. Each patch's context is the tree the task before it leaves.

| File | Tasks |
|---|---|
| `tests/wilson/meyer_cases.py` | 1, 2, 3, 4, 5 |
| `tests/wilson/test_window_threading.py` | 2, 3, 4, 5, 7 |
| `tests/wilson/vendored/test_evolutionary.py` | 2, 3, 4, 5, 7 |
| `tests/wilson/test_grid_bands.py` | 1, 2, 3, 7 |
| `src/domainia/testing.py` | 2, 3, 5 |
| `src/domainia/wilson/evolutionary.py` | 2, 5, 7 |
| `src/domainia/wilson/grid.py` | 1, 2, 7 |
| `src/domainia/wilson/stationary.py` | 2, 3, 4 |
| `src/domainia/wilson/transform.py` | 2, 5, 7 |
| `tests/test_covariance.py` | 4, 5, 6 |
| `tests/test_dom_testing.py` | 2, 3, 5 |
| `tests/wilson/test_evolutionary_kernels.py` | 3, 4, 5 |
| `README.md` | 2, 6 |
| `src/domainia/_transform.py` | 1, 2 |
| `src/domainia/wilson/matrix.py` | 2, 3 |
| `src/domainia/wilson/noise.py` | 3, 4 |
| `tests/test_package.py` | 3, 7 |
| `tests/wilson/test_meyer_baseline.py` | 1, 3 |
| `tests/wilson/test_noise_stationary.py` | 4, 5 |
| `tests/wilson/vendored/test_matrix.py` | 2, 3 |
| `tests/wilson/vendored/test_stationary.py` | 3, 4 |

## Interface contract (names later tasks rely on)

* **Task 1**, `src/domainia/wilson/grid.py`:
  * `cells(Nf, Nt) -> (m, n)`, `fold(P, Nf, Nt)` and `unfold(R, Nf, Nt)` are public, in `__all__`.
  * `RESIDUE = 1e-10` is a module constant outside `__all__`.
  * The private `_wdm_array(name, x, shape)` holds the kind and trailing-shape checks.
  * `tests/wilson/meyer_cases.py` has `load_baseline() -> dict[str, np.ndarray]`.
* **Task 2:**
  * `wilson.grid._fold_fix(shape, Nf, Nt)` returns the message suffix naming `wilson.grid.fold` or `.reshape(..., Nf, Nt)`.
  * `wilson.transform` has the padded twins `_forward`, `_inverse`, `_forward_time` (until Task 5) and `_inverse_time`, each with the public signature.
  * `meyer_cases.FOLDED` lists the families gathered raw with `cells`.
* **Task 3:**
  * `wilson.matrix.__all__` is `build_C_time`, `build_C_wdm`, `build_W`, `real_to_rfft` and `rfft_to_real`.
  * `meyer_cases.DROPPED` covers 100 keys.
  * `meyer_cases.FOLDED_ROWS` exists until Task 5.
* **Task 4:** `evolutionary_band_variance(Nf, Nt, dt, Sn, *, window=MEYER)` has no `bands`.
* **Task 5:**
  * `eq45_plan(Nf, Nt, dt, keep=None, *, P=24, t_max=None, window=MEYER)` takes `keep` as a boolean `(Nf, Nt)` mask.
  * `testing` has the private `_mask(Nf, Nt, mask)`.
  * `meyer_cases` has `BAND_ORDER` and `DENSE`.
* **Task 6:** `wdm_covariance(S, *, Nf, Nt, dt, window=MEYER) -> (K, K, Nf, Nt)`.
* **Task 7:** `wilson.grid._active_mask(Nf, Nt)` is private, and `wilson.grid.__all__` has 14 names.

## Spec coverage

| Spec section | Tasks |
|---|---|
| User decisions | the header |
| §1 The folded layout | 1 (`cells`); every task's kernel; 6 (README Conventions) |
| §2 `cells`, `fold`, `unfold`, `RESIDUE`; `active_mask` private | 1; 7 (`_active_mask`) |
| §3 `transform.py` and the twins | 2; 5 (`_forward_time` goes, deviation 1); 7 (`_inverse` trimmed, deviation 2) |
| §3 `matrix.py` | 3 |
| §3 `noise.py`, `stationary.py` | 4 |
| §3 `evolutionary.py`, `grid.py` | 5; 7 |
| §4 `transform` | 2 |
| §4 `wdm_covariance` | 6 |
| §4 `frequency_covariance` | unchanged |
| §4 `domainia.testing` | 3 (bridge); 5 |
| §4 Public names | 1 (`cells`, `fold`, `unfold`); 3 (`active_W`, `build_C_wdm_active`); 7 (`active_mask`) |
| §4 README | 2 (Data); 6 (Covariance, Conventions, Kernels) |
| §5 The Meyer replay | 1 (loader); 2, 3, 4, 5 (each task's families); deviation 3 |
| §5 `translate_reference.npz` | 2; deviation 6 |
| §5 The paper oracles | 2–5 (gathered with `cells`) |
| §5 Tests that go, change, new | each task's tests patch; deviations 7–8 |
| §6 Enchilada | not a task (read-only); see Known open points |
| §7 Gates and size | every task; 7 (four lanes) |
| §8 Order of work | Tasks 1–7, in this order |
| §9 Supersedes | no code; see Known open points |
| §10 Open points | Known open points |

## Size and suite after each task (prototype)

| After | src lines / code | tests (+ oracle 4,135) | full suite, PY and FLOOR |
|---|---|---|---|
| start (e6d1e3f) | 4,080 / 1,861 | 4,344 | 951 passed, 3 skipped |
| Task 1 | 4,173 / 1,898 | 4,393 | 956 passed, 3 skipped |
| Task 2 | 4,194 / 1,906 | 4,412 | 955 passed, 3 skipped |
| Task 3 | 4,128 / 1,881 | 4,355 | 943 passed, 3 skipped |
| Task 4 | 4,099 / 1,864 | 4,346 | 943 passed, 3 skipped |
| Task 5 | 4,084 / 1,852 | 4,354 | 943 passed, 3 skipped |
| Task 6 | 4,085 / 1,852 | 4,350 | 943 passed, 3 skipped |
| Task 7 | 4,059 / 1,841 | 4,325 | 938 passed, 3 skipped |

## Known open points

None of these blocks a task.

* **Size.** The prototype ends at src 4,059 lines / 1,841 code and tests 4,325. That is −21 src lines, −20 code lines and −19 test lines against e6d1e3f. Nothing grew. Tasks 1–2 grow first, with the new helpers, twins and tests; the removals in Tasks 3–7 more than pay it back.
* **`fold` and an active NaN.** `fold`'s peak is `max |P|` over the slice, as spec §2 words it. An active NaN makes the peak NaN, so a finite inactive value in that slice is dropped without a refusal. An inactive NaN or inf is still refused. The active NaN stays in the folded output, where the kernels' finiteness checks refuse it. Taking the peak over finite values only (`np.max(v, where=np.isfinite(v), initial=0)`) would close this, at the cost of departing from §2's wording.
* **The order of refusals in `testing.dense_W`.** It now builds `build_W` before it checks the mask. A call with both a bad `dt` and a bad mask reports the `dt` first. A single bad input gives the same kind and message as before, and nothing pins the order.
* **`eq45_plan` and a padded `keep`.** It refuses one with a plain wrong-shape `ValueError`, without naming `wilson.grid.fold`. Spec §3's rule names data arrays, and §4 drops the padded-mask refusals, so this plan reads masks as outside it.
* **From the spec (§10):**
  * an Nf/Nt transpose is invisible on a square grid;
  * Enchilada's choice between adopting the fold and converting at its boundary is its own (spec §6 recommends adopting it), and until then the two packages disagree on the WDM layout;
  * `fold`'s 1e-10 refusal and the removal of `evolutionary_band_variance(bands=)` are defaults, not explicit user decisions.
* **Spec text that execution makes stale:**
  * §3's list of twins (deviation 1);
  * §5's per-case rebuild of selections (deviation 3);
  * §5's `bands=` row comparison (deviation 8);
  * §2's `RESIDUE` naming (deviation 4).

  The spec may want these edits after execution.
* **Three skips in every lane:** `DOMAINIA_ORACLE_ROOT is not set` (`tests/wilson/conftest.py`).

---

## Tasks

### Task 1: `wilson.grid`: `cells`, `fold` and `unfold` (spec §2; §8 step 1)

**Goal:** `wilson.grid` gains the public `cells`, `fold` and `unfold`, and holds the private constant `RESIDUE`. The replay gains its loader, `load_baseline()`, which converts no family yet. No kernel changes.

**Files:**
- Modify: `src/domainia/wilson/grid.py` (`RESIDUE`, `cells`, `_wdm_array`, `fold`, `unfold`, `__all__`, the module docstring)
- Modify: `src/domainia/_transform.py` (`RESIDUE` is imported from `wilson.grid`)
- Test: `tests/wilson/test_grid_bands.py`, `tests/wilson/meyer_cases.py` (`load_baseline`), `tests/wilson/test_meyer_baseline.py` (its `baseline` fixture)

**What the patches do:**
- **`cells(Nf, Nt)`** validates the grid. It returns `np.indices((Nf, Nt))` with row 0's odd slots moved to `(Nf, j − 1)`: `m[0, 1::2] = Nf` and `n[0, 1::2] -= 1`.
- **`fold(P, Nf, Nt)`** returns `P[..., m, n]` in P's dtype.
  - It first refuses, with a `ValueError` naming the leading index and the cell, the first inactive cell that is non-finite or whose magnitude exceeds `RESIDUE · max|P|` over the slice's last two axes.
  - It takes magnitudes in floating point (`np.abs(a.astype(np.result_type(a, 1.0), copy=False))`). Otherwise a signed integer's minimum would wrap back to itself under `abs` and pass.
  - It multiplies rather than divides, so a value at exactly the tolerance passes and no warning arises.
- **`unfold(R, Nf, Nt)`** writes R into a zero `(..., Nf+1, Nt)` array of R's dtype.
- **`_wdm_array`** holds the checks the two functions share: `TypeError` for a masked array or a dtype outside `"biufc"`, and `ValueError` for a wrong trailing shape.
- **`RESIDUE = 1e-10`** moves here from `_transform.py`. It keeps its name and stays out of `__all__`.
- **`active_mask` stays public.** Task 7 makes it private.
- **The tests:**
  - the Nf = Nt = 4 slot table, written out by hand;
  - an exact round trip with leading axes, and `unfold`'s exact zeros (`False` for bool);
  - `fold`'s refusals: 2e-10 of the peak, a NaN, a padded all-True mask and an int8 minimum;
  - a value at exactly 1e-10 passes;
  - both functions refuse a wrong shape (`ValueError`) and a masked or `str` array (`TypeError`).

**Acceptance Criteria:**
- [ ] The `cells` doctest prints `([0, 4, 0, 4], [0, 0, 2, 2])` for row 0 of a 4 × 4 grid.
- [ ] `wilson.grid.__all__` contains `cells`, `fold`, `unfold` and still `active_mask`; `RESIDUE` is a module constant outside `__all__`, and `_transform.py` imports it.
- [ ] The replay prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`.
- [ ] Full suite (prototype): PY and FLOOR `956 passed, 3 skipped`, no warnings, no collection errors.
- [ ] Size (prototype): `src 4173 lines / 1898 code; tests 4393 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/test_grid_bands.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/grid.py 2>&1 | tail -1` → `43 passed, 1 skipped`

**Steps:**

- [ ] **Step 1: Apply the tests patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T1-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists exactly `tests/wilson/meyer_cases.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_meyer_baseline.py` as modified.

<!-- patch: T1-tests -->
````diff
diff --git a/tests/wilson/meyer_cases.py b/tests/wilson/meyer_cases.py
index 559bca9..e0f4aaf 100644
--- a/tests/wilson/meyer_cases.py
+++ b/tests/wilson/meyer_cases.py
@@ -54,6 +54,14 @@ def _family(key: str) -> str:
     return "::".join(key.split("::")[:2])
 
 
+def load_baseline() -> dict[str, np.ndarray]:
+    """The recorded baseline as ``{key: array}``.  The families whose kernels now return the
+    folded WDM layout are converted here, never in :func:`mismatches` (folded WDM spec,
+    section 5)."""
+    with np.load(BASELINE) as z:
+        return {k: z[k] for k in z.files}
+
+
 def mismatches(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> list[str]:
     """The keys of ``want`` that ``got`` does not reproduce.  ``want``'s :data:`DROPPED` keys are
     set aside; the other keys must be exactly ``got``'s.
@@ -253,8 +261,7 @@ def main() -> None:
     from domainia import wilson
 
     out = compute(wilson, {}, {})
-    with np.load(BASELINE) as z:
-        stored = {k: z[k] for k in z.files}
+    stored = load_baseline()
     try:
         bad = mismatches(out, stored)
     except AssertionError as exc:
diff --git a/tests/wilson/test_grid_bands.py b/tests/wilson/test_grid_bands.py
index 50fb4b2..d5f789f 100644
--- a/tests/wilson/test_grid_bands.py
+++ b/tests/wilson/test_grid_bands.py
@@ -13,7 +13,7 @@ import pytest
 
 from domainia import wilson
 from domainia.wilson.filter import phi_centered
-from domainia.wilson.grid import Grid, active_mask, endpoint_weights
+from domainia.wilson.grid import Grid, active_mask, cells, endpoint_weights, fold, unfold
 from domainia.wilson.matrix import active_W
 from domainia.wilson.transform import atom_spectrum, base_atom_spectra, forward_time
 
@@ -32,6 +32,49 @@ def test_grid_is_frozen_positional_and_unpacks():
         g.Nf = 4  # type: ignore[misc]
 
 
+def test_cells_is_the_folded_slot_table():
+    m, n = cells(4, 4)
+    assert m.tolist() == [[0, 4, 0, 4], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
+    assert n.tolist() == [[0, 0, 2, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
+
+
+def test_fold_and_unfold_round_trip_and_unfold_writes_exact_zeros(rng):
+    Nf, Nt = 8, 6
+    R = rng.standard_normal((2, 3, Nf, Nt))
+    P = unfold(R, Nf, Nt)
+    assert P.shape == (2, 3, Nf + 1, Nt) and np.array_equal(fold(P, Nf, Nt), R)
+    m, n = cells(Nf, Nt)
+    P[..., m, n] = 0.0
+    assert not P.any()
+    B = unfold(np.ones((Nf, Nt), bool), Nf, Nt)
+    assert B.dtype == bool and np.array_equal(B, active_mask(Nf, Nt))
+    assert fold(B, Nf, Nt).dtype == bool
+
+
+def test_fold_refuses_a_live_inactive_cell_and_both_refuse_a_wrong_shape_or_kind():
+    Nf, Nt = 4, 4
+    P = unfold(np.array([1.0, 4.0])[:, None, None] * np.ones((Nf, Nt)), Nf, Nt)
+    P[:, 4, 3] = [1e-10, 4e-10]                   # exactly 1e-10 of each slice's peak: accepted
+    fold(P, Nf, Nt)
+    for value in (2e-10, np.nan):                 # 2e-10 is under slice 1's tolerance, not 0's
+        Q = P.copy()
+        Q[0, 0, 1] = value
+        with pytest.raises(ValueError, match=r"P\[0\]: the inactive cell \(m=0, n=1\)"):
+            fold(Q, Nf, Nt)
+    with pytest.raises(ValueError, match=r"^P: the inactive cell \(m=0, n=1\) holds True"):
+        fold(np.ones((Nf + 1, Nt), bool), Nf, Nt)
+    Q = unfold(np.ones((Nf, Nt), np.int8), Nf, Nt)
+    Q[0, 1] = np.iinfo(np.int8).min                 # np.abs(Q) would wrap it back to -128
+    with pytest.raises(ValueError, match=r"^P: the inactive cell \(m=0, n=1\) holds -128;"):
+        fold(Q, Nf, Nt)
+    for f, wrong in ((fold, (Nf, Nt)), (unfold, (Nf + 1, Nt))):
+        with pytest.raises(ValueError, match="must end in"):
+            f(np.zeros(wrong), Nf, Nt)
+        for x in (np.ma.zeros(wrong), np.full(wrong, "a")):
+            with pytest.raises(TypeError):
+                f(x, Nf, Nt)
+
+
 def test_endpoint_weights():
     c = endpoint_weights(9)
     assert c[0] == c[-1] == 0.5 and np.all(c[1:-1] == 1.0)
diff --git a/tests/wilson/test_meyer_baseline.py b/tests/wilson/test_meyer_baseline.py
index 385a74a..2bc6922 100644
--- a/tests/wilson/test_meyer_baseline.py
+++ b/tests/wilson/test_meyer_baseline.py
@@ -13,13 +13,12 @@ import pytest
 
 from domainia import wilson
 
-from .meyer_cases import BASELINE, DROPPED, RTOL, THREADED, compute, mismatches
+from .meyer_cases import DROPPED, RTOL, THREADED, compute, load_baseline, mismatches
 
 
 @pytest.fixture(scope="module")
 def baseline() -> dict[str, np.ndarray]:
-    with np.load(BASELINE) as z:
-        return {k: z[k] for k in z.files}
+    return load_baseline()
 
 
 def test_the_dropped_families_are_96_of_the_1504_keys(baseline):
````

- [ ] **Step 2: Run the task's tests: they fail**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error --tb=no -rfE tests/wilson/test_grid_bands.py tests/wilson/test_meyer_baseline.py 2>&1 | tail -n 40
```

Expected last line (prototype): `2 errors`. The module fails to collect, and pytest reports it twice, once from the module collector and once from the doctest collector: `ImportError: cannot import name 'cells' from 'domainia.wilson.grid'`. That blocks the new tests. The replay loader changes no behaviour, so it has no failing state.

- [ ] **Step 3: Apply the implementation patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T1-impl && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists `src/domainia/_transform.py`, `src/domainia/wilson/grid.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_meyer_baseline.py` as modified, and nothing else.

<!-- patch: T1-impl -->
````diff
diff --git a/src/domainia/_transform.py b/src/domainia/_transform.py
index 625d4c4..7fecdbe 100644
--- a/src/domainia/_transform.py
+++ b/src/domainia/_transform.py
@@ -19,12 +19,11 @@ import numpy as np
 from . import wilson
 from .wilson import MEYER, Window
 from .wilson._checks import require_dtype, require_int
+from .wilson.grid import RESIDUE
 
 __all__ = ["transform"]
 
 KINDS = ("time", "frequency", "wdm")
-#: The largest canonical-form residue an input may carry, as a fraction of its slice's peak.
-RESIDUE = 1e-10
 
 
 def _grid(role: str, kind: Any, wdm: Any, N: int) -> tuple[int, int] | None:
diff --git a/src/domainia/wilson/grid.py b/src/domainia/wilson/grid.py
index 17cdb8e..f9c777a 100644
--- a/src/domainia/wilson/grid.py
+++ b/src/domainia/wilson/grid.py
@@ -1,5 +1,5 @@
-"""Time-frequency grid: the grid and input checks, pixel widths and centres, and the
-active-pixel mask.
+"""Time-frequency grid: the grid and input checks, pixel widths and centres, the
+active-pixel mask and the folded layout.
 
 The WDM transform partitions the time-frequency plane into a grid of
 $N_f+1$ frequency bands (indexed by $m = 0,\\ldots,N_f$) and $N_t$
@@ -13,6 +13,10 @@ Edge bands ($m=0$ DC, $m=N_f$ Nyquist) carry only $N_t/2$ active
 coefficients each (parity-selected); interior bands carry all $N_t$.
 The total active count is therefore
 $(N_f - 1)\\,N_t + N_t/2 + N_t/2 = N$.
+
+The folded layout ``(Nf, Nt)`` holds exactly those $N$ cells: row ``i >= 1`` is band ``i``,
+and row 0 holds band 0 in its even slots and band ``Nf`` in its odd ones (:func:`cells`).
+:func:`fold` and :func:`unfold` convert between it and the padded ``(Nf+1, Nt)`` array.
 """
 
 from __future__ import annotations
@@ -33,17 +37,23 @@ __all__ = [
     "Grid",
     "active_mask",
     "band_frequencies",
+    "cells",
     "delta_F",
     "delta_T",
     "endpoint_weights",
+    "fold",
     "n_samples",
     "sanitize_psd",
     "time_pixel_centers",
+    "unfold",
     "validate_dt",
     "validate_finite",
     "validate_grid",
 ]
 
+#: The largest canonical-form residue an input may carry, as a fraction of its slice's peak.
+RESIDUE = 1e-10
+
 
 @dataclass(frozen=True)
 class Grid:
@@ -315,6 +325,90 @@ def active_mask(Nf: int, Nt: int) -> BoolArray:
     return mask
 
 
+def cells(Nf: int, Nt: int) -> tuple[np.ndarray, np.ndarray]:
+    """The band ``m`` and time ``n`` of each folded slot, two int arrays of shape ``(Nf, Nt)``.
+
+    Slot ``(i, j)`` with ``i >= 1`` is band ``i`` at time ``j``.  Row 0 holds band 0 (DC) at
+    time ``j`` in its even slots and band ``Nf`` (Nyquist) at time ``j - 1`` in its odd ones.
+    A slot's frequency and time are ``band_frequencies(Nf, dt)[m]`` and
+    ``time_pixel_centers(Nf, Nt, dt)[n]``.
+
+    Examples
+    --------
+    >>> from domainia import wilson
+    >>> m, n = wilson.grid.cells(4, 4)
+    >>> m[0].tolist(), n[0].tolist()
+    ([0, 4, 0, 4], [0, 0, 2, 2])
+    """
+    validate_grid(Nf, Nt)
+    m, n = np.indices((Nf, Nt))
+    m[0, 1::2] = Nf
+    n[0, 1::2] -= 1
+    return m, n
+
+
+def _wdm_array(name: str, x: object, shape: tuple[int, int]) -> np.ndarray:
+    """``x`` as an array, after the kind and trailing-shape checks of :func:`fold` and
+    :func:`unfold`."""
+    if isinstance(x, np.ma.MaskedArray):
+        raise TypeError(f"{name} is a masked array; numpy.asarray would drop the mask, so fill "
+                        "it first")
+    a = np.asarray(x)
+    require_dtype(name, a, "biufc", "a numeric or boolean array")
+    if a.shape[-2:] != shape:
+        raise ValueError(f"{name} has shape {a.shape}, but must end in {shape}")
+    return a
+
+
+def fold(P: object, Nf: int, Nt: int) -> np.ndarray:
+    """The padded ``(..., Nf+1, Nt)`` array ``P`` in the folded layout ``(..., Nf, Nt)``:
+    ``P[..., m, n]`` with ``m, n = cells(Nf, Nt)``, in ``P``'s dtype (a boolean mask folds too).
+
+    Raises
+    ------
+    TypeError
+        If ``P`` is masked, or neither numeric nor boolean.
+    ValueError
+        If the grid is invalid, ``P`` does not end in ``(Nf+1, Nt)``, or an inactive cell is
+        non-finite or exceeds :data:`RESIDUE` of its slice's peak (``max |P|`` over the last two
+        axes; for a boolean mask, any ``True``): fold drops it.
+
+    Examples
+    --------
+    >>> import numpy as np; from domainia import wilson
+    >>> R = np.arange(32.0).reshape(4, 8)
+    >>> P = wilson.grid.unfold(R, 4, 8)
+    >>> P.shape, bool(np.array_equal(wilson.grid.fold(P, 4, 8), R))
+    ((5, 8), True)
+    >>> P[0, 1] = 1.0
+    >>> wilson.grid.fold(P, 4, 8)                  # doctest: +ELLIPSIS
+    Traceback (most recent call last):
+    ValueError: P: the inactive cell (m=0, n=1) holds 1.0; ...
+    """
+    m, n = cells(Nf, Nt)
+    a = _wdm_array("P", P, (Nf + 1, Nt))
+    v = np.abs(a.astype(np.result_type(a, 1.0), copy=False))
+    peak = np.max(v, axis=(-2, -1), keepdims=True)
+    bad = np.argwhere(~active_mask(Nf, Nt) & (~np.isfinite(v) | (v > RESIDUE * peak)))
+    if bad.size:
+        *lead, i, j = (int(k) for k in bad[0])
+        raise ValueError(f"P{lead if lead else ''}: the inactive cell (m={i}, n={j}) holds "
+                         f"{a[tuple(bad[0])].item()}; fold drops inactive cells, so each must be "
+                         f"finite and at most {RESIDUE:g} of its slice's peak")
+    return a[..., m, n]
+
+
+def unfold(R: object, Nf: int, Nt: int) -> np.ndarray:
+    """The folded ``(..., Nf, Nt)`` array ``R`` in the padded layout ``(..., Nf+1, Nt)``, in
+    ``R``'s dtype, with the inactive cells exactly 0 (``False``); the inverse of :func:`fold`.
+    Raises as :func:`fold` does for a wrong kind, grid or shape."""
+    m, n = cells(Nf, Nt)
+    a = _wdm_array("R", R, (Nf, Nt))
+    P = np.zeros(a.shape[:-2] + (Nf + 1, Nt), a.dtype)
+    P[..., m, n] = a
+    return P
+
+
 def endpoint_weights(n_rfft: int) -> Array:
     r"""Trapezoid weights :math:`\eta_k` of the one-sided sum: ``1/2`` at DC and
     Nyquist, ``1`` between (WDM Paper II, Eq. 51, eq:freq_edge_weight).
````

- [ ] **Step 4: Run the task's tests: they pass**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/test_grid_bands.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/grid.py 2>&1 | tail -1
```

Expected (prototype): `43 passed, 1 skipped`.

- [ ] **Step 5: Run the lanes, the replay and the size**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py
```

Expected (prototype): `956 passed, 3 skipped` in both lanes, with no warnings and no collection errors; `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`; `src 4173 lines / 1898 code; tests 4393 lines (+ oracle 4135)`.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/_transform.py src/domainia/wilson/grid.py tests/wilson/meyer_cases.py tests/wilson/test_grid_bands.py tests/wilson/test_meyer_baseline.py && git commit -q -F - <<'EOF'
Add wilson.grid.cells, fold and unfold for the folded WDM layout

cells(Nf, Nt) gives each folded slot's band and time; fold gathers a padded
(..., Nf+1, Nt) array into (..., Nf, Nt), refusing an inactive cell that is
non-finite or above RESIDUE of its slice's peak; unfold scatters back with
exact zeros. RESIDUE moves into wilson.grid, and transform imports it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
```


### Task 2: the transform kernels and `transform` fold (spec §3 `transform.py`, §4 `transform`, §5; §8 step 2)

**Goal:**
- `forward` and `forward_time` return `(..., Nf, Nt)` folded, and `inverse` and `inverse_time` take it. Each refuses a padded or a flat input, with a message that names the fix.
- The padded kernels become private twins, and every other module calls those twins, so no other output changes yet.
- `domainia.transform`'s `"wdm"` layout is folded, and its WDM residue check and zeroing go.

**Files:**
- Modify: `src/domainia/wilson/transform.py` (the twins `_forward`, `_inverse`, `_forward_time` and `_inverse_time`; the folded public kernels; docstrings and doctests)
- Modify: `src/domainia/wilson/grid.py` (the private `_fold_fix`; the `Grid` doctest)
- Modify: `src/domainia/_transform.py` (the folded `"wdm"` layout, the padded and flat refusals, no WDM residue rule)
- Modify (callers moved to the twins): `src/domainia/testing.py`, `src/domainia/wilson/evolutionary.py`, `src/domainia/wilson/matrix.py`, `src/domainia/wilson/stationary.py`
- Modify: `README.md` (the Data example)
- Test: `tests/test_transform.py`, `tests/wilson/test_transform_shim.py`, `tests/wilson/vendored/test_transform.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/meyer_cases.py`, and bridges in `tests/wilson/test_grid_bands.py`, `tests/test_dom_testing.py`, `tests/wilson/vendored/test_evolutionary.py` and `tests/wilson/vendored/test_matrix.py`

**What the patches do:**
- **The twins** hold today's padded bodies:
  - `_forward` is today's `forward`;
  - `_inverse` is today's `inverse` without the inactive-cell zeroing. `_prepare_edges_for_inverse` is deleted, and its √2 undo is inlined;
  - `_forward_time` ends in `_forward`;
  - `_inverse_time` is `irfft(_inverse(...))`.
- **The public kernels:**
  - `forward` and `forward_time` return `twin(...)[..., m, n]` with `m, n = cells(Nf, Nt)`.
  - `inverse` checks the folded input (grid, dt, real dtype, `(..., Nf, Nt)` shape, finite) and then calls `_inverse(unfold(wmn, Nf, Nt), ...)`.
  - `inverse_time` is `irfft(inverse(...))`.
- **`_fold_fix(shape, Nf, Nt)`** is the end of the wrong-shape message:
  - `"; fold a padded (..., Nf+1, Nt) array with wilson.grid.fold first"`;
  - or `"; reshape a flat (..., N) array with .reshape(..., Nf, Nt) first"`, with the numbers filled in;
  - or `""` for any other shape.

  `inverse` and `_transform._checked` share it.
- **`_transform`:**
  - The WDM shape is `(Nf, Nt)`.
  - The WDM residue branch and the WDM output zeroing go.
  - The frequency residue rule is unchanged.
- **Callers moved to the twins:**
  - `testing.empirical_variance`;
  - `matrix.build_W`;
  - `evolutionary.apply_time_mask`, `wrap_fractions` and `exact_covariance`;
  - `stationary.apply_lambda_from_psd` and `psd_from_lambda_blocks`.

  `apply_lambda_from_psd` masks its still-padded input itself (`_inverse(x_wmn * active_mask(Nf, Nt), ...)`), a bridge that Task 4 removes.
- **README Data example:** the WDM shape is `(2, 6, 12)`. The inactive-cell half of the residue comment goes, with its `forward_time` assertion and the `wilson` import that only it used.
- **Tests:**
  - `translate_reference.npz`'s WDM `x` and `y` are folded at load with `wilson.grid.fold`. Their inactive cells are exactly 0, so `fold` also checks them.
  - The oracle's padded output is gathered with `cells`; ours is never unfolded.
  - The WDM rows of the residue test go, and so does vendored `test_inactive_pixels_zero`.
  - `transform`, `inverse` and `inverse_time` must refuse a padded input and a flat one, each message naming its fix.
  - Σw² = dt·Σx² is checked on the folded output by `test_round_trips_leading_axes_and_parseval`, vendored `test_parseval_normalized` and `test_time_round_trip_and_energy`.
- **Replay:** `meyer_cases.FOLDED` lists `forward`, `forward_time`, `forward_many` and `forward_time_many`, which `load_baseline()` gathers with `cells`. Two generators bridge with `unfold` until their kernels fold, in Tasks 4 and 5.

**Acceptance Criteria:**
- [ ] `wilson.forward_time(x, 8, 16, dt).shape == (8, 16)`, and `wilson.inverse` of a `(9, 16)` array raises `ValueError` whose message contains `wilson.grid.fold`; of a `(128,)` array, one containing `.reshape(..., 8, 16)`.
- [ ] `domainia.transform` returns folded WDM arrays and refuses padded and flat WDM inputs in the same way.
- [ ] The replay prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`.
- [ ] Full suite (prototype): PY and FLOOR `955 passed, 3 skipped`, no warnings, no collection errors.
- [ ] Size (prototype): `src 4194 lines / 1906 code; tests 4412 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/test_transform.py tests/wilson/test_transform_shim.py tests/wilson/vendored/test_transform.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py tests/wilson/test_grid_bands.py tests/test_dom_testing.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_matrix.py tests/wilson/test_wrong_kinds.py tests/test_readme.py src/domainia/wilson/transform.py src/domainia/wilson/grid.py src/domainia/wilson/stationary.py 2>&1 | tail -1` → `610 passed, 1 skipped`

**Steps:**

- [ ] **Step 1: Apply the tests patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T2-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists exactly `tests/test_dom_testing.py`, `tests/test_transform.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_transform_shim.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_matrix.py`, `tests/wilson/vendored/test_transform.py` as modified.

<!-- patch: T2-tests -->
````diff
diff --git a/tests/test_dom_testing.py b/tests/test_dom_testing.py
index 4c0e841..3fe8575 100644
--- a/tests/test_dom_testing.py
+++ b/tests/test_dom_testing.py
@@ -37,7 +37,8 @@ def rel(a, b):
 def test_dense_W_rows_are_the_atoms(rng):
     W = T.dense_W(NF, NT, DT, MASK)
     x = rng.standard_normal(N)
-    assert np.max(np.abs(W @ x - wilson.forward_time(x, NF, NT, DT)[MASK])) < 1e-12
+    w = wilson.grid.unfold(wilson.forward_time(x, NF, NT, DT), NF, NT)
+    assert np.max(np.abs(W @ x - w[MASK])) < 1e-12
     assert np.max(np.abs(W @ W.T - DT * np.eye(N))) < 1e-14
     sub = MASK.copy()
     sub[:, :3] = sub[:, -1:] = False
diff --git a/tests/test_transform.py b/tests/test_transform.py
index a4eca74..ff47ba1 100644
--- a/tests/test_transform.py
+++ b/tests/test_transform.py
@@ -39,29 +39,30 @@ def test_agrees_with_the_recorded_translate_transform():
     """The six pairs on four WDM grids, time <-> frequency at odd N, one channel with no
     leading axis, and WDM -> WDM across two grids: each output within 1e-12 of its peak
     (measured: 0.0) and exactly canonical.  At odd N the last bin is not a Nyquist bin, and
-    it keeps its imaginary part."""
+    it keeps its imaginary part.  The recorded WDM arrays are padded, with exact zeros in the
+    inactive cells, and are folded at load."""
     with np.load(REFERENCE) as z:
         for i, ((s, t), (n, dt, *g)) in enumerate(zip(z["routes"], z["grids"])):
             n, sg, tg = int(n), (int(g[0]), int(g[1])), (int(g[2]), int(g[3]))
-            got = transform(z[f"x{i}"], s, t, N=n, dt=dt, source_wdm=sg if s == "wdm" else None,
+            x = wilson.grid.fold(z[f"x{i}"], *sg) if s == "wdm" else z[f"x{i}"]
+            y = wilson.grid.fold(z[f"y{i}"], *tg) if t == "wdm" else z[f"y{i}"]
+            got = transform(x, s, t, N=n, dt=dt, source_wdm=sg if s == "wdm" else None,
                             target_wdm=tg if t == "wdm" else None)
-            _close(got, z[f"y{i}"], 1e-12)
+            _close(got, y, 1e-12)
             if t == "frequency":
                 assert not np.any(got[..., 0].imag), i
                 assert np.all(got[..., -1].imag) if n % 2 else not np.any(got[..., -1].imag), i
-            if t == "wdm":
-                assert not np.any(got[..., ~wilson.grid.active_mask(*tg)]), i
 
 
 def test_round_trips_leading_axes_and_parseval():
-    """Every pair maps X's three layouts onto each other and back, with a non-default window;
-    each row is the kernel's; the energy scales by dt on WDM and dt^2 N on frequency, also at
-    odd N."""
+    """Every pair, a domain onto itself included, maps X's three layouts onto each other and
+    back, with a non-default window; each row is the kernel's; the energy scales by dt on the
+    folded WDM array and dt^2 N on frequency, also at odd N."""
     on = {k: _on(k, window=ALT_WINDOW) for k in KINDS}
-    assert np.array_equal(on["time"], X) and on["wdm"].shape == (3, 2, 9, 16)
+    assert np.array_equal(on["time"], X) and on["wdm"].shape == (3, 2, 8, 16)
     _close(on["frequency"], DT * np.fft.rfft(X), 1e-15)
     _close(on["wdm"][2, 1], wilson.forward_time(X[2, 1], *G, DT, window=ALT_WINDOW), 1e-15)
-    for s, t in itertools.permutations(KINDS, 2):
+    for s, t in itertools.product(KINDS, repeat=2):
         y = transform(on[s], s, t, **_kw(s, t, window=ALT_WINDOW))
         _close(y, on[t], 1e-12)
         _close(transform(y, t, s, **_kw(t, s, window=ALT_WINDOW)), on[s], 1e-12)
@@ -76,26 +77,23 @@ def test_round_trips_leading_axes_and_parseval():
 
 
 def test_a_residue_up_to_1e_10_of_the_peak_is_accepted_and_the_copy_is_canonical():
-    """1e-12 of the slice's peak at DC, at Nyquist or in an inactive cell passes, and
-    ``source == target`` returns a fresh canonical copy; 1e-6 is refused, naming the place."""
-    for kind, at, name in (("frequency", (0,), r"bin 0 \(DC\)"), ("frequency", (64,), r"bin 64 \(Nyquist\)"),
-                           ("wdm", (0, 1), r"cell \(m=0, n=1\)"), ("wdm", (8, 15), r"cell \(m=8, n=15\)")):
-        x = _on(kind)
-        unit = 1j if kind == "frequency" else 1.0
-        peak = np.max(np.abs(x[2, 1]))
+    """1e-12 of the slice's peak at DC or at Nyquist passes, and ``source == target`` returns a
+    fresh canonical copy; 1e-6 is refused, naming the bin."""
+    kw, x = _kw("frequency", "time"), _on("frequency")
+    peak = np.max(np.abs(x[2, 1]))
+    for at, name in ((0, r"bin 0 \(DC\)"), (64, r"bin 64 \(Nyquist\)")):
         r = x.copy()
-        r[(2, 1, *at)] += 1e-12 * peak * unit
-        same = transform(r, kind, kind, **_kw(kind, kind))
+        r[2, 1, at] += 1e-12j * peak
+        same = transform(r, "frequency", "frequency", **kw)
         assert np.array_equal(same, x) and not np.shares_memory(same, r)
-        assert np.array_equal(transform(r, kind, "time", **_kw(kind, "time")),
-                              transform(x, kind, "time", **_kw(kind, "time")))
-        r[(2, 1, *at)] += 1e-6 * peak * unit
+        assert np.array_equal(transform(r, "frequency", "time", **kw),
+                              transform(x, "frequency", "time", **kw))
+        r[2, 1, at] += 1e-6j * peak
         with pytest.raises(ValueError, match=rf"^x\[2, 1\]: .*{name} is 1e-06 of the slice's peak; "
                                              r"at most 1e-10 is accepted$"):
-            transform(r, kind, "time", **_kw(kind, "time"))
+            transform(r, "frequency", "time", **kw)
     real = np.arange(65)                                                         # integers too
-    assert np.array_equal(transform(real, "frequency", "frequency", **_kw("frequency", "frequency")),
-                          real.astype(np.complex128))
+    assert np.array_equal(transform(real, "frequency", "frequency", **kw), real.astype(np.complex128))
 
 
 def test_wdm_to_wdm_across_two_grids_is_the_chain_through_frequency():
@@ -128,7 +126,11 @@ def test_every_refusal():
         (t, "time", "frequency", {"window": "garbage"}, TypeError, "window must be a domainia.wilson"),
         (t, "time", "wdm", {"dt": 0.0}, ValueError, "dt must be"),
         (t[:-1], "time", "wdm", {}, ValueError, r"shape \(127,\)"),
-        (w.T, "wdm", "time", {}, ValueError, r"shape \(16, 9\)"),
+        (w.T, "wdm", "time", {}, ValueError, r"shape \(16, 8\)"),
+        (wilson.grid.unfold(w, *G), "wdm", "time", {}, ValueError,
+         r"shape \(9, 16\), .*; fold a padded .* with wilson\.grid\.fold first$"),
+        (w.ravel(), "wdm", "time", {}, ValueError,
+         r"shape \(128,\), .*; reshape a flat .* with \.reshape\(\.\.\., 8, 16\) first$"),
         (np.where(np.arange(N) == 3, np.nan, t), "time", "wdm", {}, ValueError, "not finite"),
         (t + 0j, "time", "frequency", {}, ValueError, "complex"),
         (w + 0j, "wdm", "time", {}, ValueError, "complex"),
diff --git a/tests/wilson/meyer_cases.py b/tests/wilson/meyer_cases.py
index e0f4aaf..3fd0b48 100644
--- a/tests/wilson/meyer_cases.py
+++ b/tests/wilson/meyer_cases.py
@@ -45,6 +45,9 @@ DROPPED = ("domainia::TimeToWDM.", "domainia::WDMToTime.", "domainia::FourierToW
            "wilson::transform.forward_bands::", "wilson::transform.inverse_bands::")
 #: The comparison: ``max |got - want| <= RTOL * s`` (see :func:`mismatches`).
 RTOL = 1e-12
+#: Families recorded padded ``(..., Nf+1, Nt)`` whose kernels now return the folded layout.
+FOLDED = ("wilson::transform.forward", "wilson::transform.forward_many",
+          "wilson::transform.forward_time", "wilson::transform.forward_time_many")
 
 Case = tuple[str, Callable[[dict], Any]]
 
@@ -57,9 +60,17 @@ def _family(key: str) -> str:
 def load_baseline() -> dict[str, np.ndarray]:
     """The recorded baseline as ``{key: array}``.  The families whose kernels now return the
     folded WDM layout are converted here, never in :func:`mismatches` (folded WDM spec,
-    section 5)."""
+    section 5): the :data:`FOLDED` ones are gathered into the folded slots, without
+    ``wilson.grid.fold``'s check of the inactive cells."""
+    from domainia.wilson.grid import cells
+
     with np.load(BASELINE) as z:
-        return {k: z[k] for k in z.files}
+        out = {k: z[k] for k in z.files}
+    for k, v in out.items():
+        if _family(k) in FOLDED:
+            m, n = cells(*map(int, k.split("::")[2].split("x")))
+            out[k] = v[..., m, n]
+    return out
 
 
 def mismatches(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> list[str]:
@@ -150,7 +161,8 @@ def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
          lambda kw: St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, **kw)),
         ("stationary.build_K_full_active", lambda kw: St.build_K_full_active(Nf, Nt, dt, psd, **kw)),
         ("stationary.psd_from_K", lambda kw: St.psd_from_K(K0, Nf, Nt, dt, **kw)),
-        ("stationary.apply_lambda_from_psd", lambda kw: St.apply_lambda_from_psd(w, Nf, Nt, dt, S, **kw)),
+        ("stationary.apply_lambda_from_psd",
+         lambda kw: St.apply_lambda_from_psd(wl.grid.unfold(w, Nf, Nt), Nf, Nt, dt, S, **kw)),
         ("stationary.psd_from_lambda_blocks", lambda kw: St.psd_from_lambda_blocks(*B0, Nf, Nt, dt, **kw)),
         ("noise.band_averaged_variance", lambda kw: No.band_averaged_variance(Nf, Nt, dt, psd, **kw)),
         ("noise.evolutionary_band_variance",
@@ -159,7 +171,8 @@ def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
          lambda kw: testing.empirical_variance(Nf, Nt, dt, psd, 3, seed=11, **kw)),
         # w2 is a length-N time-domain weight (rho_p = sum_t w2(t) g_p(t)^2 / dt).
         ("evolutionary.energy_fraction", lambda kw: E.energy_fraction(x ** 2, Nf, Nt, dt, **kw)),
-        ("evolutionary.apply_time_mask", lambda kw: E.apply_time_mask(Wm[:, mask], chi, Nf, Nt, dt, **kw)),
+        ("evolutionary.apply_time_mask",
+         lambda kw: E.apply_time_mask(wl.grid.unfold(Wm, Nf, Nt)[:, mask], chi, Nf, Nt, dt, **kw)),
         # nan (inactive cells) -> -1, as recorded; the fractions lie in [0, 1].
         ("evolutionary.wrap_fractions",
          lambda kw: np.nan_to_num(E.wrap_fractions(Nf, Nt, dt, **kw), nan=-1.0)),
diff --git a/tests/wilson/test_grid_bands.py b/tests/wilson/test_grid_bands.py
index d5f789f..7ee16e0 100644
--- a/tests/wilson/test_grid_bands.py
+++ b/tests/wilson/test_grid_bands.py
@@ -139,7 +139,7 @@ def test_unrepresentable_delta_F_raises_instead_of_returning_zeros(dt):
     calls = {
         "phi_centered": lambda: phi_centered(Nf, Nt, dt),
         "forward_time": lambda: forward_time(np.ones(Nf * Nt), Nf, Nt, dt),
-        "inverse_time": lambda: wilson.inverse_time(np.ones((Nf + 1, Nt)), Nf, Nt, dt),
+        "inverse_time": lambda: wilson.inverse_time(np.ones((Nf, Nt)), Nf, Nt, dt),
     }
     for name, call in calls.items():
         with pytest.raises(ValueError, match="delta_F is not representable"):
diff --git a/tests/wilson/test_transform_shim.py b/tests/wilson/test_transform_shim.py
index 983d8f3..7ef5a65 100644
--- a/tests/wilson/test_transform_shim.py
+++ b/tests/wilson/test_transform_shim.py
@@ -4,7 +4,9 @@ It matches to rounding on random grids, dt != 1, one record and a stack of three
 Bit-identical up to v0.11.0; the Hz window of v0.12.0 (Paper I eq:Meyer-def: phi
 flat value 1/sqrt(Delta F), kappa = sqrt(2)/Nf, lambda = 1/(sqrt(2) dt), against the
 oracle's omega-normalised 1/sqrt(2 pi Delta F), 2 sqrt(pi)/Nf, sqrt(pi)/dt) is the
-same product rounded differently, so the gate is ORACLE_RTOL * max|ref|.
+same product rounded differently, so the gate is ORACLE_RTOL * max|ref|.  The oracle is
+padded: its forward output is gathered into the folded slots, and its inverse takes our
+folded input unfolded.
 """
 
 from __future__ import annotations
@@ -14,6 +16,7 @@ import pytest
 import vendored_2f7136c.transform as oracle
 
 import domainia.wilson.transform as T
+from domainia.wilson.grid import cells, unfold
 
 GRIDS = [(2, 2, 1.0), (4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10),
          (32, 64, 10.0), (64, 256, 2.0**-10), (6, 10, 0.3)]
@@ -41,20 +44,22 @@ def test_forward_family_is_the_oracle_to_rounding(grid, rng):
         (T.forward(np.fft.rfft(X), Nf, Nt, dt),
          np.stack([oracle.forward(np.fft.rfft(r), Nf, Nt, dt) for r in X])),
     ]
+    m, n = cells(Nf, Nt)
     for got, ref in pairs:
-        _assert_is_the_oracle_to_rounding(got, ref)
+        _assert_is_the_oracle_to_rounding(got, ref[..., m, n])
 
 
 @pytest.mark.parametrize("grid", GRIDS)
 def test_inverse_family_is_the_oracle_to_rounding(grid, rng):
     Nf, Nt, dt = grid
-    w = rng.standard_normal((Nf + 1, Nt))          # junk in inactive pixels on purpose
-    W = rng.standard_normal((3, Nf + 1, Nt))
+    w = rng.standard_normal((Nf, Nt))
+    W = rng.standard_normal((3, Nf, Nt))
+    p, P = unfold(w, Nf, Nt), unfold(W, Nf, Nt)
     pairs = [
-        (T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
-        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
-        (T.inverse(W, Nf, Nt, dt), np.stack([oracle.inverse(r, Nf, Nt, dt) for r in W])),
-        (T.inverse_time(W, Nf, Nt, dt), np.stack([oracle.inverse_time(r, Nf, Nt, dt) for r in W])),
+        (T.inverse(w, Nf, Nt, dt), oracle.inverse(p, Nf, Nt, dt)),
+        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(p, Nf, Nt, dt)),
+        (T.inverse(W, Nf, Nt, dt), np.stack([oracle.inverse(r, Nf, Nt, dt) for r in P])),
+        (T.inverse_time(W, Nf, Nt, dt), np.stack([oracle.inverse_time(r, Nf, Nt, dt) for r in P])),
     ]
     for got, ref in pairs:
         _assert_is_the_oracle_to_rounding(got, ref)
@@ -62,20 +67,28 @@ def test_inverse_family_is_the_oracle_to_rounding(grid, rng):
 
 def test_leading_axes_are_the_loop_over_the_slices(rng):
     """Each kernel takes any leading axes: a (3, 2, ...) stack gives the loop over its
-    slices, and an input one entry short on its last axis is refused."""
+    slices, and an input one entry short on its last axis is refused.  The inverses refuse a
+    padded and a flat grid too, naming the fix."""
     Nf, Nt, dt = 8, 16, 0.5
     x = rng.standard_normal((3, 2, Nf * Nt))
-    w = rng.standard_normal((3, 2, Nf + 1, Nt))
+    w = rng.standard_normal((3, 2, Nf, Nt))
     for f, a in ((T.forward, np.fft.rfft(x)), (T.forward_time, x), (T.inverse, w), (T.inverse_time, w)):
         want = np.stack([np.stack([f(s, Nf, Nt, dt) for s in row]) for row in a])
         _assert_is_the_oracle_to_rounding(f(a, Nf, Nt, dt), want)
         with pytest.raises(ValueError, match="must have shape"):
             f(a[..., 1:], Nf, Nt, dt)
+    for f in (T.inverse, T.inverse_time):
+        with pytest.raises(ValueError, match=r"got \(3, 2, 9, 16\); fold a padded .* with "
+                                             r"wilson\.grid\.fold first$"):
+            f(unfold(w, Nf, Nt), Nf, Nt, dt)
+        with pytest.raises(ValueError, match=r"got \(3, 2, 128\); reshape a flat .* with "
+                                             r"\.reshape\(\.\.\., 8, 16\) first$"):
+            f(x, Nf, Nt, dt)
 
 
 def test_inputs_are_not_mutated(rng):
     Nf, Nt, dt = 8, 16, 0.5
-    w = rng.standard_normal((Nf + 1, Nt))
+    w = rng.standard_normal((Nf, Nt))
     w0 = w.copy()
     T.inverse(w, Nf, Nt, dt)
     assert np.array_equal(w, w0)
diff --git a/tests/wilson/test_window_threading.py b/tests/wilson/test_window_threading.py
index 10e47b3..bd4d15a 100644
--- a/tests/wilson/test_window_threading.py
+++ b/tests/wilson/test_window_threading.py
@@ -33,12 +33,11 @@ MEYER_HELPERS = frozenset({"phi_centered"})
 
 #: Every function that samples the window, directly or through another consumer.
 CONSUMERS = frozenset({
-    "_gather_bands", "forward", "forward_time", "inverse", "inverse_time", "base_atom_spectra",
-    "atom_spectrum",
+    "_gather_bands", "forward", "forward_time", "inverse", "inverse_time", "_forward",
+    "_forward_time", "_inverse", "_inverse_time", "base_atom_spectra", "atom_spectrum",
     "build_W", "active_W", "build_C_wdm", "build_C_wdm_active",
     "_slab_K", "build_lambda_blocks_full_active", "build_K_full_active", "psd_from_K",
-    "apply_lambda_from_psd", "psd_from_lambda_blocks", "_forward", "_inverse", "_forward_wdm",
-    "_inverse_wdm",
+    "apply_lambda_from_psd", "psd_from_lambda_blocks",
     "band_averaged_variance", "evolutionary_band_variance", "empirical_variance",
     "energy_fraction", "apply_time_mask", "wrap_fractions", "wrap_cut_columns", "exact_covariance",
     "eq45_plan",
@@ -374,12 +373,12 @@ def _close(got, want, rtol: float, what: str) -> None:
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_build_W_is_the_impulse_response(window, Nf, Nt, dt):
-    """Column ``k`` of ``W`` is ``forward_time(e_k)`` (the definition; ``build_W`` probes
-    ``2 (Nf+1)`` rows and rolls them by ``2 Nf`` samples instead), and the other matrix
-    builders follow from it."""
+    """Column ``k`` of ``W`` is ``forward_time(e_k)``, unfolded to ``build_W``'s padded rows (the
+    definition; ``build_W`` probes ``2 (Nf+1)`` rows and rolls them by ``2 Nf`` samples
+    instead), and the other matrix builders follow from it."""
     N = Nf * Nt
-    W_def = np.stack([wilson.forward_time(e, Nf, Nt, dt, window=window).ravel() for e in np.eye(N)],
-                     axis=1)
+    W_def = wilson.grid.unfold(wilson.forward_time(np.eye(N), Nf, Nt, dt, window=window), Nf, Nt)
+    W_def = W_def.reshape(N, -1).T
     mask = wilson.grid.active_mask(Nf, Nt).ravel()
     C_def = W_def @ M.build_C_time(N, dt, psd) @ W_def.T
     _close(M.build_W(Nf, Nt, dt, window=window), W_def, RTOL, "build_W")
diff --git a/tests/wilson/vendored/test_evolutionary.py b/tests/wilson/vendored/test_evolutionary.py
index 32fb1a1..13c6f60 100644
--- a/tests/wilson/vendored/test_evolutionary.py
+++ b/tests/wilson/vendored/test_evolutionary.py
@@ -10,7 +10,7 @@ import pytest
 # average and exact covariance against band_averaged_variance, build_C_time and
 # build_C_wdm_active.  tests/wilson/test_evolutionary_kernels.py checks the kernels against it.
 from wdm_dev_ev import evolutionary as ev
-from domainia.wilson.grid import active_mask
+from domainia.wilson.grid import active_mask, unfold
 from domainia.wilson.matrix import active_W
 from domainia.wilson.transform import forward_time
 
@@ -30,7 +30,8 @@ def test_packed_grid_matches_the_dense_transform(rng):
     np.testing.assert_array_equal(g.mask, active_mask(NF, NT))
     x = rng.standard_normal((3, N))
     np.testing.assert_allclose(np.stack([g.forward(r) for r in x]), x @ Wa.T, atol=1e-12)
-    np.testing.assert_allclose(g.forward(x[0]), forward_time(x[0], NF, NT, DT)[g.mask], atol=0)
+    w = unfold(forward_time(x[0], NF, NT, DT), NF, NT)
+    np.testing.assert_allclose(g.forward(x[0]), w[g.mask], atol=0)
 
 
 def test_atoms_are_the_rows_of_the_transform():
diff --git a/tests/wilson/vendored/test_matrix.py b/tests/wilson/vendored/test_matrix.py
index 3581be4..b2a5a1f 100644
--- a/tests/wilson/vendored/test_matrix.py
+++ b/tests/wilson/vendored/test_matrix.py
@@ -4,6 +4,7 @@ from __future__ import annotations
 
 import numpy as np
 
+from domainia.wilson.grid import unfold
 from domainia.wilson.matrix import (
     active_W,
     build_C_time,
@@ -30,7 +31,7 @@ def test_W_matches_FFT_transform(simple_grid):
     rng = np.random.default_rng(0)
     x = rng.standard_normal(g.N)
     w_matrix = (W @ x).reshape(g.Nf + 1, g.Nt)
-    w_fft = forward_time(x, g.Nf, g.Nt, g.dt)
+    w_fft = unfold(forward_time(x, g.Nf, g.Nt, g.dt), g.Nf, g.Nt)
     assert np.allclose(w_matrix, w_fft, atol=1e-13)
 
 
diff --git a/tests/wilson/vendored/test_transform.py b/tests/wilson/vendored/test_transform.py
index 6e0d999..2a5f718 100644
--- a/tests/wilson/vendored/test_transform.py
+++ b/tests/wilson/vendored/test_transform.py
@@ -5,7 +5,7 @@ from __future__ import annotations
 import numpy as np
 import pytest
 
-from domainia.wilson.grid import active_mask
+from domainia.wilson.grid import cells, unfold
 from domainia.wilson.transform import forward, forward_time, inverse, inverse_time
 
 
@@ -26,25 +26,14 @@ def test_parseval_normalized(simple_grid, white_noise):
 
 
 def test_output_shape_dtype(simple_grid):
-    """Forward returns shape (Nf+1, Nt) of real floats."""
+    """Forward returns shape (Nf, Nt) of real floats."""
     g = simple_grid
     x = np.random.default_rng(0).standard_normal(g.N)
     w = forward_time(x, g.Nf, g.Nt, g.dt)
-    assert w.shape == (g.Nf + 1, g.Nt)
+    assert w.shape == (g.Nf, g.Nt)
     assert w.dtype.kind == "f"
 
 
-def test_inactive_pixels_zero(simple_grid, white_noise):
-    """Coefficients at inactive (masked-False) positions vanish to machine precision."""
-    g = simple_grid
-    w = forward_time(white_noise, g.Nf, g.Nt, g.dt)
-    mask = active_mask(g.Nf, g.Nt)
-    # Inactive pixels come from algebraic cancellations and are at the
-    # floating-point noise floor, not bit-exactly zero.
-    scale = np.max(np.abs(w[mask]))
-    assert np.max(np.abs(w[~mask])) < 1e-12 * scale
-
-
 def test_monochromatic_lands_in_band(medium_grid):
     """A pure cosine at frequency m*dF lands in band m."""
     g = medium_grid
@@ -52,7 +41,7 @@ def test_monochromatic_lands_in_band(medium_grid):
     t = np.arange(g.N) * g.dt
     x = np.cos(2 * np.pi * m_signal * g.dF * t)
     w = forward_time(x, g.Nf, g.Nt, g.dt)
-    band_powers = np.sum(w**2, axis=1)
+    band_powers = np.sum(unfold(w, g.Nf, g.Nt) ** 2, axis=1)     # rows are bands 0..Nf
     assert np.argmax(band_powers) == m_signal
 
 
@@ -73,7 +62,7 @@ def test_linearity(simple_grid):
 def test_inverse_returns_complex_rfft(simple_grid):
     """inverse returns shape (N//2+1,) of complex."""
     g = simple_grid
-    w = np.zeros((g.Nf + 1, g.Nt))
+    w = np.zeros((g.Nf, g.Nt))
     w[5, 3] = 1.0
     out = inverse(w, g.Nf, g.Nt, g.dt)
     assert out.shape == (g.N // 2 + 1,)
@@ -131,5 +120,5 @@ def test_forward_matches_a_naive_reference(simple_grid, white_noise, dt):
     g = simple_grid
     x_rfft = np.fft.rfft(white_noise)
     fast = forward(x_rfft, g.Nf, g.Nt, dt)
-    naive = _forward_reference(x_rfft, g.Nf, g.Nt, dt)
+    naive = _forward_reference(x_rfft, g.Nf, g.Nt, dt)[cells(g.Nf, g.Nt)]   # padded -> folded
     assert np.max(np.abs(fast - naive)) < 1e-14 * np.max(np.abs(naive))
````

- [ ] **Step 2: Run the task's tests: they fail**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error --tb=no -rfE tests/test_transform.py tests/wilson/test_transform_shim.py tests/wilson/vendored/test_transform.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py tests/wilson/test_grid_bands.py tests/test_dom_testing.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_matrix.py tests/wilson/test_wrong_kinds.py 2>&1 | tail -n 40
```

Expected last line (prototype): `60 failed, 525 passed, 1 skipped`. The 60 ids are in `test_transform` (3), `test_transform_shim` (18), vendored `test_transform` (6), `test_window_threading` (24), the replay (3), `test_grid_bands` (3), and one each in `test_dom_testing`, vendored `test_evolutionary` and vendored `test_matrix`. The kernels still return and take `(9, 16)`. So:
- the folded shape asserts fail (`(9, 16) == (8, 16)`);
- the tests' `unfold` of kernel output refuses it (`ValueError: R has shape (9, 16), but must end in (8, 16)`);
- the shim's inverse family hits `wmn must have shape (..., 9, 16); got (8, 16)`;
- the replay fails inside its generators.

- [ ] **Step 3: Apply the implementation patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T2-impl && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists `README.md`, `src/domainia/_transform.py`, `src/domainia/testing.py`, `src/domainia/wilson/evolutionary.py`, `src/domainia/wilson/grid.py`, `src/domainia/wilson/matrix.py`, `src/domainia/wilson/stationary.py`, `src/domainia/wilson/transform.py`, `tests/test_dom_testing.py`, `tests/test_transform.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_transform_shim.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_matrix.py`, `tests/wilson/vendored/test_transform.py` as modified, and nothing else.

<!-- patch: T2-impl -->
````diff
diff --git a/README.md b/README.md
index a393456..5f47bbc 100644
--- a/README.md
+++ b/README.md
@@ -20,21 +20,19 @@ subdirectories each one expects).
 
 ```python
 import numpy as np
-from domainia import transform, wilson
+from domainia import transform
 
 Nf, Nt, dt = 6, 12, 0.3
 N = Nf * Nt
 x = np.random.default_rng(6).standard_normal((2, N))            # any leading axes
 
-w = transform(x, "time", "wdm", N=N, dt=dt, target_wdm=(Nf, Nt))       # (2, 7, 12) float64
+w = transform(x, "time", "wdm", N=N, dt=dt, target_wdm=(Nf, Nt))       # (2, 6, 12) float64
 X = transform(w, "wdm", "frequency", N=N, dt=dt, source_wdm=(Nf, Nt))  # (2, 37): dt * rfft(x)
 v = transform(X, "frequency", "wdm", N=N, dt=dt, target_wdm=(12, 6))   # another WDM grid
 assert np.allclose(transform(v, "wdm", "time", N=N, dt=dt, source_wdm=(12, 6)), x)
 
-# Raw kernel output (~1e-17 in the inactive cells) and a full-FFT slice (~1e-16 imaginary
-# at Nyquist) are taken as they are; a residue above 1e-10 of the peak is refused.
-assert np.allclose(transform(wilson.forward_time(x, Nf, Nt, dt), "wdm", "time", N=N, dt=dt,
-                             source_wdm=(Nf, Nt)), x)
+# A full-FFT slice (~1e-16 imaginary at Nyquist) is taken as it is; a residue above 1e-10
+# of the peak is refused.
 assert np.allclose(transform(dt * np.fft.fft(x)[:, : N // 2 + 1], "frequency", "time",
                              N=N, dt=dt), x)
 ```
diff --git a/src/domainia/_transform.py b/src/domainia/_transform.py
index 7fecdbe..8e54fe1 100644
--- a/src/domainia/_transform.py
+++ b/src/domainia/_transform.py
@@ -4,8 +4,8 @@ The layouts, on the trailing axes of an array with any leading axes (typically `
 
 * ``"time"``: real ``(N,)``;
 * ``"frequency"``: complex ``(N // 2 + 1,)``, ``dt * numpy.fft.rfft(x)``;
-* ``"wdm"``: real ``(Nf + 1, Nt)``, the Wilson coefficients of ``window``, edge bands
-  included.
+* ``"wdm"``: real ``(Nf, Nt)``, the Wilson coefficients of ``window`` in the folded layout
+  (:func:`domainia.wilson.grid.cells` gives each slot's band and time).
 
 Every route goes through the rfft ``numpy.fft.rfft(x)``.
 """
@@ -19,7 +19,7 @@ import numpy as np
 from . import wilson
 from .wilson import MEYER, Window
 from .wilson._checks import require_dtype, require_int
-from .wilson.grid import RESIDUE
+from .wilson.grid import RESIDUE, _fold_fix
 
 __all__ = ["transform"]
 
@@ -59,30 +59,25 @@ def _checked(x: Any, kind: str, N: int, grid: tuple[int, int] | None) -> np.ndar
     require_dtype("x", a, "iufc", "a numeric array")
     if a.dtype.kind == "c" and kind != "frequency":
         raise ValueError(f"x is complex, but {kind} data are real")
-    shape = {"time": (N,), "frequency": (N // 2 + 1,)}.get(kind) or (grid[0] + 1, grid[1])
-    nd = len(shape)
-    if a.shape[a.ndim - nd:] != shape:
-        raise ValueError(f"x has shape {a.shape}, but {kind} data end in shape {shape}")
+    shape = {"time": (N,), "frequency": (N // 2 + 1,)}.get(kind) or grid
+    if a.shape[a.ndim - len(shape):] != shape:
+        raise ValueError(f"x has shape {a.shape}, but {kind} data end in shape {shape}"
+                         + (_fold_fix(a.shape, *grid) if grid else ""))
     wilson.grid.validate_finite(a, "x")
     a = a.astype(np.complex128 if kind == "frequency" else np.float64)
-    if kind == "time":
+    if kind != "frequency":
         return a
-    if kind == "frequency":
-        ends = np.zeros(shape, bool)
-        ends[-1] = N % 2 == 0
-        ends[0] = True
-        off = np.where(ends, np.abs(a.imag), 0.0)
-    else:
-        off = np.where(wilson.grid.active_mask(*grid), 0.0, np.abs(a))
-    peak = np.max(np.abs(a), axis=tuple(range(-nd, 0)), keepdims=True)
+    ends = np.zeros(shape, bool)
+    ends[-1] = N % 2 == 0
+    ends[0] = True
+    off = np.where(ends, np.abs(a.imag), 0.0)
+    peak = np.max(np.abs(a), axis=-1, keepdims=True)
     rel = np.divide(off, peak, out=np.zeros_like(off), where=peak > 0)
     bad = np.argwhere(rel > RESIDUE)
     if bad.size:
-        at = [int(i) for i in bad[0]]
-        lead, k = at[:-nd], at[-nd:]
-        where = (f"the imaginary part of rfft bin {k[0]} ({'Nyquist' if k[0] else 'DC'})"
-                 if kind == "frequency" else f"inactive WDM cell (m={k[0]}, n={k[1]})")
-        raise ValueError(f"x{lead if lead else ''}: {where} is {rel[tuple(at)]:.2g} of the "
+        *lead, k = (int(i) for i in bad[0])
+        raise ValueError(f"x{lead if lead else ''}: the imaginary part of rfft bin {k} "
+                         f"({'Nyquist' if k else 'DC'}) is {rel[tuple(bad[0])]:.2g} of the "
                          f"slice's peak; at most {RESIDUE:g} is accepted")
     return a
 
@@ -99,17 +94,20 @@ def transform(x: Any, source: str, target: str, *, N: int, dt: float,
     regrids through the rfft.  When the source equals the target, grid included, the result
     is a validated canonical copy.
 
-    The result is float64 (complex128 on frequency) and exactly canonical: inactive WDM cells
-    are 0, and so are the imaginary parts at DC and even-``N`` Nyquist.  An input may carry a
-    residue there up to :data:`RESIDUE` of its slice's peak (``max |x|`` over the domain axes,
-    for each leading index); a larger one is refused, naming the cell or bin.
+    The result is float64 (complex128 on frequency); on frequency it is exactly canonical, with
+    the imaginary parts at DC and even-``N`` Nyquist 0.  A frequency input may carry a residue
+    there up to :data:`~domainia.wilson.grid.RESIDUE` of its slice's peak (``max |x|`` over the
+    last axis, for each leading index); a larger one is refused, naming the bin.  A WDM output
+    is the folded output of :func:`domainia.wilson.forward` as it is.
 
     Raises ``TypeError`` for a masked or non-numeric ``x`` (bool, str, object), a non-string
     ``source``/``target``, a WDM grid that is not a sequence, a non-integer ``N``, ``Nf`` or
     ``Nt``, a non-real ``dt`` or a non-window ``window``; ``ValueError`` for an unknown domain,
     a missing or misplaced WDM grid or one that is not a valid ``(Nf, Nt)`` pair, ``N < 1``, a
-    non-positive or non-finite ``dt``, complex ``x`` on time or WDM, a wrong trailing shape, a
-    non-finite value, or a residue above :data:`RESIDUE`; and ``WindowError``, a
+    non-positive or non-finite ``dt``, complex ``x`` on time or WDM, a wrong trailing shape (for
+    a padded ``(..., Nf+1, Nt)`` or a flat ``(..., N)`` WDM ``x`` the message names the fix,
+    ``wilson.grid.fold`` or ``.reshape(..., Nf, Nt)``), a non-finite value, or a residue above
+    :data:`~domainia.wilson.grid.RESIDUE`; and ``WindowError``, a
     ``ValueError``, for a window inadmissible on a WDM grid.  A real ``x`` on frequency is
     taken as complex.
     """
@@ -138,6 +136,4 @@ def transform(x: Any, source: str, target: str, *, N: int, dt: float,
         y[..., 0] = y[..., 0].real
         if N % 2 == 0:
             y[..., -1] = y[..., -1].real
-    elif target == "wdm":
-        y[..., ~wilson.grid.active_mask(*tg)] = 0.0
     return y
diff --git a/src/domainia/testing.py b/src/domainia/testing.py
index 3ffa689..2e73a83 100644
--- a/src/domainia/testing.py
+++ b/src/domainia/testing.py
@@ -38,7 +38,7 @@ from domainia import wilson
 from domainia.wilson._checks import require_callable, require_dtype, require_int
 from domainia.wilson.evolutionary import eq45_direct, eq45_plan
 from domainia.wilson.grid import n_samples, sanitize_psd, validate_dt, validate_grid
-from domainia.wilson.transform import forward
+from domainia.wilson.transform import _forward
 from domainia.wilson.windows import MEYER, SampledWindow, Window
 
 __all__ = ["ALT_WINDOW", "colored_noise", "dense_W", "empirical_variance", "eq45_dense",
@@ -159,7 +159,7 @@ def empirical_variance(
     accum = np.zeros((Nf + 1, Nt))
     for _ in range(n_realizations):
         x_rfft = colored_noise(N, dt, Sn_func, rng)
-        w = forward(x_rfft, Nf, Nt, dt, window=window)
+        w = _forward(x_rfft, Nf, Nt, dt, window=window)
         accum += w * w
     return accum / n_realizations
 
diff --git a/src/domainia/wilson/evolutionary.py b/src/domainia/wilson/evolutionary.py
index 177f56d..d2e0b2e 100644
--- a/src/domainia/wilson/evolutionary.py
+++ b/src/domainia/wilson/evolutionary.py
@@ -54,10 +54,10 @@ from .grid import (
 )
 from .noise import _psd_at
 from .transform import (
+    _forward_time,
+    _inverse_time,
     atom_spectrum,
     base_atom_spectra,
-    forward_time,
-    inverse_time,
 )
 from .windows import MEYER, Window
 
@@ -130,8 +130,8 @@ def apply_time_mask(Y: Array, chi: Array, Nf: int, Nt: int, dt: float, *,
     Y = np.atleast_2d(Y)
     full = np.zeros((Y.shape[0], Nf + 1, Nt))
     full[:, mask] = Y
-    X = inverse_time(full, Nf, Nt, dt, window=window) * chi
-    return forward_time(X, Nf, Nt, dt, window=window)[:, mask]
+    X = _inverse_time(full, Nf, Nt, dt, window=window) * chi
+    return _forward_time(X, Nf, Nt, dt, window=window)[:, mask]
 
 
 # ---------------------------------------------------------------------------
@@ -162,7 +162,7 @@ def wrap_fractions(Nf: int, Nt: int, dt: float, *, bands: Array | None = None,
                 continue
             e = np.zeros((Nf + 1, Nt))
             e[m, a] = 1.0
-            g2 = inverse_time(e, Nf, Nt, dt, window=window) ** 2        # base atom, centre a Nf
+            g2 = _inverse_time(e, Nf, Nt, dt, window=window) ** 2       # base atom, centre a Nf
             delta = (np.arange(N) - a * Nf + N // 2) % N - N // 2       # signed offset from centre
             order = np.argsort(delta, kind="stable")
             ds, cum = delta[order], np.concatenate([[0.0], np.cumsum(g2[order])])
@@ -250,8 +250,8 @@ def exact_covariance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc, *,
     validate_grid(Nf, Nt)
     N = Nf * Nt
     G = time_domain_factor(N, dt, Sn)
-    A = forward_time(np.ascontiguousarray(G.T), Nf, Nt, dt,
-                     window=window)[:, active_mask(Nf, Nt)].T
+    A = _forward_time(np.ascontiguousarray(G.T), Nf, Nt, dt,
+                      window=window)[:, active_mask(Nf, Nt)].T
     Lam = A @ A.T
     return 0.5 * (Lam + Lam.T)
 
diff --git a/src/domainia/wilson/grid.py b/src/domainia/wilson/grid.py
index f9c777a..7180eda 100644
--- a/src/domainia/wilson/grid.py
+++ b/src/domainia/wilson/grid.py
@@ -70,7 +70,7 @@ class Grid:
     (128, 0.0625, 8.0)
     >>> x = np.random.default_rng(0).standard_normal(g.N)
     >>> wilson.forward_time(x, *g).shape          # unpacks as (Nf, Nt, dt)
-    (9, 16)
+    (8, 16)
     >>> wilson.Grid(Nf=9, Nt=16, dt=1.0)
     Traceback (most recent call last):
     ValueError: Nf must be even, got 9
@@ -360,6 +360,16 @@ def _wdm_array(name: str, x: object, shape: tuple[int, int]) -> np.ndarray:
     return a
 
 
+def _fold_fix(shape: tuple[int, ...], Nf: int, Nt: int) -> str:
+    """The fix a WDM array of ``shape`` needs if it is padded or flat, as the end of the
+    kernels' wrong-shape message (``""`` for any other shape)."""
+    if shape[-2:] == (Nf + 1, Nt):
+        return "; fold a padded (..., Nf+1, Nt) array with wilson.grid.fold first"
+    if shape[-1:] == (Nf * Nt,):
+        return f"; reshape a flat (..., N) array with .reshape(..., {Nf}, {Nt}) first"
+    return ""
+
+
 def fold(P: object, Nf: int, Nt: int) -> np.ndarray:
     """The padded ``(..., Nf+1, Nt)`` array ``P`` in the folded layout ``(..., Nf, Nt)``:
     ``P[..., m, n]`` with ``m, n = cells(Nf, Nt)``, in ``P``'s dtype (a boolean mask folds too).
diff --git a/src/domainia/wilson/matrix.py b/src/domainia/wilson/matrix.py
index bc308e2..5a50c79 100644
--- a/src/domainia/wilson/matrix.py
+++ b/src/domainia/wilson/matrix.py
@@ -48,7 +48,7 @@ from .grid import (
     validate_dt,
     validate_grid,
 )
-from .transform import inverse_time
+from .transform import _inverse_time
 from .windows import MEYER, Window
 
 type Array = NDArray[np.floating]
@@ -121,7 +121,7 @@ def build_W(
                 continue
             seed[:] = 0.0
             seed[m, n0] = 1.0
-            row = dt * inverse_time(seed, Nf, Nt, dt, window=window)
+            row = dt * _inverse_time(seed, Nf, Nt, dt, window=window)
             for n in range(n0, Nt, 2):
                 W[m, n] = np.roll(row, (n - n0) * Nf)
 
diff --git a/src/domainia/wilson/stationary.py b/src/domainia/wilson/stationary.py
index 21a7244..7b8b30c 100644
--- a/src/domainia/wilson/stationary.py
+++ b/src/domainia/wilson/stationary.py
@@ -57,7 +57,7 @@ import numpy as np
 from numpy.typing import NDArray
 
 from ._checks import REAL_KINDS, require_dtype
-from .grid import n_samples, sanitize_psd, validate_dt, validate_grid
+from .grid import active_mask, n_samples, sanitize_psd, validate_dt, validate_grid
 from .windows import MEYER, Window, sample_window
 
 type Array = NDArray[np.floating]
@@ -740,15 +740,14 @@ def apply_lambda_from_psd(
     >>> import numpy as np; from domainia import wilson
     >>> Nf, Nt, dt = 8, 16, 1.0
     >>> N = Nf * Nt
-    >>> x = wilson.forward_time(
+    >>> x = wilson.grid.unfold(wilson.forward_time(
     ...     np.random.default_rng(0).standard_normal(N), Nf, Nt, dt
-    ... )
+    ... ), Nf, Nt)
     >>> Sn = 1e-3 * np.ones(N // 2 + 1)
     >>> wilson.stationary.apply_lambda_from_psd(x, Nf, Nt, dt, Sn).shape
     (9, 16)
     """
-    from .transform import forward as _forward
-    from .transform import inverse as _inverse
+    from .transform import _forward, _inverse
 
     validate_grid(Nf, Nt)
     validate_dt(dt)
@@ -759,7 +758,7 @@ def apply_lambda_from_psd(
             f"apply_lambda_from_psd: Sn_rfft must have shape ({n_rfft},) -- the "
             f"np.fft.rfftfreq(N, dt) grid; got {Sn_rfft.shape}"
         )
-    y = _inverse(x_wmn, Nf, Nt, dt, window=window)
+    y = _inverse(x_wmn * active_mask(Nf, Nt), Nf, Nt, dt, window=window)    # inactive: ignored
     z = (Sn_rfft / 2.0) * y
     return _forward(z, Nf, Nt, dt, window=window)
 
@@ -820,8 +819,7 @@ def psd_from_lambda_blocks(
     >>> Sn.shape
     (65,)
     """
-    from .transform import forward as _forward_wdm
-    from .transform import inverse as _inverse_wdm
+    from .transform import _forward, _inverse
 
     validate_grid(Nf, Nt)
     validate_dt(dt)
@@ -829,11 +827,11 @@ def psd_from_lambda_blocks(
     N = n_samples(Nf, Nt)
 
     ones_rfft = np.ones(N // 2 + 1, dtype=complex)
-    bar_w = _forward_wdm(ones_rfft, Nf, Nt, dt, window=window)
+    bar_w = _forward(ones_rfft, Nf, Nt, dt, window=window)
 
     u = _block_tridiag_matvec(diag_blocks, offd_blocks, n_active, bar_w)
 
-    X = _inverse_wdm(u, Nf, Nt, dt, window=window)
+    X = _inverse(u, Nf, Nt, dt, window=window)
 
     return 2.0 * X.real
 
diff --git a/src/domainia/wilson/transform.py b/src/domainia/wilson/transform.py
index bd387c2..bb0c6e0 100644
--- a/src/domainia/wilson/transform.py
+++ b/src/domainia/wilson/transform.py
@@ -14,6 +14,11 @@ computes in the Wilson basis of that window: $\\hat\\phi_c$ below is the window'
 centred samples, from :func:`domainia.wilson.windows.sample_window`.  The formulas
 and the edge normalisation are the same for every admissible window.
 
+The public kernels take and return the folded layout ``(..., Nf, Nt)``
+(:func:`domainia.wilson.grid.cells`).  Each computes through a private twin (``_forward``, ...)
+on the padded ``(..., Nf+1, Nt)`` grid, where row ``m`` is band ``m``; the other kernels call
+the twins.
+
 The slab is linear-centred:
 
 $$\\ell_{m,k} \\equiv k + (m - 1)\\,N_t/2,\\qquad k = 0, \\ldots, N_t - 1.$$
@@ -60,8 +65,11 @@ from numpy.typing import NDArray
 
 from ._checks import REAL_KINDS, require_dtype, require_int
 from .grid import (
+    _fold_fix,
     active_mask,
+    cells,
     n_samples,
+    unfold,
     validate_dt,
     validate_finite,
     validate_grid,
@@ -123,24 +131,6 @@ def _build_slab(X_rfft: CArray, m: int, Nf: int, Nt: int) -> CArray:
     return slab
 
 
-def _prepare_edges_for_inverse(wmn: Array, Nf: int, Nt: int) -> Array:
-    """Working copy of ``wmn`` with the edge bands put into ``inverse``'s form.
-
-    Two things, both edge-only:
-
-    1. **Zero the parity-killed entries.** They correspond to zero rows of $W$,
-       so the adjoint must ignore them; otherwise junk there leaks into the
-       active pixels of the reconstruction.
-    2. **Undo the unit-norm $\\sqrt2$** that :func:`forward` applied.
-    """
-    work = wmn.astype(np.float64)      # always a fresh copy, so updating it in place is safe
-    mask = active_mask(Nf, Nt)
-    root2 = np.sqrt(2.0)
-    work[..., 0, :] *= mask[0] * root2
-    work[..., Nf, :] *= mask[Nf] * root2
-    return work
-
-
 def _scatter_interior_bands(
     X: CArray,
     wmn_work: Array,
@@ -206,7 +196,7 @@ def _gather_bands(x_rfft: CArray, Nf: int, Nt: int, dt: float, *,
                   window: Window = MEYER) -> Array:
     """Coefficient grid(s) from spectra: ``(..., N//2+1) -> (..., Nf+1, Nt)``.
 
-    The whole of the forward algorithm; :func:`forward` validates and calls it.
+    The whole of the forward algorithm; :func:`_forward` validates and calls it.
 
     Interior bands $m = 1 \\ldots N_f-1$: every slab index
     $\\ell_{m,k} = k + (m-1)N_t/2$ lies in $[0, N/2]$
@@ -247,6 +237,21 @@ def _gather_bands(x_rfft: CArray, Nf: int, Nt: int, dt: float, *,
     return out
 
 
+def _forward(x_rfft: CArray, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
+    """:func:`forward` on the padded grid: ``(..., N//2+1)`` to ``(..., Nf+1, Nt)``."""
+    validate_grid(Nf, Nt)
+    validate_dt(dt)
+    N = n_samples(Nf, Nt)
+    x_rfft = np.asarray(x_rfft)
+    if x_rfft.shape[-1:] != (N // 2 + 1,):
+        raise ValueError(
+            f"x_rfft must have shape (..., {N // 2 + 1}) for Nf={Nf}, Nt={Nt}; "
+            f"got {x_rfft.shape}"
+        )
+    validate_finite(x_rfft, "forward: x_rfft")
+    return _gather_bands(x_rfft, Nf, Nt, dt, window=window)
+
+
 def forward(
     x_rfft: CArray,
     Nf: int,
@@ -266,11 +271,9 @@ def forward(
 
     Returns
     -------
-    ndarray, float, shape (..., Nf+1, Nt)
-        Real WDM coefficients.  Edge bands have $N_t/2$ active values;
-        the inactive entries vanish by Hermitian symmetry of the slab,
-        realized to floating-point rounding (~1e-16 relative to the
-        peak) rather than bit-exactly.  Compare them with a tolerance.
+    ndarray, float, shape (..., Nf, Nt)
+        Real WDM coefficients in the folded layout: slot ``(i, j)`` holds band ``m[i, j]`` at
+        time ``n[i, j]``, with ``m, n = wilson.grid.cells(Nf, Nt)``.
 
     Raises
     ------
@@ -294,25 +297,38 @@ def forward(
     >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
     >>> w = wilson.forward(np.fft.rfft(x), Nf, Nt, dt)
     >>> w.shape
-    (9, 16)
+    (8, 16)
     >>> bool(np.isclose(np.sum(w**2), dt * np.sum(x**2)))  # Parseval
     True
     >>> xs = np.random.default_rng(1).standard_normal((2, 3, Nf * Nt))   # leading axes
     >>> wilson.forward(np.fft.rfft(xs), Nf, Nt, dt).shape
-    (2, 3, 9, 16)
+    (2, 3, 8, 16)
     """
+    m, n = cells(Nf, Nt)
+    return _forward(x_rfft, Nf, Nt, dt, window=window)[..., m, n]
+
+
+def _inverse(wmn: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> CArray:
+    """:func:`inverse` on the padded grid ``(..., Nf+1, Nt)``, whose inactive cells must be 0."""
     validate_grid(Nf, Nt)
     validate_dt(dt)
     N = n_samples(Nf, Nt)
-    x_rfft = np.asarray(x_rfft)
-    if x_rfft.shape[-1:] != (N // 2 + 1,):
-        raise ValueError(
-            f"x_rfft must have shape (..., {N // 2 + 1}) for Nf={Nf}, Nt={Nt}; "
-            f"got {x_rfft.shape}"
-        )
-    validate_finite(x_rfft, "forward: x_rfft")
-    out = _gather_bands(x_rfft, Nf, Nt, dt, window=window)
-    return out
+    wmn = np.asarray(wmn)
+    # A complex grid would have its imaginary part silently discarded.
+    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
+    if wmn.shape[-2:] != (Nf + 1, Nt):
+        raise ValueError(f"wmn must have shape (..., {Nf + 1}, {Nt}); got {wmn.shape}")
+    validate_finite(wmn, "inverse: wmn")
+
+    phi_c = sample_window(window, Nf, Nt, dt)
+    wmn_work = wmn.astype(np.float64)              # a fresh copy, so scaling it in place is safe
+    wmn_work[..., [0, Nf], :] *= np.sqrt(2.0)      # undo forward's unit-norm edge scaling
+
+    lam = 1.0 / (np.sqrt(2.0) * dt)
+    X = np.zeros((*wmn.shape[:-2], N // 2 + 1), dtype=complex)
+    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
+    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
+    return X
 
 
 def inverse(
@@ -326,12 +342,12 @@ def inverse(
     """Inverse WDM transform (Algorithm 2): $w_{mn} \\to \\text{rfft}(x)$.
 
     Each band scatters into the one-sided rfft accumulator at its linear-centred slab
-    indices.  The inactive edge cells of ``wmn`` are ignored.
+    indices.
 
     Parameters
     ----------
-    wmn : ndarray, float, shape (..., Nf+1, Nt)
-        WDM coefficient grid, with any leading axes.
+    wmn : ndarray, float, shape (..., Nf, Nt)
+        WDM coefficients in the folded layout of :func:`forward`, with any leading axes.
 
     Returns
     -------
@@ -345,8 +361,10 @@ def inverse(
         is not a real numeric array: a complex grid would have its imaginary part
         silently discarded, and a bool, str or object grid is refused too.
     ValueError
-        If the grid is invalid, if ``wmn`` is wrongly shaped, or
-        if it holds a ``nan`` or ``inf`` (see :func:`domainia.wilson.grid.validate_finite`).
+        If the grid is invalid, if ``wmn`` does not end in ``(Nf, Nt)`` (for a padded
+        ``(..., Nf+1, Nt)`` or a flat ``(..., N)`` array the message names the fix,
+        :func:`domainia.wilson.grid.fold` or ``.reshape(..., Nf, Nt)``), or if it holds a
+        ``nan`` or ``inf`` (see :func:`domainia.wilson.grid.validate_finite`).
 
     Examples
     --------
@@ -362,22 +380,32 @@ def inverse(
     """
     validate_grid(Nf, Nt)
     validate_dt(dt)
-    N = n_samples(Nf, Nt)
     wmn = np.asarray(wmn)
-    # A complex grid would have its imaginary part silently discarded.
     require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
-    if wmn.shape[-2:] != (Nf + 1, Nt):
-        raise ValueError(f"wmn must have shape (..., {Nf + 1}, {Nt}); got {wmn.shape}")
+    if wmn.shape[-2:] != (Nf, Nt):
+        raise ValueError(f"wmn must have shape (..., {Nf}, {Nt}); got {wmn.shape}"
+                         + _fold_fix(wmn.shape, Nf, Nt))
     validate_finite(wmn, "inverse: wmn")
+    return _inverse(unfold(wmn, Nf, Nt), Nf, Nt, dt, window=window)
 
-    phi_c = sample_window(window, Nf, Nt, dt)
-    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt)
 
-    lam = 1.0 / (np.sqrt(2.0) * dt)
-    X = np.zeros((*wmn.shape[:-2], N // 2 + 1), dtype=complex)
-    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
-    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
-    return X
+def _forward_time(x: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
+    """:func:`forward_time` on the padded grid: ``(..., N)`` to ``(..., Nf+1, Nt)``."""
+    validate_grid(Nf, Nt)
+    x = np.asarray(x)
+    expected = n_samples(Nf, Nt)
+    if x.shape[-1:] != (expected,):
+        raise ValueError(
+            f"x must have shape (..., {expected}) = (..., Nf*Nt) for Nf={Nf}, Nt={Nt}; "
+            f"got {x.shape}"
+        )
+    # Check here rather than only on the rfft: reporting "index 5 of x" beats
+    # reporting that every bin of the transformed spectrum went non-finite.
+    validate_finite(x, "forward_time: x")
+    # At least double before the rfft, which keeps a lower-precision float's precision.
+    # A complex x stays complex, and the rfft refuses it.
+    x_rfft = np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False))
+    return _forward(x_rfft, Nf, Nt, dt, window=window)
 
 
 def forward_time(
@@ -389,7 +417,7 @@ def forward_time(
     window: Window = MEYER,
 ) -> Array:
     """Time-domain forward transform, ``forward(np.fft.rfft(x), ...)``: a real ``x`` of shape
-    ``(..., N)``, $N = N_f N_t$, to ``(..., Nf+1, Nt)``.
+    ``(..., N)``, $N = N_f N_t$, to the folded ``(..., Nf, Nt)``.
 
     Raises
     ------
@@ -406,27 +434,19 @@ def forward_time(
     >>> Nf, Nt, dt = 8, 16, 1.0
     >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
     >>> wilson.forward_time(x, Nf, Nt, dt).shape
-    (9, 16)
+    (8, 16)
     >>> x[5] = np.nan                      # one gap-filled sample
     >>> wilson.forward_time(x, Nf, Nt, dt)    # doctest: +ELLIPSIS
     Traceback (most recent call last):
     ValueError: forward_time: x: 1 of 128 entries are not finite...
     """
-    validate_grid(Nf, Nt)
-    x = np.asarray(x)
-    expected = n_samples(Nf, Nt)
-    if x.shape[-1:] != (expected,):
-        raise ValueError(
-            f"x must have shape (..., {expected}) = (..., Nf*Nt) for Nf={Nf}, Nt={Nt}; "
-            f"got {x.shape}"
-        )
-    # Check here rather than only on the rfft: reporting "index 5 of x" beats
-    # reporting that every bin of the transformed spectrum went non-finite.
-    validate_finite(x, "forward_time: x")
-    # At least double before the rfft, which keeps a lower-precision float's precision.
-    # A complex x stays complex, and the rfft refuses it.
-    x_rfft = np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False))
-    return forward(x_rfft, Nf, Nt, dt, window=window)
+    m, n = cells(Nf, Nt)
+    return _forward_time(x, Nf, Nt, dt, window=window)[..., m, n]
+
+
+def _inverse_time(wmn: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
+    """:func:`inverse_time` on the padded grid, whose inactive cells must be 0."""
+    return np.fft.irfft(_inverse(wmn, Nf, Nt, dt, window=window), n=n_samples(Nf, Nt))
 
 
 def inverse_time(
@@ -438,7 +458,7 @@ def inverse_time(
     window: Window = MEYER,
 ) -> Array:
     """Time-domain inverse transform, ``np.fft.irfft(inverse(wmn, ...), n=N)``:
-    ``(..., Nf+1, Nt)`` to ``(..., N)``.
+    the folded ``(..., Nf, Nt)`` to ``(..., N)``.
 
     Examples
     --------
@@ -453,10 +473,7 @@ def inverse_time(
     ...                  atol=1e-12))
     True
     """
-    N = n_samples(Nf, Nt)
-    x_rfft = inverse(wmn, Nf, Nt, dt, window=window)
-    x = np.fft.irfft(x_rfft, n=N)
-    return x
+    return np.fft.irfft(inverse(wmn, Nf, Nt, dt, window=window), n=n_samples(Nf, Nt))
 
 
 # ---------------------------------------------------------------------------
@@ -482,7 +499,7 @@ def base_atom_spectra(Nf: int, Nt: int, dt: float, *,
                 continue
             e = np.zeros((Nf + 1, Nt))
             e[m, a] = 1.0
-            out[m, a] = dt * inverse(e, Nf, Nt, dt, window=window)
+            out[m, a] = dt * _inverse(e, Nf, Nt, dt, window=window)
     return out
 
 
````

- [ ] **Step 4: Run the task's tests: they pass**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/test_transform.py tests/wilson/test_transform_shim.py tests/wilson/vendored/test_transform.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py tests/wilson/test_grid_bands.py tests/test_dom_testing.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_matrix.py tests/wilson/test_wrong_kinds.py tests/test_readme.py src/domainia/wilson/transform.py src/domainia/wilson/grid.py src/domainia/wilson/stationary.py 2>&1 | tail -1
```

Expected (prototype): `610 passed, 1 skipped`.

- [ ] **Step 5: Run the lanes, the replay and the size**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py
```

Expected (prototype): `955 passed, 3 skipped` in both lanes, with no warnings and no collection errors; `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`; `src 4194 lines / 1906 code; tests 4412 lines (+ oracle 4135)`.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add README.md src/domainia/_transform.py src/domainia/testing.py src/domainia/wilson/evolutionary.py src/domainia/wilson/grid.py src/domainia/wilson/matrix.py src/domainia/wilson/stationary.py src/domainia/wilson/transform.py tests/test_dom_testing.py tests/test_transform.py tests/wilson/meyer_cases.py tests/wilson/test_grid_bands.py tests/wilson/test_transform_shim.py tests/wilson/test_window_threading.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_matrix.py tests/wilson/vendored/test_transform.py && git commit -q -F - <<'EOF'
Fold the transform kernels' and transform's WDM arrays to (..., Nf, Nt)

forward and forward_time return the folded layout and inverse and inverse_time take it,
through private padded twins (_forward, _inverse, _forward_time, _inverse_time) that the
other kernels call, so their outputs are unchanged until their own task. inverse no longer
zeroes inactive cells: unfold writes the zeros, and apply_lambda_from_psd masks its padded
input itself until it folds. transform drops the WDM residue check and the WDM zeroing.
inverse and transform refuse a padded or flat WDM array, naming wilson.grid.fold or
.reshape(..., Nf, Nt). README's Data example shows the folded shape.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
```


### Task 3: `wilson.matrix` folds; `active_W` and `build_C_wdm_active` go (spec §3 `matrix.py`, §5; §8 step 3)

**Goal:** `build_W` returns `(N, N)` with rows in folded flat order and W Wᵀ = dt·I. `build_C_wdm` returns the folded `(N, N)`. `active_W` and `build_C_wdm_active` are removed, because they would now equal these two. Their 4 replay keys join `DROPPED` (96 → 100).

**Files:**
- Modify: `src/domainia/wilson/matrix.py`
- Modify: `src/domainia/testing.py` (`dense_W` bridge), `src/domainia/wilson/noise.py` (doctest), `src/domainia/wilson/stationary.py` (a message)
- Test: `tests/wilson/meyer_cases.py`, `tests/wilson/test_meyer_baseline.py`, `tests/wilson/vendored/test_matrix.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_stationary.py`, `tests/test_dom_testing.py`, `tests/wilson/test_wrong_kinds.py`, `tests/test_package.py`

**What the patches do:**
- **`build_W`** fills a folded `(Nf, Nt, N)` array directly.
  - For each row i and j0 ∈ {0, 1}, it seeds the padded cell `cells(Nf, Nt)[:, i, j0]` into `_inverse_time`. It then rolls the result by `(j − j0)·Nf` into the slots j ≡ j0 (mod 2).
  - That is 2·Nf transforms with no inactive skip.
  - The result equals the recorded `build_W` gathered by rows, bit for bit.
- **`build_C_wdm`'s body is unchanged** (`W @ C_time @ W.T`). It is folded because W is, and its docstring names the (0, Nf−1) corner pair.
- **`active_W` and `build_C_wdm_active`** leave the module and `__all__`. Their doctests move into `build_W` (W Wᵀ ≈ dt·I) and `build_C_wdm` (diag ≈ S/2).
- **The `noise.py` doctest** reads cell (1, 0) at folded index `Nt` instead of the band-order index `Nt // 2`.
- **`testing.dense_W`** permutes `build_W`'s rows into the band order of its still-padded mask. This is a bridge that Task 5 removes.
- **Replay:**
  - `DROPPED` gains `wilson::matrix.active_W::` and `wilson::matrix.build_C_wdm_active::`. The trailing `::` keeps them from matching `build_C_wdm`.
  - `load_baseline()` gathers `build_W`'s rows, and `build_C_wdm`'s rows and columns, into folded order (`FOLDED_ROWS`).
- **Tests:**
  - Shapes become `(N, N)`. Band structure is checked with the band of each slot from `cells`.
  - `test_package` pins `wilson.matrix.__all__`.
  - Three vendored tests that stronger tests already imply are deleted: `test_W_complete` and `test_W_inverse_via_transpose` (W Wᵀ = dt·I on a square W implies both), and `test_C_wdm_block_tridiagonal` (`test_window_threading`'s per-entry test is 1e-12 on the same grid).
  - Vendored `test_full_active_blocks_match_dense_active` is deleted too: the assembly test after it compares the same blocks bit for bit.
  - Other tests that still want band order permute with a `BAND` index (bridges for Tasks 4–5).

**Acceptance Criteria:**
- [ ] `wilson.matrix.__all__ == ["build_C_time", "build_C_wdm", "build_W", "real_to_rfft", "rfft_to_real"]`; `build_W(4, 8, 1.0).shape == (32, 32)`.
- [ ] The replay prints `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`.
- [ ] Full suite (prototype): PY and FLOOR `943 passed, 3 skipped`, no warnings, no collection errors.
- [ ] Size (prototype): `src 4128 lines / 1881 code; tests 4355 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/vendored/test_matrix.py tests/wilson/test_window_threading.py tests/wilson/test_grid_bands.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_wrong_kinds.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_stationary.py tests/test_dom_testing.py tests/test_package.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/matrix.py src/domainia/wilson/noise.py src/domainia/testing.py 2>&1 | tail -1` → `592 passed, 3 skipped`

**Steps:**

- [ ] **Step 1: Apply the tests patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T3-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists exactly `tests/test_dom_testing.py`, `tests/test_package.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_meyer_baseline.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/test_wrong_kinds.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_matrix.py`, `tests/wilson/vendored/test_stationary.py` as modified.

<!-- patch: T3-tests -->
````diff
diff --git a/tests/test_dom_testing.py b/tests/test_dom_testing.py
index 3fe8575..8cdd802 100644
--- a/tests/test_dom_testing.py
+++ b/tests/test_dom_testing.py
@@ -18,6 +18,8 @@ N = NF * NT
 F_MIN, F_REF = 0.5, 1.0
 #: Every active cell.
 MASK = wilson.grid.active_mask(NF, NT)
+#: ``wilson.matrix.build_W``'s folded rows in ``MASK``'s band order.
+BAND = np.argsort(np.ravel_multi_index(wilson.grid.cells(NF, NT), (NF + 1, NT)), axis=None)
 
 
 def L_of(f):
@@ -69,17 +71,17 @@ def test_eq45_differs_from_exact_at_second_order_with_drift():
 
 
 def test_the_oracles_compute_in_the_window_they_are_given():
-    """With ALT_WINDOW: ``dense_W`` is ``active_W``, the congruence is ``W S W^T``, and Eq. 45
+    """With ALT_WINDOW: ``dense_W`` is ``build_W``, the congruence is ``W S W^T``, and Eq. 45
     of a stationary spectrum is the exact covariance; each differs from Meyer's."""
     W = T.dense_W(NF, NT, DT, MASK, window=T.ALT_WINDOW)
-    np.testing.assert_array_equal(W, wilson.matrix.active_W(NF, NT, DT, window=T.ALT_WINDOW))
+    np.testing.assert_array_equal(W, wilson.matrix.build_W(NF, NT, DT, window=T.ALT_WINDOW)[BAND])
     S = np.diag(1.0 + np.arange(N) / N)                  # not the identity: W I W^T = dt I for any window
     alt = T.wdm_congruence(S, NF, NT, DT, MASK, window=T.ALT_WINDOW)
     np.testing.assert_allclose(alt, W @ S @ W.T, rtol=0, atol=1e-12)
     assert not np.allclose(alt, T.wdm_congruence(S, NF, NT, DT, MASK))
     got = T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0), window=T.ALT_WINDOW)
-    exact = wilson.matrix.build_C_wdm_active(NF, NT, DT, lambda f: drift(-1.2, 0.0)(f, 0.0),
-                                             window=T.ALT_WINDOW)
+    exact = wilson.matrix.build_C_wdm(NF, NT, DT, lambda f: drift(-1.2, 0.0)(f, 0.0),
+                                      window=T.ALT_WINDOW)[np.ix_(BAND, BAND)]
     assert rel(got, exact) < 1e-10
     assert not np.allclose(got, T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0)))
 
diff --git a/tests/test_package.py b/tests/test_package.py
index 84bba6b..d6f56e8 100644
--- a/tests/test_package.py
+++ b/tests/test_package.py
@@ -49,13 +49,17 @@ def test_the_project_metadata():
 
 def test_the_public_names():
     """A name that left a top level stays in its submodule (``wilson.grid.active_mask``,
-    ``wilson.windows.WindowError``)."""
+    ``wilson.windows.WindowError``).  ``wilson.matrix`` has no ``active_W`` or
+    ``build_C_wdm_active``: ``build_W`` and ``build_C_wdm`` are already ``N x N``."""
     assert set(domainia.__all__) == PUBLIC
     assert set(wilson.__all__) == WILSON_PUBLIC
     assert all(hasattr(domainia, name) for name in domainia.__all__)
     assert all(hasattr(wilson, name) for name in wilson.__all__)
     assert not hasattr(domainia, "MEYER") and not hasattr(wilson, "active_mask")
     assert wilson.MEYER is wilson.windows.MEYER and wilson.Grid is wilson.grid.Grid
+    assert set(wilson.matrix.__all__) == {"build_C_time", "build_C_wdm", "build_W",
+                                          "real_to_rfft", "rfft_to_real"}
+    assert not hasattr(wilson.matrix, "active_W") and not hasattr(wilson.matrix, "build_C_wdm_active")
 
 
 def test_no_release_leftovers():
diff --git a/tests/wilson/meyer_cases.py b/tests/wilson/meyer_cases.py
index 3fd0b48..f94a803 100644
--- a/tests/wilson/meyer_cases.py
+++ b/tests/wilson/meyer_cases.py
@@ -38,16 +38,21 @@ BASELINE = Path(__file__).resolve().parent / "data" / "meyer_baseline.npz"
 #: Case-name prefixes whose kernels take ``window=``.  The single source for
 #: ``test_meyer_baseline.py`` and ``test_window_threading.py``.
 THREADED = ("transform.", "matrix.", "stationary.", "noise.", "evolutionary.")
-#: Baseline key prefixes of code that domainia 3.0 deletes: maps (54 keys), inner_product (24)
-#: and the band-limited entry points (18).  They are not replayed.
+#: Baseline key prefixes of code that domainia 3.0 deletes: maps (54 keys), inner_product (24),
+#: the band-limited entry points (18) and the active-cell matrices (4), which the folded layout
+#: made ``build_W`` and ``build_C_wdm``.  They are not replayed.
 DROPPED = ("domainia::TimeToWDM.", "domainia::WDMToTime.", "domainia::FourierToWDM.",
            "wilson::inner_product.", "wilson::transform.band_atoms::",
-           "wilson::transform.forward_bands::", "wilson::transform.inverse_bands::")
+           "wilson::transform.forward_bands::", "wilson::transform.inverse_bands::",
+           "wilson::matrix.active_W::", "wilson::matrix.build_C_wdm_active::")
 #: The comparison: ``max |got - want| <= RTOL * s`` (see :func:`mismatches`).
 RTOL = 1e-12
 #: Families recorded padded ``(..., Nf+1, Nt)`` whose kernels now return the folded layout.
 FOLDED = ("wilson::transform.forward", "wilson::transform.forward_many",
           "wilson::transform.forward_time", "wilson::transform.forward_time_many")
+#: Dense families recorded with padded ``(Nf+1) Nt`` rows (and columns, ``build_C_wdm``), now
+#: in folded flat order.
+FOLDED_ROWS = ("wilson::matrix.build_W", "wilson::matrix.build_C_wdm")
 
 Case = tuple[str, Callable[[dict], Any]]
 
@@ -60,16 +65,21 @@ def _family(key: str) -> str:
 def load_baseline() -> dict[str, np.ndarray]:
     """The recorded baseline as ``{key: array}``.  The families whose kernels now return the
     folded WDM layout are converted here, never in :func:`mismatches` (folded WDM spec,
-    section 5): the :data:`FOLDED` ones are gathered into the folded slots, without
+    section 5): the :data:`FOLDED` ones are gathered into the folded slots, and the
+    :data:`FOLDED_ROWS` rows (and the covariance's columns) into folded flat order, without
     ``wilson.grid.fold``'s check of the inactive cells."""
     from domainia.wilson.grid import cells
 
     with np.load(BASELINE) as z:
         out = {k: z[k] for k in z.files}
     for k, v in out.items():
-        if _family(k) in FOLDED:
-            m, n = cells(*map(int, k.split("::")[2].split("x")))
-            out[k] = v[..., m, n]
+        fam = _family(k)
+        if fam in FOLDED + FOLDED_ROWS:
+            Nf, Nt = map(int, k.split("::")[2].split("x"))
+            m, n = cells(Nf, Nt)
+            p = (m * Nt + n).ravel()                # each folded slot's padded flat index
+            out[k] = (v[..., m, n] if fam in FOLDED else v[p] if fam == "wilson::matrix.build_W"
+                      else v[np.ix_(p, p)])
     return out
 
 
@@ -182,9 +192,7 @@ def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
     if (Nf, Nt) == DENSE_GRID:
         cases += [
             ("matrix.build_W", lambda kw: M.build_W(Nf, Nt, dt, **kw)),
-            ("matrix.active_W", lambda kw: M.active_W(Nf, Nt, dt, **kw)),
             ("matrix.build_C_wdm", lambda kw: M.build_C_wdm(Nf, Nt, dt, psd, **kw)),
-            ("matrix.build_C_wdm_active", lambda kw: M.build_C_wdm_active(Nf, Nt, dt, psd, **kw)),
             ("evolutionary.exact_covariance", lambda kw: E.exact_covariance(Nf, Nt, dt, Sn, **kw)),
             ("evolutionary.eq45_direct", lambda kw: E.eq45_direct(E.eq45_plan(Nf, Nt, dt, **kw), Sn)),
         ]
diff --git a/tests/wilson/test_evolutionary_kernels.py b/tests/wilson/test_evolutionary_kernels.py
index 8bd1907..9532743 100644
--- a/tests/wilson/test_evolutionary_kernels.py
+++ b/tests/wilson/test_evolutionary_kernels.py
@@ -21,7 +21,8 @@ import pytest
 from wdm_dev_ev import evolutionary as ev
 
 from domainia.wilson import evolutionary as E
-from domainia.wilson.matrix import active_W
+from domainia.wilson.grid import cells
+from domainia.wilson.matrix import build_C_wdm, build_W
 from domainia.wilson.noise import evolutionary_band_variance
 
 from .conftest import _oracle_paths
@@ -34,6 +35,8 @@ F_NYQ = 0.5 / DT
 LINEAGE_A = ("toy_models/vendor/wdm/src", "wdm_paper_2/toy_models/src")
 #: max|got - ref| over max|ref| against the omega-era sources (float64).
 ORACLE_RTOL = 1e-14
+#: ``build_W``'s folded rows in band order (``numpy.nonzero`` of the padded mask).
+BAND = np.argsort(np.ravel_multi_index(cells(NF, NT), (NF + 1, NT)), axis=None)
 
 
 def _gap(got, ref, scale=None):
@@ -105,7 +108,7 @@ def test_moved_functions_are_the_oracle_to_rounding(rng):
 
 
 def test_energy_fraction_and_time_mask_match_the_dense_mask(rng):
-    Wa = active_W(NF, NT, DT)
+    Wa = build_W(NF, NT, DT)[BAND]
     w2 = rng.uniform(0.0, 1.0, N)
     Mm = Wa @ np.diag(w2) @ Wa.T / DT
     Y = rng.standard_normal((2, N))
@@ -165,11 +168,9 @@ def test_interp_blocks_and_direct_sum_are_the_oracle_to_rounding(keep):
 
 
 def test_eq45_is_exact_for_stationary_noise():
-    from domainia.wilson.matrix import build_C_wdm_active
-
     plan = E.eq45_plan(NF, NT, DT)
     Sst = lambda f, t: _S_stat(f) + 0 * t  # noqa: E731
-    ref = build_C_wdm_active(NF, NT, DT, _S_stat)
+    ref = build_C_wdm(NF, NT, DT, _S_stat)[np.ix_(BAND, BAND)]
     assert np.max(np.abs(E.eq45_direct(plan, Sst) - ref)) <= 1e-12 * np.abs(ref).max()
     coeffs = E.eq45_coeffs_interp(Sst, plan.f, plan.t_max, plan.P, k=plan.k_used)
     C = _assemble(plan, *E.eq45_blocks_cheb(plan, coeffs))
diff --git a/tests/wilson/test_grid_bands.py b/tests/wilson/test_grid_bands.py
index 7ee16e0..64052f3 100644
--- a/tests/wilson/test_grid_bands.py
+++ b/tests/wilson/test_grid_bands.py
@@ -14,7 +14,7 @@ import pytest
 from domainia import wilson
 from domainia.wilson.filter import phi_centered
 from domainia.wilson.grid import Grid, active_mask, cells, endpoint_weights, fold, unfold
-from domainia.wilson.matrix import active_W
+from domainia.wilson.matrix import build_W
 from domainia.wilson.transform import atom_spectrum, base_atom_spectra, forward_time
 
 from .conftest import _oracle_paths
@@ -170,14 +170,14 @@ def test_rfft_packing_rejects_odd_N():
 
 def test_atoms_are_the_rows_of_the_transform():
     Nf, Nt, dt = 8, 16, 0.5
-    Wa = active_W(Nf, Nt, dt)
+    W = build_W(Nf, Nt, dt)
     base = base_atom_spectra(Nf, Nt, dt)
     assert sorted(base) == [(m, a) for m in range(Nf + 1) for a in (0, 1)
                             if active_mask(Nf, Nt)[m, a]]
-    m, n = np.nonzero(active_mask(Nf, Nt))
+    m, n = (c.ravel() for c in cells(Nf, Nt))           # row p of W is folded slot p
     for p in range(m.size):
         atom = np.fft.irfft(atom_spectrum(int(m[p]), int(n[p]), Nf, Nt, dt, base=base), n=Nf * Nt)
-        assert np.allclose(atom, Wa[p], rtol=0, atol=1e-12)
+        assert np.allclose(atom, W[p], rtol=0, atol=1e-12)
     with pytest.raises(ValueError, match="not an active cell"):
         atom_spectrum(0, 1, Nf, Nt, dt, base=base)
 
diff --git a/tests/wilson/test_meyer_baseline.py b/tests/wilson/test_meyer_baseline.py
index 2bc6922..f390a68 100644
--- a/tests/wilson/test_meyer_baseline.py
+++ b/tests/wilson/test_meyer_baseline.py
@@ -21,10 +21,11 @@ def baseline() -> dict[str, np.ndarray]:
     return load_baseline()
 
 
-def test_the_dropped_families_are_96_of_the_1504_keys(baseline):
-    """maps 54, inner_product 24 and the band-limited entry points 18."""
+def test_the_dropped_families_are_100_of_the_1504_keys(baseline):
+    """maps 54, inner_product 24, the band-limited entry points 18 and the active-cell
+    matrices 4."""
     assert len(baseline) == 1504
-    assert sum(k.startswith(DROPPED) for k in baseline) == 96
+    assert sum(k.startswith(DROPPED) for k in baseline) == 100
 
 
 def test_the_comparator():
diff --git a/tests/wilson/test_window_threading.py b/tests/wilson/test_window_threading.py
index bd4d15a..857afa4 100644
--- a/tests/wilson/test_window_threading.py
+++ b/tests/wilson/test_window_threading.py
@@ -35,7 +35,7 @@ MEYER_HELPERS = frozenset({"phi_centered"})
 CONSUMERS = frozenset({
     "_gather_bands", "forward", "forward_time", "inverse", "inverse_time", "_forward",
     "_forward_time", "_inverse", "_inverse_time", "base_atom_spectra", "atom_spectrum",
-    "build_W", "active_W", "build_C_wdm", "build_C_wdm_active",
+    "build_W", "build_C_wdm",
     "_slab_K", "build_lambda_blocks_full_active", "build_K_full_active", "psd_from_K",
     "apply_lambda_from_psd", "psd_from_lambda_blocks",
     "band_averaged_variance", "evolutionary_band_variance", "empirical_variance",
@@ -326,7 +326,7 @@ def test_atom_spectrum_checks_the_window_even_when_base_is_given(bogus):
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_every_admissible_window_gives_an_orthonormal_basis(window, Nf, Nt, dt):
-    A = wilson.matrix.active_W(Nf, Nt, dt, window=window)
+    A = wilson.matrix.build_W(Nf, Nt, dt, window=window)
     np.testing.assert_allclose(A @ A.T, dt * np.eye(A.shape[0]), rtol=0, atol=1e-12 * dt)
 
 
@@ -344,9 +344,9 @@ def test_time_round_trip_and_energy(window, Nf, Nt, dt):
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 def test_stationary_covariance_is_block_tridiagonal(window):
     Nf, Nt, dt = 8, 16, 1.0
-    C = wilson.matrix.build_C_wdm(Nf, Nt, dt, psd, window=window).reshape(Nf + 1, Nt, Nf + 1, Nt)
-    far = max(float(np.max(np.abs(C[m, :, k, :])))
-              for m in range(Nf + 1) for k in range(Nf + 1) if abs(m - k) > 1)
+    C = wilson.matrix.build_C_wdm(Nf, Nt, dt, psd, window=window)
+    m = wilson.grid.cells(Nf, Nt)[0].ravel()               # the band of each folded slot
+    far = float(np.max(np.abs(C[np.abs(m[:, None] - m[None, :]) > 1])))
     assert far <= 1e-12 * float(np.max(np.abs(C)))
 
 
@@ -373,19 +373,14 @@ def _close(got, want, rtol: float, what: str) -> None:
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_build_W_is_the_impulse_response(window, Nf, Nt, dt):
-    """Column ``k`` of ``W`` is ``forward_time(e_k)``, unfolded to ``build_W``'s padded rows (the
-    definition; ``build_W`` probes ``2 (Nf+1)`` rows and rolls them by ``2 Nf`` samples
-    instead), and the other matrix builders follow from it."""
+    """Column ``k`` of ``W`` is ``forward_time(e_k)`` in folded flat order (the definition;
+    ``build_W`` probes ``2 Nf`` rows and rolls them by ``2 Nf`` samples instead), and
+    ``build_C_wdm`` follows from it."""
     N = Nf * Nt
-    W_def = wilson.grid.unfold(wilson.forward_time(np.eye(N), Nf, Nt, dt, window=window), Nf, Nt)
-    W_def = W_def.reshape(N, -1).T
-    mask = wilson.grid.active_mask(Nf, Nt).ravel()
+    W_def = wilson.forward_time(np.eye(N), Nf, Nt, dt, window=window).reshape(N, N).T
     C_def = W_def @ M.build_C_time(N, dt, psd) @ W_def.T
     _close(M.build_W(Nf, Nt, dt, window=window), W_def, RTOL, "build_W")
-    _close(M.active_W(Nf, Nt, dt, window=window), W_def[mask], RTOL, "active_W")
     _close(M.build_C_wdm(Nf, Nt, dt, psd, window=window), C_def, RTOL, "build_C_wdm")
-    _close(M.build_C_wdm_active(Nf, Nt, dt, psd, window=window), C_def[np.ix_(mask, mask)], RTOL,
-           "build_C_wdm_active")
 
 
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@@ -395,20 +390,21 @@ def test_atom_spectra_are_the_rfft_of_the_rows_of_W(window, Nf, Nt, dt):
     one in each edge band -- built from ``window=`` alone (``atom_spectrum``'s own
     ``base_atom_spectra(window=)`` call)."""
     W = M.build_W(Nf, Nt, dt, window=window)
+    row = wilson.grid.unfold(np.arange(Nf * Nt).reshape(Nf, Nt), Nf, Nt)    # cell (m, n)'s row
     base = T.base_atom_spectra(Nf, Nt, dt, window=window)
     for (m, a), G in base.items():
-        _close(G, np.fft.rfft(W[m * Nt + a]), RTOL, f"base_atom_spectra[{m}, {a}]")
+        _close(G, np.fft.rfft(W[row[m, a]]), RTOL, f"base_atom_spectra[{m}, {a}]")
     for m, n in zip(*np.nonzero(wilson.grid.active_mask(Nf, Nt)), strict=True):
         _close(T.atom_spectrum(int(m), int(n), Nf, Nt, dt, base=base, window=window),
-               np.fft.rfft(W[m * Nt + n]), RTOL_ATOM, f"atom_spectrum({m}, {n}, base=)")
+               np.fft.rfft(W[row[m, n]]), RTOL_ATOM, f"atom_spectrum({m}, {n}, base=)")
     for m, n in ((1, 3), (0, 2), (Nf, 2)):
-        _close(T.atom_spectrum(m, n, Nf, Nt, dt, window=window), np.fft.rfft(W[m * Nt + n]),
+        _close(T.atom_spectrum(m, n, Nf, Nt, dt, window=window), np.fft.rfft(W[row[m, n]]),
                RTOL_ATOM, f"atom_spectrum({m}, {n})")
 
 
 # ---------------------------------------------------------------------------
 # The stationary, noise, evolutionary and inner-product kernels, each against a dense reference
-# built from the value-verified matrix kernels above (W, W_a, build_C_wdm, build_C_wdm_active)
+# built from the value-verified matrix kernels above (build_W and build_C_wdm)
 # with the same window.  A kernel that samples the window but computes with MEYER fails these
 # for every window but MEYER.
 # ---------------------------------------------------------------------------
@@ -432,20 +428,20 @@ def _read_only(a):
 
 @lru_cache(maxsize=None)
 def _dense(window, Nf, Nt, dt):
-    """``(W, C, W_a, C_a)``: ``build_W``, ``build_C_wdm``, ``active_W`` and
-    ``build_C_wdm_active`` of ``psd`` with ``window`` (read-only)."""
-    return tuple(_read_only(a) for a in (
-        M.build_W(Nf, Nt, dt, window=window), M.build_C_wdm(Nf, Nt, dt, psd, window=window),
-        M.active_W(Nf, Nt, dt, window=window), M.build_C_wdm_active(Nf, Nt, dt, psd, window=window)))
-
-
-def _blocks_of(C, n_active, Nf, Nt):
-    """The diagonal and upper band blocks of a dense ``(Nf+1) Nt`` square matrix, restricted to
-    the active cells (the ``build_lambda_blocks_full_active`` layout)."""
-    C4 = C.reshape(Nf + 1, Nt, Nf + 1, Nt)
-    diag = [C4[m, :, m][np.ix_(n_active[m], n_active[m])] for m in range(Nf + 1)]
-    offd = [C4[m, :, m + 1][np.ix_(n_active[m], n_active[m + 1])] for m in range(Nf)]
-    return diag, offd
+    """``(W, C, W_a, C_a)``: ``build_W`` and ``build_C_wdm`` of ``psd`` with ``window``, and the
+    same with their rows (and columns) in band order, ``numpy.nonzero`` of the padded mask
+    (read-only)."""
+    W, C = M.build_W(Nf, Nt, dt, window=window), M.build_C_wdm(Nf, Nt, dt, psd, window=window)
+    a = np.argsort(np.ravel_multi_index(wilson.grid.cells(Nf, Nt), (Nf + 1, Nt)), axis=None)
+    return tuple(_read_only(x) for x in (W, C, W[a], C[np.ix_(a, a)]))
+
+
+def _blocks_of(C, Nf, Nt):
+    """The diagonal and upper band blocks of a dense folded ``N x N`` matrix (the
+    ``build_lambda_blocks_full_active`` layout: band ``m``'s cells, time ascending)."""
+    band = [np.flatnonzero(wilson.grid.cells(Nf, Nt)[0] == m) for m in range(Nf + 1)]
+    return ([C[np.ix_(b, b)] for b in band],
+            [C[np.ix_(b, c)] for b, c in zip(band[:-1], band[1:], strict=True)])
 
 
 def _K_blocks(K_diag, K_off, n_active, Nf, Nt):
@@ -469,12 +465,13 @@ def test_stationary_builders_are_the_blocks_of_the_dense_covariance(window, Nf,
     """``build_lambda_blocks_full_active``, and the blocks ``build_K_full_active``'s ``K``
     stand for, assemble to ``build_C_wdm``."""
     C = _dense(window, Nf, Nt, dt)[1]
+    p = np.ravel_multi_index(wilson.grid.cells(Nf, Nt), (Nf + 1, Nt)).ravel()
     d, o, na = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, window=window)
-    _close(St.assemble_lambda_full_active(d, o, na, Nf, Nt), C, RTOL_KERNEL,
+    _close(St.assemble_lambda_full_active(d, o, na, Nf, Nt)[np.ix_(p, p)], C, RTOL_KERNEL,
            "build_lambda_blocks_full_active")
     K_diag, K_off = St.build_K_full_active(Nf, Nt, dt, psd, window=window)
-    _close(St.assemble_lambda_full_active(*_K_blocks(K_diag, K_off, na, Nf, Nt), na, Nf, Nt), C,
-           RTOL_KERNEL, "build_K_full_active")
+    _close(St.assemble_lambda_full_active(*_K_blocks(K_diag, K_off, na, Nf, Nt), na, Nf, Nt)
+           [np.ix_(p, p)], C, RTOL_KERNEL, "build_K_full_active")
 
 
 @pytest.mark.parametrize("Nf, Nt, dt, window", [
@@ -491,7 +488,7 @@ def test_stationary_backward_maps_recover_the_psd(window, Nf, Nt, dt):
     d, o, na = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, window=window)
     _close(St.psd_from_lambda_blocks(d, o, na, Nf, Nt, dt, window=window), S, RTOL_KERNEL,
            "psd_from_lambda_blocks")
-    dense = _blocks_of(_dense(window, Nf, Nt, dt)[1], na, Nf, Nt)
+    dense = _blocks_of(_dense(window, Nf, Nt, dt)[1], Nf, Nt)
     _close(St.psd_from_lambda_blocks(*dense, na, Nf, Nt, dt, window=window), S, RTOL_KERNEL,
            "psd_from_lambda_blocks(blocks of build_C_wdm)")
 
@@ -499,7 +496,7 @@ def test_stationary_backward_maps_recover_the_psd(window, Nf, Nt, dt):
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_apply_lambda_from_psd_is_the_dense_covariance(window, Nf, Nt, dt):
-    """``apply_lambda_from_psd(x)`` is ``build_C_wdm_active @ x`` on the active cells (junk in
+    """``apply_lambda_from_psd(x)`` is ``build_C_wdm @ x`` on the active cells (junk in
     the inactive cells of ``x`` is ignored)."""
     N = Nf * Nt
     mask = wilson.grid.active_mask(Nf, Nt)
@@ -515,7 +512,7 @@ def test_apply_lambda_from_psd_is_the_dense_covariance(window, Nf, Nt, dt):
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_band_variances_are_the_diagonal_of_the_dense_covariance(window, Nf, Nt, dt):
     """``band_averaged_variance`` and the stationary limit of ``evolutionary_band_variance``
-    (all bands, and selected ``bands=``) are ``diag(build_C_wdm_active)`` on the active cells."""
+    (all bands, and selected ``bands=``) are ``diag(build_C_wdm)`` on the active cells."""
     mask = wilson.grid.active_mask(Nf, Nt)
     want = np.diag(_dense(window, Nf, Nt, dt)[3])
     _close(No.band_averaged_variance(Nf, Nt, dt, psd, window=window)[mask], want, RTOL_KERNEL,
@@ -538,8 +535,9 @@ def test_empirical_variance_is_the_mean_square_of_the_same_draws(window, Nf, Nt,
     W = _dense(window, Nf, Nt, dt)[0]
     rng = np.random.default_rng(5)
     draws = [W @ np.fft.irfft(colored_noise(N, dt, psd, rng), n=N) for _ in range(3)]
-    want = np.mean(np.square(draws), axis=0).reshape(Nf + 1, Nt)
-    _close(empirical_variance(Nf, Nt, dt, psd, 3, seed=5, window=window), want, RTOL_KERNEL,
+    want = np.mean(np.square(draws), axis=0).reshape(Nf, Nt)
+    m, n = wilson.grid.cells(Nf, Nt)
+    _close(empirical_variance(Nf, Nt, dt, psd, 3, seed=5, window=window)[m, n], want, RTOL_KERNEL,
            "empirical_variance")
 
 
@@ -562,14 +560,14 @@ def test_time_masks_are_the_dense_mask(window, Nf, Nt, dt):
            RTOL_KERNEL, "apply_time_mask(1-D)")
 
 
-def _wrap_reference(W, Nf, Nt):
-    """The definition of ``wrap_fractions`` on the rows of ``W``: for the active cell
+def _wrap_reference(W_a, Nf, Nt):
+    """The definition of ``wrap_fractions`` on the band-order rows ``W_a``: for the active cell
     ``(m, n)``, centred on sample ``c = n Nf``, the share of ``sum_t g_mn(t)^2`` at samples
     whose unwrapped position ``c + delta`` (``delta`` the signed offset from the centre, in
     ``[-N/2, N/2)``) lies outside ``[0, N)``; ``nan`` at inactive cells."""
     N = Nf * Nt
     m, n = np.nonzero(wilson.grid.active_mask(Nf, Nt))
-    g2 = W.reshape(Nf + 1, Nt, N)[m, n] ** 2
+    g2 = W_a ** 2
     c = n[:, None] * Nf
     pos = c + (np.arange(N)[None, :] - c + N // 2) % N - N // 2
     out = np.full((Nf + 1, Nt), np.nan)
@@ -592,7 +590,7 @@ def _cut_reference(frac, tol):
 def test_wrap_rule_is_its_definition_on_the_rows_of_W(window, Nf, Nt, dt):
     """``wrap_fractions`` (all bands and a subset) is its definition on the rows of
     ``build_W``, and ``wrap_cut_columns`` is the cut rule on those fractions."""
-    want = _wrap_reference(_dense(window, Nf, Nt, dt)[0], Nf, Nt)
+    want = _wrap_reference(_dense(window, Nf, Nt, dt)[2], Nf, Nt)
     got = E.wrap_fractions(Nf, Nt, dt, window=window)
     assert np.array_equal(np.isnan(got), np.isnan(want))
     _close(np.nan_to_num(got, nan=-1.0), np.nan_to_num(want, nan=-1.0), RTOL_KERNEL, "wrap_fractions")
@@ -607,7 +605,7 @@ def test_wrap_rule_is_its_definition_on_the_rows_of_W(window, Nf, Nt, dt):
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_exact_covariance_is_the_dense_congruence(window, Nf, Nt, dt):
     """``exact_covariance`` is ``W_a Sigma W_a^T`` (``Sigma = time_domain_covariance``), and
-    ``build_C_wdm_active`` in the stationary limit."""
+    ``build_C_wdm`` in the stationary limit."""
     N = Nf * Nt
     Wa, Ca = _dense(window, Nf, Nt, dt)[2:]
     _close(E.exact_covariance(Nf, Nt, dt, _stationary, window=window), Ca, RTOL_KERNEL,
@@ -648,7 +646,7 @@ def _assemble_gram(gplan, diag, up, n):
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_eq45_plan_carries_the_window_to_every_eq45_builder(window, Nf, Nt, dt):
     """``plan.base`` is ``base_atom_spectra(window=)``, and in the stationary limit the plan's
-    ``eq45_direct``, ``eq45_blocks_cheb`` and ``eq45_blocks_gram`` are ``build_C_wdm_active``,
+    ``eq45_direct``, ``eq45_blocks_cheb`` and ``eq45_blocks_gram`` are ``build_C_wdm``,
     on all cells and on a ``keep`` subset (the kept rows and columns)."""
     Ca = _dense(window, Nf, Nt, dt)[3]
     base = T.base_atom_spectra(Nf, Nt, dt, window=window)
diff --git a/tests/wilson/test_wrong_kinds.py b/tests/wilson/test_wrong_kinds.py
index c1850c7..19dbe8a 100644
--- a/tests/wilson/test_wrong_kinds.py
+++ b/tests/wilson/test_wrong_kinds.py
@@ -246,7 +246,6 @@ def test_float32_input_is_computed_in_double():
         "empirical_variance": lambda dt: testing.empirical_variance(6, 10, dt, psd, 2, seed=0),
         "build_C_time": lambda dt: M.build_C_time(60, dt, psd),
         "build_C_wdm": lambda dt: M.build_C_wdm(6, 10, dt, psd),
-        "build_C_wdm_active": lambda dt: M.build_C_wdm_active(6, 10, dt, psd),
         "build_lambda_blocks_full_active": lambda dt: St.build_lambda_blocks_full_active(
             6, 10, dt, psd)[0][1],
         "build_K_full_active": lambda dt: St.build_K_full_active(6, 10, dt, psd)[0][1],
diff --git a/tests/wilson/vendored/test_evolutionary.py b/tests/wilson/vendored/test_evolutionary.py
index 13c6f60..38444bc 100644
--- a/tests/wilson/vendored/test_evolutionary.py
+++ b/tests/wilson/vendored/test_evolutionary.py
@@ -6,16 +6,19 @@ import numpy as np
 import pytest
 
 # The frozen wdm_dev oracle (tests/wilson/oracle/wdm_dev_ev) against domainia.wilson's dense
-# references: its packed grid, atoms and time mask against active_W and forward_time, its band
-# average and exact covariance against band_averaged_variance, build_C_time and
-# build_C_wdm_active.  tests/wilson/test_evolutionary_kernels.py checks the kernels against it.
+# references: its packed grid, atoms and time mask against build_W and forward_time, its band
+# average and exact covariance against band_averaged_variance, build_C_time and build_C_wdm.
+# The oracle packs the cells in band order; tests/wilson/test_evolutionary_kernels.py checks
+# the kernels against it.
 from wdm_dev_ev import evolutionary as ev
-from domainia.wilson.grid import active_mask, unfold
-from domainia.wilson.matrix import active_W
+from domainia.wilson.grid import active_mask, cells, unfold
+from domainia.wilson.matrix import build_W
 from domainia.wilson.transform import forward_time
 
 NF, NT, DT = 8, 16, 0.5
 N = NF * NT
+#: ``build_W``'s folded rows in the oracle's band order (``numpy.nonzero`` of the padded mask).
+BAND = np.argsort(np.ravel_multi_index(cells(NF, NT), (NF + 1, NT)), axis=None)
 
 
 @pytest.fixture
@@ -25,7 +28,7 @@ def rng():
 
 def test_packed_grid_matches_the_dense_transform(rng):
     g = ev.PackedGrid(NF, NT, DT)
-    Wa = active_W(NF, NT, DT)
+    Wa = build_W(NF, NT, DT)[BAND]
     assert g.size == N
     np.testing.assert_array_equal(g.mask, active_mask(NF, NT))
     x = rng.standard_normal((3, N))
@@ -36,14 +39,14 @@ def test_packed_grid_matches_the_dense_transform(rng):
 
 def test_atoms_are_the_rows_of_the_transform():
     g = ev.PackedGrid(NF, NT, DT)
-    Wa = active_W(NF, NT, DT)
+    Wa = build_W(NF, NT, DT)[BAND]
     for p in range(g.size):
         np.testing.assert_allclose(g.atom(p), Wa[p], atol=1e-12)
 
 
 def test_apply_mask_and_energy_fraction_match_the_dense_mask(rng):
     g = ev.PackedGrid(NF, NT, DT)
-    Wa = active_W(NF, NT, DT)
+    Wa = build_W(NF, NT, DT)[BAND]
     w2 = rng.uniform(0.0, 1.0, N)
     Mm = Wa @ np.diag(w2) @ Wa.T / DT
     Y = rng.standard_normal((2, N))
@@ -55,7 +58,7 @@ def test_apply_mask_and_energy_fraction_match_the_dense_mask(rng):
 # General S(f, t): band average and exact covariance
 # ---------------------------------------------------------------------------
 
-from domainia.wilson.matrix import build_C_time, build_C_wdm_active  # noqa: E402
+from domainia.wilson.matrix import build_C_time, build_C_wdm  # noqa: E402
 from domainia.wilson.noise import band_averaged_variance  # noqa: E402
 
 T_REC = N * DT
@@ -93,10 +96,10 @@ def test_stationary_limit_of_the_exact_covariance():
     np.testing.assert_allclose(ev.time_domain_covariance(N, DT, Sst), build_C_time(N, DT, _S_stat),
                                atol=1e-12)
     Lam = ev.exact_covariance(NF, NT, DT, Sst)
-    np.testing.assert_allclose(Lam, build_C_wdm_active(NF, NT, DT, _S_stat), atol=1e-12)
+    np.testing.assert_allclose(Lam, build_C_wdm(NF, NT, DT, _S_stat)[np.ix_(BAND, BAND)], atol=1e-12)
 
 
 def test_exact_covariance_is_the_transformed_time_covariance():
-    Wa = active_W(NF, NT, DT)
+    Wa = build_W(NF, NT, DT)[BAND]
     ref = Wa @ ev.time_domain_covariance(N, DT, _S_evo) @ Wa.T
     np.testing.assert_allclose(ev.exact_covariance(NF, NT, DT, _S_evo), ref, atol=1e-12 * np.abs(ref).max())
diff --git a/tests/wilson/vendored/test_matrix.py b/tests/wilson/vendored/test_matrix.py
index b2a5a1f..1bee0dd 100644
--- a/tests/wilson/vendored/test_matrix.py
+++ b/tests/wilson/vendored/test_matrix.py
@@ -4,12 +4,9 @@ from __future__ import annotations
 
 import numpy as np
 
-from domainia.wilson.grid import unfold
 from domainia.wilson.matrix import (
-    active_W,
     build_C_time,
     build_C_wdm,
-    build_C_wdm_active,
     build_W,
     real_to_rfft,
     rfft_to_real,
@@ -17,50 +14,26 @@ from domainia.wilson.matrix import (
 from domainia.wilson.transform import forward_time
 
 
-def test_W_shape(simple_grid):
-    """build_W returns shape ((Nf+1)*Nt, N)."""
-    g = simple_grid
-    W = build_W(g.Nf, g.Nt, g.dt)
-    assert W.shape == ((g.Nf + 1) * g.Nt, g.N)
-
-
 def test_W_matches_FFT_transform(simple_grid):
-    """W @ x == forward(rfft(x)) to machine precision."""
+    """W @ x == forward(rfft(x)) to machine precision, rows in folded flat order."""
     g = simple_grid
     W = build_W(g.Nf, g.Nt, g.dt)
     rng = np.random.default_rng(0)
     x = rng.standard_normal(g.N)
-    w_matrix = (W @ x).reshape(g.Nf + 1, g.Nt)
-    w_fft = unfold(forward_time(x, g.Nf, g.Nt, g.dt), g.Nf, g.Nt)
+    w_matrix = (W @ x).reshape(g.Nf, g.Nt)
+    w_fft = forward_time(x, g.Nf, g.Nt, g.dt)
     assert np.allclose(w_matrix, w_fft, atol=1e-13)
 
 
-def test_active_W_orthogonal(simple_grid):
-    """W_active @ W_active.T = I (orthogonal rows)."""
+def test_W_orthogonal(simple_grid):
+    """W @ W.T = dt I (orthogonal rows)."""
     g = simple_grid
-    Wa = active_W(g.Nf, g.Nt, g.dt)
-    assert Wa.shape == (g.N, g.N)
-    err = np.max(np.abs(Wa @ Wa.T - np.eye(g.N)))
-    assert err < 1e-12
-
-
-def test_active_W_complete(simple_grid):
-    """W_active.T @ W_active = I (orthogonal columns / completeness)."""
-    g = simple_grid
-    Wa = active_W(g.Nf, g.Nt, g.dt)
-    err = np.max(np.abs(Wa.T @ Wa - np.eye(g.N)))
+    W = build_W(g.Nf, g.Nt, g.dt)
+    assert W.shape == (g.N, g.N)
+    err = np.max(np.abs(W @ W.T - g.dt * np.eye(g.N)))
     assert err < 1e-12
 
 
-def test_active_W_inverse_via_transpose(simple_grid):
-    """W_active^T W_active x == x for any x."""
-    g = simple_grid
-    Wa = active_W(g.Nf, g.Nt, g.dt)
-    rng = np.random.default_rng(0)
-    x = rng.standard_normal(g.N)
-    assert np.allclose(Wa.T @ Wa @ x, x, atol=1e-12)
-
-
 def test_C_time_symmetric(simple_grid, lisa_psd):
     """C_time is symmetric."""
     g = simple_grid
@@ -79,39 +52,18 @@ def test_C_time_toeplitz(simple_grid, flat_psd):
         assert np.max(np.abs(diag - diag[0])) < 1e-12 * np.max(np.abs(C))
 
 
-def test_C_wdm_block_tridiagonal(simple_grid, lisa_psd):
-    """C_wdm is block-tridiagonal in m: |m - m'| >= 2 blocks are tiny."""
-    g = simple_grid
-    C = build_C_wdm(g.Nf, g.Nt, g.dt, lisa_psd)
-    Nb = g.Nf + 1
-    # Compute block-Frobenius norms.
-    diag_norm = np.linalg.norm(C[: g.Nt, : g.Nt])
-    for m in range(Nb):
-        for mp in range(Nb):
-            if abs(m - mp) >= 2:
-                block = C[m * g.Nt : (m + 1) * g.Nt, mp * g.Nt : (mp + 1) * g.Nt]
-                # Distant blocks should be << diag_norm.
-                assert np.linalg.norm(block) < 1e-3 * diag_norm
-
-
 def test_C_wdm_diag_matches_whittle(simple_grid, flat_psd):
-    """The active diagonal of C_wdm equals the Whittle variance for flat PSD."""
+    """The diagonal of C_wdm equals the Whittle variance for flat PSD."""
     g = simple_grid
     C = build_C_wdm(g.Nf, g.Nt, g.dt, flat_psd)
-    diag = np.diag(C).reshape(g.Nf + 1, g.Nt)
-    # Average over the active interior pixels; should equal S_n / 2.
+    assert C.shape == (g.N, g.N)
+    diag = np.diag(C).reshape(g.Nf, g.Nt)
+    # Average over the interior bands (rows 1..Nf-1); should equal S_n / 2.
     expected = 1e-3 / 2
     interior = diag[1 : g.Nf]
     assert np.isclose(np.mean(interior), expected, rtol=0.01)
 
 
-def test_C_wdm_active_shape(simple_grid, lisa_psd):
-    """C_wdm_active has shape (N, N)."""
-    g = simple_grid
-    C = build_C_wdm_active(g.Nf, g.Nt, g.dt, lisa_psd)
-    assert C.shape == (g.N, g.N)
-
-
 def test_rfft_real_roundtrip(simple_grid):
     """rfft_to_real and real_to_rfft are mutual inverses."""
     g = simple_grid
diff --git a/tests/wilson/vendored/test_stationary.py b/tests/wilson/vendored/test_stationary.py
index 8d3ada6..c8fdf5b 100644
--- a/tests/wilson/vendored/test_stationary.py
+++ b/tests/wilson/vendored/test_stationary.py
@@ -5,7 +5,8 @@ from __future__ import annotations
 import numpy as np
 import pytest
 
-from domainia.wilson.matrix import active_W, build_C_time, build_W
+from domainia.wilson.grid import cells
+from domainia.wilson.matrix import build_C_time, build_W
 from domainia.wilson.stationary import (
     assemble_lambda_full_active,
     build_lambda_blocks_full_active,
@@ -145,39 +146,13 @@ def test_invalid_2d_shape_raises(simple_grid):
         build_lambda_blocks(g.Nf, g.Nt, g.dt, bad)
 
 
-def test_full_active_blocks_match_dense_active(simple_grid, lisa_psd):
-    """Full-active slab-IFFT (interior + edges) matches W_a @ C @ W_a^T at machine precision."""
-    g = simple_grid
-    diag, offd, n_active = build_lambda_blocks_full_active(g.Nf, g.Nt, g.dt, lisa_psd)
-    W_a = active_W(g.Nf, g.Nt, g.dt)
-    C_time = build_C_time(g.N, g.dt, lisa_psd)
-    Lambda_a = W_a @ C_time @ W_a.T
-
-    # Active-W ordering: m=0 (Nt/2 rows), m=1..Nf-1 (Nt each), m=Nf (Nt/2).
-    offsets = [0]
-    for m in range(g.Nf + 1):
-        offsets.append(offsets[-1] + len(n_active[m]))
-
-    for m in range(g.Nf + 1):
-        ref = Lambda_a[offsets[m] : offsets[m + 1], offsets[m] : offsets[m + 1]]
-        rel = np.max(np.abs(diag[m] - ref)) / max(np.max(np.abs(ref)), 1.0)
-        assert rel < 1e-10, f"diag[m={m}] rel err = {rel:.3e}"
-
-    for m in range(g.Nf):
-        ref = Lambda_a[offsets[m] : offsets[m + 1], offsets[m + 1] : offsets[m + 2]]
-        rel = np.max(np.abs(offd[m] - ref)) / max(np.max(np.abs(ref)), 1.0)
-        assert rel < 1e-10, f"offd[m={m}] rel err = {rel:.3e}"
-
-
 def test_full_active_assembly_matches_dense_W_C_W(simple_grid, lisa_psd):
     """Assembled full-active Λ matches the dense W @ C_time @ W^T at machine precision."""
     g = simple_grid
     diag, offd, n_active = build_lambda_blocks_full_active(g.Nf, g.Nt, g.dt, lisa_psd)
-    Lambda_fast = assemble_lambda_full_active(diag, offd, n_active, g.Nf, g.Nt)
-
-    W = build_W(g.Nf, g.Nt, g.dt)
-    C_time = build_C_time(g.N, g.dt, lisa_psd)
-    Lambda_dense = W @ C_time @ W.T
+    p = np.ravel_multi_index(cells(g.Nf, g.Nt), (g.Nf + 1, g.Nt)).ravel()
+    Lambda_fast = assemble_lambda_full_active(diag, offd, n_active, g.Nf, g.Nt)[np.ix_(p, p)]
+    Lambda_dense = _dense_lambda(g.Nf, g.Nt, g.dt, lisa_psd)
 
     rel = np.max(np.abs(Lambda_fast - Lambda_dense)) / np.max(np.abs(Lambda_dense))
     assert rel < 1e-10, f"|Λ_fast − W C W^T| / |Λ| = {rel:.3e}"
````

- [ ] **Step 2: Run the task's tests: they fail**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error --tb=no -rfE tests/wilson/vendored/test_matrix.py tests/wilson/test_window_threading.py tests/wilson/test_grid_bands.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_wrong_kinds.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_stationary.py tests/test_dom_testing.py tests/test_package.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/matrix.py src/domainia/wilson/noise.py 2>&1 | tail -n 40
```

Expected last line (prototype): `264 failed, 330 passed, 3 skipped`. 247 of the ids are `test_window_threading`'s parametrizations. `build_W` is still `(144, 128)` with zero rows, and `build_C_wdm` is `(144, 144)`. So:
- the shape, W Wᵀ = dt·I, per-slot band and folded-order checks fail;
- `test_package` sees `active_W` and `build_C_wdm_active`;
- the replay reports the `build_W` and `build_C_wdm` keys.

- [ ] **Step 3: Apply the implementation patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T3-impl && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists `src/domainia/testing.py`, `src/domainia/wilson/matrix.py`, `src/domainia/wilson/noise.py`, `src/domainia/wilson/stationary.py`, `tests/test_dom_testing.py`, `tests/test_package.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_meyer_baseline.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/test_wrong_kinds.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_matrix.py`, `tests/wilson/vendored/test_stationary.py` as modified, and nothing else.

<!-- patch: T3-impl -->
````diff
diff --git a/src/domainia/testing.py b/src/domainia/testing.py
index 2e73a83..6b092f8 100644
--- a/src/domainia/testing.py
+++ b/src/domainia/testing.py
@@ -2,7 +2,7 @@
 ``O(N^2)`` memory or worse: keep ``N`` <~ 2048.
 
 * :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a set of cells
-  (``wilson.matrix.active_W``) and ``W_K Sigma W_K^T``.
+  (``wilson.matrix.build_W``) and ``W_K Sigma W_K^T``.
 * :func:`eq45_dense`: Eq. (45) on a set of cells by the direct sum over rfft bins
   (``wilson.evolutionary.eq45_direct``).
 * :func:`colored_noise`, :func:`empirical_variance`: one draw of stationary noise (its rfft)
@@ -63,10 +63,11 @@ def _keep(Nf: int, Nt: int, mask: Any) -> NDArray[np.bool_]:
 
 def dense_W(Nf: int, Nt: int, dt: float, mask: Any, *, window: Window = MEYER) -> Array:
     """``(cells, N)``: row ``p`` is the atom of the ``p``-th cell of ``mask`` sampled in
-    time, so ``W x`` are those cells of ``x`` (``wilson.matrix.active_W`` rows,
-    ``W_a W_a^T = dt I``) in the Wilson basis of ``window``."""
+    time, so ``W x`` are those cells of ``x`` (``wilson.matrix.build_W`` rows,
+    ``W W^T = dt I``) in the Wilson basis of ``window``."""
     keep = _keep(Nf, Nt, mask)
-    return wilson.matrix.active_W(Nf, Nt, dt, window=window)[keep]
+    band = np.argsort(np.ravel_multi_index(wilson.grid.cells(Nf, Nt), (Nf + 1, Nt)), axis=None)
+    return wilson.matrix.build_W(Nf, Nt, dt, window=window)[band[keep]]
 
 
 def wdm_congruence(Sigma: Array, Nf: int, Nt: int, dt: float, mask: Any, *,
diff --git a/src/domainia/wilson/matrix.py b/src/domainia/wilson/matrix.py
index 5a50c79..7d1084a 100644
--- a/src/domainia/wilson/matrix.py
+++ b/src/domainia/wilson/matrix.py
@@ -1,15 +1,15 @@
 """Explicit WDM transform matrix and noise covariance.
 
 The WDM forward transform is a linear map $w = W\\,x$ where $W$ is a
-real $(N_f+1)N_t \\times N$ matrix.  Restricted to its $N$ active rows
-(skipping the inactive edge-band entries), $W_a$ is square and
+real $N \\times N$ matrix whose rows are the folded slots in flat order (slot $(i, j)$ is
+row $i\\,N_t + j$; :func:`~domainia.wilson.grid.cells` gives its band and time).  $W$ is
 orthogonal **up to the sampling scale**:
 
-$$W_a W_a^T = W_a^T W_a = \\delta t\\,I,$$
+$$W W^T = W^T W = \\delta t\\,I,$$
 
-i.e. $W_a = \\sqrt{\\delta t}\\,Q$ with $Q$ orthogonal.  Invert with
-$x = W_a^T w / \\delta t$.  The $\\delta t$ is the same factor that
-appears in Parseval ($\\sum w^2 = \\delta t \\sum x^2$); $W_a$ is
+i.e. $W = \\sqrt{\\delta t}\\,Q$ with $Q$ orthogonal.  Invert with
+$x = W^T w / \\delta t$.  The $\\delta t$ is the same factor that
+appears in Parseval ($\\sum w^2 = \\delta t \\sum x^2$); $W$ is
 exactly orthogonal only when $\\delta t = 1$.
 
 Every builder takes ``window=`` (default :data:`domainia.wilson.MEYER`) and returns the
@@ -18,8 +18,7 @@ window (:mod:`domainia.wilson.windows`).
 
 This module exposes:
 
-* :func:`build_W` — full $(N_f+1)N_t \\times N$ matrix.
-* :func:`active_W` — the $N \\times N$ restriction ($W_a W_a^T = \\delta t\\,I$).
+* :func:`build_W` — the $N \\times N$ matrix ($W W^T = \\delta t\\,I$).
 * :func:`build_C_time` — Toeplitz time-domain noise covariance from
   a PSD (vectorized via ``irfft`` of the power spectrum).
 * :func:`build_C_wdm` — WDM-domain noise covariance via the similarity
@@ -42,7 +41,7 @@ from numpy.typing import NDArray
 from .grid import (
     _require_even_positive_int,
     _require_int,
-    active_mask,
+    cells,
     n_samples,
     sanitize_psd,
     validate_dt,
@@ -56,10 +55,8 @@ type CArray = NDArray[np.complexfloating]
 type PsdFunc = Callable[[Array], Array]
 
 __all__ = [
-    "active_W",
     "build_C_time",
     "build_C_wdm",
-    "build_C_wdm_active",
     "build_W",
     "real_to_rfft",
     "rfft_to_real",
@@ -73,93 +70,53 @@ def build_W(
     *,
     window: Window = MEYER,
 ) -> Array:
-    """Build the full $(N_f+1)N_t \\times N$ WDM transform matrix for ``window``.
+    """Build the $N \\times N$ WDM transform matrix for ``window``.
 
-    Row $\\alpha = (m, n)$ is the basis function $g_{mn}$ sampled in time, so
-    $W$ is *defined* by "column $k_0$ is the WDM transform of a delta at sample
-    $k_0$".  It is *built* from the lattice's shift-covariance instead, which
-    agrees with the definition to rounding.
+    Row $p = i\\,N_t + j$ is the basis function $g_{mn}$ of folded slot $(i, j)$ sampled in
+    time ($(m, n)$ from :func:`~domainia.wilson.grid.cells`), so $W$ is *defined* by "column
+    $k_0$ is the flattened WDM transform of a delta at sample $k_0$".  It is *built* from the
+    lattice's shift-covariance instead, which agrees with the definition to rounding.
 
     Returns
     -------
-    ndarray, shape ((Nf+1)*Nt, N)
-        Real matrix.  Rows are indexed by the flattened ``(m, n)``
-        pixel index $\\alpha = m\\,N_t + n$; columns by the time
-        sample $k_0 = 0, \\ldots, N - 1$.  Inactive edge-band rows are exactly zero.
+    ndarray, shape (N, N)
+        Real matrix with $W W^T = \\delta t\\,I$.  Rows are the folded slots in flat order
+        (``W @ x`` is ``forward_time(x).reshape(N)``); columns are the time samples
+        $k_0 = 0, \\ldots, N - 1$.
 
     Notes
     -----
-    Cost: $2(N_f+1)$ inverse transforms plus $O(N^2)$ of copying, and
+    Cost: $2 N_f$ inverse transforms plus $O(N^2)$ of copying, and
     $O(N^2)$ memory.
 
     Examples
     --------
     >>> import numpy as np; from domainia import wilson
     >>> W = wilson.matrix.build_W(4, 8, 1.0)
-    >>> W.shape
-    (40, 32)
+    >>> W.shape, bool(np.allclose(W @ W.T, 1.0 * np.eye(32)))
+    ((32, 32), True)
     """
     validate_grid(Nf, Nt)
     validate_dt(dt)
     N = n_samples(Nf, Nt)
-    W = np.zeros((Nf + 1, Nt, N))
-
-    # Since W_a^T W_a = dt*I, row (m, n) is exactly dt * inverse(e_mn).  The
-    # lattice is shift-covariant in steps of TWO time pixels -- one step flips
-    # the carrier parity (-1)^{(m+1)n} -- so rows n and n+2 differ only by a
-    # circular shift of 2*Nf samples.  Probing n = 0 and n = 1 per band and
-    # rolling the rest turns N transforms into 2(Nf+1).
-    #
-    # Only ACTIVE pixels are seeded: inactive edge pixels correspond to zero
-    # rows of W, and those rows come out exactly zero here rather than carrying
-    # the ~1e-17 of cancellation noise that impulse probing leaves.
-    mask = active_mask(Nf, Nt)
+    m, n = cells(Nf, Nt)
+    W = np.empty((Nf, Nt, N))
+
+    # Since W^T W = dt*I, row p is exactly dt * inverse(e_p).  The lattice is
+    # shift-covariant in steps of TWO time pixels -- one step flips the carrier
+    # parity (-1)^{(m+1)n} -- so slots j and j+2 of a row (the same band, in row 0
+    # too) differ only by a circular shift of 2*Nf samples.  Probing j = 0 and
+    # j = 1 per row and rolling the rest turns N transforms into 2*Nf.
     seed = np.zeros((Nf + 1, Nt))
-    for n0 in (0, 1):
-        for m in range(Nf + 1):
-            if not mask[m, n0]:
-                continue
+    for i in range(Nf):
+        for j0 in (0, 1):
             seed[:] = 0.0
-            seed[m, n0] = 1.0
+            seed[m[i, j0], n[i, j0]] = 1.0
             row = dt * _inverse_time(seed, Nf, Nt, dt, window=window)
-            for n in range(n0, Nt, 2):
-                W[m, n] = np.roll(row, (n - n0) * Nf)
-
-    return W.reshape((Nf + 1) * Nt, N)
-
-
-def active_W(
-    Nf: int,
-    Nt: int,
-    dt: float,
-    *,
-    window: Window = MEYER,
-) -> Array:
-    """Restrict :func:`build_W` to its $N$ active rows.
+            for j in range(j0, Nt, 2):
+                W[i, j] = np.roll(row, (j - j0) * Nf)
 
-    Returns the square $N \\times N$ matrix obtained by keeping only
-    the rows corresponding to active pixels (interior bands and the
-    parity-selected edge-band entries).  This matrix is orthogonal up to
-    the sampling scale:
-    $W_a W_a^T = W_a^T W_a = \\delta t\\,I$.  Invert with
-    $x = W_a^T w / \\delta t$; the plain transpose is the inverse only
-    at $\\delta t = 1$.
-
-    Returns
-    -------
-    ndarray, shape (N, N)
-        Square real matrix satisfying $W_a W_a^T = \\delta t\\,I$.
-
-    Examples
-    --------
-    >>> import numpy as np; from domainia import wilson
-    >>> Wa = wilson.matrix.active_W(4, 8, 1.0)
-    >>> Wa.shape, bool(np.allclose(Wa @ Wa.T, 1.0 * np.eye(32)))
-    ((32, 32), True)
-    """
-    W = build_W(Nf, Nt, dt, window=window)
-    mask = active_mask(Nf, Nt).ravel()
-    return W[mask, :]
+    return W.reshape(N, N)
 
 
 def build_C_time(N: int, dt: float, Sn_func: PsdFunc) -> Array:
@@ -227,53 +184,29 @@ def build_C_wdm(
 ) -> Array:
     """WDM-domain covariance via the similarity transform $W\\,C^{(\\text{time})}\\,W^T$.
 
-    Returns the full $(N_f+1)N_t \\times (N_f+1)N_t$ covariance matrix
-    of the WDM coefficients under stationary noise with PSD $S_n(f)$.
+    Returns the $N \\times N$ covariance matrix of the WDM coefficients, rows and columns in
+    :func:`build_W`'s folded flat order, under stationary noise with PSD $S_n(f)$.
     Block-tridiagonal in the band index $m$ for every admissible ``window``
     (compact support inside one slab): only $|m - m'| \\le 1$ blocks are
-    nonzero (up to numerical noise).
+    nonzero (up to numerical noise).  Row 0 holds both edge bands, so in slot rows this is
+    block-tridiagonal plus the $(0, N_f - 1)$ corner pair.
 
     Returns
     -------
-    ndarray, shape ((Nf+1)*Nt, (Nf+1)*Nt)
+    ndarray, shape (N, N)
         Symmetric real covariance matrix.
 
     Examples
     --------
     >>> import numpy as np; from domainia import wilson
     >>> C = wilson.matrix.build_C_wdm(4, 8, 1.0, lambda f: 1e-3 * np.ones_like(f))
-    >>> C.shape
-    (40, 40)
-    """
-    N = n_samples(Nf, Nt)
-    W = build_W(Nf, Nt, dt, window=window)
-    C_time = build_C_time(N, dt, Sn_func)
-    return W @ C_time @ W.T
-
-
-def build_C_wdm_active(
-    Nf: int,
-    Nt: int,
-    dt: float,
-    Sn_func: PsdFunc,
-    *,
-    window: Window = MEYER,
-) -> Array:
-    """Active-pixel restriction of :func:`build_C_wdm`: the $N \\times N$ covariance of the
-    active WDM coefficients, i.e. ``build_C_wdm`` on the active rows and columns.
-
-    Examples
-    --------
-    >>> import numpy as np; from domainia import wilson
-    >>> flat = lambda f: 1e-3 * np.ones_like(f)
-    >>> C = wilson.matrix.build_C_wdm_active(4, 8, 1.0, flat)
     >>> C.shape, float(np.round(np.diag(C)[0], 6))   # ~ S_n / 2
     ((32, 32), 0.0005)
     """
     N = n_samples(Nf, Nt)
-    Wa = active_W(Nf, Nt, dt, window=window)
+    W = build_W(Nf, Nt, dt, window=window)
     C_time = build_C_time(N, dt, Sn_func)
-    return Wa @ C_time @ Wa.T
+    return W @ C_time @ W.T
 
 
 def rfft_to_real(x_rfft: CArray, N: int) -> Array:
diff --git a/src/domainia/wilson/noise.py b/src/domainia/wilson/noise.py
index d5faa44..d5d31c2 100644
--- a/src/domainia/wilson/noise.py
+++ b/src/domainia/wilson/noise.py
@@ -81,8 +81,8 @@ def band_averaged_variance(
     >>> var.shape
     (9, 16)
     >>> # exact: reproduces the diagonal of the WDM covariance
-    >>> truth = np.diag(wilson.matrix.build_C_wdm_active(Nf, Nt, dt, steep))
-    >>> bool(abs(var[1, 0] - truth[Nt // 2]) / truth[Nt // 2] < 1e-12)
+    >>> truth = np.diag(wilson.matrix.build_C_wdm(Nf, Nt, dt, steep))
+    >>> bool(abs(var[1, 0] - truth[Nt]) / truth[Nt] < 1e-12)
     True
     """
     validate_grid(Nf, Nt)
diff --git a/src/domainia/wilson/stationary.py b/src/domainia/wilson/stationary.py
index 7b8b30c..52cabe1 100644
--- a/src/domainia/wilson/stationary.py
+++ b/src/domainia/wilson/stationary.py
@@ -179,7 +179,7 @@ def _normalize_noise_to_sigma2(
                     "The fast stationary algorithm requires off-diagonal "
                     f"entries < {rtol:g} times the max diagonal entry. "
                     "For non-stationary noise build the dense pipeline "
-                    "explicitly: W = domainia.wilson.matrix.build_W(...) (or active_W), "
+                    "explicitly: W = domainia.wilson.matrix.build_W(...), "
                     "then W @ C @ W.T with your own (N, N) covariance. Note "
                     "domainia.wilson.matrix.build_C_wdm does NOT accept one -- it takes a "
                     "stationary PSD callable and builds a Toeplitz C_time."
````

- [ ] **Step 4: Run the task's tests: they pass**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/vendored/test_matrix.py tests/wilson/test_window_threading.py tests/wilson/test_grid_bands.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_wrong_kinds.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_stationary.py tests/test_dom_testing.py tests/test_package.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/matrix.py src/domainia/wilson/noise.py src/domainia/testing.py 2>&1 | tail -1
```

Expected (prototype): `592 passed, 3 skipped`.

- [ ] **Step 5: Run the lanes, the replay and the size**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py
```

Expected (prototype): `943 passed, 3 skipped` in both lanes, with no warnings and no collection errors; `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`; `src 4128 lines / 1881 code; tests 4355 lines (+ oracle 4135)`.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/testing.py src/domainia/wilson/matrix.py src/domainia/wilson/noise.py src/domainia/wilson/stationary.py tests/test_dom_testing.py tests/test_package.py tests/wilson/meyer_cases.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_grid_bands.py tests/wilson/test_meyer_baseline.py tests/wilson/test_window_threading.py tests/wilson/test_wrong_kinds.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_matrix.py tests/wilson/vendored/test_stationary.py && git commit -q -F - <<'EOF'
Fold build_W and build_C_wdm to N x N and remove active_W and build_C_wdm_active

build_W returns the N x N matrix with rows in folded flat order (W W^T = dt I): it seeds
the first two slots of each folded row through cells and rolls the rest, 2 Nf transforms,
so no row is structurally zero. build_C_wdm is the folded N x N covariance. active_W and
build_C_wdm_active would equal them and go. testing.dense_W permutes build_W's rows to the
band order of its padded mask until that mask folds. The noise doctest reads cell (1, 0)
at folded index Nt, and the stationary message no longer names active_W.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
```


### Task 4: `wilson.noise` and `wilson.stationary` fold (spec §3 `noise.py` and `stationary.py`, §5; §8 step 4)

**Goal:**
- `band_averaged_variance` and `evolutionary_band_variance` return `(Nf, Nt)` folded, and `evolutionary_band_variance(bands=)` is removed.
- `apply_lambda_from_psd` takes and returns folded arrays.
- `assemble_lambda_full_active` returns `(N, N)` in folded flat order.
- The band blocks are unchanged.

**Files:**
- Modify: `src/domainia/wilson/noise.py`, `src/domainia/wilson/stationary.py`
- Test: `tests/wilson/meyer_cases.py`, `tests/wilson/test_noise_stationary.py`, `tests/wilson/vendored/test_stationary.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_window_threading.py`, `tests/test_covariance.py`

**What the patches do:**
- **`band_averaged_variance`** is `0.5 * Sn_eff[cells(Nf, Nt)[0]]`, so row 0's odd slots get the Nyquist band's value.
- **`evolutionary_band_variance`** computes its padded grid as before and gathers it with `[cells(Nf, Nt)]`. `bands=` goes; nothing in src used it.
- **`apply_lambda_from_psd`** is `forward((S/2) · inverse(x))` through the public kernels.
  - It takes and returns the folded layout, and refuses a padded or flat input with the named fix.
  - Task 2's masking bridge goes.
  - This departs from spec §3, which has it call the twins. It makes the same two boundary conversions, with no round trip.
- **`assemble_lambda_full_active`** places each band block, and its transpose, straight at its cells' folded flat indices (`unfold(arange(N))`). This is about 15 lines smaller than building the padded matrix and gathering it.
- **Unchanged:** `psd_from_lambda_blocks`, its padded matvec and the band blocks.
- **Replay:**
  - `FOLDED` adds `apply_lambda_from_psd`, `band_averaged_variance` and `evolutionary_band_variance`. These are gathered raw, because their recorded inactive cells hold the band value.
  - The `BandAveragedVariance` generator reads `unfold(...)[m, n]`, a bridge that Task 5 moves to load time.
- **Tests:**
  - The oracle's padded grids are gathered with `cells`.
  - The `bands=` comparisons in `test_evolutionary_kernels` and `test_window_threading` go. With `bands=` removed, rows of our full result against the oracle would only repeat the full-grid comparison already there.
  - `test_covariance` compares `wilson.grid.fold` of the still-padded `wdm_covariance` with the folded variance, until Task 6.

**Acceptance Criteria:**
- [ ] `band_averaged_variance(8, 16, 1.0, S).shape == (8, 16)`; `evolutionary_band_variance` has no `bands` parameter; `assemble_lambda_full_active` on an 8 × 16 grid is `(128, 128)`.
- [ ] The replay prints `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`.
- [ ] Full suite (prototype): PY and FLOOR `943 passed, 3 skipped`, no warnings, no collection errors.
- [ ] Size (prototype): `src 4099 lines / 1864 code; tests 4346 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/test_noise_stationary.py tests/wilson/vendored/test_noise.py tests/wilson/vendored/test_stationary.py tests/wilson/vendored/test_evolutionary.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_window_threading.py tests/test_covariance.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/noise.py src/domainia/wilson/stationary.py 2>&1 | tail -1` → `495 passed, 2 skipped`

**Steps:**

- [ ] **Step 1: Apply the tests patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T4-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists exactly `tests/test_covariance.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_noise_stationary.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_stationary.py` as modified.

<!-- patch: T4-tests -->
````diff
diff --git a/tests/test_covariance.py b/tests/test_covariance.py
index 683b0d0..b78bef3 100644
--- a/tests/test_covariance.py
+++ b/tests/test_covariance.py
@@ -175,7 +175,7 @@ def test_wdm_covariance_of_an_evolutionary_model(monkeypatch):
             one = wdm_covariance(lambda f, t, S=S, a=a: S(f, t)[a:a + 1, a:a + 1],
                                  Nf=Nf, Nt=Nt, dt=DT)
             for v in (got[a, a], one[0, 0]):
-                np.testing.assert_allclose(v[mask], want[mask], rtol=1e-14, atol=0)
+                np.testing.assert_allclose(wilson.grid.fold(v, Nf, Nt), want, rtol=1e-14, atol=0)
                 assert not np.any(v[~mask])
     points = ([], [])
 
diff --git a/tests/wilson/meyer_cases.py b/tests/wilson/meyer_cases.py
index f94a803..d39f9c8 100644
--- a/tests/wilson/meyer_cases.py
+++ b/tests/wilson/meyer_cases.py
@@ -49,7 +49,9 @@ DROPPED = ("domainia::TimeToWDM.", "domainia::WDMToTime.", "domainia::FourierToW
 RTOL = 1e-12
 #: Families recorded padded ``(..., Nf+1, Nt)`` whose kernels now return the folded layout.
 FOLDED = ("wilson::transform.forward", "wilson::transform.forward_many",
-          "wilson::transform.forward_time", "wilson::transform.forward_time_many")
+          "wilson::transform.forward_time", "wilson::transform.forward_time_many",
+          "wilson::stationary.apply_lambda_from_psd", "wilson::noise.band_averaged_variance",
+          "wilson::noise.evolutionary_band_variance")
 #: Dense families recorded with padded ``(Nf+1) Nt`` rows (and columns, ``build_C_wdm``), now
 #: in folded flat order.
 FOLDED_ROWS = ("wilson::matrix.build_W", "wilson::matrix.build_C_wdm")
@@ -171,8 +173,7 @@ def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
          lambda kw: St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, **kw)),
         ("stationary.build_K_full_active", lambda kw: St.build_K_full_active(Nf, Nt, dt, psd, **kw)),
         ("stationary.psd_from_K", lambda kw: St.psd_from_K(K0, Nf, Nt, dt, **kw)),
-        ("stationary.apply_lambda_from_psd",
-         lambda kw: St.apply_lambda_from_psd(wl.grid.unfold(w, Nf, Nt), Nf, Nt, dt, S, **kw)),
+        ("stationary.apply_lambda_from_psd", lambda kw: St.apply_lambda_from_psd(w, Nf, Nt, dt, S, **kw)),
         ("stationary.psd_from_lambda_blocks", lambda kw: St.psd_from_lambda_blocks(*B0, Nf, Nt, dt, **kw)),
         ("noise.band_averaged_variance", lambda kw: No.band_averaged_variance(Nf, Nt, dt, psd, **kw)),
         ("noise.evolutionary_band_variance",
@@ -221,7 +222,8 @@ def domainia_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
                                                              k=plan.k_used))
 
     cases: list[Case] = [
-        ("BandAveragedVariance", lambda kw: wl.noise.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)[m, n]),
+        ("BandAveragedVariance", lambda kw: wl.grid.unfold(
+            wl.noise.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw), Nf, Nt)[m, n]),
         ("EvolutionaryCovariance.interp", interp),
     ]
     if (Nf, Nt) == DENSE_GRID:
diff --git a/tests/wilson/test_evolutionary_kernels.py b/tests/wilson/test_evolutionary_kernels.py
index 9532743..4ac4f9c 100644
--- a/tests/wilson/test_evolutionary_kernels.py
+++ b/tests/wilson/test_evolutionary_kernels.py
@@ -99,11 +99,9 @@ def test_moved_functions_are_the_oracle_to_rounding(rng):
                 ev.exact_covariance(NF, NT, DT, _S_evo)) <= ORACLE_RTOL
     assert np.array_equal(E.draw_noise(N, DT, _S_evo, np.random.default_rng(5)),
                           ev.draw_noise(N, DT, _S_evo, np.random.default_rng(5)))
-    # Hz window (v0.12.0): measured 1.8e-16 (all bands and bands [0, 3, NF])
+    # Hz window (v0.12.0): measured 1.8e-16; the oracle's padded grid is gathered
     assert _gap(evolutionary_band_variance(NF, NT, DT, _S_evo),
-                ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo)) <= ORACLE_RTOL
-    assert _gap(evolutionary_band_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF]),
-                ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF])
+                ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo)[cells(NF, NT)]
                 ) <= ORACLE_RTOL
 
 
diff --git a/tests/wilson/test_noise_stationary.py b/tests/wilson/test_noise_stationary.py
index edcaf4d..b56eeaf 100644
--- a/tests/wilson/test_noise_stationary.py
+++ b/tests/wilson/test_noise_stationary.py
@@ -19,6 +19,7 @@ import vendored_2f7136c.stationary as ostat
 import domainia.wilson.noise as noise
 import domainia.wilson.stationary as stat
 from domainia import testing
+from domainia.wilson.grid import cells, unfold
 
 GRIDS = [(4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10)]
 #: max|got - ref| over max|ref| against the omega-era oracle.
@@ -55,7 +56,7 @@ def test_band_averaged_variance_is_the_oracle_to_rounding(grid):
     """band_averaged_variance (a phi^2-weighted average) to rounding."""
     Nf, Nt, dt = grid
     got = noise.band_averaged_variance(Nf, Nt, dt, _psd)
-    ref = onoise.band_averaged_variance(Nf, Nt, dt, _psd)
+    ref = onoise.band_averaged_variance(Nf, Nt, dt, _psd)[cells(Nf, Nt)]
     assert got.dtype == ref.dtype
     assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 4.8e-16
 
@@ -139,6 +140,8 @@ def test_stationary_builders_are_the_oracle_to_rounding(grid, rng):
     2 pi times the oracle's K (prefactor 1/N, was 2 pi/N); the active-index lists bit for bit."""
     Nf, Nt, dt = grid
     N = Nf * Nt
+    m, n = cells(Nf, Nt)
+    p = (m * Nt + n).ravel()                    # each folded slot's padded flat index
     S_rfft = _psd(np.fft.rfftfreq(N, d=dt))
     for noise_in in (_psd, S_rfft):
         d1, o1, n1 = stat.build_lambda_blocks_full_active(Nf, Nt, dt, noise_in)
@@ -153,7 +156,7 @@ def test_stationary_builders_are_the_oracle_to_rounding(grid, rng):
         assert max(_gap(a, b, lam) for a, b in zip(d1 + o1, d2 + o2, strict=True)) <= RTOL64
         assert all(np.array_equal(a, b) for a, b in zip(n1, n2, strict=True))
         got = stat.assemble_lambda_full_active(d1, o1, n1, Nf, Nt)
-        ref = ostat.assemble_lambda_full_active(d2, o2, n2, Nf, Nt)
+        ref = ostat.assemble_lambda_full_active(d2, o2, n2, Nf, Nt)[np.ix_(p, p)]
         assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 1.8e-16
         K1, L1 = stat.build_K_full_active(Nf, Nt, dt, noise_in)
         K2, L2 = ostat.build_K_full_active(Nf, Nt, dt, noise_in)
@@ -164,9 +167,9 @@ def test_stationary_builders_are_the_oracle_to_rounding(grid, rng):
         assert _gap(got, ostat.lambda_to_psd_slab(K2, Nf, Nt, dt)) <= RTOL64   # Hz: 2.3e-16
         got = stat.psd_from_lambda_blocks(d1, o1, n1, Nf, Nt, dt)
         assert _gap(got, ostat.lambda_to_psd(d2, o2, n2, Nf, Nt, dt)) <= RTOL64  # Hz: 7.1e-16
-    x = rng.standard_normal((Nf + 1, Nt))
+    x = rng.standard_normal((Nf, Nt))
     got = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S_rfft)
-    ref = ostat.apply_lambda_from_psd(x, Nf, Nt, dt, S_rfft)
+    ref = ostat.apply_lambda_from_psd(unfold(x, Nf, Nt), Nf, Nt, dt, S_rfft)[m, n]
     assert got.dtype == ref.dtype
     assert _gap(got, ref) <= RTOL64              # Hz window (v0.12.0): measured 2.0e-16
 
@@ -223,7 +226,7 @@ def test_apply_lambda_from_psd_inf_psd_is_zero_power(rng):
     """An inf PSD sample (the LISA f = 0 case) is zero power: ``sanitize_psd`` maps it to 0.0."""
     Nf, Nt, dt = 8, 16, 0.5
     S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
-    x = rng.standard_normal((Nf + 1, Nt))
+    x = rng.standard_normal((Nf, Nt))
     S[0] = 0.0
     want = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
     S[0] = np.inf
diff --git a/tests/wilson/test_window_threading.py b/tests/wilson/test_window_threading.py
index 857afa4..2de0f5c 100644
--- a/tests/wilson/test_window_threading.py
+++ b/tests/wilson/test_window_threading.py
@@ -465,13 +465,12 @@ def test_stationary_builders_are_the_blocks_of_the_dense_covariance(window, Nf,
     """``build_lambda_blocks_full_active``, and the blocks ``build_K_full_active``'s ``K``
     stand for, assemble to ``build_C_wdm``."""
     C = _dense(window, Nf, Nt, dt)[1]
-    p = np.ravel_multi_index(wilson.grid.cells(Nf, Nt), (Nf + 1, Nt)).ravel()
     d, o, na = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, window=window)
-    _close(St.assemble_lambda_full_active(d, o, na, Nf, Nt)[np.ix_(p, p)], C, RTOL_KERNEL,
+    _close(St.assemble_lambda_full_active(d, o, na, Nf, Nt), C, RTOL_KERNEL,
            "build_lambda_blocks_full_active")
     K_diag, K_off = St.build_K_full_active(Nf, Nt, dt, psd, window=window)
-    _close(St.assemble_lambda_full_active(*_K_blocks(K_diag, K_off, na, Nf, Nt), na, Nf, Nt)
-           [np.ix_(p, p)], C, RTOL_KERNEL, "build_K_full_active")
+    _close(St.assemble_lambda_full_active(*_K_blocks(K_diag, K_off, na, Nf, Nt), na, Nf, Nt),
+           C, RTOL_KERNEL, "build_K_full_active")
 
 
 @pytest.mark.parametrize("Nf, Nt, dt, window", [
@@ -496,14 +495,11 @@ def test_stationary_backward_maps_recover_the_psd(window, Nf, Nt, dt):
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_apply_lambda_from_psd_is_the_dense_covariance(window, Nf, Nt, dt):
-    """``apply_lambda_from_psd(x)`` is ``build_C_wdm @ x`` on the active cells (junk in
-    the inactive cells of ``x`` is ignored)."""
+    """``apply_lambda_from_psd(x)`` is ``build_C_wdm @ x``."""
     N = Nf * Nt
-    mask = wilson.grid.active_mask(Nf, Nt)
-    x = np.random.default_rng(N + Nf + 3).standard_normal((Nf + 1, Nt))
+    x = np.random.default_rng(N + Nf + 3).standard_normal((Nf, Nt))
     S = psd(np.fft.rfftfreq(N, dt))
-    want = np.zeros_like(x)
-    want[mask] = _dense(window, Nf, Nt, dt)[3] @ x[mask]
+    want = (_dense(window, Nf, Nt, dt)[1] @ x.ravel()).reshape(Nf, Nt)
     got = St.apply_lambda_from_psd(x, Nf, Nt, dt, S, window=window)
     _close(got, want, RTOL_KERNEL, "apply_lambda_from_psd")
 
@@ -512,19 +508,12 @@ def test_apply_lambda_from_psd_is_the_dense_covariance(window, Nf, Nt, dt):
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_band_variances_are_the_diagonal_of_the_dense_covariance(window, Nf, Nt, dt):
     """``band_averaged_variance`` and the stationary limit of ``evolutionary_band_variance``
-    (all bands, and selected ``bands=``) are ``diag(build_C_wdm)`` on the active cells."""
-    mask = wilson.grid.active_mask(Nf, Nt)
-    want = np.diag(_dense(window, Nf, Nt, dt)[3])
-    _close(No.band_averaged_variance(Nf, Nt, dt, psd, window=window)[mask], want, RTOL_KERNEL,
+    are ``diag(build_C_wdm)``."""
+    want = np.diag(_dense(window, Nf, Nt, dt)[1]).reshape(Nf, Nt)
+    _close(No.band_averaged_variance(Nf, Nt, dt, psd, window=window), want, RTOL_KERNEL,
            "band_averaged_variance")
-    _close(No.evolutionary_band_variance(Nf, Nt, dt, _stationary, window=window)[mask], want,
+    _close(No.evolutionary_band_variance(Nf, Nt, dt, _stationary, window=window), want,
            RTOL_KERNEL, "evolutionary_band_variance")
-    bands = np.array([0, Nf // 2, Nf])
-    grid = np.zeros((Nf + 1, Nt))
-    grid[mask] = want
-    sub = No.evolutionary_band_variance(Nf, Nt, dt, _stationary, bands=bands, window=window)
-    _close(sub[mask[bands]], grid[bands][mask[bands]], RTOL_KERNEL,
-           "evolutionary_band_variance(bands)")
 
 
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
diff --git a/tests/wilson/vendored/test_evolutionary.py b/tests/wilson/vendored/test_evolutionary.py
index 38444bc..5291d28 100644
--- a/tests/wilson/vendored/test_evolutionary.py
+++ b/tests/wilson/vendored/test_evolutionary.py
@@ -79,14 +79,15 @@ def _S_evo(f, t):
 def test_band_average_reduces_to_the_stationary_band_average():
     got = ev.band_averaged_evolutionary_variance(NF, NT, DT, lambda f, t: _S_stat(f) + 0 * t)
     ref = band_averaged_variance(NF, NT, DT, _S_stat)
-    np.testing.assert_allclose(got, ref, rtol=1e-13)
+    np.testing.assert_allclose(got[cells(NF, NT)], ref, rtol=1e-13)
 
 
 def test_band_average_samples_the_cell_centres():
     sep = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo)
     stat = band_averaged_variance(NF, NT, DT, _S_stat)
-    tn = np.arange(NT) * NF * DT
-    np.testing.assert_allclose(sep, stat * (1 + 0.4 * np.sin(2 * np.pi * tn / T_REC))[None, :], rtol=1e-13)
+    m, n = cells(NF, NT)
+    tn = n * NF * DT                                    # each folded slot's cell time
+    np.testing.assert_allclose(sep[m, n], stat * (1 + 0.4 * np.sin(2 * np.pi * tn / T_REC)), rtol=1e-13)
     sub = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF])
     np.testing.assert_array_equal(sub, sep[[0, 3, NF]])
 
diff --git a/tests/wilson/vendored/test_stationary.py b/tests/wilson/vendored/test_stationary.py
index c8fdf5b..12a1828 100644
--- a/tests/wilson/vendored/test_stationary.py
+++ b/tests/wilson/vendored/test_stationary.py
@@ -5,7 +5,6 @@ from __future__ import annotations
 import numpy as np
 import pytest
 
-from domainia.wilson.grid import cells
 from domainia.wilson.matrix import build_C_time, build_W
 from domainia.wilson.stationary import (
     assemble_lambda_full_active,
@@ -150,8 +149,7 @@ def test_full_active_assembly_matches_dense_W_C_W(simple_grid, lisa_psd):
     """Assembled full-active Λ matches the dense W @ C_time @ W^T at machine precision."""
     g = simple_grid
     diag, offd, n_active = build_lambda_blocks_full_active(g.Nf, g.Nt, g.dt, lisa_psd)
-    p = np.ravel_multi_index(cells(g.Nf, g.Nt), (g.Nf + 1, g.Nt)).ravel()
-    Lambda_fast = assemble_lambda_full_active(diag, offd, n_active, g.Nf, g.Nt)[np.ix_(p, p)]
+    Lambda_fast = assemble_lambda_full_active(diag, offd, n_active, g.Nf, g.Nt)
     Lambda_dense = _dense_lambda(g.Nf, g.Nt, g.dt, lisa_psd)
 
     rel = np.max(np.abs(Lambda_fast - Lambda_dense)) / np.max(np.abs(Lambda_dense))
````

- [ ] **Step 2: Run the task's tests: they fail**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error --tb=no -rfE tests/wilson/test_noise_stationary.py tests/wilson/vendored/test_noise.py tests/wilson/vendored/test_stationary.py tests/wilson/vendored/test_evolutionary.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_window_threading.py tests/test_covariance.py tests/wilson/test_meyer_baseline.py 2>&1 | tail -n 40
```

Expected last line (prototype): `80 failed, 407 passed, 2 skipped`. The noise grids are still `(9, 16)`, and `assemble_lambda_full_active` is `(144, 144)`. So comparisons with folded references fail with `operands could not be broadcast together with shapes (9,16) (8,16)`, or `(144,144) (128,128)`. Task 2's bridge in `apply_lambda_from_psd` cannot take a folded input, and the replay generators fail the same way.

- [ ] **Step 3: Apply the implementation patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T4-impl && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists `src/domainia/wilson/noise.py`, `src/domainia/wilson/stationary.py`, `tests/test_covariance.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_noise_stationary.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_stationary.py` as modified, and nothing else.

<!-- patch: T4-impl -->
````diff
diff --git a/src/domainia/wilson/noise.py b/src/domainia/wilson/noise.py
index d5d31c2..83d47b0 100644
--- a/src/domainia/wilson/noise.py
+++ b/src/domainia/wilson/noise.py
@@ -16,9 +16,8 @@ band average of WDM Paper II, Eq. 36:
 
 A Monte-Carlo estimator, :func:`domainia.testing.empirical_variance`, checks them.
 
-Both grids have shape ``(Nf+1, Nt)``.  They fill the inactive edge pixels
-($m = 0$ and $m = N_f$ at odd $n$) by the same formula as the active ones, so they hold
-about $S_n/2$ there, not 0; :func:`domainia.wdm_covariance` sets those cells to 0.
+Both grids are in the folded layout ``(Nf, Nt)`` (:func:`domainia.wilson.grid.cells`): row 0's
+odd slots hold the Nyquist band.
 
 Every kernel here that reaches the window takes ``window=`` (a
 :class:`domainia.wilson.windows.Window`; default :data:`domainia.wilson.MEYER`) and
@@ -34,6 +33,7 @@ from numpy.typing import NDArray
 
 from .grid import (
     band_frequencies,
+    cells,
     delta_F,
     delta_T,
     n_samples,
@@ -66,8 +66,8 @@ def band_averaged_variance(
     = \\frac{\\sum_j \\hat\\phi^2(f_j - f_m)\\,S_n(f_j)}
             {\\sum_j \\hat\\phi^2(f_j - f_m)}$$
 
-    and returns $\\sigma_{mn}^2 = S_n^{\\text{eff}}(m)/2$ broadcast
-    over $n$, shape ``(Nf+1, Nt)``.  This is exact: it reproduces the diagonal of the WDM
+    and returns $\\sigma_{mn}^2 = S_n^{\\text{eff}}(m)/2$ at every cell, folded
+    ``(Nf, Nt)``.  This is exact: it reproduces the diagonal of the WDM
     covariance $W\\,C\\,W^T$ to machine precision, because the pixel
     variance is the $\\hat\\phi^2$-weighted average of the PSD across
     the band.
@@ -79,7 +79,7 @@ def band_averaged_variance(
     >>> steep = lambda f: 1e-3 * (1.0 + 1e2 / (f + 1 / (2 * dt * Nf)) ** 2)
     >>> var = wilson.band_averaged_variance(Nf, Nt, dt, steep)
     >>> var.shape
-    (9, 16)
+    (8, 16)
     >>> # exact: reproduces the diagonal of the WDM covariance
     >>> truth = np.diag(wilson.matrix.build_C_wdm(Nf, Nt, dt, steep))
     >>> bool(abs(var[1, 0] - truth[Nt]) / truth[Nt] < 1e-12)
@@ -117,7 +117,7 @@ def band_averaged_variance(
     weights = phi2[in_band]
     Sn_eff = Sn_vals[:, in_band] @ weights / np.sum(weights)
 
-    return np.broadcast_to(0.5 * Sn_eff[:, None], (Nf + 1, Nt)).copy()
+    return 0.5 * Sn_eff[cells(Nf, Nt)[0]]
 
 
 # ---------------------------------------------------------------------------
@@ -136,19 +136,15 @@ def _psd_at(Sn: PsdFtFunc, f: Array, t: Array, context: str) -> Array:
 
 
 def evolutionary_band_variance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc, *,
-                               bands: Array | None = None,
                                window: Window = MEYER) -> Array:
     r"""The filter-weighted band average of :math:`S(f, t_n)` at every cell centre,
 
     .. math::  \sigma^2_{mn} = \frac12\,
                \frac{\sum_j \tilde\phi^2(f_j - f_m)\,S(f_j, t_n)}{\sum_j \tilde\phi^2(f_j - f_m)},
 
-    WDM Paper II, Eq. 36 (eq:wdm_variance_band_average; the edge-band factor is not
-    applied -- inactive edge cells carry the same value, as in
-    :func:`band_averaged_variance`).  ``Sn(f, t)`` is elementwise and broadcasting
-    (argument order ``(f, t)``).  In the stationary limit this is
-    :func:`band_averaged_variance`.  Returns ``(Nf + 1, Nt)``, or
-    ``(len(bands), Nt)`` for the bands ``bands`` (all by default).
+    WDM Paper II, Eq. 36 (eq:wdm_variance_band_average).  ``Sn(f, t)`` is elementwise and
+    broadcasting (argument order ``(f, t)``).  In the stationary limit this is
+    :func:`band_averaged_variance`.  Returns the folded ``(Nf, Nt)``.
     """
     validate_grid(Nf, Nt)
     N = Nf * Nt
@@ -159,12 +155,11 @@ def evolutionary_band_variance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc, *,
     sup = np.flatnonzero(phi2)
     phi2 = phi2[sup]
     f_off = (np.arange(Nt) - Nt // 2)[sup] / (N * dt)
-    m = np.arange(Nf + 1) if bands is None else np.asarray(bands, int)
-    fb = band_frequencies(Nf, dt)[m][:, None] + f_off[None, :]
+    fb = band_frequencies(Nf, dt)[:, None] + f_off[None, :]
     f_nyq = 0.5 / dt
     ff = np.abs(fb)
-    ff = np.where(ff > f_nyq, 2.0 * f_nyq - ff, ff)                       # (B, J)
+    ff = np.where(ff > f_nyq, 2.0 * f_nyq - ff, ff)                    # (Nf+1, J)
     t = np.arange(Nt) * delta_T(Nf, dt)
     S = _psd_at(Sn, ff[:, None, :], t[None, :, None],
-                "evolutionary_band_variance")                              # (B, Nt, J)
-    return 0.5 * (S * phi2).sum(axis=-1) / phi2.sum()
+                "evolutionary_band_variance")                           # (Nf+1, Nt, J)
+    return (0.5 * (S * phi2).sum(axis=-1) / phi2.sum())[cells(Nf, Nt)]
diff --git a/src/domainia/wilson/stationary.py b/src/domainia/wilson/stationary.py
index 52cabe1..a17e36e 100644
--- a/src/domainia/wilson/stationary.py
+++ b/src/domainia/wilson/stationary.py
@@ -27,7 +27,7 @@ versus $\\mathcal O(N^3)$ for the dense $W\\,C_{\\rm time}\\,W^T$.
 including the parity-restricted edge bands $m \\in \\{0, N_f\\}$, via
 half-slab Hermitian-extended IFFTs, with the edge $\\sqrt 2$ factors of the unit-norm
 basis (:mod:`domainia.wilson.transform`) built in.  It matches
-the dense $W_a\\,C_{\\rm time}\\,W_a^T$ at machine precision.
+the dense $W\\,C_{\\rm time}\\,W^T$ at machine precision.
 
 Backward maps (Paper I Alg. 4)
 ------------------------------
@@ -57,7 +57,7 @@ import numpy as np
 from numpy.typing import NDArray
 
 from ._checks import REAL_KINDS, require_dtype
-from .grid import active_mask, n_samples, sanitize_psd, validate_dt, validate_grid
+from .grid import n_samples, sanitize_psd, unfold, validate_dt, validate_grid
 from .windows import MEYER, Window, sample_window
 
 type Array = NDArray[np.floating]
@@ -422,7 +422,7 @@ def build_lambda_blocks_full_active(
     ...     diag, offd, n_active, Nf, Nt
     ... )
     >>> Lam.shape
-    (144, 144)
+    (128, 128)
     """
     validate_grid(Nf, Nt)
     validate_dt(dt)
@@ -715,17 +715,16 @@ def apply_lambda_from_psd(
 
     Parameters
     ----------
-    x_wmn : ndarray, float, shape (Nf+1, Nt)
-        Input WDM coefficient array.  Inactive (parity-killed) edge entries
-        are ignored.
+    x_wmn : ndarray, float, shape (Nf, Nt)
+        Input WDM coefficients, folded (:func:`domainia.wilson.transform.forward`).
     Sn_rfft : ndarray, float, shape (N//2 + 1,)
         One-sided PSD evaluated on the ``rfftfreq(N, dt)`` grid.  ``inf`` samples are
         zero power (:func:`domainia.wilson.grid.sanitize_psd`).
 
     Returns
     -------
-    ndarray, float, shape (Nf+1, Nt)
-        $\Lambda \bm x$.
+    ndarray, float, shape (Nf, Nt)
+        $\Lambda \bm x$, folded.
 
     Notes
     -----
@@ -740,14 +739,12 @@ def apply_lambda_from_psd(
     >>> import numpy as np; from domainia import wilson
     >>> Nf, Nt, dt = 8, 16, 1.0
     >>> N = Nf * Nt
-    >>> x = wilson.grid.unfold(wilson.forward_time(
-    ...     np.random.default_rng(0).standard_normal(N), Nf, Nt, dt
-    ... ), Nf, Nt)
+    >>> x = wilson.forward_time(np.random.default_rng(0).standard_normal(N), Nf, Nt, dt)
     >>> Sn = 1e-3 * np.ones(N // 2 + 1)
     >>> wilson.stationary.apply_lambda_from_psd(x, Nf, Nt, dt, Sn).shape
-    (9, 16)
+    (8, 16)
     """
-    from .transform import _forward, _inverse
+    from .transform import forward, inverse
 
     validate_grid(Nf, Nt)
     validate_dt(dt)
@@ -758,9 +755,8 @@ def apply_lambda_from_psd(
             f"apply_lambda_from_psd: Sn_rfft must have shape ({n_rfft},) -- the "
             f"np.fft.rfftfreq(N, dt) grid; got {Sn_rfft.shape}"
         )
-    y = _inverse(x_wmn * active_mask(Nf, Nt), Nf, Nt, dt, window=window)    # inactive: ignored
-    z = (Sn_rfft / 2.0) * y
-    return _forward(z, Nf, Nt, dt, window=window)
+    z = (Sn_rfft / 2.0) * inverse(x_wmn, Nf, Nt, dt, window=window)
+    return forward(z, Nf, Nt, dt, window=window)
 
 
 def psd_from_lambda_blocks(
@@ -843,11 +839,12 @@ def assemble_lambda_full_active(
     Nf: int,
     Nt: int,
 ) -> Array:
-    """Assemble the full-active blocks into a dense $(N_f+1)N_t \\times (N_f+1)N_t$ matrix.
+    """Assemble the full-active blocks into the dense $N \\times N$ matrix, rows and columns in
+    folded flat order.
 
-    The blocks are :func:`build_lambda_blocks_full_active`'s.  Parity-killed edge entries
-    stay exactly zero, and the lower off-diagonal is filled by transposition, so the result
-    is symmetric: the dense $W\\,C_{\\rm time}\\,W^T$ at machine precision.
+    The blocks are :func:`build_lambda_blocks_full_active`'s.  The lower off-diagonal is
+    filled by transposition, so the result is symmetric: the dense
+    $W\\,C_{\\rm time}\\,W^T$ at machine precision.
 
     Examples
     --------
@@ -857,38 +854,17 @@ def assemble_lambda_full_active(
     ...     8, 16, 1.0, flat
     ... )
     >>> wilson.stationary.assemble_lambda_full_active(d, o, na, 8, 16).shape
-    (144, 144)
+    (128, 128)
     """
     validate_grid(Nf, Nt)
     _check_block_layout(diag_blocks, offd_blocks, n_active, Nf)
-    M = (Nf + 1) * Nt
-    Lambda = np.zeros((M, M))
-    band_offset = [m * Nt for m in range(Nf + 2)]
-
-    # Diagonal blocks: place in active rows/cols of band-m's slot.  Interior
-    # bands are contiguous, so a plain slice beats np.ix_'s fancy indexing.
+    N = n_samples(Nf, Nt)
+    slot = unfold(np.arange(N).reshape(Nf, Nt), Nf, Nt)      # each cell's folded flat index
+    idx = [slot[m, n] for m, n in enumerate(n_active)]
+    Lambda = np.zeros((N, N))
     for m, D in enumerate(diag_blocks):
-        if m in (0, Nf):
-            idx = band_offset[m] + n_active[m]
-            Lambda[np.ix_(idx, idx)] = D
-        else:
-            s0 = band_offset[m]
-            Lambda[s0 : s0 + Nt, s0 : s0 + Nt] = D
-
-    # Off-diagonal (m, m+1) blocks: edge dimensions on the edge side only.
+        Lambda[np.ix_(idx[m], idx[m])] = D
     for m, L in enumerate(offd_blocks):
-        row_idx: NDArray[np.integer]
-        col_idx: NDArray[np.integer]
-        if m == 0:
-            row_idx = band_offset[0] + n_active[0]  # edge restriction
-            col_idx = band_offset[1] + np.arange(Nt)  # interior full
-        elif m == Nf - 1:
-            row_idx = band_offset[Nf - 1] + np.arange(Nt)
-            col_idx = band_offset[Nf] + n_active[Nf]  # edge restriction
-        else:
-            row_idx = band_offset[m] + np.arange(Nt)
-            col_idx = band_offset[m + 1] + np.arange(Nt)
-        Lambda[np.ix_(row_idx, col_idx)] = L
-        Lambda[np.ix_(col_idx, row_idx)] = L.T
-
+        Lambda[np.ix_(idx[m], idx[m + 1])] = L
+        Lambda[np.ix_(idx[m + 1], idx[m])] = L.T
     return Lambda
````

- [ ] **Step 4: Run the task's tests: they pass**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/test_noise_stationary.py tests/wilson/vendored/test_noise.py tests/wilson/vendored/test_stationary.py tests/wilson/vendored/test_evolutionary.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_window_threading.py tests/test_covariance.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/noise.py src/domainia/wilson/stationary.py 2>&1 | tail -1
```

Expected (prototype): `495 passed, 2 skipped`.

- [ ] **Step 5: Run the lanes, the replay and the size**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py
```

Expected (prototype): `943 passed, 3 skipped` in both lanes, with no warnings and no collection errors; `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`; `src 4099 lines / 1864 code; tests 4346 lines (+ oracle 4135)`.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/wilson/noise.py src/domainia/wilson/stationary.py tests/test_covariance.py tests/wilson/meyer_cases.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_noise_stationary.py tests/wilson/test_window_threading.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_stationary.py && git commit -q -F - <<'EOF'
Fold the noise variance grids, apply_lambda_from_psd and assemble_lambda_full_active

band_averaged_variance and evolutionary_band_variance return the folded (Nf, Nt), row 0's
odd slots holding the Nyquist band; evolutionary_band_variance loses bands=, which nothing
in src used. apply_lambda_from_psd is forward((S/2) inverse(x)) through the public kernels,
so it takes and returns the folded layout and refuses a padded or flat input with the named
fix. assemble_lambda_full_active places each band block at its cells' folded flat indices
and returns N x N. The band blocks and psd_from_lambda_blocks' padded matvec are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
```


### Task 5: `wilson.evolutionary` and `domainia.testing` fold (spec §3 `evolutionary.py`, §4 `domainia.testing`, §5; §8 step 5)

**Goal:**
- **Folded `(Nf, Nt)`:** `energy_fraction`, `testing.empirical_variance`, and `wrap_fractions` with `bands=None`.
- **`(..., Nf, Nt)` in and out:** `apply_time_mask`.
- **Folded flat order:** `exact_covariance` and `eq45_direct`.
- **Folded `(Nf, Nt)` masks:** `eq45_plan(keep)`, `testing.dense_W`, `wdm_congruence` and `eq45_dense`.
- **`_forward_time` goes:** its last internal callers move to the public kernels.

**Files:**
- Modify: `src/domainia/wilson/evolutionary.py`, `src/domainia/testing.py`, `src/domainia/wilson/transform.py` (`_forward_time` goes)
- Test: `tests/wilson/meyer_cases.py` (in both patches), `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/test_dom_testing.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/test_noise_stationary.py`, `tests/test_covariance.py`

**What the patches do:**
- **`energy_fraction`** fills a folded `(Nf, Nt)` grid through `cells`. The FFT correlation loop is unchanged.
- **`apply_time_mask`** is `forward_time(inverse_time(Y) · chi)` through the public kernels, so a padded or flat `Y` is refused with the named fix. It no longer promotes a 1-D input.
- **`exact_covariance`** is `forward_time(G.T).reshape(-1, N).T`.
- **`wrap_fractions`:**
  - It computes the band grid as before.
  - With `bands=None` it returns it gathered with `cells`: folded, with no NaN.
  - With `bands=` it keeps returning band rows, with NaN at an edge band's odd columns.
  - `wrap_cut_columns` always asks for band rows, so its column maxima never see a folded column.
- **`eq45_plan(keep)`:**
  - It takes a boolean `(Nf, Nt)` mask, with `m, n` taken from `cells`.
  - Its shape refusal reads `keep must have shape (Nf, Nt) = (8, 16), got (128,)`. The `TypeError` text is unchanged.
  - The per-cell arrays follow the folded flat order of the kept cells. Within a band, times still ascend, so the Eq. 45 blocks and `eq45_direct`'s code are unchanged.
- **`testing`:**
  - `_keep` is replaced by a smaller `_mask`, which keeps the kind check (`TypeError`), the grid check and the shape check (`ValueError`). The inactive-cell refusal goes.
  - `dense_W` is `build_W(...)[mask.ravel()]`, which removes Task 3's bridge.
  - `eq45_dense` passes its mask straight to `eq45_plan`.
  - `empirical_variance` gathers its padded accumulation with `cells`.
- **`_forward_time`:** with `apply_time_mask` and `exact_covariance` on the public kernels, nothing else calls this twin. It goes, and `forward_time`'s body is again the e6d1e3f one, ending in `forward`.
- **Replay:**
  - `FOLDED` adds `empirical_variance` and `wrap_fractions`, gathered raw because `wrap_fractions` records −1 in inactive cells.
  - `BAND_ORDER` (`energy_fraction`, `apply_time_mask`, `BandAveragedVariance`) is permuted from band order and reshaped to `(..., Nf, Nt)`.
  - `DENSE` (`build_W`, `build_C_wdm`, `exact_covariance`, `eq45_direct`, the three `testing.*` cases and `EvolutionaryCovariance.direct`) is permuted on rows, and on columns except for the `_W` families.
  - Every recorded subset case selects every cell, so one permutation per grid serves them all. Spec §5's per-case rebuild of selections is not needed; a later subset case would fail on shape, not silently.
  - The generators pass folded masks and folded `Y`. The implementation patch merges the two dense tuples into `DENSE`.
- **Tests:**
  - Our outputs are permuted into the oracle's band order with `BAND`, where the oracle is band-ordered.
  - The padded oracle grids are gathered with `cells`.
  - The NaN-pattern checks go, and so do `test_dom_testing`'s padded-mask and inactive-cell refusals; a padded mask is now a wrong-shape refusal.
  - `test_dom_testing`'s atom assertion is one line again.

**Acceptance Criteria:**
- [ ] `energy_fraction(...)` and `testing.empirical_variance(...)` are `(Nf, Nt)`; `eq45_plan(8, 16, 1.0, np.ones(128, bool))` raises `ValueError`; `testing.dense_W(8, 16, 1.0, np.ones((8, 16), bool)).shape == (128, 128)`.
- [ ] `grep -n "_forward_time" src` prints nothing.
- [ ] The replay prints `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`.
- [ ] Full suite (prototype): PY and FLOOR `943 passed, 3 skipped`, no warnings, no collection errors.
- [ ] Size (prototype): `src 4084 lines / 1852 code; tests 4354 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/test_evolutionary_kernels.py tests/wilson/vendored/test_evolutionary.py tests/test_dom_testing.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py tests/wilson/test_noise_stationary.py tests/test_covariance.py tests/wilson/test_wrong_kinds.py src/domainia/wilson/evolutionary.py src/domainia/wilson/transform.py 2>&1 | tail -1` → `583 passed, 2 skipped`

**Steps:**

- [ ] **Step 1: Apply the tests patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T5-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists exactly `tests/test_covariance.py`, `tests/test_dom_testing.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_noise_stationary.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py` as modified.

<!-- patch: T5-tests -->
````diff
diff --git a/tests/test_covariance.py b/tests/test_covariance.py
index b78bef3..03a37a4 100644
--- a/tests/test_covariance.py
+++ b/tests/test_covariance.py
@@ -14,8 +14,7 @@ LEVELS = np.array([[1.0, 0.6, 0.3], [0.6, 1.2, -0.4], [0.3, -0.4, 0.9]])
 DELAYS, PHASES = np.array([0.0, 7.3, -11.9]) * DT, np.array([0.0, 0.4, -1.1])
 #: A complex mixing matrix for the loop-built model.
 B = np.random.default_rng(20261114).standard_normal((K, K, 2)) @ [1.0, 1.0j]
-#: The WDM grid of the stationary case, and its active cells in numpy.nonzero order (the rows
-#: of testing.dense_W).
+#: The WDM grid of the stationary case, and its active cells.
 NF, NT = 16, 32
 MASK = wilson.grid.active_mask(NF, NT)
 #: The routes from a stationary model S(f) to a covariance; the WDM route keeps the active cells.
@@ -95,11 +94,11 @@ def sigma(S, N, dt):
 
 
 def same_cell(Nf, Nt, Tp):
-    """The same-cell blocks of spec 6.2's dense evolutionary reference on the active cells,
-    (K, K, N): (I_K x W) Sigma (I_K x W)^T with Sigma_ab[i, j] = [B(t_i) B(t_j)^T]_ab R(i - j),
+    """The same-cell blocks of spec 6.2's dense evolutionary reference, (K, K, N) in folded flat
+    order: (I_K x W) Sigma (I_K x W)^T with Sigma_ab[i, j] = [B(t_i) B(t_j)^T]_ab R(i - j),
     where R is the stationary reference of psd."""
     N = Nf * Nt
-    W = testing.dense_W(Nf, Nt, DT, wilson.grid.active_mask(Nf, Nt))
+    W = testing.dense_W(Nf, Nt, DT, np.ones((Nf, Nt), bool))
     R = sigma(lambda f: psd(f)[None, None], N, DT)[0, 0]
     Bt = np.moveaxis(mixing(np.arange(N) * DT, Tp), 0, -1).copy()     # Bt[a, c, i] = B_ac(t_i)
     out = 0.0
@@ -144,11 +143,11 @@ def test_wdm_covariance_of_a_stationary_model_is_the_dense_reference():
     size-1 time axis gives the explicitly broadcast output, bit for bit."""
     Sigma = sigma(delayed, NF * NT, DT)
     for window in (wilson.MEYER, testing.ALT_WINDOW):
-        W = testing.dense_W(NF, NT, DT, MASK, window=window)
+        W = testing.dense_W(NF, NT, DT, np.ones((NF, NT), bool), window=window)
         want = np.einsum("pi,abij,pj->abp", W, Sigma, W, optimize=True)
         got = wdm_covariance(lambda f, t: delayed(f), Nf=NF, Nt=NT, dt=DT, window=window)
         assert got.shape == (K, K, NF + 1, NT) and got.dtype == np.float64
-        assert error(got[..., MASK], want) < 1e-12
+        assert error(wilson.grid.fold(got, NF, NT).reshape(K, K, -1), want) < 1e-12
         assert not np.any(got[..., ~MASK])
     full = wdm_covariance(lambda f, t: delayed(f + 0 * t), Nf=NF, Nt=NT, dt=DT, window=window)
     assert np.array_equal(got, full)
@@ -163,12 +162,13 @@ def test_wdm_covariance_of_an_evolutionary_model(monkeypatch):
     Nf, Nt = 16, 64
     mask = wilson.grid.active_mask(Nf, Nt)
     lead, trail = wilson.evolutionary.wrap_cut_columns(Nf, Nt, DT)
-    n = np.nonzero(mask)[1]
+    n = wilson.grid.cells(Nf, Nt)[1].ravel()
     kept = (n >= lead) & (n < Nt - trail)
     for Tp, bound in ((32, 5e-2), (64, 1.5e-2)):
         S = modulated(Tp * Nf * DT)
         got = wdm_covariance(S, Nf=Nf, Nt=Nt, dt=DT)
-        assert error(got[..., mask][..., kept], same_cell(Nf, Nt, Tp * Nf * DT)[..., kept]) < bound
+        assert error(wilson.grid.fold(got, Nf, Nt).reshape(K, K, -1)[..., kept],
+                     same_cell(Nf, Nt, Tp * Nf * DT)[..., kept]) < bound
         for a in range(K):
             want = wilson.noise.evolutionary_band_variance(
                 Nf, Nt, DT, lambda f, t, S=S, a=a: S(f, t)[a, a])
diff --git a/tests/test_dom_testing.py b/tests/test_dom_testing.py
index 8cdd802..514967b 100644
--- a/tests/test_dom_testing.py
+++ b/tests/test_dom_testing.py
@@ -16,10 +16,8 @@ from domainia import testing as T
 NF, NT, DT = 8, 16, 1.0 / 16
 N = NF * NT
 F_MIN, F_REF = 0.5, 1.0
-#: Every active cell.
-MASK = wilson.grid.active_mask(NF, NT)
-#: ``wilson.matrix.build_W``'s folded rows in ``MASK``'s band order.
-BAND = np.argsort(np.ravel_multi_index(wilson.grid.cells(NF, NT), (NF + 1, NT)), axis=None)
+#: Every cell.
+MASK = np.ones((NF, NT), bool)
 
 
 def L_of(f):
@@ -39,8 +37,7 @@ def rel(a, b):
 def test_dense_W_rows_are_the_atoms(rng):
     W = T.dense_W(NF, NT, DT, MASK)
     x = rng.standard_normal(N)
-    w = wilson.grid.unfold(wilson.forward_time(x, NF, NT, DT), NF, NT)
-    assert np.max(np.abs(W @ x - w[MASK])) < 1e-12
+    assert np.max(np.abs(W @ x - wilson.forward_time(x, NF, NT, DT)[MASK])) < 1e-12
     assert np.max(np.abs(W @ W.T - DT * np.eye(N))) < 1e-14
     sub = MASK.copy()
     sub[:, :3] = sub[:, -1:] = False
@@ -65,7 +62,7 @@ def test_eq45_differs_from_exact_at_second_order_with_drift():
     leading and 1 trailing column)."""
     C = T.eq45_dense(NF, NT, DT, MASK, drift(-2.0, 1.2))
     exact = exact_covariance(NF, NT, DT, drift(-2.0, 1.2))
-    interior = ~np.isin(np.nonzero(MASK)[1], np.r_[0:3, NT - 1])
+    interior = ~np.isin(wilson.grid.cells(NF, NT)[1].ravel(), np.r_[0:3, NT - 1])
     d = np.diag(exact)[interior]
     assert np.max(np.abs(np.diag(C)[interior] / d - 1)) < 0.05
 
@@ -74,26 +71,26 @@ def test_the_oracles_compute_in_the_window_they_are_given():
     """With ALT_WINDOW: ``dense_W`` is ``build_W``, the congruence is ``W S W^T``, and Eq. 45
     of a stationary spectrum is the exact covariance; each differs from Meyer's."""
     W = T.dense_W(NF, NT, DT, MASK, window=T.ALT_WINDOW)
-    np.testing.assert_array_equal(W, wilson.matrix.build_W(NF, NT, DT, window=T.ALT_WINDOW)[BAND])
+    np.testing.assert_array_equal(W, wilson.matrix.build_W(NF, NT, DT, window=T.ALT_WINDOW))
     S = np.diag(1.0 + np.arange(N) / N)                  # not the identity: W I W^T = dt I for any window
     alt = T.wdm_congruence(S, NF, NT, DT, MASK, window=T.ALT_WINDOW)
     np.testing.assert_allclose(alt, W @ S @ W.T, rtol=0, atol=1e-12)
     assert not np.allclose(alt, T.wdm_congruence(S, NF, NT, DT, MASK))
     got = T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0), window=T.ALT_WINDOW)
     exact = wilson.matrix.build_C_wdm(NF, NT, DT, lambda f: drift(-1.2, 0.0)(f, 0.0),
-                                      window=T.ALT_WINDOW)[np.ix_(BAND, BAND)]
+                                      window=T.ALT_WINDOW)
     assert rel(got, exact) < 1e-10
     assert not np.allclose(got, T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0)))
 
 
 def test_the_oracles_refuse_a_mask_of_the_wrong_kind_or_value():
-    """A mask that is not boolean is a wrong kind; a wrong shape or an inactive cell is a wrong
-    value.  ``eq45_dense`` also needs a callable spectrum."""
+    """A mask that is not boolean is a wrong kind; a wrong shape (flat, or padded
+    ``(Nf + 1, Nt)``) is a wrong value.  ``eq45_dense`` also needs a callable spectrum."""
     for mask, err, message in (
         (MASK.astype(int), TypeError, "mask must be a boolean cell mask, got dtype int64"),
         (None, TypeError, "mask must be a boolean cell mask, got dtype object"),
-        (MASK.reshape(-1), ValueError, "mask must have shape (Nf + 1, Nt) = (9, 16), got (144,)"),
-        (np.ones((NF + 1, NT), bool), ValueError, "mask selects the inactive cell (m=0, n=1)"),
+        (MASK.reshape(-1), ValueError, "mask must have shape (Nf, Nt) = (8, 16), got (128,)"),
+        (np.ones((NF + 1, NT), bool), ValueError, "mask must have shape (Nf, Nt) = (8, 16), got (9, 16)"),
     ):
         match = f"^{re.escape(message)}$"
         with pytest.raises(err, match=match):
diff --git a/tests/wilson/meyer_cases.py b/tests/wilson/meyer_cases.py
index d39f9c8..0d2e84f 100644
--- a/tests/wilson/meyer_cases.py
+++ b/tests/wilson/meyer_cases.py
@@ -47,14 +47,25 @@ DROPPED = ("domainia::TimeToWDM.", "domainia::WDMToTime.", "domainia::FourierToW
            "wilson::matrix.active_W::", "wilson::matrix.build_C_wdm_active::")
 #: The comparison: ``max |got - want| <= RTOL * s`` (see :func:`mismatches`).
 RTOL = 1e-12
-#: Families recorded padded ``(..., Nf+1, Nt)`` whose kernels now return the folded layout.
+#: Families recorded padded ``(..., Nf+1, Nt)`` whose kernels now return the folded layout
+#: (``wrap_fractions`` recorded -1 at the inactive cells).
 FOLDED = ("wilson::transform.forward", "wilson::transform.forward_many",
           "wilson::transform.forward_time", "wilson::transform.forward_time_many",
           "wilson::stationary.apply_lambda_from_psd", "wilson::noise.band_averaged_variance",
-          "wilson::noise.evolutionary_band_variance")
+          "wilson::noise.evolutionary_band_variance", "wilson::noise.empirical_variance",
+          "wilson::evolutionary.wrap_fractions")
 #: Dense families recorded with padded ``(Nf+1) Nt`` rows (and columns, ``build_C_wdm``), now
 #: in folded flat order.
 FOLDED_ROWS = ("wilson::matrix.build_W", "wilson::matrix.build_C_wdm")
+#: Families recorded over the active cells in band order (``numpy.nonzero`` of the padded mask),
+#: now ``(..., Nf, Nt)`` folded.
+BAND_ORDER = ("wilson::evolutionary.energy_fraction", "wilson::evolutionary.apply_time_mask",
+              "domainia::BandAveragedVariance")
+#: Dense families recorded with band-order rows (``dense_W``) or rows and columns, now in folded
+#: flat order.  Every recorded case selects every cell, so one permutation serves them all.
+BAND_ROWS = ("domainia::testing.dense_W", "domainia::testing.wdm_congruence",
+             "domainia::testing.eq45_dense", "domainia::EvolutionaryCovariance.direct",
+             "wilson::evolutionary.exact_covariance", "wilson::evolutionary.eq45_direct")
 
 Case = tuple[str, Callable[[dict], Any]]
 
@@ -67,21 +78,28 @@ def _family(key: str) -> str:
 def load_baseline() -> dict[str, np.ndarray]:
     """The recorded baseline as ``{key: array}``.  The families whose kernels now return the
     folded WDM layout are converted here, never in :func:`mismatches` (folded WDM spec,
-    section 5): the :data:`FOLDED` ones are gathered into the folded slots, and the
-    :data:`FOLDED_ROWS` rows (and the covariance's columns) into folded flat order, without
-    ``wilson.grid.fold``'s check of the inactive cells."""
+    section 5): the :data:`FOLDED` ones are gathered into the folded slots, without
+    ``wilson.grid.fold``'s check of the inactive cells, and the others are permuted into folded
+    order: :data:`BAND_ORDER`'s last axis, the rows of ``build_W`` and ``dense_W``, and the rows
+    and columns of the other :data:`FOLDED_ROWS` and :data:`BAND_ROWS` matrices."""
     from domainia.wilson.grid import cells
 
     with np.load(BASELINE) as z:
         out = {k: z[k] for k in z.files}
     for k, v in out.items():
         fam = _family(k)
-        if fam in FOLDED + FOLDED_ROWS:
+        if fam in FOLDED + FOLDED_ROWS + BAND_ORDER + BAND_ROWS:
             Nf, Nt = map(int, k.split("::")[2].split("x"))
             m, n = cells(Nf, Nt)
             p = (m * Nt + n).ravel()                # each folded slot's padded flat index
-            out[k] = (v[..., m, n] if fam in FOLDED else v[p] if fam == "wilson::matrix.build_W"
-                      else v[np.ix_(p, p)])
+            b = p.argsort().argsort()               # ... and its band-order index
+            if fam in FOLDED:
+                out[k] = v[..., m, n]
+            elif fam in BAND_ORDER:
+                out[k] = v[..., b].reshape(v.shape[:-1] + (Nf, Nt))
+            else:
+                i = p if fam in FOLDED_ROWS else b
+                out[k] = v[i] if fam.endswith("_W") else v[np.ix_(i, i)]
     return out
 
 
@@ -151,7 +169,6 @@ def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
     # Fixed WDM-domain inputs, computed once with the default (Meyer) window.
     w = wl.forward(xr, Nf, Nt, dt)
     Wm = wl.forward(Xr, Nf, Nt, dt)
-    mask = wl.grid.active_mask(Nf, Nt)
     Tr, M, St, No, E = wl.transform, wl.matrix, wl.stationary, wl.noise, wl.evolutionary
     K0 = St.build_K_full_active(Nf, Nt, dt, psd)[0]
     B0 = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd)
@@ -182,11 +199,8 @@ def wilson_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
          lambda kw: testing.empirical_variance(Nf, Nt, dt, psd, 3, seed=11, **kw)),
         # w2 is a length-N time-domain weight (rho_p = sum_t w2(t) g_p(t)^2 / dt).
         ("evolutionary.energy_fraction", lambda kw: E.energy_fraction(x ** 2, Nf, Nt, dt, **kw)),
-        ("evolutionary.apply_time_mask",
-         lambda kw: E.apply_time_mask(wl.grid.unfold(Wm, Nf, Nt)[:, mask], chi, Nf, Nt, dt, **kw)),
-        # nan (inactive cells) -> -1, as recorded; the fractions lie in [0, 1].
-        ("evolutionary.wrap_fractions",
-         lambda kw: np.nan_to_num(E.wrap_fractions(Nf, Nt, dt, **kw), nan=-1.0)),
+        ("evolutionary.apply_time_mask", lambda kw: E.apply_time_mask(Wm, chi, Nf, Nt, dt, **kw)),
+        ("evolutionary.wrap_fractions", lambda kw: E.wrap_fractions(Nf, Nt, dt, **kw)),
         ("evolutionary.wrap_cut_columns", lambda kw: np.asarray(E.wrap_cut_columns(Nf, Nt, dt, **kw))),
         ("evolutionary.eq45_plan", lambda kw: E.eq45_plan(Nf, Nt, dt, **kw).base),
     ]
@@ -204,8 +218,8 @@ def domainia_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
     """The cases recorded on the domainia layer, under their recorded names.  The classes
     that recorded them are computed by the kernels they wrapped: ``EvolutionaryCovariance``
     by ``eq45_plan`` (``P = 24``) + ``eq45_coeffs_interp`` + ``eq45_blocks_cheb`` and by
-    ``eq45_direct``, ``BandAveragedVariance`` by ``noise.evolutionary_band_variance`` on the
-    active cells.  ``kw`` goes to every window-taking call."""
+    ``eq45_direct``, ``BandAveragedVariance`` by ``noise.evolutionary_band_variance``.  ``kw``
+    goes to every window-taking call."""
     from domainia import testing
 
     N = Nf * Nt
@@ -213,8 +227,7 @@ def domainia_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
     rng.standard_normal(5 * N + 2 * Nt)        # the recorded maps cases' inputs: Sigma unchanged
     Sn = psd_ft(N * dt, depth=0.4)
     E = wl.evolutionary
-    mask = wl.grid.active_mask(Nf, Nt)
-    m, n = np.nonzero(mask)
+    mask = np.ones((Nf, Nt), bool)             # every cell
 
     def interp(kw: dict) -> Any:
         plan = E.eq45_plan(Nf, Nt, dt, **kw)
@@ -222,8 +235,7 @@ def domainia_cases(wl: Any, Nf: int, Nt: int, dt: float) -> list[Case]:
                                                              k=plan.k_used))
 
     cases: list[Case] = [
-        ("BandAveragedVariance", lambda kw: wl.grid.unfold(
-            wl.noise.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw), Nf, Nt)[m, n]),
+        ("BandAveragedVariance", lambda kw: wl.noise.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)),
         ("EvolutionaryCovariance.interp", interp),
     ]
     if (Nf, Nt) == DENSE_GRID:
diff --git a/tests/wilson/test_evolutionary_kernels.py b/tests/wilson/test_evolutionary_kernels.py
index 4ac4f9c..1663a19 100644
--- a/tests/wilson/test_evolutionary_kernels.py
+++ b/tests/wilson/test_evolutionary_kernels.py
@@ -35,7 +35,7 @@ F_NYQ = 0.5 / DT
 LINEAGE_A = ("toy_models/vendor/wdm/src", "wdm_paper_2/toy_models/src")
 #: max|got - ref| over max|ref| against the omega-era sources (float64).
 ORACLE_RTOL = 1e-14
-#: ``build_W``'s folded rows in band order (``numpy.nonzero`` of the padded mask).
+#: The folded flat indices in the oracle's band order (``numpy.nonzero`` of the padded mask).
 BAND = np.argsort(np.ravel_multi_index(cells(NF, NT), (NF + 1, NT)), axis=None)
 
 
@@ -81,21 +81,22 @@ def test_moved_functions_are_the_oracle_to_rounding(rng):
     time-domain kernels (no window) bit for bit."""
     g = ev.PackedGrid(NF, NT, DT)
     w2 = rng.uniform(0.0, 1.0, N)
-    Y = rng.standard_normal((2, N))
-    # Hz window (v0.12.0): measured 5.1e-16 (energy_fraction), 2.2e-16 (apply_time_mask)
-    assert _gap(E.energy_fraction(w2, NF, NT, DT), g.energy_fraction(w2)) <= ORACLE_RTOL
-    assert _gap(E.apply_time_mask(Y, w2, NF, NT, DT), g.apply_mask(Y, w2)) <= ORACLE_RTOL
-    got, ref = E.wrap_fractions(NF, NT, DT), ev.wrap_fractions(NF, NT, DT)
-    assert np.array_equal(np.isnan(got), np.isnan(ref))
-    # Hz window (v0.12.0): measured 4.2e-16
-    assert _gap(np.nan_to_num(got), np.nan_to_num(ref)) <= ORACLE_RTOL
+    Y = rng.standard_normal((2, NF, NT))
+    # Hz window (v0.12.0): measured 5.1e-16 (energy_fraction), 2.2e-16 (apply_time_mask).  The
+    # oracle packs the cells in band order (BAND).
+    assert _gap(E.energy_fraction(w2, NF, NT, DT).ravel()[BAND], g.energy_fraction(w2)) <= ORACLE_RTOL
+    assert _gap(E.apply_time_mask(Y, w2, NF, NT, DT).reshape(2, N)[:, BAND],
+                g.apply_mask(Y.reshape(2, N)[:, BAND], w2)) <= ORACLE_RTOL
+    # Hz window (v0.12.0): measured 4.2e-16; the oracle's padded grid is gathered
+    assert _gap(E.wrap_fractions(NF, NT, DT), ev.wrap_fractions(NF, NT, DT)[cells(NF, NT)]
+                ) <= ORACLE_RTOL
     for tol in (1e-2, 1e-3, 1e-4):
         assert E.wrap_cut_columns(NF, NT, DT, tol) == ev.wrap_cut_columns(NF, NT, DT, tol)
     assert np.array_equal(E.time_domain_factor(N, DT, _S_evo), ev.time_domain_factor(N, DT, _S_evo))
     assert np.array_equal(E.time_domain_covariance(N, DT, _S_evo),
                           ev.time_domain_covariance(N, DT, _S_evo))
     # Hz window (v0.12.0): measured 2.8e-16
-    assert _gap(E.exact_covariance(NF, NT, DT, _S_evo),
+    assert _gap(E.exact_covariance(NF, NT, DT, _S_evo)[np.ix_(BAND, BAND)],
                 ev.exact_covariance(NF, NT, DT, _S_evo)) <= ORACLE_RTOL
     assert np.array_equal(E.draw_noise(N, DT, _S_evo, np.random.default_rng(5)),
                           ev.draw_noise(N, DT, _S_evo, np.random.default_rng(5)))
@@ -106,18 +107,19 @@ def test_moved_functions_are_the_oracle_to_rounding(rng):
 
 
 def test_energy_fraction_and_time_mask_match_the_dense_mask(rng):
-    Wa = build_W(NF, NT, DT)[BAND]
+    W = build_W(NF, NT, DT)
     w2 = rng.uniform(0.0, 1.0, N)
-    Mm = Wa @ np.diag(w2) @ Wa.T / DT
-    Y = rng.standard_normal((2, N))
-    assert np.allclose(E.apply_time_mask(Y, w2, NF, NT, DT), Y @ Mm.T, rtol=0, atol=1e-12)
-    assert np.allclose(E.energy_fraction(w2, NF, NT, DT), np.diag(Mm), rtol=0, atol=1e-12)
+    Mm = W @ np.diag(w2) @ W.T / DT
+    Y = rng.standard_normal((2, NF, NT))
+    assert np.allclose(E.apply_time_mask(Y, w2, NF, NT, DT), (Y.reshape(2, N) @ Mm.T).reshape(Y.shape),
+                       rtol=0, atol=1e-12)
+    assert np.allclose(E.energy_fraction(w2, NF, NT, DT), np.diag(Mm).reshape(NF, NT), rtol=0, atol=1e-12)
 
 
 def test_wrap_rule_on_the_sinusoid_grid():
     """The cut is (6, 5) at (64, 256, 2**-10).  The last cut columns wrap 0.11200% (col 5)
     and 0.11708% (col 251); the first kept ones 0.0631% (cols 6 and 250)."""
-    frac = np.nanmax(E.wrap_fractions(64, 256, 2.0**-10), axis=0)
+    frac = np.nanmax(E.wrap_fractions(64, 256, 2.0**-10, bands=np.arange(65)), axis=0)
     assert E.wrap_cut_columns(64, 256, 2.0**-10) == (6, 5)
     assert frac[5] == pytest.approx(1.1200e-3, abs=5e-8)
     assert frac[251] == pytest.approx(1.1708e-3, abs=5e-8)
@@ -153,7 +155,8 @@ def test_interp_blocks_and_direct_sum_are_the_oracle_to_rounding(keep):
     g = ev.PackedGrid(NF, NT, DT)
     mask = None if keep is None else ~np.isin(g.n, (0, 1, NT - 1))
     tri = ev.TriCovariance(g, mask)
-    plan = E.eq45_plan(NF, NT, DT, mask)
+    keep = None if keep is None else ~np.isin(cells(NF, NT)[1], (0, 1, NT - 1))   # the same, folded
+    plan = E.eq45_plan(NF, NT, DT, keep)
     coeffs = E.eq45_coeffs_interp(_S_evo, plan.f, plan.t_max, plan.P, k=plan.k_used)
     assert np.array_equal(coeffs, tri.chebyshev_coefficients(_S_evo))
     diag, up = E.eq45_blocks_cheb(plan, coeffs)
@@ -161,27 +164,30 @@ def test_interp_blocks_and_direct_sum_are_the_oracle_to_rounding(keep):
     # Hz window (v0.12.0): measured 4.6e-17 over every block (4.2e-16 of a diagonal block's own
     # max; the upper blocks 2.1e-18, up to 1.2e-14 of their own max, block m = 4)
     assert _block_gap((diag, up), (d0, u0)) <= ORACLE_RTOL
+    o = plan.pos[BAND]
+    o = o[o >= 0]                                      # the kept cells' rows, in band order
     # Hz window (v0.12.0): measured 9.1e-17
-    assert _gap(E.eq45_direct(plan, _S_evo), tri.dense(_S_evo)) <= ORACLE_RTOL
+    assert _gap(E.eq45_direct(plan, _S_evo)[np.ix_(o, o)], tri.dense(_S_evo)) <= ORACLE_RTOL
 
 
 def test_eq45_is_exact_for_stationary_noise():
     plan = E.eq45_plan(NF, NT, DT)
     Sst = lambda f, t: _S_stat(f) + 0 * t  # noqa: E731
-    ref = build_C_wdm(NF, NT, DT, _S_stat)[np.ix_(BAND, BAND)]
+    ref = build_C_wdm(NF, NT, DT, _S_stat)
     assert np.max(np.abs(E.eq45_direct(plan, Sst) - ref)) <= 1e-12 * np.abs(ref).max()
     coeffs = E.eq45_coeffs_interp(Sst, plan.f, plan.t_max, plan.P, k=plan.k_used)
     C = _assemble(plan, *E.eq45_blocks_cheb(plan, coeffs))
     assert np.max(np.abs(C - ref)) <= 1e-11 * np.abs(ref).max()
 
 
-def test_eq45_plan_rejects_a_non_boolean_keep():
-    # np.arange(N) has the mask's shape: cast to bool it would silently drop cell 0
-    N = NF * NT
-    for keep in (np.arange(N), np.ones(N, int), list(range(N))):
+def test_eq45_plan_rejects_a_non_boolean_keep_or_a_band_order_one():
+    # np.arange(N) in the mask's shape: cast to bool it would silently drop cell 0
+    for keep in (np.arange(N).reshape(NF, NT), np.ones((NF, NT), int), [list(range(NT))] * NF):
         with pytest.raises(TypeError, match="boolean mask"):
             E.eq45_plan(NF, NT, DT, keep)
-    assert E.eq45_plan(NF, NT, DT, [True] * N).cells.size == N
+    with pytest.raises(ValueError, match=r"^keep must have shape \(Nf, Nt\) = \(8, 16\), got \(128,\)$"):
+        E.eq45_plan(NF, NT, DT, np.ones(N, bool))
+    assert E.eq45_plan(NF, NT, DT, [[True] * NT] * NF).cells.size == N
 
 
 @pytest.mark.parametrize("Nt_grid, keep_cols, r", [(NT, None, 1.0), (NT + 4, None, (NT + 4) / NT)])
diff --git a/tests/wilson/test_noise_stationary.py b/tests/wilson/test_noise_stationary.py
index b56eeaf..0a7fc2d 100644
--- a/tests/wilson/test_noise_stationary.py
+++ b/tests/wilson/test_noise_stationary.py
@@ -68,7 +68,7 @@ def test_colored_noise_and_empirical_variance_are_the_oracle_to_rounding():
     b = onoise._generate_colored_noise(Nf * Nt, dt, _psd, np.random.default_rng(7))
     assert np.array_equal(a, b)
     got = testing.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3)
-    ref = onoise.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3)
+    ref = onoise.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3)[cells(Nf, Nt)]
     assert got.dtype == ref.dtype
     assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 2.1e-16
 
diff --git a/tests/wilson/test_window_threading.py b/tests/wilson/test_window_threading.py
index 2de0f5c..823e6b3 100644
--- a/tests/wilson/test_window_threading.py
+++ b/tests/wilson/test_window_threading.py
@@ -34,7 +34,7 @@ MEYER_HELPERS = frozenset({"phi_centered"})
 #: Every function that samples the window, directly or through another consumer.
 CONSUMERS = frozenset({
     "_gather_bands", "forward", "forward_time", "inverse", "inverse_time", "_forward",
-    "_forward_time", "_inverse", "_inverse_time", "base_atom_spectra", "atom_spectrum",
+    "_inverse", "_inverse_time", "base_atom_spectra", "atom_spectrum",
     "build_W", "build_C_wdm",
     "_slab_K", "build_lambda_blocks_full_active", "build_K_full_active", "psd_from_K",
     "apply_lambda_from_psd", "psd_from_lambda_blocks",
@@ -428,12 +428,9 @@ def _read_only(a):
 
 @lru_cache(maxsize=None)
 def _dense(window, Nf, Nt, dt):
-    """``(W, C, W_a, C_a)``: ``build_W`` and ``build_C_wdm`` of ``psd`` with ``window``, and the
-    same with their rows (and columns) in band order, ``numpy.nonzero`` of the padded mask
-    (read-only)."""
-    W, C = M.build_W(Nf, Nt, dt, window=window), M.build_C_wdm(Nf, Nt, dt, psd, window=window)
-    a = np.argsort(np.ravel_multi_index(wilson.grid.cells(Nf, Nt), (Nf + 1, Nt)), axis=None)
-    return tuple(_read_only(x) for x in (W, C, W[a], C[np.ix_(a, a)]))
+    """``(W, C)``: ``build_W`` and ``build_C_wdm`` of ``psd`` with ``window`` (read-only)."""
+    return (_read_only(M.build_W(Nf, Nt, dt, window=window)),
+            _read_only(M.build_C_wdm(Nf, Nt, dt, psd, window=window)))
 
 
 def _blocks_of(C, Nf, Nt):
@@ -525,8 +522,7 @@ def test_empirical_variance_is_the_mean_square_of_the_same_draws(window, Nf, Nt,
     rng = np.random.default_rng(5)
     draws = [W @ np.fft.irfft(colored_noise(N, dt, psd, rng), n=N) for _ in range(3)]
     want = np.mean(np.square(draws), axis=0).reshape(Nf, Nt)
-    m, n = wilson.grid.cells(Nf, Nt)
-    _close(empirical_variance(Nf, Nt, dt, psd, 3, seed=5, window=window)[m, n], want, RTOL_KERNEL,
+    _close(empirical_variance(Nf, Nt, dt, psd, 3, seed=5, window=window), want, RTOL_KERNEL,
            "empirical_variance")
 
 
@@ -534,29 +530,30 @@ def test_empirical_variance_is_the_mean_square_of_the_same_draws(window, Nf, Nt,
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_time_masks_are_the_dense_mask(window, Nf, Nt, dt):
     """``energy_fraction`` is ``diag(M)`` and ``apply_time_mask`` applies ``M`` (to a stack of
-    rows and to one 1-D row), ``M = W_a diag(w2) W_a^T / dt``."""
+    grids and to one grid), ``M = W diag(w2) W^T / dt``."""
     N = Nf * Nt
     rng = np.random.default_rng(N + Nf + 4)
-    Wa = _dense(window, Nf, Nt, dt)[2]
+    W = _dense(window, Nf, Nt, dt)[0]
     w2 = rng.uniform(0.0, 1.0, N)
-    Mm = Wa @ (w2[:, None] * Wa.T) / dt
-    Y = rng.standard_normal((2, N))
-    _close(E.energy_fraction(w2, Nf, Nt, dt, window=window), np.diag(Mm), RTOL_KERNEL,
-           "energy_fraction")
-    _close(E.apply_time_mask(Y, w2, Nf, Nt, dt, window=window), Y @ Mm.T, RTOL_KERNEL,
-           "apply_time_mask")
-    _close(E.apply_time_mask(Y[0], w2, Nf, Nt, dt, window=window), (Mm @ Y[0])[None, :],
-           RTOL_KERNEL, "apply_time_mask(1-D)")
-
-
-def _wrap_reference(W_a, Nf, Nt):
-    """The definition of ``wrap_fractions`` on the band-order rows ``W_a``: for the active cell
-    ``(m, n)``, centred on sample ``c = n Nf``, the share of ``sum_t g_mn(t)^2`` at samples
-    whose unwrapped position ``c + delta`` (``delta`` the signed offset from the centre, in
-    ``[-N/2, N/2)``) lies outside ``[0, N)``; ``nan`` at inactive cells."""
+    Mm = W @ (w2[:, None] * W.T) / dt
+    Y = rng.standard_normal((2, Nf, Nt))
+    _close(E.energy_fraction(w2, Nf, Nt, dt, window=window), np.diag(Mm).reshape(Nf, Nt),
+           RTOL_KERNEL, "energy_fraction")
+    _close(E.apply_time_mask(Y, w2, Nf, Nt, dt, window=window),
+           (Y.reshape(2, N) @ Mm.T).reshape(Y.shape), RTOL_KERNEL, "apply_time_mask")
+    _close(E.apply_time_mask(Y[0], w2, Nf, Nt, dt, window=window),
+           (Mm @ Y[0].ravel()).reshape(Nf, Nt), RTOL_KERNEL, "apply_time_mask(one grid)")
+
+
+def _wrap_reference(W, Nf, Nt):
+    """The definition of ``wrap_fractions`` on the folded rows ``W``, as band rows ``(Nf + 1,
+    Nt)``: for the cell ``(m, n)``, centred on sample ``c = n Nf``, the share of
+    ``sum_t g_mn(t)^2`` at samples whose unwrapped position ``c + delta`` (``delta`` the signed
+    offset from the centre, in ``[-N/2, N/2)``) lies outside ``[0, N)``; ``nan`` at inactive
+    cells."""
     N = Nf * Nt
-    m, n = np.nonzero(wilson.grid.active_mask(Nf, Nt))
-    g2 = W_a ** 2
+    m, n = (a.ravel() for a in wilson.grid.cells(Nf, Nt))
+    g2 = W ** 2
     c = n[:, None] * Nf
     pos = c + (np.arange(N)[None, :] - c + N // 2) % N - N // 2
     out = np.full((Nf + 1, Nt), np.nan)
@@ -577,12 +574,11 @@ def _cut_reference(frac, tol):
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_wrap_rule_is_its_definition_on_the_rows_of_W(window, Nf, Nt, dt):
-    """``wrap_fractions`` (all bands and a subset) is its definition on the rows of
-    ``build_W``, and ``wrap_cut_columns`` is the cut rule on those fractions."""
-    want = _wrap_reference(_dense(window, Nf, Nt, dt)[2], Nf, Nt)
-    got = E.wrap_fractions(Nf, Nt, dt, window=window)
-    assert np.array_equal(np.isnan(got), np.isnan(want))
-    _close(np.nan_to_num(got, nan=-1.0), np.nan_to_num(want, nan=-1.0), RTOL_KERNEL, "wrap_fractions")
+    """``wrap_fractions`` (folded, and band rows for a subset) is its definition on the rows of
+    ``build_W``, and ``wrap_cut_columns`` is the cut rule on the band rows."""
+    want = _wrap_reference(_dense(window, Nf, Nt, dt)[0], Nf, Nt)
+    _close(E.wrap_fractions(Nf, Nt, dt, window=window), want[wilson.grid.cells(Nf, Nt)], RTOL_KERNEL,
+           "wrap_fractions")
     bands = np.array([0, Nf // 2, Nf])
     _close(np.nan_to_num(E.wrap_fractions(Nf, Nt, dt, bands=bands, window=window), nan=-1.0),
            np.nan_to_num(want[bands], nan=-1.0), RTOL_KERNEL, "wrap_fractions(bands)")
@@ -593,15 +589,15 @@ def test_wrap_rule_is_its_definition_on_the_rows_of_W(window, Nf, Nt, dt):
 @pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
 @pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
 def test_exact_covariance_is_the_dense_congruence(window, Nf, Nt, dt):
-    """``exact_covariance`` is ``W_a Sigma W_a^T`` (``Sigma = time_domain_covariance``), and
+    """``exact_covariance`` is ``W Sigma W^T`` (``Sigma = time_domain_covariance``), and
     ``build_C_wdm`` in the stationary limit."""
     N = Nf * Nt
-    Wa, Ca = _dense(window, Nf, Nt, dt)[2:]
-    _close(E.exact_covariance(Nf, Nt, dt, _stationary, window=window), Ca, RTOL_KERNEL,
+    W, C = _dense(window, Nf, Nt, dt)
+    _close(E.exact_covariance(Nf, Nt, dt, _stationary, window=window), C, RTOL_KERNEL,
            "exact_covariance(stationary)")
     Sn = psd_ft(N * dt)
     _close(E.exact_covariance(Nf, Nt, dt, Sn, window=window),
-           Wa @ E.time_domain_covariance(N, dt, Sn) @ Wa.T, RTOL_KERNEL, "exact_covariance")
+           W @ E.time_domain_covariance(N, dt, Sn) @ W.T, RTOL_KERNEL, "exact_covariance")
 
 
 def _assemble_cheb(plan, diag, up):
@@ -637,12 +633,13 @@ def test_eq45_plan_carries_the_window_to_every_eq45_builder(window, Nf, Nt, dt):
     """``plan.base`` is ``base_atom_spectra(window=)``, and in the stationary limit the plan's
     ``eq45_direct``, ``eq45_blocks_cheb`` and ``eq45_blocks_gram`` are ``build_C_wdm``,
     on all cells and on a ``keep`` subset (the kept rows and columns)."""
-    Ca = _dense(window, Nf, Nt, dt)[3]
+    C = _dense(window, Nf, Nt, dt)[1]
     base = T.base_atom_spectra(Nf, Nt, dt, window=window)
-    keep = ~np.isin(np.nonzero(wilson.grid.active_mask(Nf, Nt))[1], (0, 1, Nt - 1))
-    for tag, plan, ref in (("", E.eq45_plan(Nf, Nt, dt, window=window), Ca),
+    keep = ~np.isin(wilson.grid.cells(Nf, Nt)[1], (0, 1, Nt - 1))
+    k = keep.ravel()
+    for tag, plan, ref in (("", E.eq45_plan(Nf, Nt, dt, window=window), C),
                            ("(keep)", E.eq45_plan(Nf, Nt, dt, keep, window=window),
-                            Ca[np.ix_(keep, keep)])):
+                            C[np.ix_(k, k)])):
         assert sorted(plan.base) == sorted(base)
         assert all(np.array_equal(plan.base[key], G) for key, G in base.items())
         _close(E.eq45_direct(plan, _stationary), ref, RTOL_KERNEL, f"eq45_direct{tag}")
diff --git a/tests/wilson/vendored/test_evolutionary.py b/tests/wilson/vendored/test_evolutionary.py
index 5291d28..ff5dffc 100644
--- a/tests/wilson/vendored/test_evolutionary.py
+++ b/tests/wilson/vendored/test_evolutionary.py
@@ -11,7 +11,7 @@ import pytest
 # The oracle packs the cells in band order; tests/wilson/test_evolutionary_kernels.py checks
 # the kernels against it.
 from wdm_dev_ev import evolutionary as ev
-from domainia.wilson.grid import active_mask, cells, unfold
+from domainia.wilson.grid import active_mask, cells
 from domainia.wilson.matrix import build_W
 from domainia.wilson.transform import forward_time
 
@@ -33,8 +33,7 @@ def test_packed_grid_matches_the_dense_transform(rng):
     np.testing.assert_array_equal(g.mask, active_mask(NF, NT))
     x = rng.standard_normal((3, N))
     np.testing.assert_allclose(np.stack([g.forward(r) for r in x]), x @ Wa.T, atol=1e-12)
-    w = unfold(forward_time(x[0], NF, NT, DT), NF, NT)
-    np.testing.assert_allclose(g.forward(x[0]), w[g.mask], atol=0)
+    np.testing.assert_allclose(g.forward(x[0]), forward_time(x[0], NF, NT, DT).ravel()[BAND], atol=0)
 
 
 def test_atoms_are_the_rows_of_the_transform():
````

- [ ] **Step 2: Run the task's tests: they fail**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error --tb=no -rfE tests/wilson/test_evolutionary_kernels.py tests/wilson/vendored/test_evolutionary.py tests/test_dom_testing.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py tests/wilson/test_noise_stationary.py tests/test_covariance.py 2>&1 | tail -n 40
```

Expected last line (prototype): `127 failed, 355 passed, 2 skipped`. Five kinds of failure:
- the dense oracles still want a padded mask: `ValueError: mask must have shape (Nf + 1, Nt) = (9, 16), got (8, 16)`;
- `eq45_plan` still wants a band-order keep: `keep must be a mask over the 128 packed cells; got (8, 16)`;
- `apply_time_mask` still takes band-order `(B, N)`, which gives a `shape mismatch`;
- the folded comparisons of `energy_fraction`, `wrap_fractions`, `exact_covariance`, `eq45_direct` and `empirical_variance` fail;
- `test_eq45_plan_rejects_a_non_boolean_keep_or_a_band_order_one` reports `DID NOT RAISE`.

- [ ] **Step 3: Apply the implementation patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T5-impl && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists `src/domainia/testing.py`, `src/domainia/wilson/evolutionary.py`, `src/domainia/wilson/transform.py`, `tests/test_covariance.py`, `tests/test_dom_testing.py`, `tests/wilson/meyer_cases.py`, `tests/wilson/test_evolutionary_kernels.py`, `tests/wilson/test_noise_stationary.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py` as modified, and nothing else.

<!-- patch: T5-impl -->
````diff
diff --git a/src/domainia/testing.py b/src/domainia/testing.py
index 6b092f8..1a4421a 100644
--- a/src/domainia/testing.py
+++ b/src/domainia/testing.py
@@ -9,15 +9,15 @@
   and the Monte-Carlo WDM variance of such draws, which checks
   ``wilson.band_averaged_variance``.
 
-The cells are a boolean ``mask`` of shape ``(Nf + 1, Nt)``, the WDM layout of
-``wilson.grid.active_mask``, that selects active cells only.  Rows and columns follow
-``numpy.nonzero(mask)``: band by band, time within a band.
+The cells are a boolean ``mask`` of shape ``(Nf, Nt)``, the folded WDM layout
+(``wilson.grid.cells``).  Rows and columns follow ``numpy.nonzero(mask)``: the folded flat order
+of the selected cells.
 
 :func:`dense_W`, :func:`wdm_congruence`, :func:`eq45_dense` and :func:`empirical_variance` take
 ``window=`` (default Meyer: the WDM transform) and compute in the Wilson basis of that window.
 
 The oracles follow the package's rule: a wrong kind (a ``mask`` that is not boolean) raises
-``TypeError``, a wrong value (a wrong shape, an inactive cell) ``ValueError``.
+``TypeError``, a wrong value (a wrong shape) ``ValueError``.
 
 Test windows (``domainia.wilson.windows``):
 
@@ -47,27 +47,21 @@ __all__ = ["ALT_WINDOW", "colored_noise", "dense_W", "empirical_variance", "eq45
 Array = NDArray[np.float64]
 
 
-def _keep(Nf: int, Nt: int, mask: Any) -> NDArray[np.bool_]:
-    """The cells of ``mask`` as a mask over the ``N`` packed active cells (``eq45_plan``'s
-    ``keep``), after the checks of the module docstring."""
+def _mask(Nf: int, Nt: int, mask: Any) -> NDArray[np.bool_]:
+    """``mask`` (``eq45_plan``'s ``keep``) after the checks of the module docstring."""
     require_dtype("mask", mask, "b", "a boolean cell mask")
+    validate_grid(Nf, Nt)
     mask = np.asarray(mask)
-    active = wilson.grid.active_mask(Nf, Nt)
-    if mask.shape != active.shape:
-        raise ValueError(f"mask must have shape (Nf + 1, Nt) = {active.shape}, got {mask.shape}")
-    bad = np.argwhere(mask & ~active)
-    if bad.size:
-        raise ValueError(f"mask selects the inactive cell (m={bad[0, 0]}, n={bad[0, 1]})")
-    return mask[active]
+    if mask.shape != (Nf, Nt):
+        raise ValueError(f"mask must have shape (Nf, Nt) = {(Nf, Nt)}, got {mask.shape}")
+    return mask
 
 
 def dense_W(Nf: int, Nt: int, dt: float, mask: Any, *, window: Window = MEYER) -> Array:
     """``(cells, N)``: row ``p`` is the atom of the ``p``-th cell of ``mask`` sampled in
     time, so ``W x`` are those cells of ``x`` (``wilson.matrix.build_W`` rows,
     ``W W^T = dt I``) in the Wilson basis of ``window``."""
-    keep = _keep(Nf, Nt, mask)
-    band = np.argsort(np.ravel_multi_index(wilson.grid.cells(Nf, Nt), (Nf + 1, Nt)), axis=None)
-    return wilson.matrix.build_W(Nf, Nt, dt, window=window)[band[keep]]
+    return wilson.matrix.build_W(Nf, Nt, dt, window=window)[_mask(Nf, Nt, mask).ravel()]
 
 
 def wdm_congruence(Sigma: Array, Nf: int, Nt: int, dt: float, mask: Any, *,
@@ -82,7 +76,7 @@ def eq45_dense(Nf: int, Nt: int, dt: float, mask: Any, Sn: Callable[[Any, Any],
                P: int = 24, t_max: float | None = None, window: Window = MEYER) -> Array:
     """Eq. (45) on the cells of ``mask`` by the direct sum, ``Sn(f, t)`` elementwise (``A_n``
     inside), with the atoms of ``window``."""
-    keep = _keep(Nf, Nt, mask)
+    keep = _mask(Nf, Nt, mask)
     require_callable("Sn", Sn)
     plan = eq45_plan(Nf, Nt, dt, keep, P=P, t_max=t_max, window=window)
     return eq45_direct(plan, Sn)
@@ -147,7 +141,7 @@ def empirical_variance(
     *,
     window: Window = MEYER,
 ) -> Array:
-    """Monte-Carlo estimate of $\\langle w_{mn}^2 \\rangle$, shape ``(Nf + 1, Nt)``: the mean
+    """Monte-Carlo estimate of $\\langle w_{mn}^2 \\rangle$, folded ``(Nf, Nt)``: the mean
     squared WDM coefficients of ``n_realizations`` draws of :func:`colored_noise` from
     ``numpy.random.default_rng(seed)``.  It checks
     :func:`domainia.wilson.band_averaged_variance`."""
@@ -162,7 +156,7 @@ def empirical_variance(
         x_rfft = colored_noise(N, dt, Sn_func, rng)
         w = _forward(x_rfft, Nf, Nt, dt, window=window)
         accum += w * w
-    return accum / n_realizations
+    return accum[wilson.grid.cells(Nf, Nt)] / n_realizations
 
 
 def smoothstep(x: Any) -> Array:
diff --git a/src/domainia/wilson/evolutionary.py b/src/domainia/wilson/evolutionary.py
index d2e0b2e..db87d2a 100644
--- a/src/domainia/wilson/evolutionary.py
+++ b/src/domainia/wilson/evolutionary.py
@@ -31,6 +31,9 @@ the plan's atoms (``plan.base``) carry the window to every builder downstream.
 PSDs are callables ``Sn(f, t)``: elementwise, broadcasting, argument order
 ``(f, t)`` (WDM Paper II writes :math:`S(t, f)`), samples through
 :func:`domainia.wilson.grid.sanitize_psd`.  Cell times are :math:`t_n = n\Delta T`.
+
+WDM arrays are folded ``(..., Nf, Nt)`` and dense matrices index the cells in folded flat order
+(:func:`domainia.wilson.grid.cells`); the Eq. (45) blocks are per band, time ascending.
 """
 
 from __future__ import annotations
@@ -47,6 +50,7 @@ from ._checks import require_dtype, require_real
 from .grid import (
     _require_int,
     active_mask,
+    cells,
     delta_T,
     endpoint_weights,
     validate_dt,
@@ -54,10 +58,11 @@ from .grid import (
 )
 from .noise import _psd_at
 from .transform import (
-    _forward_time,
     _inverse_time,
     atom_spectrum,
     base_atom_spectra,
+    forward_time,
+    inverse_time,
 )
 from .windows import MEYER, Window
 
@@ -98,20 +103,20 @@ __all__ = [
 
 
 def energy_fraction(w2: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
-    """``rho_p = sum_t w2(t) g_p(t)^2 / dt`` for every active cell ``p`` (band order), from the
+    """``rho_p = sum_t w2(t) g_p(t)^2 / dt`` for every cell ``p``, folded ``(Nf, Nt)``, from the
     time-domain weight ``w2`` of shape ``(N,)``: one FFT correlation per base atom.
     """
     validate_grid(Nf, Nt)
     validate_dt(dt)
     N = Nf * Nt
-    m_arr, n_arr = np.nonzero(active_mask(Nf, Nt))
+    m_arr, n_arr = cells(Nf, Nt)
     base = base_atom_spectra(Nf, Nt, dt, window=window)
-    rho = np.empty(m_arr.size)
+    rho = np.empty((Nf, Nt))
     W2 = np.fft.rfft(w2)
     for (m, a), G in base.items():
         g2 = np.fft.irfft(G, n=N) ** 2
         corr = np.fft.irfft(W2 * np.conj(np.fft.rfft(g2)), n=N)  # sum_t w2[t+s] g2[t]
-        sel = np.flatnonzero((m_arr == m) & (n_arr % 2 == a))
+        sel = (m_arr == m) & (n_arr % 2 == a)
         shifts = (n_arr[sel] - a) * Nf
         rho[sel] = corr[shifts % N] / dt
     return rho
@@ -119,19 +124,11 @@ def energy_fraction(w2: Array, Nf: int, Nt: int, dt: float, *, window: Window =
 
 def apply_time_mask(Y: Array, chi: Array, Nf: int, Nt: int, dt: float, *,
                     window: Window = MEYER) -> Array:
-    """``M y`` for each packed row ``y`` of ``Y``, ``M = W chi W^T / dt``: back to time,
-    mask, forward again.
-
-    ``Y`` is ``(N,)`` or ``(B, N)``: packed rows (active cells, band order).  ``chi`` is the
-    time-domain mask, ``(N,)``.
+    """``M y`` for each folded grid ``y`` of ``Y`` (``(..., Nf, Nt)``), ``M = W chi W^T / dt``:
+    back to time, mask, forward again.  ``chi`` is the time-domain mask, ``(N,)``.
     """
-    validate_grid(Nf, Nt)
-    mask = active_mask(Nf, Nt)
-    Y = np.atleast_2d(Y)
-    full = np.zeros((Y.shape[0], Nf + 1, Nt))
-    full[:, mask] = Y
-    X = _inverse_time(full, Nf, Nt, dt, window=window) * chi
-    return _forward_time(X, Nf, Nt, dt, window=window)[:, mask]
+    X = inverse_time(Y, Nf, Nt, dt, window=window) * chi
+    return forward_time(X, Nf, Nt, dt, window=window)
 
 
 # ---------------------------------------------------------------------------
@@ -146,8 +143,8 @@ def wrap_fractions(Nf: int, Nt: int, dt: float, *, bands: Array | None = None,
     For the cell :math:`(m, n)`, centred on sample :math:`n N_f`, the fraction of
     :math:`\sum_j g_{mn}(t_j)^2` at samples whose unwrapped position (measured from
     the centre along the shorter way round) lies outside :math:`[0, N)`.
-    Returns ``(Nf + 1, Nt)`` (or ``(len(bands), Nt)`` for the bands ``bands``), ``nan`` at
-    inactive cells.
+    Returns the folded ``(Nf, Nt)``, or for the bands ``bands`` their rows ``(len(bands), Nt)``,
+    ``nan`` at an edge band's inactive (odd) columns.
     """
     validate_grid(Nf, Nt)
     N = Nf * Nt
@@ -171,7 +168,7 @@ def wrap_fractions(Nf: int, Nt: int, dt: float, *, bands: Array | None = None,
             left = cum[np.searchsorted(ds, -c, side="left")]            # delta < -c
             right = total - cum[np.searchsorted(ds, N - c, side="left")]  # delta >= N - c
             out[row, cols] = (left + right) / total
-    return out
+    return out[cells(Nf, Nt)] if bands is None else out
 
 
 def wrap_cut_columns(Nf: int, Nt: int, dt: float, tol: float = 1e-3, *,
@@ -189,8 +186,9 @@ def wrap_cut_columns(Nf: int, Nt: int, dt: float, tol: float = 1e-3, *,
     >>> wilson.evolutionary.wrap_cut_columns(64, 256, 2**-10)
     (6, 5)
     """
-    # worst band per column
-    frac = np.nanmax(wrap_fractions(Nf, Nt, dt, bands=bands, window=window), axis=0)
+    # worst band per column, on band rows (a folded column mixes times n and n - 1)
+    rows = np.arange(Nf + 1) if bands is None else bands
+    frac = np.nanmax(wrap_fractions(Nf, Nt, dt, bands=rows, window=window), axis=0)
     bad = np.flatnonzero(frac > tol)
     half = Nt // 2
     lead = int(bad[bad < half].max()) + 1 if np.any(bad < half) else 0
@@ -243,15 +241,14 @@ def draw_noise(N: int, dt: float, Sn: PsdFtFunc, rng: np.random.Generator) -> Ar
 
 def exact_covariance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc, *,
                      window: Window = MEYER) -> Array:
-    r"""Exact :math:`\Lambda = W_a \Sigma W_a^{\mathsf T}` on the active cells (band order),
-    built in Gram form :math:`(W_a G)(W_a G)^{\mathsf T}`, so PSD by construction.
+    r"""Exact :math:`\Lambda = W \Sigma W^{\mathsf T}`, ``(N, N)`` in folded flat order,
+    built in Gram form :math:`(W G)(W G)^{\mathsf T}`, so PSD by construction.
     ``Sn(f, t)`` is elementwise and broadcasting.
     """
     validate_grid(Nf, Nt)
     N = Nf * Nt
     G = time_domain_factor(N, dt, Sn)
-    A = _forward_time(np.ascontiguousarray(G.T), Nf, Nt, dt,
-                      window=window)[:, active_mask(Nf, Nt)].T
+    A = forward_time(np.ascontiguousarray(G.T), Nf, Nt, dt, window=window).reshape(-1, N).T
     Lam = A @ A.T
     return 0.5 * (Lam + Lam.T)
 
@@ -265,7 +262,7 @@ def exact_covariance(Nf: int, Nt: int, dt: float, Sn: PsdFtFunc, *,
 class Eq45BlockSpec:
     """One band block ``(m, m2)``, ``m2 in {m, m+1}``, of an :class:`Eq45Plan`.
 
-    ``R``/``C`` (row/column packed cells), ``combo = 2a + b`` (base-atom parities),
+    ``R``/``C`` (row/column cells, folded flat indices), ``combo = 2a + b`` (base-atom parities),
     ``d`` (circular column lag), ``s = n + n'`` (mid-time index; ``t_bar = s dT/2``)
     and ``B = {combo: (k, eta_k G_ma(k) G*_m2b(k) / T)}`` over the support bins.
     """
@@ -284,8 +281,9 @@ class Eq45BlockSpec:
 class Eq45Plan:
     """Constants of Eq. (45) on the kept cells of one grid.
 
-    ``specs`` holds one :class:`Eq45BlockSpec` per band block ``(m, m2)``,
-    ``m2 in {m, m+1}``.
+    The per-cell arrays ``m``, ``n``, ``t`` and ``pos`` (row in the kept-cell matrix, ``-1`` if
+    not kept) are in folded flat order, and ``cells`` lists the kept ones.  ``specs`` holds one
+    :class:`Eq45BlockSpec` per band block ``(m, m2)``, ``m2 in {m, m+1}``.
     """
 
     Nf: int
@@ -335,9 +333,9 @@ def _block_spec(Nf: int, Nt: int, T: float, n: IntArray, eta: Array,
 def eq45_plan(Nf: int, Nt: int, dt: float, keep: BoolArray | None = None, *,
               P: int = 24, t_max: float | None = None,
               window: Window = MEYER) -> Eq45Plan:
-    """Static plan of Eq. (45) on the cells ``keep`` (a boolean mask over the ``N`` packed
-    cells, band order; all by default), Chebyshev order ``P`` in time on
-    ``t_bar in [0, t_max]`` (default: this grid's ``T = N dt``).
+    """Static plan of Eq. (45) on the cells ``keep`` (a boolean folded ``(Nf, Nt)`` mask; all
+    by default), Chebyshev order ``P`` in time on ``t_bar in [0, t_max]`` (default: this grid's
+    ``T = N dt``).  The kept cells are in folded flat order.
 
     Pair mid-times of the kept cells must lie in ``[0, t_max]``.  A padded grid
     passes its extended ``t_max``; the spectrum's time dependence is then
@@ -364,28 +362,27 @@ def eq45_plan(Nf: int, Nt: int, dt: float, keep: BoolArray | None = None, *,
     Nf, Nt, dt = int(Nf), int(Nt), float(dt)
     N = Nf * Nt
     T = N * dt
-    mask = active_mask(Nf, Nt)
-    m, n = np.nonzero(mask)                                       # band-major
+    m, n = (a.ravel() for a in cells(Nf, Nt))                     # folded flat order
     t = n * delta_T(Nf, dt)                                       # cell times
     K = N // 2 + 1
     f = np.fft.rfftfreq(N, d=dt)
     eta = np.ones(K)
     eta[[0, -1]] = 0.5
     base = base_atom_spectra(Nf, Nt, dt, window=window)
-    keep = np.ones(m.size, bool) if keep is None else np.asarray(keep)
+    keep = np.ones((Nf, Nt), bool) if keep is None else np.asarray(keep)
     # An index array (e.g. np.arange(N)) would silently drop cell 0.
     require_dtype("keep", keep, "b", "a boolean mask over the packed cells")
-    if keep.shape != (m.size,):
-        raise ValueError(f"keep must be a mask over the {m.size} packed cells; got {keep.shape}")
+    if keep.shape != (Nf, Nt):
+        raise ValueError(f"keep must have shape (Nf, Nt) = {(Nf, Nt)}, got {keep.shape}")
     t_max = T if t_max is None else require_real("t_max", t_max)
-    cells = np.flatnonzero(keep)
+    kept = np.flatnonzero(keep)
     pos = -np.ones(m.size, int)
-    pos[cells] = np.arange(cells.size)
-    band_cells = [cells[m[cells] == mm] for mm in range(Nf + 1)]
+    pos[kept] = np.arange(kept.size)
+    band_cells = [kept[m[kept] == mm] for mm in range(Nf + 1)]
     # Chebyshev basis on the pair mid-times t_bar = s dT / 2, s = n + n'
     s_grid = np.arange(2 * Nt - 1)
     u = 2.0 * (s_grid * 0.5 * delta_T(Nf, dt) / t_max - 0.5)
-    used = s_grid <= 2 * int(n[cells].max())
+    used = s_grid <= 2 * int(n[kept].max())
     if np.any(np.abs(u[used]) > 1 + 1e-12):
         raise ValueError("pair mid-times must lie inside [0, t_max]")
     Tp = np.empty((P, s_grid.size))
@@ -398,7 +395,7 @@ def eq45_plan(Nf: int, Nt: int, dt: float, keep: BoolArray | None = None, *,
              for mm in range(Nf + 1) for m2 in (mm, mm + 1) if m2 <= Nf]
     k_all = [k for spec in specs for (k, _) in spec.B.values()]
     k_used = np.unique(np.concatenate(k_all)) if k_all else np.zeros(0, int)
-    return Eq45Plan(Nf, Nt, dt, N, T, K, P, t_max, f, eta, m, n, t, base, keep, cells, pos,
+    return Eq45Plan(Nf, Nt, dt, N, T, K, P, t_max, f, eta, m, n, t, base, keep, kept, pos,
                     band_cells, Tp, specs, k_used)
 
 
diff --git a/src/domainia/wilson/transform.py b/src/domainia/wilson/transform.py
index bb0c6e0..d7f0c22 100644
--- a/src/domainia/wilson/transform.py
+++ b/src/domainia/wilson/transform.py
@@ -389,25 +389,6 @@ def inverse(
     return _inverse(unfold(wmn, Nf, Nt), Nf, Nt, dt, window=window)
 
 
-def _forward_time(x: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
-    """:func:`forward_time` on the padded grid: ``(..., N)`` to ``(..., Nf+1, Nt)``."""
-    validate_grid(Nf, Nt)
-    x = np.asarray(x)
-    expected = n_samples(Nf, Nt)
-    if x.shape[-1:] != (expected,):
-        raise ValueError(
-            f"x must have shape (..., {expected}) = (..., Nf*Nt) for Nf={Nf}, Nt={Nt}; "
-            f"got {x.shape}"
-        )
-    # Check here rather than only on the rfft: reporting "index 5 of x" beats
-    # reporting that every bin of the transformed spectrum went non-finite.
-    validate_finite(x, "forward_time: x")
-    # At least double before the rfft, which keeps a lower-precision float's precision.
-    # A complex x stays complex, and the rfft refuses it.
-    x_rfft = np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False))
-    return _forward(x_rfft, Nf, Nt, dt, window=window)
-
-
 def forward_time(
     x: Array,
     Nf: int,
@@ -440,8 +421,21 @@ def forward_time(
     Traceback (most recent call last):
     ValueError: forward_time: x: 1 of 128 entries are not finite...
     """
-    m, n = cells(Nf, Nt)
-    return _forward_time(x, Nf, Nt, dt, window=window)[..., m, n]
+    validate_grid(Nf, Nt)
+    x = np.asarray(x)
+    expected = n_samples(Nf, Nt)
+    if x.shape[-1:] != (expected,):
+        raise ValueError(
+            f"x must have shape (..., {expected}) = (..., Nf*Nt) for Nf={Nf}, Nt={Nt}; "
+            f"got {x.shape}"
+        )
+    # Check here rather than only on the rfft: reporting "index 5 of x" beats
+    # reporting that every bin of the transformed spectrum went non-finite.
+    validate_finite(x, "forward_time: x")
+    # At least double before the rfft, which keeps a lower-precision float's precision.
+    # A complex x stays complex, and the rfft refuses it.
+    x_rfft = np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False))
+    return forward(x_rfft, Nf, Nt, dt, window=window)
 
 
 def _inverse_time(wmn: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> Array:
diff --git a/tests/wilson/meyer_cases.py b/tests/wilson/meyer_cases.py
index 0d2e84f..bf395ce 100644
--- a/tests/wilson/meyer_cases.py
+++ b/tests/wilson/meyer_cases.py
@@ -54,18 +54,16 @@ FOLDED = ("wilson::transform.forward", "wilson::transform.forward_many",
           "wilson::stationary.apply_lambda_from_psd", "wilson::noise.band_averaged_variance",
           "wilson::noise.evolutionary_band_variance", "wilson::noise.empirical_variance",
           "wilson::evolutionary.wrap_fractions")
-#: Dense families recorded with padded ``(Nf+1) Nt`` rows (and columns, ``build_C_wdm``), now
-#: in folded flat order.
-FOLDED_ROWS = ("wilson::matrix.build_W", "wilson::matrix.build_C_wdm")
 #: Families recorded over the active cells in band order (``numpy.nonzero`` of the padded mask),
 #: now ``(..., Nf, Nt)`` folded.
 BAND_ORDER = ("wilson::evolutionary.energy_fraction", "wilson::evolutionary.apply_time_mask",
               "domainia::BandAveragedVariance")
-#: Dense families recorded with band-order rows (``dense_W``) or rows and columns, now in folded
-#: flat order.  Every recorded case selects every cell, so one permutation serves them all.
-BAND_ROWS = ("domainia::testing.dense_W", "domainia::testing.wdm_congruence",
-             "domainia::testing.eq45_dense", "domainia::EvolutionaryCovariance.direct",
-             "wilson::evolutionary.exact_covariance", "wilson::evolutionary.eq45_direct")
+#: Dense families, now in folded flat order: rows (the ``_W`` ones) or rows and columns, recorded
+#: padded (``wilson::matrix.``) or in band order.  Every recorded case selects every cell.
+DENSE = ("wilson::matrix.build_W", "wilson::matrix.build_C_wdm", "domainia::testing.dense_W",
+         "domainia::testing.wdm_congruence", "domainia::testing.eq45_dense",
+         "domainia::EvolutionaryCovariance.direct", "wilson::evolutionary.exact_covariance",
+         "wilson::evolutionary.eq45_direct")
 
 Case = tuple[str, Callable[[dict], Any]]
 
@@ -78,17 +76,16 @@ def _family(key: str) -> str:
 def load_baseline() -> dict[str, np.ndarray]:
     """The recorded baseline as ``{key: array}``.  The families whose kernels now return the
     folded WDM layout are converted here, never in :func:`mismatches` (folded WDM spec,
-    section 5): the :data:`FOLDED` ones are gathered into the folded slots, without
-    ``wilson.grid.fold``'s check of the inactive cells, and the others are permuted into folded
-    order: :data:`BAND_ORDER`'s last axis, the rows of ``build_W`` and ``dense_W``, and the rows
-    and columns of the other :data:`FOLDED_ROWS` and :data:`BAND_ROWS` matrices."""
+    section 5): :data:`FOLDED` is gathered into the folded slots, without ``wilson.grid.fold``'s
+    check of the inactive cells, and :data:`BAND_ORDER` and :data:`DENSE` are permuted into
+    folded (flat) order."""
     from domainia.wilson.grid import cells
 
     with np.load(BASELINE) as z:
         out = {k: z[k] for k in z.files}
     for k, v in out.items():
         fam = _family(k)
-        if fam in FOLDED + FOLDED_ROWS + BAND_ORDER + BAND_ROWS:
+        if fam in FOLDED + BAND_ORDER + DENSE:
             Nf, Nt = map(int, k.split("::")[2].split("x"))
             m, n = cells(Nf, Nt)
             p = (m * Nt + n).ravel()                # each folded slot's padded flat index
@@ -98,7 +95,7 @@ def load_baseline() -> dict[str, np.ndarray]:
             elif fam in BAND_ORDER:
                 out[k] = v[..., b].reshape(v.shape[:-1] + (Nf, Nt))
             else:
-                i = p if fam in FOLDED_ROWS else b
+                i = p if fam.startswith("wilson::matrix.") else b
                 out[k] = v[i] if fam.endswith("_W") else v[np.ix_(i, i)]
     return out
 
````

- [ ] **Step 4: Run the task's tests: they pass**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/wilson/test_evolutionary_kernels.py tests/wilson/vendored/test_evolutionary.py tests/test_dom_testing.py tests/wilson/test_window_threading.py tests/wilson/test_meyer_baseline.py tests/wilson/test_noise_stationary.py tests/test_covariance.py tests/wilson/test_wrong_kinds.py src/domainia/wilson/evolutionary.py src/domainia/wilson/transform.py 2>&1 | tail -1
```

Expected (prototype): `583 passed, 2 skipped`.

- [ ] **Step 5: Run the lanes, the replay and the size**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py
```

Expected (prototype): `943 passed, 3 skipped` in both lanes, with no warnings and no collection errors; `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`; `src 4084 lines / 1852 code; tests 4354 lines (+ oracle 4135)`.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/testing.py src/domainia/wilson/evolutionary.py src/domainia/wilson/transform.py tests/test_covariance.py tests/test_dom_testing.py tests/wilson/meyer_cases.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_noise_stationary.py tests/wilson/test_window_threading.py tests/wilson/vendored/test_evolutionary.py && git commit -q -F - <<'EOF'
Fold the evolutionary kernels' and the testing oracles' WDM arrays and cell order

energy_fraction and testing.empirical_variance return (Nf, Nt) folded, apply_time_mask
takes and returns (..., Nf, Nt) through the public inverse_time and forward_time (so a
padded or flat input is refused with the named fix), and wrap_fractions returns the folded
grid with no NaN; with bands= it keeps band rows, which wrap_cut_columns now always asks
for. exact_covariance and eq45_direct are in folded flat order, and eq45_plan takes a
boolean (Nf, Nt) keep, its per-cell arrays in folded flat order; the band blocks are
unchanged. testing.dense_W, wdm_congruence and eq45_dense take a folded mask: _keep and its
inactive-cell refusal go, the kind and shape checks stay. The replay's two dense tuples
merge into one. With apply_time_mask and exact_covariance on the public kernels, nothing
but forward_time calls the padded twin _forward_time, so it goes: forward_time is again
the rfft followed by forward.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
```


### Task 6: `wdm_covariance` folds; the README describes the layout (spec §4 `wdm_covariance` and README; §8 step 6)

**Goal:** `wdm_covariance` returns `(K, K, Nf, Nt)` folded, with no zeroed cells. The README's Covariance example, Conventions and Kernels sections describe the folded layout.

**Files:**
- Modify: `src/domainia/_covariance.py`, `README.md`
- Test: `tests/test_covariance.py`

**What the patches do:**
- **The chunk loop is unchanged:** same model calls, same chunk shapes. It fills the padded `(K, K, Nf+1, Nt)` grid, then returns `C[:, :, m, n]` with `m, n = cells(Nf, Nt)`.
  - This is a raw gather, because the loop computes the inactive cells too.
  - The final zeroing goes.
- **The docstring** states `C[a, b, i, j] = ½ BA_m[Re H_ab(·, t_n)]` with `(m, n) = cells(Nf, Nt)[:, i, j]`. "Folded at DC and Nyquist", the frequency reflection, becomes "reflected", because "folded" now names the layout.
- **README:**
  - The Covariance example is `(2, 2, 16, 64)` without `not D[:, :, 0, 1].any()`.
  - The Conventions WDM bullet describes the folded slots, `.reshape(..., N)`, `wilson.grid.cells`, and `fold`/`unfold` for padded arrays.
  - The Kernels section names `cells`, `fold` and `unfold`.
- **Tests:**
  - `ROUTES["wdm"]` drops `[..., MASK]`.
  - The inactive-zero checks and Task 4's `fold` bridges go.
  - The evolutionary test pins row 0's odd slots to Nyquist at time j − 1, against the folded `evolutionary_band_variance`.
  - The refusal test that pins the model's chunk shape `(K, K, 17, 1, 25)` is unchanged, because the model contract is.

**Acceptance Criteria:**
- [ ] `wdm_covariance(S, Nf=16, Nt=64, dt=dt)` for K = 2 is `(2, 2, 16, 64)`; the README examples run (`tests/test_readme.py`).
- [ ] The replay prints `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`; no recorded key covers `wdm_covariance`, so the loader is unchanged.
- [ ] Full suite (prototype): PY and FLOOR `943 passed, 3 skipped`, no warnings, no collection errors.
- [ ] Size (prototype): `src 4085 lines / 1852 code; tests 4350 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/test_covariance.py tests/test_readme.py 2>&1 | tail -1` → `8 passed`

**Steps:**

- [ ] **Step 1: Apply the tests patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T6-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists exactly `tests/test_covariance.py` as modified.

<!-- patch: T6-tests -->
````diff
diff --git a/tests/test_covariance.py b/tests/test_covariance.py
index 03a37a4..5bc789e 100644
--- a/tests/test_covariance.py
+++ b/tests/test_covariance.py
@@ -14,13 +14,12 @@ LEVELS = np.array([[1.0, 0.6, 0.3], [0.6, 1.2, -0.4], [0.3, -0.4, 0.9]])
 DELAYS, PHASES = np.array([0.0, 7.3, -11.9]) * DT, np.array([0.0, 0.4, -1.1])
 #: A complex mixing matrix for the loop-built model.
 B = np.random.default_rng(20261114).standard_normal((K, K, 2)) @ [1.0, 1.0j]
-#: The WDM grid of the stationary case, and its active cells.
+#: The WDM grid of the stationary case.
 NF, NT = 16, 32
-MASK = wilson.grid.active_mask(NF, NT)
-#: The routes from a stationary model S(f) to a covariance; the WDM route keeps the active cells.
+#: The routes from a stationary model S(f) to a covariance.
 ROUTES = {"frequency": lambda S, N=128: frequency_covariance(S, N=N, dt=DT),
           "wdm": lambda S: wdm_covariance((lambda f, t: S(f)) if callable(S) else S, Nf=NF,
-                                          Nt=NT, dt=DT)[..., MASK]}
+                                          Nt=NT, dt=DT)}
 #: x = B(t) y for K independent channels y of PSD psd (spec 6.2), with
 #: B(t) = L0 (I + sin(2 pi t / Tp) M1 + cos(4 pi t / Tp) M2).
 L0 = np.linalg.cholesky(LEVELS)
@@ -138,17 +137,16 @@ def test_frequency_covariance_is_the_per_bin_covariance_of_the_dense_reference()
 
 
 def test_wdm_covariance_of_a_stationary_model_is_the_dense_reference():
-    """The same-cell blocks of (I_K x W) Sigma (I_K x W)^T, active edge-band cells included,
-    within 1e-12 (measured: 3.3e-15) in two windows; every block is 0 at the inactive cells; a
-    size-1 time axis gives the explicitly broadcast output, bit for bit."""
+    """The same-cell blocks of (I_K x W) Sigma (I_K x W)^T in folded order, edge-band cells
+    included, within 1e-12 (measured: 3.3e-15) in two windows; a size-1 time axis gives the
+    explicitly broadcast output, bit for bit."""
     Sigma = sigma(delayed, NF * NT, DT)
     for window in (wilson.MEYER, testing.ALT_WINDOW):
         W = testing.dense_W(NF, NT, DT, np.ones((NF, NT), bool), window=window)
         want = np.einsum("pi,abij,pj->abp", W, Sigma, W, optimize=True)
         got = wdm_covariance(lambda f, t: delayed(f), Nf=NF, Nt=NT, dt=DT, window=window)
-        assert got.shape == (K, K, NF + 1, NT) and got.dtype == np.float64
-        assert error(wilson.grid.fold(got, NF, NT).reshape(K, K, -1), want) < 1e-12
-        assert not np.any(got[..., ~MASK])
+        assert got.shape == (K, K, NF, NT) and got.dtype == np.float64
+        assert error(got.reshape(K, K, -1), want) < 1e-12
     full = wdm_covariance(lambda f, t: delayed(f + 0 * t), Nf=NF, Nt=NT, dt=DT, window=window)
     assert np.array_equal(got, full)
 
@@ -157,17 +155,16 @@ def test_wdm_covariance_of_an_evolutionary_model(monkeypatch):
     """x = B(t) y on 16 x 64, outside the columns wrap_cut_columns names: the error is below 5e-2
     at T_p = 32 dT and below 1.5e-2 at 64 dT (measured: 3.5e-2 and 9.3e-3); each diagonal
     entry, also as a K = 1 model, is noise.evolutionary_band_variance to 1e-14 (measured: bit
-    for bit) and 0 at the inactive cells; smaller chunks, with partial last chunks, change no
-    bit and keep every call within CHUNK points."""
+    for bit); smaller chunks, with partial last chunks, change no bit and keep every call
+    within CHUNK points."""
     Nf, Nt = 16, 64
-    mask = wilson.grid.active_mask(Nf, Nt)
     lead, trail = wilson.evolutionary.wrap_cut_columns(Nf, Nt, DT)
     n = wilson.grid.cells(Nf, Nt)[1].ravel()
     kept = (n >= lead) & (n < Nt - trail)
     for Tp, bound in ((32, 5e-2), (64, 1.5e-2)):
         S = modulated(Tp * Nf * DT)
         got = wdm_covariance(S, Nf=Nf, Nt=Nt, dt=DT)
-        assert error(wilson.grid.fold(got, Nf, Nt).reshape(K, K, -1)[..., kept],
+        assert error(got.reshape(K, K, -1)[..., kept],
                      same_cell(Nf, Nt, Tp * Nf * DT)[..., kept]) < bound
         for a in range(K):
             want = wilson.noise.evolutionary_band_variance(
@@ -175,8 +172,7 @@ def test_wdm_covariance_of_an_evolutionary_model(monkeypatch):
             one = wdm_covariance(lambda f, t, S=S, a=a: S(f, t)[a:a + 1, a:a + 1],
                                  Nf=Nf, Nt=Nt, dt=DT)
             for v in (got[a, a], one[0, 0]):
-                np.testing.assert_allclose(wilson.grid.fold(v, Nf, Nt), want, rtol=1e-14, atol=0)
-                assert not np.any(v[~mask])
+                np.testing.assert_allclose(v, want, rtol=1e-14, atol=0)
     points = ([], [])
 
     def counted(f, t, stationary=False):
````

- [ ] **Step 2: Run the task's tests: they fail**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error --tb=no -rfE tests/test_covariance.py tests/test_readme.py 2>&1 | tail -n 40
```

Expected last line (prototype): `3 failed, 5 passed`. The three failures:
- the shape assert `(3, 3, 17, 32) == (3, 3, 16, 32)`;
- a boolean index of 1024 against an axis of 1088;
- in `test_linear_and_exactly_hermitian_and_a_zero_point_passes`, a `RuntimeWarning: invalid value encountered in divide`, which `-W error` turns into a failure. It is 0/0 at the padded output's zeroed cells.

`tests/test_readme.py` passes here, because the README change is in the implementation patch.

- [ ] **Step 3: Apply the implementation patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T6-impl && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists `README.md`, `src/domainia/_covariance.py`, `tests/test_covariance.py` as modified, and nothing else.

<!-- patch: T6-impl -->
````diff
diff --git a/README.md b/README.md
index 5f47bbc..9be0886 100644
--- a/README.md
+++ b/README.md
@@ -51,20 +51,21 @@ def S(f):                                       # one-sided 2 x 2 cross-spectral
     return p * np.array([[np.ones_like(c), c], [np.conj(c), np.ones_like(c)]])
 
 C = frequency_covariance(S, N=N, dt=dt)                                  # (2, 2, 513) complex
-D = wdm_covariance(lambda f, t: S(f), Nf=Nf, Nt=Nt, dt=dt)               # (2, 2, 17, 64) real
+D = wdm_covariance(lambda f, t: S(f), Nf=Nf, Nt=Nt, dt=dt)               # (2, 2, 16, 64) real
 E = wdm_covariance(lambda f, t: S(f) * (1.3 + 0.3 * np.cos(2 * np.pi * t / (N * dt))),
                    Nf=Nf, Nt=Nt, dt=dt)                                  # evolutionary
 assert np.array_equal(C, C.conj().transpose(1, 0, 2))
-assert np.array_equal(D, D.transpose(1, 0, 2, 3)) and not D[:, :, 0, 1].any()
+assert np.array_equal(D, D.transpose(1, 0, 2, 3))
 ```
 
 ## Conventions
 
 - Trailing-axis layouts: time is real `(N,)`; frequency is complex `(N // 2 + 1,)`,
-  `dt * rfft(x)`; WDM is real `(Nf + 1, Nt)` with the edge bands, `Nf` and `Nt` even and
-  `Nf * Nt == N`. The atoms have unit norm (`sum(w**2) == dt * sum(x**2)`), and the inactive
-  WDM cells (bands 0 and `Nf` at odd `n`) are 0 in every output of `transform` and
-  `wdm_covariance`.
+  `dt * rfft(x)`; WDM is real `(Nf, Nt)` folded, `Nf` and `Nt` even and `Nf * Nt == N`: row
+  `i >= 1` is band `i`, and row 0 holds band 0 (DC) at time `j` in even slots `j` and band
+  `Nf` (Nyquist) at time `j - 1` in odd ones. `wilson.grid.cells` gives each slot's band and
+  time, `.reshape(..., N)` is the flat form, and `wilson.grid.fold` and `unfold` convert from
+  and to padded `(Nf + 1, Nt)`. The atoms have unit norm (`sum(w**2) == dt * sum(x**2)`).
 - A model `S(f)` or `S(f, t)` (`t` in seconds from the first sample) returns `(K, K, ...)`,
   or `(K, K, 1)` if it is constant, with `S[b, a] = conj(S[a, b])`, oriented so that
   `E[X_a conj(X_b)] = (N dt / 2) S_ab`.
@@ -79,5 +80,6 @@ assert np.array_equal(D, D.transpose(1, 0, 2, 3)) and not D[:, :, 0, 1].any()
 ## Kernels
 
 `domainia.wilson` holds the kernels: `forward`, `inverse`, `forward_time` and `inverse_time`
-(any leading axes), the windows that `transform` and `wdm_covariance` take as `window=`
-(`MEYER`, the default, `Meyer(d)`, `RampWindow`, `SampledWindow`), and the dense references.
+(any leading axes), the layout's `grid.cells`, `grid.fold` and `grid.unfold`, the windows that
+`transform` and `wdm_covariance` take as `window=` (`MEYER`, the default, `Meyer(d)`,
+`RampWindow`, `SampledWindow`), and the dense references.
diff --git a/src/domainia/_covariance.py b/src/domainia/_covariance.py
index da545c0..609e2e4 100644
--- a/src/domainia/_covariance.py
+++ b/src/domainia/_covariance.py
@@ -14,7 +14,7 @@ from typing import Any
 import numpy as np
 
 from .wilson._checks import require_callable, require_dtype, require_int, validate_dt
-from .wilson.grid import active_mask, band_frequencies, delta_T, validate_grid
+from .wilson.grid import band_frequencies, cells, delta_T, validate_grid
 from .wilson.windows import MEYER, Window, sample_window
 
 __all__ = ["frequency_covariance", "wdm_covariance"]
@@ -105,11 +105,12 @@ def wdm_covariance(S: Callable[[np.ndarray, np.ndarray], Any], *, Nf: int, Nt: i
                    window: Window = MEYER) -> np.ndarray:
     """The covariance of each WDM cell's K coefficients under an evolutionary model ``S(f, t)``.
 
-    Returns ``C[a, b, m, n] = BA_m[Re H_ab(., t_n)] / 2``, float64 of shape ``(K, K, Nf + 1, Nt)``,
-    exactly symmetric in ``a, b`` and 0 at the inactive cells.  ``H = (S + S^H) / 2`` after the
-    check (:func:`_hermitian`), ``t_n = n Nf dt``, and ``BA_m`` is the phi-hat^2-weighted mean over
-    band ``m``'s frequency samples, folded at DC and Nyquist (WDM Paper II, Eq. 36), in the Wilson
-    basis of ``window``.  For K = 1 this is ``wilson.noise.evolutionary_band_variance``.
+    Returns ``C[a, b, i, j] = BA_m[Re H_ab(., t_n)] / 2`` with ``(m, n) = cells(Nf, Nt)[:, i, j]``
+    (:func:`domainia.wilson.grid.cells`), float64 of the folded shape ``(K, K, Nf, Nt)`` and
+    exactly symmetric in ``a, b``.  ``H = (S + S^H) / 2`` after the check (:func:`_hermitian`),
+    ``t_n = n Nf dt``, and ``BA_m`` is the phi-hat^2-weighted mean over band ``m``'s frequency
+    samples, reflected at DC and Nyquist (WDM Paper II, Eq. 36), in the Wilson basis of
+    ``window``.  For K = 1 this is ``wilson.noise.evolutionary_band_variance``.
 
     ``S`` gets broadcasting read-only ``f`` (bands, 1, bins) [Hz] and ``t`` (1, columns, 1) [s
     from the record's first sample]: once on band 0 and two columns, then once per chunk of at
@@ -149,5 +150,5 @@ def wdm_covariance(S: Callable[[np.ndarray, np.ndarray], Any], *, Nf: int, Nt: i
                 raise ValueError(f"{name} is {H.shape[0]} x {H.shape[0]}, but was {K} x {K}")
             C[a, b, m:m + nb, n:n + nc] = C[b, a, m:m + nb, n:n + nc] = (
                 0.5 * (H.real[a, b] * w).sum(axis=-1) / w.sum())
-    C[:, :, ~active_mask(Nf, Nt)] = 0.0
-    return C
+    m, n = cells(Nf, Nt)
+    return C[:, :, m, n]
````

- [ ] **Step 4: Run the task's tests: they pass**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/test_covariance.py tests/test_readme.py 2>&1 | tail -1
```

Expected (prototype): `8 passed`.

- [ ] **Step 5: Run the lanes, the replay and the size**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py
```

Expected (prototype): `943 passed, 3 skipped` in both lanes, with no warnings and no collection errors; `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`; `src 4085 lines / 1852 code; tests 4350 lines (+ oracle 4135)`.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add README.md src/domainia/_covariance.py tests/test_covariance.py && git commit -q -F - <<'EOF'
Fold wdm_covariance to (K, K, Nf, Nt) and describe the folded layout in the README

wdm_covariance still fills the padded (K, K, Nf+1, Nt) grid chunk by chunk with the same
model calls, then gathers its cells with wilson.grid.cells; the zeroing of the inactive
cells goes. The README's Covariance example shows (2, 2, 16, 64) without the inactive-cell
check, Conventions describes the folded slots, the flat form, cells and fold/unfold in
place of the zero inactive cells, and Kernels names cells, fold and unfold.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
```


### Task 7: final sweep: `active_mask` private, dead checks, four-lane gates (spec §2, §7; §8 step 7)

**Goal:** `active_mask` leaves `wilson.grid.__all__` as the private `_active_mask`, and the remaining padded-layout wording and dead checks go. The final gates pass in all four lanes.

**Files:**
- Modify: `src/domainia/wilson/grid.py`, `src/domainia/wilson/transform.py`, `src/domainia/wilson/evolutionary.py`
- Test: `tests/test_package.py`, `tests/wilson/vendored/test_grid.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py`

**What the patches do:**
- **`_active_mask(Nf, Nt)`:**
  - It is one line: `mask[[0, Nf], 1::2] = False`. Nf is even, and every caller has validated the grid.
  - It loses its doctest and its own `validate_grid` call.
  - Only `fold` and `wrap_fractions` use it.
- **`base_atom_spectra`** skips the two inactive base atoms with `if a and m in (0, Nf):`, without building the mask.
- **`_inverse`** keeps only its finiteness check, which `psd_from_lambda_blocks` relies on.
  - Every caller builds or checks the padded float grid, and `sample_window` checks the grid and dt.
  - A 274-case probe of single bad inputs over every public caller gave the same exception kinds and messages before and after.
  - `wrap_fractions` now calls `validate_dt` itself, so a bad dt is still reported before a bad window.
- **The sweep:** the grid module docstring no longer lists the mask, and `transform.py`'s docstrings write W for the removed W_a.
- **Padded wording left on purpose:**
  - private code that is padded: the twins and `_block_tridiag_matvec`;
  - `eq45_plan`'s pinned `TypeError` text ("packed cells");
  - `noise.py`'s "folds about 0", which is the frequency reflection, next to `f_folded`.
- **Tests:**
  - `test_package` pins `wilson.grid.__all__`: 14 names, with `cells`, `fold` and `unfold` and without `active_mask`.
  - The four vendored `active_mask` tests go.
  - The remaining uses import `_active_mask`.

**Acceptance Criteria:**
- [ ] `hasattr(wilson.grid, "active_mask")` is False; `wilson.grid.__all__` has 14 names including `cells`, `fold`, `unfold`.
- [ ] `grep -rn "active_mask" src` shows only `_active_mask` (in `grid.py` and `evolutionary.py`).
- [ ] The replay prints `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`.
- [ ] Full suite (prototype): `938 passed, 3 skipped` in each of PY, FLOOR, `cur313` and `cur314`, under `-W error`, with no warnings and no collection errors.
- [ ] Size (prototype): `src 4059 lines / 1841 code; tests 4325 lines (+ oracle 4135)`. Against e6d1e3f (src 4,080 lines / 1,861 code, tests 4,344), src −21 lines / −20 code and tests −19.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/test_package.py tests/wilson/vendored/test_grid.py tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/wilson/test_evolutionary_kernels.py tests/wilson/vendored/test_evolutionary.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/grid.py src/domainia/wilson/transform.py src/domainia/wilson/evolutionary.py 2>&1 | tail -1` → `492 passed, 3 skipped`

**Steps:**

- [ ] **Step 1: Apply the tests patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T7-tests && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists exactly `tests/test_package.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_grid.py` as modified.

<!-- patch: T7-tests -->
````diff
diff --git a/tests/test_package.py b/tests/test_package.py
index d6f56e8..6b13519 100644
--- a/tests/test_package.py
+++ b/tests/test_package.py
@@ -48,15 +48,21 @@ def test_the_project_metadata():
 
 
 def test_the_public_names():
-    """A name that left a top level stays in its submodule (``wilson.grid.active_mask``,
-    ``wilson.windows.WindowError``).  ``wilson.matrix`` has no ``active_W`` or
-    ``build_C_wdm_active``: ``build_W`` and ``build_C_wdm`` are already ``N x N``."""
+    """A name that left a top level stays in its submodule (``wilson.windows.WindowError``).
+    The folded layout has no inactive cells, so ``wilson.grid`` has no ``active_mask`` and
+    ``wilson.matrix`` no ``active_W`` or ``build_C_wdm_active``: ``build_W`` and
+    ``build_C_wdm`` are already ``N x N``."""
     assert set(domainia.__all__) == PUBLIC
     assert set(wilson.__all__) == WILSON_PUBLIC
     assert all(hasattr(domainia, name) for name in domainia.__all__)
     assert all(hasattr(wilson, name) for name in wilson.__all__)
     assert not hasattr(domainia, "MEYER") and not hasattr(wilson, "active_mask")
     assert wilson.MEYER is wilson.windows.MEYER and wilson.Grid is wilson.grid.Grid
+    assert set(wilson.grid.__all__) == {"Grid", "band_frequencies", "cells", "delta_F", "delta_T",
+                                        "endpoint_weights", "fold", "n_samples", "sanitize_psd",
+                                        "time_pixel_centers", "unfold", "validate_dt",
+                                        "validate_finite", "validate_grid"}
+    assert not hasattr(wilson.grid, "active_mask")
     assert set(wilson.matrix.__all__) == {"build_C_time", "build_C_wdm", "build_W",
                                           "real_to_rfft", "rfft_to_real"}
     assert not hasattr(wilson.matrix, "active_W") and not hasattr(wilson.matrix, "build_C_wdm_active")
diff --git a/tests/wilson/test_grid_bands.py b/tests/wilson/test_grid_bands.py
index 64052f3..e85db06 100644
--- a/tests/wilson/test_grid_bands.py
+++ b/tests/wilson/test_grid_bands.py
@@ -13,7 +13,7 @@ import pytest
 
 from domainia import wilson
 from domainia.wilson.filter import phi_centered
-from domainia.wilson.grid import Grid, active_mask, cells, endpoint_weights, fold, unfold
+from domainia.wilson.grid import Grid, _active_mask, cells, endpoint_weights, fold, unfold
 from domainia.wilson.matrix import build_W
 from domainia.wilson.transform import atom_spectrum, base_atom_spectra, forward_time
 
@@ -47,7 +47,7 @@ def test_fold_and_unfold_round_trip_and_unfold_writes_exact_zeros(rng):
     P[..., m, n] = 0.0
     assert not P.any()
     B = unfold(np.ones((Nf, Nt), bool), Nf, Nt)
-    assert B.dtype == bool and np.array_equal(B, active_mask(Nf, Nt))
+    assert B.dtype == bool and np.array_equal(B, _active_mask(Nf, Nt))
     assert fold(B, Nf, Nt).dtype == bool
 
 
@@ -173,7 +173,7 @@ def test_atoms_are_the_rows_of_the_transform():
     W = build_W(Nf, Nt, dt)
     base = base_atom_spectra(Nf, Nt, dt)
     assert sorted(base) == [(m, a) for m in range(Nf + 1) for a in (0, 1)
-                            if active_mask(Nf, Nt)[m, a]]
+                            if _active_mask(Nf, Nt)[m, a]]
     m, n = (c.ravel() for c in cells(Nf, Nt))           # row p of W is folded slot p
     for p in range(m.size):
         atom = np.fft.irfft(atom_spectrum(int(m[p]), int(n[p]), Nf, Nt, dt, base=base), n=Nf * Nt)
diff --git a/tests/wilson/test_window_threading.py b/tests/wilson/test_window_threading.py
index 823e6b3..28a469b 100644
--- a/tests/wilson/test_window_threading.py
+++ b/tests/wilson/test_window_threading.py
@@ -394,7 +394,7 @@ def test_atom_spectra_are_the_rfft_of_the_rows_of_W(window, Nf, Nt, dt):
     base = T.base_atom_spectra(Nf, Nt, dt, window=window)
     for (m, a), G in base.items():
         _close(G, np.fft.rfft(W[row[m, a]]), RTOL, f"base_atom_spectra[{m}, {a}]")
-    for m, n in zip(*np.nonzero(wilson.grid.active_mask(Nf, Nt)), strict=True):
+    for m, n in zip(*np.nonzero(wilson.grid._active_mask(Nf, Nt)), strict=True):
         _close(T.atom_spectrum(int(m), int(n), Nf, Nt, dt, base=base, window=window),
                np.fft.rfft(W[row[m, n]]), RTOL_ATOM, f"atom_spectrum({m}, {n}, base=)")
     for m, n in ((1, 3), (0, 2), (Nf, 2)):
diff --git a/tests/wilson/vendored/test_evolutionary.py b/tests/wilson/vendored/test_evolutionary.py
index ff5dffc..7dd06d2 100644
--- a/tests/wilson/vendored/test_evolutionary.py
+++ b/tests/wilson/vendored/test_evolutionary.py
@@ -11,7 +11,7 @@ import pytest
 # The oracle packs the cells in band order; tests/wilson/test_evolutionary_kernels.py checks
 # the kernels against it.
 from wdm_dev_ev import evolutionary as ev
-from domainia.wilson.grid import active_mask, cells
+from domainia.wilson.grid import _active_mask, cells
 from domainia.wilson.matrix import build_W
 from domainia.wilson.transform import forward_time
 
@@ -30,7 +30,7 @@ def test_packed_grid_matches_the_dense_transform(rng):
     g = ev.PackedGrid(NF, NT, DT)
     Wa = build_W(NF, NT, DT)[BAND]
     assert g.size == N
-    np.testing.assert_array_equal(g.mask, active_mask(NF, NT))
+    np.testing.assert_array_equal(g.mask, _active_mask(NF, NT))
     x = rng.standard_normal((3, N))
     np.testing.assert_allclose(np.stack([g.forward(r) for r in x]), x @ Wa.T, atol=1e-12)
     np.testing.assert_allclose(g.forward(x[0]), forward_time(x[0], NF, NT, DT).ravel()[BAND], atol=0)
diff --git a/tests/wilson/vendored/test_grid.py b/tests/wilson/vendored/test_grid.py
index b0d7383..97ffa6d 100644
--- a/tests/wilson/vendored/test_grid.py
+++ b/tests/wilson/vendored/test_grid.py
@@ -6,7 +6,6 @@ import numpy as np
 import pytest
 
 from domainia.wilson.grid import (
-    active_mask,
     band_frequencies,
     delta_F,
     delta_T,
@@ -16,36 +15,6 @@ from domainia.wilson.grid import (
 )
 
 
-def test_active_mask_shape_and_dtype(simple_grid):
-    """active_mask returns shape (Nf+1, Nt) of bool."""
-    g = simple_grid
-    mask = active_mask(g.Nf, g.Nt)
-    assert mask.shape == (g.Nf + 1, g.Nt)
-    assert mask.dtype == bool
-
-
-def test_active_mask_interior_all_true(simple_grid):
-    """Interior bands (m=1..Nf-1) are fully active."""
-    g = simple_grid
-    mask = active_mask(g.Nf, g.Nt)
-    for m in range(1, g.Nf):
-        assert np.all(mask[m])
-
-
-def test_active_mask_dc_even_only(simple_grid):
-    """DC band (m=0) has only even n active."""
-    g = simple_grid
-    mask = active_mask(g.Nf, g.Nt)
-    assert np.all(mask[0] == (np.arange(g.Nt) % 2 == 0))
-
-
-def test_active_mask_nyquist_parity(simple_grid):
-    """Nyquist band (m=Nf) is active where (Nf + n) is even."""
-    g = simple_grid
-    mask = active_mask(g.Nf, g.Nt)
-    assert np.all(mask[g.Nf] == ((g.Nf + np.arange(g.Nt)) % 2 == 0))
-
-
 def test_band_frequencies_endpoints(simple_grid):
     """freqs[0] == 0 and freqs[Nf] == Nyquist."""
     g = simple_grid
````

- [ ] **Step 2: Run the task's tests: they fail**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error --tb=no -rfE --continue-on-collection-errors tests/test_package.py tests/wilson/vendored/test_grid.py tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/wilson/test_evolutionary_kernels.py tests/wilson/vendored/test_evolutionary.py 2>&1 | tail -n 40
```

Expected last line (prototype): `21 failed, 417 passed, 2 skipped, 4 errors`. `--continue-on-collection-errors` lets the run go past the import errors.
- `test_grid_bands.py` and `vendored/test_evolutionary.py` fail to collect, each reported twice: `ImportError: cannot import name '_active_mask' from 'domainia.wilson.grid'`.
- `test_package` sees `active_mask` in `wilson.grid.__all__`.
- The 20 `test_atom_spectra_are_the_rfft_of_the_rows_of_W` cases raise `AttributeError: module 'domainia.wilson.grid' has no attribute '_active_mask'`.

`vendored/test_grid.py` passes, because its patch only deletes.

- [ ] **Step 3: Apply the implementation patch**

```bash
cd /Users/aaron/Documents/lisa/domainia && ID=T7-impl && awk -v id="$ID" '$0=="<!-- patch: " id " -->"{f=1;next} f&&/^````diff$/{p=1;next} p&&/^````$/{exit} p' docs/superpowers/plans/2026-11-13-domainia-folded-wdm.md > $TMPDIR/fw-$ID.patch && git apply --check $TMPDIR/fw-$ID.patch && git apply $TMPDIR/fw-$ID.patch && git status --short
```

Expected: `git status --short` lists `src/domainia/wilson/evolutionary.py`, `src/domainia/wilson/grid.py`, `src/domainia/wilson/transform.py`, `tests/test_package.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/test_grid.py` as modified, and nothing else.

<!-- patch: T7-impl -->
````diff
diff --git a/src/domainia/wilson/evolutionary.py b/src/domainia/wilson/evolutionary.py
index db87d2a..bdb21e2 100644
--- a/src/domainia/wilson/evolutionary.py
+++ b/src/domainia/wilson/evolutionary.py
@@ -48,8 +48,8 @@ from scipy.special import ive
 
 from ._checks import require_dtype, require_real
 from .grid import (
+    _active_mask,
     _require_int,
-    active_mask,
     cells,
     delta_T,
     endpoint_weights,
@@ -147,8 +147,9 @@ def wrap_fractions(Nf: int, Nt: int, dt: float, *, bands: Array | None = None,
     ``nan`` at an edge band's inactive (odd) columns.
     """
     validate_grid(Nf, Nt)
+    validate_dt(dt)
     N = Nf * Nt
-    mask = active_mask(Nf, Nt)
+    mask = _active_mask(Nf, Nt)
     m_list = np.arange(Nf + 1) if bands is None else np.asarray(bands, int)
     out = np.full((m_list.size, Nt), np.nan)
     n = np.arange(Nt)
diff --git a/src/domainia/wilson/grid.py b/src/domainia/wilson/grid.py
index 7180eda..3b89761 100644
--- a/src/domainia/wilson/grid.py
+++ b/src/domainia/wilson/grid.py
@@ -1,5 +1,5 @@
-"""Time-frequency grid: the grid and input checks, pixel widths and centres, the
-active-pixel mask and the folded layout.
+"""Time-frequency grid: the grid and input checks, pixel widths and centres, and the folded
+layout.
 
 The WDM transform partitions the time-frequency plane into a grid of
 $N_f+1$ frequency bands (indexed by $m = 0,\\ldots,N_f$) and $N_t$
@@ -35,7 +35,6 @@ type BoolArray = NDArray[np.bool_]
 
 __all__ = [
     "Grid",
-    "active_mask",
     "band_frequencies",
     "cells",
     "delta_F",
@@ -300,28 +299,11 @@ def time_pixel_centers(Nf: int, Nt: int, dt: float) -> Array:
     return np.arange(Nt) * delta_T(Nf, dt)
 
 
-def active_mask(Nf: int, Nt: int) -> BoolArray:
-    """Boolean mask of shape ``(Nf+1, Nt)`` marking active (nonzero) pixels.
-
-    - Interior bands ($1 \\le m \\le N_f - 1$): all $N_t$ time bins active.
-    - DC band ($m=0$): only even $n$ are active ($N_t/2$ pixels).
-    - Nyquist band ($m=N_f$): only $n$ with $(N_f + n)$ even are active
-      ($N_t/2$ pixels).
-
-    Total active count equals $N = N_f N_t$.
-
-    Examples
-    --------
-    >>> import numpy as np; from domainia import wilson
-    >>> mask = wilson.grid.active_mask(8, 16)
-    >>> mask.shape, int(mask.sum())   # total active == N
-    ((9, 16), 128)
-    """
-    validate_grid(Nf, Nt)
+def _active_mask(Nf: int, Nt: int) -> BoolArray:
+    """The active cells of the padded ``(Nf+1, Nt)`` grid: every interior cell, and the even
+    times of the DC and Nyquist bands (``Nf`` is even).  The caller validates the grid."""
     mask = np.ones((Nf + 1, Nt), dtype=bool)
-    n = np.arange(Nt)
-    mask[0] = n % 2 == 0
-    mask[Nf] = (Nf + n) % 2 == 0
+    mask[[0, Nf], 1::2] = False
     return mask
 
 
@@ -399,7 +381,7 @@ def fold(P: object, Nf: int, Nt: int) -> np.ndarray:
     a = _wdm_array("P", P, (Nf + 1, Nt))
     v = np.abs(a.astype(np.result_type(a, 1.0), copy=False))
     peak = np.max(v, axis=(-2, -1), keepdims=True)
-    bad = np.argwhere(~active_mask(Nf, Nt) & (~np.isfinite(v) | (v > RESIDUE * peak)))
+    bad = np.argwhere(~_active_mask(Nf, Nt) & (~np.isfinite(v) | (v > RESIDUE * peak)))
     if bad.size:
         *lead, i, j = (int(k) for k in bad[0])
         raise ValueError(f"P{lead if lead else ''}: the inactive cell (m={i}, n={j}) holds "
diff --git a/src/domainia/wilson/transform.py b/src/domainia/wilson/transform.py
index d7f0c22..1aaf70f 100644
--- a/src/domainia/wilson/transform.py
+++ b/src/domainia/wilson/transform.py
@@ -46,8 +46,8 @@ with writes outside $[0, N/2]$ discarded.
 
 The DC and Nyquist edge-band coefficients are divided by $\\sqrt{2}$ so
 that all basis functions have unit norm.  This makes the WDM transform
-matrix orthogonal up to the sampling scale ($W_a W_a^T = \\delta t\\,I$,
-so invert with $x = W_a^T w/\\delta t$) and the Parseval identity exact:
+matrix orthogonal up to the sampling scale ($W W^T = \\delta t\\,I$,
+so invert with $x = W^T w/\\delta t$) and the Parseval identity exact:
 $\\sum_{m,n} w_{mn}^2 = \\delta t \\sum_k x_k^2$.  This is the package's only
 coefficient convention.
 
@@ -66,7 +66,6 @@ from numpy.typing import NDArray
 from ._checks import REAL_KINDS, require_dtype, require_int
 from .grid import (
     _fold_fix,
-    active_mask,
     cells,
     n_samples,
     unfold,
@@ -310,17 +309,10 @@ def forward(
 
 def _inverse(wmn: Array, Nf: int, Nt: int, dt: float, *, window: Window = MEYER) -> CArray:
     """:func:`inverse` on the padded grid ``(..., Nf+1, Nt)``, whose inactive cells must be 0."""
-    validate_grid(Nf, Nt)
-    validate_dt(dt)
     N = n_samples(Nf, Nt)
-    wmn = np.asarray(wmn)
-    # A complex grid would have its imaginary part silently discarded.
-    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
-    if wmn.shape[-2:] != (Nf + 1, Nt):
-        raise ValueError(f"wmn must have shape (..., {Nf + 1}, {Nt}); got {wmn.shape}")
     validate_finite(wmn, "inverse: wmn")
 
-    phi_c = sample_window(window, Nf, Nt, dt)
+    phi_c = sample_window(window, Nf, Nt, dt)   # checks the grid and dt
     wmn_work = wmn.astype(np.float64)              # a fresh copy, so scaling it in place is safe
     wmn_work[..., [0, Nf], :] *= np.sqrt(2.0)      # undo forward's unit-norm edge scaling
 
@@ -481,15 +473,14 @@ def base_atom_spectra(Nf: int, Nt: int, dt: float, *,
 
     Every other atom of band ``m`` is a circular shift of one of these by an
     even number of columns (:func:`atom_spectrum`).  ``dt * inverse(e)``:
-    :func:`inverse` returns ``rfft(W_a^{-1} e) = rfft(atom) / dt``.
+    :func:`inverse` returns ``rfft(W^{-1} e) = rfft(atom) / dt``.
     """
     validate_grid(Nf, Nt)
     validate_dt(dt)
-    mask = active_mask(Nf, Nt)
     out: dict[tuple[int, int], CArray] = {}
     for m in range(Nf + 1):
         for a in (0, 1):
-            if not mask[m, a]:
+            if a and m in (0, Nf):
                 continue
             e = np.zeros((Nf + 1, Nt))
             e[m, a] = 1.0
@@ -503,7 +494,7 @@ def _check_base_spectrum(base: dict[tuple[int, int], CArray], G: CArray, Nf: int
 
     An ``O(N)`` fingerprint: ``2 Nf`` entries pin ``Nf``; the length ``N//2+1`` of ``G``
     then pins ``Nt``; the energy of the atom pins ``dt``, because every active atom has
-    energy ``dt`` to rounding (``W_a W_a^T = dt I``).  ``N = Nf Nt`` is even, so the last
+    energy ``dt`` to rounding (``W W^T = dt I``).  ``N = Nf Nt`` is even, so the last
     bin is the Nyquist bin.  The window is not pinned: the atoms of every admissible window
     have energy ``dt``.
     """
````

- [ ] **Step 4: Run the task's tests: they pass**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error tests/test_package.py tests/wilson/vendored/test_grid.py tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/wilson/test_evolutionary_kernels.py tests/wilson/vendored/test_evolutionary.py tests/wilson/test_meyer_baseline.py src/domainia/wilson/grid.py src/domainia/wilson/transform.py src/domainia/wilson/evolutionary.py 2>&1 | tail -1
```

Expected (prototype): `492 passed, 3 skipped`.

- [ ] **Step 5: Run the lanes, the replay and the size**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/cur313/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/cur314/bin/python -P -m pytest -q -W error 2>&1 | tail -1
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py
```

Expected (prototype): `938 passed, 3 skipped` in every lane, with no warnings and no collection errors; `baseline reproduces: 1404 arrays (100 DROPPED keys set aside)`; `src 4059 lines / 1841 code; tests 4325 lines (+ oracle 4135)`.

- [ ] **Step 6: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/wilson/evolutionary.py src/domainia/wilson/grid.py src/domainia/wilson/transform.py tests/test_package.py tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/wilson/vendored/test_evolutionary.py tests/wilson/vendored/test_grid.py && git commit -q -F - <<'EOF'
Make wilson.grid.active_mask private and sweep the last padded-layout wording

active_mask leaves wilson.grid.__all__ as the private _active_mask: the folded layout has
no inactive cells, and only fold and wrap_fractions still need the padded mask. Its padded
(9, 16) doctest goes, and its body clears the edge bands' odd times in one line. The grid
module docstring no longer lists the mask, and transform's docstrings write W for the
removed W_a. base_atom_spectra skips the two inactive base atoms, (0, 1) and (Nf, 1),
without building the mask. _inverse keeps only its finiteness check, which
psd_from_lambda_blocks relies on: its callers build or check the padded float grid, and
sample_window checks the grid and dt. wrap_fractions checks dt itself, so a bad dt is still
reported before a bad window.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
```

