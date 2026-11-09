# domainia 2.0 Follow-ups II Implementation Plan: the open decisions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved spec `docs/superpowers/specs/2026-11-08-domainia-open-decisions-design.md` on branch `domainia-open-decisions`:
* the maps compute in double;
* "wrong kind → `TypeError`, wrong value → `ValueError`" holds across the whole package;
* `DenseOperator` keeps one `rtol`, with its fail-open bugs fixed;
* the sdist lists what it includes;
* the test gaps, text and record items are done.

**Architecture:** The changes are made in place, mostly in the spec's Order.
* Task 1 adds the shared private module `domainia.wilson._checks`. Every later check uses it.
* Tasks 2–6 apply the wrong-kind rule one module group at a time: wilson, domains, maps, covariance, and the rest. Each task applies every kind of check (integers, reals, flags, strings, `isinstance`, index and value arrays, the `", got"` message form) to the files it owns, and adds that group's wrong-kind test file (spec §2.10).
* Task 7 checks the shape of `b`, and Task 8 fixes `DenseOperator`.
* Task 9 closes §2: a zero-hit gate on the old `"; got"` kind messages, then the rule's four documents.
* Tasks 10–11 are precision: the maps compute in double, and a guard is added on single-precision kernel output.
* Tasks 12–15 are packaging, test gaps, text and the record.

A float64 bit baseline recorded before Task 1 (Task 0) is the regression gate throughout.

**Deviation from the spec's §2.11 order, recorded here:** §2.11 orders §2 by kind of check across the package: silent conversions, then `AttributeError`, then arrays, then shapes, then messages. This plan orders §2 by module group instead. Each module's file then has one owner, and that owner's diff can be reviewed as a whole. The end state is the same. §2.11's step 6 (the message format, then the four documents) stays last in §2 (Task 9). It runs after Task 8, so that `operators.py`'s messages are converted before the zero-hit gate.

**Tech Stack:** Python ≥3.12, NumPy ≥2.1, SciPy ≥1.14.1, optional JAX ≥0.9, pytest ≥8, hatchling ≥1.27 (scratch environment only). Test env: `~/miniconda3/envs/wdmrel` (Python 3.12.4, numpy 2.3.5, scipy 1.16.3, jax 0.9.0.1).

**User decisions (already made):**
- 2026-11-08, scope: "One domainia spec (Recommended)". The Enchilada adapter gets its own spec later.
- 2026-11-08, precision: "Maps compute in double (Recommended)". The wilson kernels keep float32 when called directly.
- 2026-11-08, validation: "Whole package (Recommended)". One set of shared helpers lives in `domainia.wilson`.
- 2026-11-08, array dtype: "dtype is kind: TypeError (Recommended)".
- 2026-11-08, `rtol`: "One rtol, both jobs (Recommended)".
- 2026-11-08, release: "Rename, stay private (Recommended)". Name: "I hate all of these keep the name and we'll rework it once the package is complete." The distribution stays `domainia`, and nothing is published.
- 2026-11-08, design parts: Part 1 "looks good"; Part 2 "yes, go on to part 3"; Part 3 "yes, write the spec". The items marked *(my call)* in the spec were approved with them. One of them (the per-entry symmetry check) was reversed on evidence (spec §3.4).
- 2026-11-08, written spec: "do your best". This approves the spec as written, with no trims, and both §3.5 exceptions stay.
- Standing: commit on `domainia-open-decisions` only; never push; Enchilada is read-only.

**Scope check:** one plan. The spec is one subsystem, the package's own contracts, and it has one Gates section. The Enchilada adapter is parked in the spec ("Parked for the Enchilada spec").

---

## Conventions (every task)

* **Repository.** `REPO=/Users/aaron/Documents/lisa/domainia`, the main checkout, on branch `domainia-open-decisions`. There is no worktree. Work only in `$REPO`. Never `git checkout` another branch.
* **`$PY`** is shorthand for the prefix `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P`, run from `$REPO`.
  * The Bash tool runs zsh, which does not word-split a variable, so `PY="…"; $PY -m pytest` fails. `${=PY}` fails too: the `NAME=value` words it yields are not taken as assignments. Write the prefix out in full in each command, as every task's commands do.
  * Quote globs too (`--include='*.py'`): an unmatched glob is an error in zsh.
  * A probe script outside pytest needs `PYTHONPATH=src` (the package is not installed in `wdmrel`).
* **Full suite:** `$PY -m pytest -q`. **NumPy lane:** `WDM_TESTS_HIDE_JAX=1 $PY -m pytest -q`.
  * Both must pass, with no collection errors and no new warnings, at the end of every task.
  * Steps state pass counts only for the task's own test files. For the rest they say "passes": absolute suite totals drift between tasks.
* **Meyer baseline.** `tests/wilson/test_meyer_baseline.py` replays `tests/wilson/meyer_cases.py` against `tests/wilson/data/meyer_baseline.npz`. Never re-record the npz.
* **Float64 bit gate** (Task 0's script, in `$TMPDIR`):
  * Task 0 records the baseline before any change.
  * Tasks 2, 4, 10 and 11 (the ones that touch `wilson/transform.py` or `maps/`) and Task 15 run `check`. The result must be `0 of 1504 arrays differ`.
  * `--check` alone is not enough: it compares the RESCALED families only to 1e-13.
* **Edits are anchored to code, not line numbers.** Every step names the function, class or exact old text it changes. The spec cites lines at 779550d, and earlier tasks shift them.
* **Each task starts with "Assumes the tree after Task N".** The state is the end of that task, as defined by the Interface contract below.
* **Pin lists are a starting point** (spec, Test pin lists). Before editing, a task greps for every pattern it changes. Each grep step states the expected hits, and the task updates every hit.
* **Wrong-kind tests** follow `tests/wilson/test_wrong_kinds.py`:
  * parametrised per slot kind, with `pytest.raises(TypeError, match=...)`;
  * one accepted NumPy integer (`np.int64`) and one accepted NumPy float (`np.float64`) pinned per slot kind.
* **Git.** The Claude Code sandbox blocks writes inside `.git`, so every `git add`, `git mv`, `git rm` and `git commit` runs with the sandbox disabled (`dangerouslyDisableSandbox: true`).
  * End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  * Nothing is pushed. Never amend, never `git stash`.
* **Scratch** scripts go in `$TMPDIR`, never in the repo.
* **Errata.** Deviations agreed during execution go in an "Execution errata" section at the end of this plan.

## File structure

```
src/domainia/
  wilson/_checks.py     NEW (T1): require_int, require_real, require_bool, require_str,
                        require_instance, require_callable, require_dtype, REAL_KINDS,
                        index_array, validate_dt, validate_finite
  wilson/grid.py        re-exports _require_int/validate_dt/validate_finite; validate_grid and
                        n_samples call the helper (T1); sanitize_psd, carrier_phase,
                        edge_columns, messages (T2)
  wilson/{filter,noise,windows}.py   integer check via the helper (T1); noise
                        n_realizations, windows messages (T2)
  wilson/{backend,evolutionary,inner_product,stationary}.py   kinds (T2); evolutionary
                        units note (T14)
  wilson/transform.py   kinds (T2); module docstring (T10); single-precision guard (T11);
                        citation dropped (T14)
  domains.py            _int wraps require_int (T1); every domain-side kind (T3); _cells_of (T13)
  data.py, spectra.py, translate.py   kinds (T6)
  linalg.py, testing.py kinds (T6); trim and docstrings (T14)
  variance.py           window= keyword, _cells_of (T13)
  maps/{base,time}.py   kinds (T4); dtype rule (T10)
  maps/fourier.py       kinds (T4); dtype rule (T10); trim (T14)
  covariance/{diagonal,dense}.py   kinds (T5)
  covariance/{base,blocktri,stack}.py   kinds (T5); b shape (T7)
  covariance/evolutionary.py   kinds, P/t_max (T5); window= keyword (T13)
  operators.py          DenseOperator rtol, fail-open fixes, stored rtol, symmetry factor,
                        messages (T8)
pyproject.toml          sdist only-include, Typing classifier, jax extra (T12)
.gitignore              .superpowers/ (T12)
README.md, docs/superpowers/specs/2026-10-31-*.md   the rule's text (T9)
docs/superpowers/specs/2026-11-04-*.md   dated notes (T9, T15)
docs/superpowers/plans/2026-11-07-*.md   Corrections (2026-11-08) (T15)
tests/
  wilson/test_checks.py            NEW (T1); citations dropped (T14)
  wilson/test_wrong_kinds.py       dt kinds (T1); the other wilson sites (T2); citations (T14)
  wilson/test_window_threading.py  SCANNED (T1); CONSUMERS (T13)
  test_wrong_kinds_domains.py      NEW (T3); _cells_of (T13); citations (T14)
  test_wrong_kinds_maps.py         NEW (T4); citations (T14)
  test_wrong_kinds_covariance.py   NEW (T5); citations (T14)
  test_wrong_kinds_misc.py         NEW (T6); citations (T14)
  test_dom_cov_multiply.py         b shapes (T7); never-dense layouts (T13); citations (T14)
  test_operators_dense.py, test_operators_covariance.py   DenseOperator cases, messages (T8)
  test_dom_map_dtypes.py           NEW (T10); citation (T14)
  wilson/test_single_precision.py  guard cases (T11); citation (T14)
  test_package.py                  only-include, metadata (T12)
  test_dom_maps_fourier_wdm.py     jit of the after() adjoint (T13)
  (existing files)                 pins updated per task (greps); 1.x tags (T14)
```

## Interface contract (end state of each name; later tasks build on these exactly)

| Name | Final form | Task |
|---|---|---|
| `domainia.wilson._checks` | The exact module text is in Task 1. `REAL_KINDS = "iuf"`. Every helper takes `name` first and returns the checked value. <br>• `require_int(name, value) -> int`: Python/NumPy integers; `bool`, `np.bool_`, floats, str, `None`, arrays → `TypeError` `"{name} must be an integer, got {type}"`. <br>• `require_real(name, value) -> float`: `numbers.Real` minus `bool`/`np.bool_`; str, complex, `Decimal`, `None`, arrays of any shape (0-d too) → `TypeError` `"{name} must be a real number, got {type}"`; too large for a float → `ValueError`. <br>• `require_bool(name, value) -> bool`: `bool`/`np.bool_` only → else `"{name} must be a bool, got {type}"`. <br>• `require_str(name, value) -> str`: kind only, `"{name} must be a string, got {type}"`. The site keeps its own `ValueError` listing the choices. <br>• `require_instance(name, value, cls, expected)`: `"{name} must be {expected}, got {type}"`, where `expected` carries its article. <br>• `require_callable(name, value)`: `"{name} must be callable, got {type}"`. <br>• `require_dtype(name, a, kinds, expected) -> None`: reads `a.dtype` without converting (JAX arrays and tracers pass); `"{name} must be {expected}, got dtype {dtype}"`. <br>• `index_array(name, value) -> ndarray[int64]`: integer dtypes pass; finite integral floats are converted; any other float → `ValueError` `"{name} must hold integer indices, got {first}"`; any other dtype → `TypeError` `"{name} must be an integer index array, got dtype {dtype}"`. <br>• `validate_dt(dt) -> None`: `require_real`, then `ValueError` unless positive and finite (message unchanged). <br>• `validate_finite(values, context) -> None`: moved unchanged. | T1 |
| `wilson.grid._require_int`, `wilson.grid.validate_dt`, `wilson.grid.validate_finite` | Re-exported in `grid.py` with `from ._checks import require_int as _require_int, validate_dt, validate_finite`. The public names (`wilson.validate_dt`, `wilson.validate_finite`, `wilson.grid.__all__`) are unchanged. `validate_grid` and `n_samples` call `_require_int`. | T1 |
| `domains._int(name, value, *, minimum)` | `v = require_int(name, value)`, then `ValueError` `"{name} must be >= {minimum}; got {value}"`. | T1 |
| Kind message form | Every `TypeError` message in `src` has the form `"…, got …"`. A `ValueError` keeps its text. Each of Tasks 2–6 and 8 converts the `TypeError` messages in the files it owns. Task 9's gate: `grep -rn "; got" src` shows no line inside a `raise TypeError(...)`. | T2–T9 |
| "never complex" checks (spec §3.5) | `require_dtype(name, a, REAL_KINDS, "<phrase>")`, raising `TypeError`. Bool, str and object dtypes are refused there too, as wrong kinds under §2.1. The sites are `from_coordinates` `R` (T6), `sanitize_psd`, `_reject_complex_psd`, `inverse`/`inverse_many` `wmn`, and `inverse_bands` `w` (T2). `translate.py`'s complex-on-time/WDM check stays `ValueError`, and `windows.py`'s two stay `WindowError`. | T2, T6 |
| Static fields and JAX rebuilds (spec §2.7) | Checks run in `__post_init__`/`__init__` on static meta fields and map arguments only, never on the array fields a map carries (a rebuild passes `None` or `object()` there). A derived static field defaulting to `_BUILD` is checked only after it is filled. | T3, T4, T5 |
| `Covariance._b_shape() -> tuple[int, ...]` and `covariance.base._check_leading(b, expected, what) -> None` | `_b_shape()` is `(n,)`, or `(len(covs), n_c)` for a `ChannelStack` whose channels share `n_c` (a mixed-size stack raises `ValueError`). `_check_leading` raises `ValueError` `"{what}: b must have leading shape {expected}, got shape {b.shape}"` unless `b.shape[:len(expected)] == expected`. Each factor has its own `_b_shape()` from its constants: `DiagFactor` `(v.shape[0],)`, `CholFactor` `(L.shape[0],)`, ragged `BlockTriCovFactor` the sum of its block sizes, padded `(layout.n,)`, `StackFactor` `(len(channels),)` plus each channel factor's own check. Checked in `multiply`, `solve`, `whiten` and `quad` (every form). | T7 |
| `operators.DenseOperator(matrix, domain, channel_names, *, rtol=None)` | Spec §3: `rtol >= 1` → `ValueError`; `tol = top * (n * eps)`; `S = 0.5 * C + 0.5 * C.T`; non-finite `eigh` eigenvalues → `ValueError` before `top`/`tol` are formed; zero matrix → `ValueError`; the round-off message when `rtol` was given and `λ_min >= -n·eps·top`; `op.rtol` is the effective relative tolerance and never `None`; symmetry factor `1e-12`, or `1e-12·eps32/eps64` for float32 input, printed in the refusal. | T8 |
| Maps' dtypes (spec §1.1) | Maps that compute return float64/complex128, and so does a push through them. Select/slice/zero-fill maps and their adjoints return the input's dtype, except that `BinSelect.adjoint` returns the input's complex counterpart, computed on the host as `np.result_type(np.dtype(y.dtype), np.complex64)`. The five real-taking methods call `require_dtype(..., REAL_KINDS, ...)` before `be.xp.asarray(x, dtype=np.float64)`. | T10 |
| `wilson.transform._check_single_output(out, entry) -> None` | Called by the ten entry points (spec §1.4) on the arrays they compute, only when the working dtype is single and `out` is concrete. It raises `ValueError` naming `entry` and the word `float64` if any entry is not finite. It does not change `np.errstate`. | T11 |
| sdist | `[tool.hatch.build.targets.sdist] only-include = ["src", "tests"]`; `exclude` is gone. | T12 |
| `domains._cells_of(cells) -> CellSet` | `WDMDomain` → `.cells`; `CellSet` passes; anything else raises the existing `_cells_and_window` `TypeError` text. `MidpointVariance` uses it. | T13 |

## Known collisions between tasks

Several tasks edit the same file. The later task works on the earlier task's result, and its steps name the earlier change. Every anchor below was checked in plan order: the drafted patches applied one task at a time, each to the previous task's result.

* `wilson/grid.py`, `wilson/noise.py`, `wilson/windows.py`, `tests/wilson/test_wrong_kinds.py`: T1, then T2.
* `wilson/transform.py`: T2 (the `k0`/`n_bins` ints, `_check_interior_bands`, the `wmn`/`w` `TypeError`s, messages), then T10 (the module docstring's "Where it reaches, and where it stops" only), then T11 (the guard), then T14 (one citation).
* `domains.py`: T1 (`_int`), then T3, then T13 (`_cells_of`).
* `maps/base.py`, `maps/time.py`, `maps/fourier.py`: T4, then T10; `maps/fourier.py` then T14 (the `FourierToWDM.window` trim).
* `covariance/base.py`, `blocktri.py`, `stack.py`: T5, then T7. `covariance/evolutionary.py`: T5, then T13 (`window=` keyword).
* `linalg.py`, `testing.py`: T6, then T14. `wilson/evolutionary.py`: T2, then T14.
* `tests/test_dom_domains.py`: T3, then T13 (`match=`), then T14 (line 1). `tests/test_translate_coordinates.py`: T6, then T13. `tests/test_dom_cov_multiply.py`: T7, then T13, then T14. `tests/test_wrong_kinds_domains.py`: T3, then T13, then T14. `tests/wilson/test_window_threading.py`: T1 (`SCANNED`), then T13 (`CONSUMERS`).
* `tests/test_dom_maps_dirichlet.py`, `tests/test_dom_spectra.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_noise_stationary.py`: an earlier task's pin, then T14's tag strip (line 1 only).
* T14 runs last among the code tasks. It changes only text: the 1.x tags, four sentences, two docstrings, two trims, and the spec citations Tasks 1–13 put in code.

## Spec coverage (each spec item → its task)

| Spec | Item | Task |
|---|---|---|
| §1.1 | the rule sentence in `maps/base.py`'s module docstring | T10 |
| §1.2 | the five real-taking methods (`require_dtype`, then `float64`); `apply_complex`; `TimeToFourier.adjoint` complex128; `FourierToWDM(impl="fft")` complex128; `BinSelect.adjoint` dtype; the `Grid.dt` leak; the docstrings (`maps/base.py`, `FourierToWDM`, `wilson/transform.py` module, `maps/time.py` module, `TimeToFourier`) | T10 |
| §1.3, Gates | float64 bit identity | T0 (record); T2, T4, T10, T11, T15 (check) |
| §1.4 | the guard at the ten entry points | T11 |
| §1.5 | `FourierDomain` stores `dt`/`sigma0` as Python `float` after the kind check | T3 |
| §1.6 | `tests/test_dom_map_dtypes.py`; the large-`dt` complex64 case; the complex64 pins that change in maps tests | T10 |
| §1.6 | the guard tests (`forward_bands`, `forward_time`, `inverse_time`, under `np.errstate`); the `test_single_precision.py` pins | T11 |
| §2.1 | the rule (a wrong kind raises `TypeError`, a wrong value `ValueError`; `WindowError` stays for what a window returns) | T1–T8 (code, each in its modules); T9 (the documents) |
| §2.2 | `_checks.py`; the seven integer copies; the re-exports; `SCANNED` | T1 |
| §2.2 | `", got"` for every `TypeError` message | T2–T6, T8 (own files); T9 (gate) |
| §2.3 | `wilson` integers (`forward_bands`/`inverse_bands` `k0`, `atom_spectrum` `m`/`n`, `carrier_phase` `m`, `edge_columns` `lead`/`trail`); `empirical_variance` `n_realizations`; `_check_interior_bands` bands | T2 |
| §2.3 | `CellSet` `m`/`n`, `CellSet.bands`, `CellSet.from_mask` `keep`; `tests/test_dom_domains.py:145-148` | T3 |
| §2.3 | `CropTime` `start`/`n`, `ZeroPadTime` `n_extra`, `FourierCrop` `start`; `pad_maps`/`crop_maps` counts, `k_out`, `j_in`; `BinSelect` `(k0, n)`; the five `rows` `idx` sites | T4 |
| §2.3 | `DiagonalCovariance` `n`/`channel`, `DenseCovariance` `n`, `ChannelStack` `params_per_channel`, `BlockTridiagonalCovariance` sizes | T5 |
| §2.3 | `cblock_factor` `b`, `pcg` `maxit`, `BandLayout.from_sizes` sizes; `TabulatedSpectrum.at` bins and `.values`; `eq54_covariance` `margin`, `s_I`, `s_G_src` | T6 |
| §2.4 | `validate_dt` (0-d arrays and `Decimal` refused); `tests/wilson/test_wrong_kinds.py:66` | T1 |
| §2.4 | `eq45_plan` `t_max` | T2 |
| §2.4 | `domains._positive_float` (`dt`, `sigma0`), `t0` | T3 |
| §2.4 | `TimeToFourier` `fmin` | T4 |
| §2.4 | `pcg` `true_rtol` and `tol`, and any other real scalar in the misc modules | T6 |
| §2.4 | `DenseOperator` `rtol` | T8 |
| §2.5 | `include_logdet`; `cells` (`inner_product`); `backend` (`get_backend`); `impl` (`Backend.cholesky`) | T2 |
| §2.5 | `units` | T3 |
| §2.5 | `restrict`; `impl` (`FourierToWDM`) | T4 |
| §2.5 | `symmetrize` (×3); `factor`'s `jit`; `chol_impl`; `layout` (`BlockTridiagonalCovariance`); `builder`; `assembly`; `form` | T5 |
| §2.5 | `on_fail`; `layout` (`linalg`) | T6 |
| §2.6 | `CellSet` class methods (`grid`, `sub`), `WDMDomain.full`, `ChannelAxis` bare `str` | T3 |
| §2.6 | map constructors' `src`/`dst`, `after`'s `inner`, `Restrict`, `BinSelect`, `Compose`, `Identity` | T4 |
| §2.6 | `ChannelStack` `covs` | T5 |
| §2.6 | `TabulatedSpectrum` `domain` and `.on`, `GenericPSD` callable, `BandLayout.from_cells`, `testing`'s `dense_W`/`wdm_congruence`/`eq45_dense`, `data`'s `channels` | T6 |
| §2.7, §2.10 | the JAX rebuild carve-out and its `tree_map` test on all 12 map classes and an `after()` map | T4 |
| §2.8, §2.10 | the shape of `b`; the `tests/test_dom_cov_multiply.py` cases | T7 |
| §2.9 | `EvolutionaryCovariance` `P`/`t_max` at construction | T5 |
| §2.9 | the `to_coordinates` `sqrt(2)` overflow; `domainia.testing` follows the rule | T6 |
| §2.10 | `tests/wilson/test_wrong_kinds.py` additions; `test_wrong_kinds_{domains,maps,covariance,misc}.py` | T2; T3, T4, T5, T6 |
| §2.10 | the pins `tests/test_dom_evolutionary_cov.py:101` and `tests/test_dom_variance_generic.py:77` (both pin `domains._cells_and_window`'s message) | T3 |
| §2.10 | `Meyer d must be an int`, `must be an int;` (no test pins either; the grep proves it) | T1 |
| §2.11 step 6 | the four documents (spec 2026-10-31 §5, `README.md:165-167`, spec 2026-11-04 line 73, `tests/wilson/test_wrong_kinds.py:1-2`) | T9 |
| §3.1–§3.4, §3.6 | `DenseOperator` | T8 |
| §3.5 | `from_coordinates` `R` (and its docstring); `tests/test_translate_coordinates.py:168-169` | T6 |
| §3.5 | `sanitize_psd`, `_reject_complex_psd`, `wmn` (×2), `inverse_bands` `w`; `tests/wilson/test_noise_stationary.py:226-238`, `tests/wilson/test_grid_bands.py:300` | T2 |
| §4 | `pyproject.toml`, `.gitignore`, `tests/test_package.py` | T12 |
| §4, Gates | the built-sdist inspection (scratch env) | T12 (and T15) |
| §5 | every item (the jitted `after()` adjoint is two asserts in `tests/test_dom_maps_fourier_wdm.py`'s existing jit-parity loop, which already builds those maps) | T13 |
| §6 | every item; plus no spec citation in `src`/`tests` code (the Nov 7 text sweep's rule; the specs are not in the sdist) | T14 |
| §7 | the Corrections list; the spec 2026-11-04 §1.7 note; the Task 8 `solve` probe | T15 |
| Gates | full suite, NumPy lane, `--check`, bit identity, `git status`, sdist | every task; T15 (final) |

---

### Task 0: Record the float64 bit baseline (spec Gates)

**Goal:** Before any change, record the output of `tests/wilson/meyer_cases.py`'s `compute()` (all 1504 arrays) into `$TMPDIR`. Tasks 2, 4, 10, 11 and 15 then check against it with `np.array_equal`.

**Files:**
- Create (scratch, outside the repo): `$TMPDIR/domainia_bits.py`, `$TMPDIR/domainia-bits-779550d.npz`
- No repo file changes; no commit.

**Acceptance Criteria:**
- [ ] `git diff --quiet 779550d HEAD -- src tests` exits 0 (the tree's code is 779550d's).
- [ ] `record` prints `recorded 1504 arrays`, and an immediate `check` prints `0 of 1504 arrays differ`.
- [ ] The full suite and the NumPy lane pass. Their summary lines are noted in the task report as the starting point.

**Verify:** `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz` → `0 of 1504 arrays differ`

**Steps:**

- [ ] **Step 1: Confirm the code is 779550d's**

Run: `cd /Users/aaron/Documents/lisa/domainia && git diff --quiet 779550d HEAD -- src tests && echo same`
Expected: `same`. Since 779550d, only the spec and this plan were committed. If the command prints nothing, stop: the baseline would not be 779550d's.

- [ ] **Step 2: Write the bit-gate script**

If `$TMPDIR/domainia_bits.py` already exists with exactly this text, keep it. Otherwise write it:

```python
"""Float64 bit gate for the 2026-11-08 plan: every Meyer case is bit-identical (same dtype,
np.array_equal) to the baseline recorded before Task 1.  Run from the repo root:
``record <npz>`` once before Task 1, ``check <npz>`` after."""
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

- [ ] **Step 3: Record, then check**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py record $TMPDIR/domainia-bits-779550d.npz`
Expected: `recorded 1504 arrays`. If the npz already exists from the planning session, re-recording it at this tree gives the same arrays.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz`
Expected: `0 of 1504 arrays differ`

- [ ] **Step 4: Record the starting suite**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: all pass (at 779550d: `1768 passed, 3 skipped`, the 3 skips being the oracle tests), with no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: all pass (at 779550d: `1661 passed, 110 skipped`), with no warnings summary.

No commit: nothing in the repo changed (`git status --short` shows only this plan's `.tasks.json`, if the coordinator has touched it).

---

### Task 1: The shared checks module and the seven integer checks (spec §2.2, §2.4)

**Goal:** Add the private module `domainia.wilson._checks`, which holds the kind checks that every later task uses. The seven copies of the integer check call its `require_int`. `wilson.grid` re-exports `_require_int`, `validate_dt` and `validate_finite` from it, and `validate_dt` now refuses a `Decimal` and arrays of any shape, 0-d included (spec §2.4).

**Assumes the tree after Task 0.** Task 0 changes no repo file, so the code is 779550d's and the spec's `file:line` references hold.

**Files:**
- Create: `src/domainia/wilson/_checks.py`, `tests/wilson/test_checks.py`
- Modify: `src/domainia/wilson/grid.py` (the imports; `_require_int`, `validate_dt` and `validate_finite` are deleted; the loops of `validate_grid` and `n_samples`)
- Modify: `src/domainia/wilson/filter.py` (`nu_d`), `src/domainia/wilson/noise.py` (`colored_noise`), `src/domainia/wilson/windows.py` (`Meyer.__post_init__`), `src/domainia/domains.py` (`_int`), each with one new import line
- Modify: `tests/wilson/test_wrong_kinds.py` (the two `dt` kind lists), `tests/wilson/test_window_threading.py` (`SCANNED`)

**Acceptance Criteria:**
- [ ] `src/domainia/wilson/_checks.py` is the text of Step 2, byte for byte, and its 10 doctests pass.
- [ ] `wilson.grid._require_int is _checks.require_int`. `grid.validate_dt`, `grid.validate_finite`, `wilson.validate_dt` and `wilson.validate_finite` are the `_checks` functions. `grid.__all__` and `wilson.__all__` are unchanged.
- [ ] `grep -rn "not isinstance(.*(int, np.integer))" src` shows one line, in `_checks.py`. Each of the seven sites raises `TypeError("{name} must be an integer, got {type}")`. Two messages change: `Meyer(2.5)` says `Meyer d must be an integer, got float`, and `domains._int` says `N must be an integer, got float` (both said `must be an int; got`). Every `ValueError` text is unchanged.
- [ ] `validate_dt` raises `TypeError("dt must be a real number, got <type>")` for `Decimal("0.5")`, `np.array(0.5)` and `np.array([0.5])`, and accepts `Fraction(1, 2)` and `np.int64(2)`.
- [ ] `tests/wilson/test_checks.py`: 103 passed (NumPy lane: 102 passed, 1 skipped). `tests/wilson/test_wrong_kinds.py`: 41 passed.
- [ ] The full suite and the NumPy lane pass, with no warnings summary.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_checks.py tests/wilson/test_wrong_kinds.py src/domainia/wilson/_checks.py` → `154 passed`

**Steps:**

- [ ] **Step 1: Grep the sites and the pins**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "not isinstance(.*(int, np.integer))" src`
Expected: exactly the seven copies the spec's Context lists (grep may print them in another order):

```text
src/domainia/domains.py:53:    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
src/domainia/wilson/grid.py:111:    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
src/domainia/wilson/grid.py:156:        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
src/domainia/wilson/grid.py:363:        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
src/domainia/wilson/noise.py:372:    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
src/domainia/wilson/windows.py:176:        if isinstance(self.d, bool) or not isinstance(self.d, (int, np.integer)):
src/domainia/wilson/filter.py:70:    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
```

`operators.py:102` (`rtol`, `(int, float, np.integer, np.floating)`) does not match: it is a real-number check, Task 8's.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "Meyer d must be an int\|must be an int;" src tests`
Expected: two lines, both in `src`: `domains.py:54` (`_int`) and `wilson/windows.py:177` (`Meyer`). No test pins either message.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "must be an int([^e]|$)" tests README.md`
Expected: no output. A `match="... must be an int"` would still match the new "must be an integer", so only a pin of the old `; got` tail could break, and there is none. (`tests/wilson/test_windows.py::test_meyer_rejects_a_bad_d` checks `Meyer`'s exception types only.)

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "_require_int" src tests | grep -v "tests/wilson/oracle"`
Expected: `wilson/matrix.py:49` and `:231`, `wilson/evolutionary.py:57` and `:462` (both import it from `.grid`), `wilson/grid.py:108` (the definition) and `:119` (`_require_even_positive_int`), and `tests/wilson/test_wrong_kinds.py:11` and `:38-43`. The re-export keeps every one of them working, so none of them changes.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "np.array(0.5)\|Decimal\|Fraction" src tests | grep -v "tests/wilson/oracle"`
Expected: two lines: `tests/wilson/test_wrong_kinds.py:66` (the `np.array(0.5)` that Step 4 moves) and `src/domainia/wilson/evolutionary.py:173` (the word "Fraction" in a docstring).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "__module__" src tests | grep -v "tests/wilson/oracle"`
Expected: one line, `tests/test_dom_cov_base.py:101` (a JAX type check). Nothing reads `validate_dt.__module__`, so the move is invisible there.

Checked, and needing no change:
* `wilson/__init__.py` imports `validate_dt` and `validate_finite` from `.grid`. The re-export keeps that import working; `tests/test_package.py::test_the_public_names` covers it.
* `pyproject.toml`'s `testpaths` include `src/domainia/wilson`, with `--doctest-modules`, so pytest collects the 10 doctests of `_checks.py`. The doctests of `validate_dt` and `validate_finite` move with the functions (doctest collects a module's own functions, not imported ones): `grid.py` goes from 17 doctest items to 15, and `src/domainia/wilson` from 64 to 72.
* `tests/test_package.py::test_importing_every_module_resolves_to_src_and_does_not_import_jax` walks every module, `domainia.wilson._checks` included. `_checks` imports only `math`, `numbers`, `typing` and NumPy, so the walk stays free of JAX.

With `_checks.py` (Step 2) and Steps 6 and 7 applied, and no test edited, the full suite fails exactly 2 tests: `test_the_scan_covers_every_module_but_filter_and_windows` (`wilson/_checks.py` is not in `SCANNED`) and `test_real_dt_kinds_still_pass[dt5]` (`np.array(0.5)`). No other test passes an array or a `Decimal` as `dt`. So Steps 2 and 4 are the whole pin list.

- [ ] **Step 2: Create `src/domainia/wilson/_checks.py` and list it in `SCANNED`**

Create `src/domainia/wilson/_checks.py` with exactly this text (it is the Interface contract's first row; later tasks rely on every name and message in it):

```python
"""Argument checks shared by :mod:`domainia.wilson` and the rest of domainia.

One rule for the whole package: an argument whose *type* (or an array whose *dtype*) can
never be valid in that place raises ``TypeError``; the right type with a wrong value (range,
shape, finiteness, integrality, an unknown option string) raises ``ValueError``.

Every kind message has the form ``"{name} must be <kind>, got <type or dtype>"``.

Private: this module imports only the standard library and NumPy, so both
``domainia.wilson`` and the core modules can import it without a cycle.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np

#: dtype kinds of a real numeric array: signed and unsigned integers and floats.
#: bool (``"b"``), complex (``"c"``), str/bytes (``"U"``/``"S"``) and object (``"O"``)
#: are wrong kinds wherever a real numeric array is taken.
REAL_KINDS = "iuf"


def require_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``; ``TypeError`` unless it is a Python or NumPy integer.

    ``bool`` and ``np.bool_`` are refused, and so is every float (``2.0`` included): ``int()``
    would truncate ``8.7`` to ``8`` silently, and ``True`` is an ``int`` subclass.

    >>> from domainia.wilson._checks import require_int
    >>> require_int("n", 3)
    3
    >>> require_int("n", 3.0)
    Traceback (most recent call last):
    TypeError: n must be an integer, got float
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def require_real(name: str, value: object) -> float:
    """Return ``value`` as a Python ``float``; ``TypeError`` unless it is a real number.

    Accepted: any :class:`numbers.Real` except ``bool``/``np.bool_`` -- ``int``, ``float``,
    every NumPy integer and floating scalar, :class:`fractions.Fraction`. Refused with
    ``TypeError``: ``bool``, ``str``, ``None``, complex, :class:`decimal.Decimal` and arrays
    of any shape, 0-d included. Range and finiteness are the caller's ``ValueError``; a real
    too large for a float (``10**400``) raises ``ValueError`` here.

    >>> from domainia.wilson._checks import require_real
    >>> require_real("dt", 1)
    1.0
    >>> require_real("dt", "0.5")
    Traceback (most recent call last):
    TypeError: dt must be a real number, got str
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"{name} is too large for a float") from None


def require_bool(name: str, value: object) -> bool:
    """Return ``value`` as a ``bool``; ``TypeError`` unless it is a ``bool`` or ``np.bool_``.

    A truthy ``"no"``, ``1`` or ``0.5`` is refused rather than read as a flag.

    >>> from domainia.wilson._checks import require_bool
    >>> require_bool("restrict", True)
    True
    >>> require_bool("restrict", "no")
    Traceback (most recent call last):
    TypeError: restrict must be a bool, got str
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


def require_str(name: str, value: object) -> str:
    """Return ``value``; ``TypeError`` unless it is a ``str``.

    Only the kind: whether the string is one of the choices stays the caller's
    ``ValueError``, with its own message listing them.

    >>> from domainia.wilson._checks import require_str
    >>> require_str("impl", 3)
    Traceback (most recent call last):
    TypeError: impl must be a string, got int
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


def require_instance(name: str, value: object, cls: type | tuple[type, ...], expected: str) -> Any:
    """Return ``value``; ``TypeError`` unless ``isinstance(value, cls)``.

    ``expected`` is the phrase after "must be", article included
    (``"a TimeDomain"``, ``"a CellSet or a WDMDomain"``).

    >>> from domainia.wilson._checks import require_instance
    >>> require_instance("src", 3, str, "a string domain")
    Traceback (most recent call last):
    TypeError: src must be a string domain, got int
    """
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be {expected}, got {type(value).__name__}")
    return value


def require_callable(name: str, value: object) -> Any:
    """Return ``value``; ``TypeError`` unless it is callable.

    >>> from domainia.wilson._checks import require_callable
    >>> require_callable("psd", 1.0)
    Traceback (most recent call last):
    TypeError: psd must be callable, got float
    """
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value).__name__}")
    return value


def require_dtype(name: str, a: object, kinds: str, expected: str) -> None:
    """``TypeError`` unless the dtype kind of ``a`` is one of ``kinds``.

    Reads ``a.dtype`` without converting ``a`` when it has one, so a JAX array or tracer
    passes through untouched; anything else (a list, a scalar) is read through
    :func:`numpy.asarray`. ``expected`` is the phrase after "must be", article included
    (``"a real numeric array"``). Use :data:`REAL_KINDS` for a real numeric array.

    >>> import numpy as np; from domainia.wilson._checks import require_dtype, REAL_KINDS
    >>> require_dtype("x", np.zeros(2), REAL_KINDS, "a real numeric array")
    >>> require_dtype("x", np.zeros(2, complex), REAL_KINDS, "a real numeric array")
    Traceback (most recent call last):
    TypeError: x must be a real numeric array, got dtype complex128
    """
    dtype = getattr(a, "dtype", None)
    dtype = np.asarray(a).dtype if dtype is None else np.dtype(dtype)
    if dtype.kind not in kinds:
        raise TypeError(f"{name} must be {expected}, got dtype {dtype}")


def index_array(name: str, value: object) -> np.ndarray:
    """Return ``value`` as an ``int64`` ndarray of indices.

    Integer dtypes pass. A float array whose entries are all finite integers (``[2.0]``) is
    accepted and converted. A float entry that is not integral or not finite raises
    ``ValueError``; a bool, str, bytes, object or complex array raises ``TypeError``.
    Range and shape are the caller's checks.

    >>> import numpy as np; from domainia.wilson._checks import index_array
    >>> index_array("m", [1, 2.0])
    array([1, 2])
    >>> index_array("m", [1.5])
    Traceback (most recent call last):
    ValueError: m must hold integer indices, got 1.5
    >>> index_array("m", [True])
    Traceback (most recent call last):
    TypeError: m must be an integer index array, got dtype bool
    """
    raw = np.asarray(value)
    kind = raw.dtype.kind
    if kind in "iu":
        return raw.astype(np.int64, copy=False)
    if kind != "f":
        raise TypeError(f"{name} must be an integer index array, got dtype {raw.dtype}")
    bad = ~np.isfinite(raw) | (raw != np.trunc(raw))
    if bad.any():
        first = raw.ravel()[int(np.flatnonzero(bad.ravel())[0])]
        raise ValueError(f"{name} must hold integer indices, got {float(first)!r}")
    return raw.astype(np.int64)


def validate_dt(dt: object) -> None:
    """Raise unless ``dt`` is a positive finite real number.

    A negative or zero sampling interval produces a nonsensical frequency grid
    (``delta_F`` flips sign or diverges) without raising anywhere, so the
    transform happily returns numbers that mean nothing.

    Parameters
    ----------
    dt : float
        Sampling interval.

    Raises
    ------
    TypeError
        If ``dt`` is not a real number (:func:`require_real`): ``None``, a ``bool``, a
        string, complex, a ``Decimal``, or an array of any shape, 0-d included.
        ``float()`` would take ``True`` as 1.0 and ``"0.5"`` as 0.5 silently, and drop
        the imaginary part of a NumPy complex scalar.
    ValueError
        If ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.validate_dt(0.5)
    >>> wilson.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    >>> wilson.validate_dt(True)
    Traceback (most recent call last):
    TypeError: dt must be a real number, got bool
    """
    value = require_real("dt", dt)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt!r}")


def validate_finite(values: object, context: str) -> None:
    """Raise ``ValueError`` if ``values`` holds a ``nan`` or an ``inf``.

    The WDM transform is *global* within a band: every coefficient is an
    inner product against a window spanning many samples, and the
    implementation routes each band through an FFT.  A single non-finite
    entry therefore does not stay local.  Measured on an $N_f=8$, $N_t=16$
    grid, one ``nan`` anywhere in the input series makes **all 144** output
    coefficients ``nan`` -- and numpy emits no warning on the way, because
    ``nan`` arithmetic inside an FFT is not an invalid operation.  The result
    is a coefficient grid that is entirely meaningless yet raises nothing,
    which is exactly the failure this rejects.

    Detector data with gaps routinely arrives with ``nan`` fills, so this is
    a live path rather than a theoretical one.  The check is one $O(N)$ pass
    against an $O(N\\log N_t)$ transform, so it is not measurable in the
    package's own benchmarks.

    Deliberately stricter than :func:`sanitize_psd`, which maps ``inf`` to
    ``0.0``: an infinite *PSD* sample has a physical reading ("no measurable
    power in this bin"), an infinite *data* sample has none.

    Parameters
    ----------
    values : array-like
        Data, coefficient, or spectrum array.  Complex input is checked on
        both components.
    context : str
        Name used in the error message.

    Raises
    ------
    ValueError
        If any entry is ``nan`` or infinite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.validate_finite([1.0, 2.0], 'x')      # all finite: fine
    >>> wilson.validate_finite([1.0, np.nan], 'x')   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: x: 1 of 2 entries are not finite (1 nan, 0 inf); first at index 1...
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    first = int(np.flatnonzero(bad.ravel())[0])
    raise ValueError(
        f"{context}: {int(np.count_nonzero(bad))} of {arr.size} entries are "
        f"not finite ({n_nan} nan, {int(np.count_nonzero(bad)) - n_nan} inf); "
        f"first at index {first}. A non-finite sample does not stay local -- "
        "it spreads through the transform's FFT to every coefficient, "
        "silently. Fill or excise the gaps before transforming."
    )
```

In `tests/wilson/test_window_threading.py`, `SCANNED` (spec §2.2: `test_the_scan_covers_every_module_but_filter_and_windows` requires every module to be listed, and `_checks.py` calls no window consumer):

```python
           "wilson/special.py", "wilson/__init__.py",
```
→
```python
           "wilson/special.py", "wilson/__init__.py", "wilson/_checks.py",
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q src/domainia/wilson/_checks.py`
Expected: `10 passed` (the doctests).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py -k "scan_covers or drops_the_window"`
Expected: `32 passed, 533 deselected` (31 `SCANNED` modules, then the coverage test).

- [ ] **Step 3: Write the failing tests `tests/wilson/test_checks.py`**

```python
"""The shared argument checks of ``domainia.wilson._checks`` (spec 2026-11-08 §2.2): each helper
returns the value it accepts and refuses every other kind with ``TypeError`` and the message
``"{name} must be <kind>, got <type or dtype>"``; a wrong value is the caller's ``ValueError``."""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from domainia import domains, wilson
from domainia.wilson import grid
from domainia.wilson._checks import (
    REAL_KINDS,
    index_array,
    require_bool,
    require_callable,
    require_dtype,
    require_instance,
    require_int,
    require_real,
    require_str,
    validate_dt,
    validate_finite,
)
from domainia.wilson.filter import nu_d
from domainia.wilson.noise import colored_noise


def _exact(message):
    """A ``match=`` pattern for exactly ``message``."""
    return f"^{re.escape(message)}$"


def test_real_kinds_are_signed_and_unsigned_integers_and_floats():
    assert REAL_KINDS == "iuf"


@pytest.mark.parametrize("value", [3, -2, np.int64(3), np.int32(-2), np.uint8(7)])
def test_require_int_returns_a_python_int(value):
    out = require_int("n", value)
    assert type(out) is int and out == int(value)


@pytest.mark.parametrize(("value", "kind"), [
    (True, "bool"), (np.True_, "bool"), (2.0, "float"), (np.float64(2), "float64"),
    ("2", "str"), (None, "NoneType"), (2 + 0j, "complex"), (np.array(2), "ndarray"),
    (np.array([2]), "ndarray"),
])
def test_require_int_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"n must be an integer, got {kind}")):
        require_int("n", value)


@pytest.mark.parametrize("value", [1, 0.5, np.float32(0.25), np.float64(0.5), np.int64(2), np.uint16(3),
                                   Fraction(1, 2), np.inf])
def test_require_real_returns_a_python_float(value):
    out = require_real("x", value)
    assert type(out) is float and out == float(value)


@pytest.mark.parametrize(("value", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (None, "NoneType"), (1 + 0j, "complex"),
    (np.complex64(1), "complex64"), (Decimal("0.5"), "Decimal"), (np.array(0.5), "ndarray"),
    (np.array([0.5]), "ndarray"),
])
def test_require_real_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"x must be a real number, got {kind}")):
        require_real("x", value)


@pytest.mark.parametrize("value", [10**400, Fraction(10**400)])
def test_require_real_refuses_a_real_too_large_for_a_float_with_value_error(value):
    with pytest.raises(ValueError, match=_exact("x is too large for a float")):
        require_real("x", value)


@pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False), (np.True_, True),
                                                 (np.False_, False)])
def test_require_bool_returns_a_python_bool(value, expected):
    out = require_bool("flag", value)
    assert type(out) is bool and out is expected


@pytest.mark.parametrize(("value", "kind"), [
    ("no", "str"), (1, "int"), (0, "int"), (np.int64(1), "int64"), (0.5, "float"), (None, "NoneType"),
    (np.array(True), "ndarray"),
])
def test_require_bool_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"flag must be a bool, got {kind}")):
        require_bool("flag", value)


@pytest.mark.parametrize("value", ["fft", "", np.str_("dense")])
def test_require_str_returns_the_string(value):
    assert require_str("impl", value) is value


@pytest.mark.parametrize(("value", "kind"), [
    (3, "int"), (None, "NoneType"), (b"fft", "bytes"), (["fft"], "list"), (np.float64(1), "float64"),
])
def test_require_str_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"impl must be a string, got {kind}")):
        require_str("impl", value)


def test_require_instance_checks_against_the_class_passed_in():
    g = wilson.Grid(8, 16, 1.0)
    assert require_instance("grid", g, wilson.Grid, "a domainia.wilson.Grid") is g
    assert require_instance("x", 2.5, (int, float), "a Python int or float") == 2.5
    with pytest.raises(TypeError, match=_exact("grid must be a domainia.wilson.Grid, got tuple")):
        require_instance("grid", (8, 16, 1.0), wilson.Grid, "a domainia.wilson.Grid")
    with pytest.raises(TypeError, match=_exact("x must be a Python int or float, got str")):
        require_instance("x", "2.5", (int, float), "a Python int or float")


def test_require_callable():
    assert require_callable("psd", np.ones_like) is np.ones_like
    assert require_callable("psd", wilson.Grid) is wilson.Grid
    with pytest.raises(TypeError, match=_exact("psd must be callable, got float")):
        require_callable("psd", 1.0)
    with pytest.raises(TypeError, match=_exact("psd must be callable, got NoneType")):
        require_callable("psd", None)


@pytest.mark.parametrize("a", [[1, 2], [0.5], 1.0, np.float64(1.0), np.zeros(2), np.zeros(2, np.float32),
                               np.zeros((2, 3), np.int8), np.zeros(2, np.uint64)])
def test_require_dtype_passes_a_real_numeric_list_scalar_or_array(a):
    assert require_dtype("x", a, REAL_KINDS, "a real numeric array") is None


@pytest.mark.parametrize(("a", "dtype"), [
    ([1j], "complex128"), ([True], "bool"), (["a"], "<U1"), (1j, "complex128"),
    (np.complex64(1), "complex64"), (np.zeros(2, complex), "complex128"), (np.zeros(2, bool), "bool"),
    (np.array(["a"]), "<U1"), (np.array([b"a"]), "|S1"), (np.array([None]), "object"),
])
def test_require_dtype_refuses_a_dtype_outside_the_kinds(a, dtype):
    with pytest.raises(TypeError, match=_exact(f"x must be a real numeric array, got dtype {dtype}")):
        require_dtype("x", a, REAL_KINDS, "a real numeric array")


def test_require_dtype_takes_any_kinds_and_reads_dtype_without_converting():
    assert require_dtype("keep", np.ones(3, bool), "b", "a bool array") is None
    with pytest.raises(TypeError, match=_exact("keep must be a bool array, got dtype float64")):
        require_dtype("keep", np.ones(3), "b", "a bool array")

    class Unconvertible:
        dtype = np.dtype(np.float32)

        def __array__(self, *args, **kwargs):
            raise AssertionError("require_dtype converted an object that has a dtype")

    assert require_dtype("x", Unconvertible(), REAL_KINDS, "a real numeric array") is None


@pytest.mark.jax
def test_require_dtype_passes_a_jax_array_and_a_jit_tracer_through_untouched(jax_be):
    """A conversion would show: ``np.asarray`` on a tracer raises ``TracerArrayConversionError``,
    a ``TypeError`` without the helper's message."""
    import jax
    import jax.numpy as jnp

    from domainia.wilson.backend import is_tracer

    x = jnp.arange(3.0)
    assert require_dtype("x", x, REAL_KINDS, "a real numeric array") is None
    with pytest.raises(TypeError, match=_exact("x must be a real numeric array, got dtype complex128")):
        require_dtype("x", x + 0j, REAL_KINDS, "a real numeric array")
    traced = []

    @jax.jit
    def double(v):
        require_dtype("v", v, REAL_KINDS, "a real numeric array")
        traced.append(is_tracer(v))
        return 2.0 * v

    out = double(x)
    assert traced == [True] and isinstance(out, jax.Array)
    assert np.array_equal(np.asarray(out), [0.0, 2.0, 4.0])
    with pytest.raises(TypeError) as info:              # raised while tracing; JAX adds a note
        double(x + 0j)
    assert str(info.value) == "v must be a real numeric array, got dtype complex128"


@pytest.mark.parametrize(("value", "expected"), [
    ([1, 2], [1, 2]), (np.array([1, 2], np.int32), [1, 2]), (np.array([3], np.uint8), [3]),
    ([1, 2.0], [1, 2]), (np.array([[0.0, 4.0]], np.float32), [[0, 4]]), (3, 3), ([], []),
])
def test_index_array_returns_int64(value, expected):
    out = index_array("m", value)
    assert isinstance(out, np.ndarray) and out.dtype == np.int64
    assert out.shape == np.shape(expected) and np.array_equal(out, expected)


@pytest.mark.parametrize(("value", "first"), [
    ([1.5], "1.5"), ([2.9999999], "2.9999999"), ([0.0, np.nan], "nan"), ([np.inf], "inf"),
    ([[1.0], [-0.5]], "-0.5"),
])
def test_index_array_refuses_a_non_integral_or_non_finite_float_with_value_error(value, first):
    with pytest.raises(ValueError, match="integer") as info:
        index_array("m", value)
    assert str(info.value) == f"m must hold integer indices, got {first}"


@pytest.mark.parametrize(("value", "dtype"), [
    ([True], "bool"), (np.array([1, 0], bool), "bool"), ([1j], "complex128"), (["1"], "<U1"),
    ([b"1"], "|S1"), (np.array([1], object), "object"),
])
def test_index_array_refuses_a_non_numeric_dtype_with_type_error(value, dtype):
    with pytest.raises(TypeError, match=_exact(f"m must be an integer index array, got dtype {dtype}")):
        index_array("m", value)


def test_validate_dt_and_validate_finite():
    assert validate_dt(np.float32(0.5)) is None and validate_dt(Fraction(1, 2)) is None
    with pytest.raises(TypeError, match=_exact("dt must be a real number, got Decimal")):
        validate_dt(Decimal("0.5"))
    with pytest.raises(ValueError, match=_exact("dt must be a positive finite number, got -1")):
        validate_dt(-1)
    with pytest.raises(ValueError, match=_exact("dt is too large for a float")):
        validate_dt(10**400)
    assert validate_finite([1.0, 2.0], "x") is None and validate_finite([], "x") is None
    with pytest.raises(ValueError, match=r"^y: 1 of 3 entries are not finite \(0 nan, 1 inf\); "
                                         r"first at index 2\. "):
        validate_finite([0.0, 1j, complex(1.0, np.inf)], "y")


def test_grid_re_exports_the_checks_under_their_old_names():
    """``wilson.grid`` keeps ``_require_int``, ``validate_dt`` and ``validate_finite`` (imported
    from ``grid`` by ``matrix.py``, ``evolutionary.py`` and the tests), and its ``__all__``."""
    assert grid._require_int is require_int
    assert grid.validate_dt is validate_dt is wilson.validate_dt
    assert grid.validate_finite is validate_finite is wilson.validate_finite
    assert {"validate_dt", "validate_finite", "validate_grid", "n_samples"} <= set(grid.__all__)
    assert not [name for name in grid.__all__ if name.startswith("_")]


def _noise(N):
    return colored_noise(N, 1.0, np.ones_like, np.random.default_rng(0))


@pytest.mark.parametrize(("call", "message"), [
    (lambda: grid._require_int("N", 8.0), "N must be an integer, got float"),
    (lambda: wilson.validate_grid(8, np.float64(16)), "Nt must be an integer, got float64"),
    (lambda: wilson.n_samples(True, 16), "Nf must be an integer, got bool"),
    (lambda: nu_d(np.array([0.5]), d="4"), "d must be an integer, got str"),
    (lambda: _noise(np.array(8)), "N must be an integer, got ndarray"),
    (lambda: wilson.Meyer(2.5), "Meyer d must be an integer, got float"),
    (lambda: domains.TimeDomain(8.0, 1.0), "N must be an integer, got float"),
], ids=["_require_int", "validate_grid", "n_samples", "nu_d", "colored_noise", "Meyer", "domains._int"])
def test_the_seven_integer_checks_raise_the_helpers_message(call, message):
    """The seven copies of the integer check (spec 2026-11-08, Context) call ``require_int``."""
    with pytest.raises(TypeError, match=_exact(message)):
        call()


def test_the_range_checks_after_require_int_are_unchanged():
    assert type(wilson.Meyer(np.int64(3)).d) is int and wilson.Meyer(np.int64(3)) == wilson.Meyer(3)
    with pytest.raises(ValueError, match=_exact("Meyer d must be >= 1; got 0")):
        wilson.Meyer(0)
    assert type(domains._int("k0", np.int64(2), minimum=0)) is int
    with pytest.raises(ValueError, match=_exact("k0 must be >= 0; got -1")):
        domains._int("k0", -1, minimum=0)
    with pytest.raises(TypeError, match=_exact("k0 must be an integer, got bool")):
        domains._int("k0", np.True_, minimum=0)
    with pytest.raises(ValueError, match=_exact("d must be >= 1, got 0")):
        nu_d(np.array([0.5]), d=0)
    with pytest.raises(ValueError, match=_exact("N must be even, got 7")):
        _noise(7)
    assert _noise(np.int64(8)).shape == (5,)
```

The JAX test follows `tests/wilson`'s pattern: the `jax_be` fixture (`tests/wilson/conftest.py`) skips it when JAX cannot be imported, and the `jax` marker is the one `pyproject.toml` registers. Inside `jax.jit` the complex case raises at trace time; JAX adds a note to the exception, so that case compares `str(info.value)` instead of an anchored `match=` (pytest matches the notes too).

- [ ] **Step 4: `tests/wilson/test_wrong_kinds.py`: the `dt` kinds of spec §2.4**

The imports:

```python
from __future__ import annotations

import numpy as np
```
→
```python
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
```

The refused list gains `Decimal("0.5")`, `np.array(0.5)` (moved from the accepted list) and `np.array([0.5])`:

```python
@pytest.mark.parametrize(("dt", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (None, "NoneType"),
    (1.0 + 0.0j, "complex"), (np.complex64(1.0), "complex64"), (np.array(1.0 + 0.0j), "ndarray"),
])
def test_dt_of_the_wrong_kind_raises_type_error(dt, kind):
    """``float()`` used to accept ``True`` (as 1.0), ``"0.5"`` and a NumPy complex scalar."""
```
→
```python
@pytest.mark.parametrize(("dt", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (None, "NoneType"),
    (1.0 + 0.0j, "complex"), (np.complex64(1.0), "complex64"), (np.array(1.0 + 0.0j), "ndarray"),
    (Decimal("0.5"), "Decimal"), (np.array(0.5), "ndarray"), (np.array([0.5]), "ndarray"),
])
def test_dt_of_the_wrong_kind_raises_type_error(dt, kind):
    """``float()`` used to accept ``True`` (as 1.0), ``"0.5"`` and a NumPy complex scalar; the
    old check also let a ``Decimal``, a 0-d array and a size-1 array through."""
```

The accepted list loses `np.array(0.5)` and gains `Fraction(1, 2)`. `np.int64(2)` is already there, so the NumPy integer kind is pinned and no `np.int64(1)` is added:

```python
@pytest.mark.parametrize("dt", [0.5, 1, np.float32(0.1), np.float64(0.25), np.int64(2), np.array(0.5)])
def test_real_dt_kinds_still_pass(dt):
```
→
```python
@pytest.mark.parametrize("dt", [0.5, 1, np.float32(0.1), np.float64(0.25), np.int64(2), Fraction(1, 2)])
def test_real_dt_kinds_still_pass(dt):
```

- [ ] **Step 5: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_checks.py tests/wilson/test_wrong_kinds.py`
Expected: `7 failed, 137 passed, 1 warning`:
- `test_checks.py`, 4 failed:
  - `test_grid_re_exports_the_checks_under_their_old_names`: `assert <function _require_int at 0x...> is require_int` (`grid.py` still has its own copies);
  - `test_the_seven_integer_checks_raise_the_helpers_message[Meyer]`: the old message `Meyer d must be an int; got float`;
  - `test_the_seven_integer_checks_raise_the_helpers_message[domains._int]`: `N must be an int; got float`;
  - `test_the_range_checks_after_require_int_are_unchanged`: `k0 must be an int; got bool`.
- `test_wrong_kinds.py`, 3 failed: `test_dt_of_the_wrong_kind_raises_type_error[dt7-Decimal]`, `[dt8-ndarray]` and `[dt9-ndarray]`, each `Failed: DID NOT RAISE <class 'TypeError'>`.
- The warning is NumPy's `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated`, from the old `validate_dt`'s `float(dt)` on `np.array([0.5])` (`wilson/grid.py:199`).

The other helper tests pass already: they test `_checks.py`, which Step 2 added. The other 5 integer sites pass too, because their message is the helper's already.

- [ ] **Step 6: `wilson/grid.py` re-exports the checks**

The import (the Interface contract's second row):

```python
import numpy as np
from numpy.typing import NDArray

from .backend import Backend, get_backend
```
→
```python
import numpy as np
from numpy.typing import NDArray

from ._checks import require_int as _require_int, validate_dt, validate_finite
from .backend import Backend, get_backend
```

Delete `_require_int`:

```python
def _require_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``, raising ``TypeError`` if it is not one
    (``bool`` is rejected too)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _require_even_positive_int(name: str, value: object) -> int:
```
→
```python
def _require_even_positive_int(name: str, value: object) -> int:
```

`validate_grid`'s loop calls it (the messages are unchanged):

```python
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(val).__name__}")
        if val <= 0:
```
→
```python
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        _require_int(name, val)
        if val <= 0:
```

Delete `validate_dt` and `validate_finite`, whole, up to `sanitize_psd`:

```python
def validate_dt(dt: float) -> None:
    """Raise unless ``dt`` is a positive finite real number.

    A negative or zero sampling interval produces a nonsensical frequency grid
    (``delta_F`` flips sign or diverges) without raising anywhere, so the
    transform happily returns numbers that mean nothing.

    Parameters
    ----------
    dt : float
        Sampling interval.

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
    >>> import numpy as np; from domainia import wilson
    >>> wilson.validate_dt(0.5)
    >>> wilson.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    >>> wilson.validate_dt(True)
    Traceback (most recent call last):
    TypeError: dt must be a real number, got bool
    """
    kind = getattr(getattr(dt, "dtype", None), "kind", None)
    if dt is None or isinstance(dt, (bool, str, bytes, complex)) or kind in ("b", "c", "S", "U"):
        raise TypeError(f"dt must be a real number, got {type(dt).__name__}")
    value = float(dt)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt!r}")


def validate_finite(values: object, context: str) -> None:
    """Raise ``ValueError`` if ``values`` holds a ``nan`` or an ``inf``.

    The WDM transform is *global* within a band: every coefficient is an
    inner product against a window spanning many samples, and the
    implementation routes each band through an FFT.  A single non-finite
    entry therefore does not stay local.  Measured on an $N_f=8$, $N_t=16$
    grid, one ``nan`` anywhere in the input series makes **all 144** output
    coefficients ``nan`` -- and numpy emits no warning on the way, because
    ``nan`` arithmetic inside an FFT is not an invalid operation.  The result
    is a coefficient grid that is entirely meaningless yet raises nothing,
    which is exactly the failure this rejects.

    Detector data with gaps routinely arrives with ``nan`` fills, so this is
    a live path rather than a theoretical one.  The check is one $O(N)$ pass
    against an $O(N\\log N_t)$ transform, so it is not measurable in the
    package's own benchmarks.

    Deliberately stricter than :func:`sanitize_psd`, which maps ``inf`` to
    ``0.0``: an infinite *PSD* sample has a physical reading ("no measurable
    power in this bin"), an infinite *data* sample has none.

    Parameters
    ----------
    values : array-like
        Data, coefficient, or spectrum array.  Complex input is checked on
        both components.
    context : str
        Name used in the error message.

    Raises
    ------
    ValueError
        If any entry is ``nan`` or infinite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.validate_finite([1.0, 2.0], 'x')      # all finite: fine
    >>> wilson.validate_finite([1.0, np.nan], 'x')   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: x: 1 of 2 entries are not finite (1 nan, 0 inf); first at index 1...
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    first = int(np.flatnonzero(bad.ravel())[0])
    raise ValueError(
        f"{context}: {int(np.count_nonzero(bad))} of {arr.size} entries are "
        f"not finite ({n_nan} nan, {int(np.count_nonzero(bad)) - n_nan} inf); "
        f"first at index {first}. A non-finite sample does not stay local -- "
        "it spreads through the transform's FFT to every coefficient, "
        "silently. Fill or excise the gaps before transforming."
    )


def sanitize_psd(values: object, context: str) -> Array:
```
→
```python
def sanitize_psd(values: object, context: str) -> Array:
```

`n_samples`' loop calls `_require_int` (the comment stays):

```python
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            # int() would silently truncate: n_samples(8.7, 16.2) == 128, and
            # bool is an int subclass, so n_samples(True, 16) would give 16.
            raise TypeError(f"{name} must be an integer, got {type(val).__name__}")
    return int(Nf) * int(Nt)
```
→
```python
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        # int() would silently truncate: n_samples(8.7, 16.2) == 128, and
        # bool is an int subclass, so n_samples(True, 16) would give 16.
        _require_int(name, val)
    return int(Nf) * int(Nt)
```

`grid.__all__` is unchanged: it still lists `validate_dt` and `validate_finite`, now imported names. `numpy` stays imported (`delta_F` and the type aliases use it).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "def _require_int\|def validate_dt\|def validate_finite\|_checks" src/domainia/wilson/grid.py`
Expected: one line, `25:from ._checks import require_int as _require_int, validate_dt, validate_finite`.

- [ ] **Step 7: The other four copies call `require_int`**

`wilson/filter.py`, the import:

```python
from scipy.special import betainc

from .grid import delta_F, validate_dt, validate_grid
```
→
```python
from scipy.special import betainc

from ._checks import require_int
from .grid import delta_F, validate_dt, validate_grid
```

and `nu_d`'s check:

```python
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise TypeError(f"d must be an integer, got {type(d).__name__}")
    if d < 1:
```
→
```python
    require_int("d", d)
    if d < 1:
```

`wilson/noise.py`, the import:

```python
from numpy.typing import NDArray

from .grid import (
    band_frequencies,
```
→
```python
from numpy.typing import NDArray

from ._checks import require_int
from .grid import (
    band_frequencies,
```

and `colored_noise`'s check (the comment above it, "They only raise, so valid input takes the oracle's path bit for bit", stays true: the call does not rebind `N`):

```python
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        raise TypeError(f"N must be an integer, got {type(N).__name__}")
    if N <= 0:
```
→
```python
    require_int("N", N)
    if N <= 0:
```

`wilson/windows.py`, the import:

```python
from numpy.typing import NDArray

from .filter import phi_centered
```
→
```python
from numpy.typing import NDArray

from ._checks import require_int
from .filter import phi_centered
```

and `Meyer.__post_init__` (the `ValueError` text is unchanged):

```python
    def __post_init__(self) -> None:
        if isinstance(self.d, bool) or not isinstance(self.d, (int, np.integer)):
            raise TypeError(f"Meyer d must be an int; got {type(self.d).__name__}")
        if self.d < 1:
            raise ValueError(f"Meyer d must be >= 1; got {self.d}")
        object.__setattr__(self, "d", int(self.d))
```
→
```python
    def __post_init__(self) -> None:
        d = require_int("Meyer d", self.d)
        if d < 1:
            raise ValueError(f"Meyer d must be >= 1; got {self.d}")
        object.__setattr__(self, "d", d)
```

`domains.py`, the import:

```python
from domainia import wilson
from domainia.wilson.grid import packed_cells
```
→
```python
from domainia import wilson
from domainia.wilson._checks import require_int
from domainia.wilson.grid import packed_cells
```

and `_int` (the Interface contract's third row):

```python
def _int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int; got {type(value).__name__}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return int(value)
```
→
```python
def _int(name: str, value: object, *, minimum: int) -> int:
    v = require_int(name, value)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return v
```

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "not isinstance(.*(int, np.integer))" src`
Expected: one line, `src/domainia/wilson/_checks.py:40`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "must be an int([^e]|$)" src`
Expected: no output.

- [ ] **Step 8: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_checks.py tests/wilson/test_wrong_kinds.py src/domainia/wilson/_checks.py`
Expected: `154 passed` (103 + 41 + the 10 doctests), no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_checks.py tests/wilson/test_wrong_kinds.py src/domainia/wilson/_checks.py`
Expected: `153 passed, 1 skipped` (the JAX test skips).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q src/domainia/wilson tests/test_package.py`
Expected: `78 passed` (the 72 doctests of `domainia.wilson` and the 6 package tests).

- [ ] **Step 9: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, with no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, with no warnings summary.

- [ ] **Step 10: Commit** (sandbox disabled for both git commands)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/wilson/_checks.py src/domainia/wilson/grid.py src/domainia/wilson/filter.py src/domainia/wilson/noise.py src/domainia/wilson/windows.py src/domainia/domains.py tests/wilson/test_checks.py tests/wilson/test_wrong_kinds.py tests/wilson/test_window_threading.py
git commit -m "domainia.wilson._checks: the shared argument checks

A private module holds the kind checks that the rest of the plan uses
(require_int, require_real, require_bool, require_str, require_instance,
require_callable, require_dtype, index_array, REAL_KINDS), and validate_dt
and validate_finite, moved from wilson.grid. grid re-exports _require_int,
validate_dt and validate_finite under their old names. The seven copies of
the integer check call require_int, so Meyer d and domains._int now say
\"must be an integer, got X\". validate_dt refuses a Decimal and arrays of
any shape, 0-d included (spec 2026-11-08 §2.2, §2.4).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `cd /Users/aaron/Documents/lisa/domainia && git status --short`
Expected: no line for any file of this task.

---

### Task 2: Wrong kinds in domainia.wilson (spec §2.3–§2.5, §3.5)

**Goal:** The rest of `domainia.wilson` follows the rule: a wrong kind raises `TypeError` with a `", got"` message, and a wrong value raises `ValueError`. The integer arguments that `int()`, `range()` or a parity test took silently use `require_int`. The band lists of the band kernels use `index_array`. `eq45_plan`'s `t_max` uses `require_real`. `include_logdet` uses `require_bool`. The `cells`, `backend` and `impl` options use `require_str` before their membership `ValueError`. The five "never complex" checks of spec §3.5 become `require_dtype(..., REAL_KINDS, ...)` and raise `TypeError`. Every `TypeError` message in `wilson/` has the `", got"` form.

**Assumes the tree after Task 1.** Task 1 added `src/domainia/wilson/_checks.py`, the `_checks` import line of `grid.py` (`validate_dt`, `validate_finite` and `_require_int` are no longer defined there), and a `from ._checks import require_int` line in `filter.py`, `noise.py` and `windows.py`. Task 1 did not touch `transform.py`, `stationary.py`, `evolutionary.py`, `inner_product.py` or `backend.py`, so the spec's `file:line` references hold there. In `grid.py` they have moved (Task 1 deleted about 100 lines). The old texts below are Task 1's result.

**Files:**
- Modify: `src/domainia/wilson/transform.py` (the `_checks` import; `inverse` (Raises, one doctest, the `wmn` check); `inverse_many` (the `wmn` check); `atom_spectrum`; `_check_interior_bands`; `forward_bands`; `inverse_bands`)
- Modify: `src/domainia/wilson/grid.py` (one new `_checks` import line; `sanitize_psd` (text, Raises, one doctest, the check); `carrier_phase`; `edge_columns`)
- Modify: `src/domainia/wilson/stationary.py` (the `_checks` import; the Raises of `_normalize_noise_to_sigma2`; `_reject_complex_psd`)
- Modify: `src/domainia/wilson/noise.py` (`empirical_variance`), `src/domainia/wilson/evolutionary.py` (the `_checks` import; `eq45_plan`: Raises, `keep`, `t_max`)
- Modify: `src/domainia/wilson/inner_product.py` (the `_checks` import; `_cell_mask`; `loglike_whittle_many`, `loglike_whittle`, `loglike_gaussian_many`, `loglike_gaussian`)
- Modify: `src/domainia/wilson/backend.py` (the `_checks` import; `_NumpyBackend.cholesky`, `_JaxBackend.cholesky`, `get_backend`)
- Modify: `src/domainia/wilson/windows.py` (the `_checks` import; `_check_name`, `_check_callable`, `require_window`)
- Modify: `tests/wilson/test_wrong_kinds.py` (`import re`; the new tests appended), `tests/wilson/test_grid_bands.py` (`test_inverse_bands_rejects_complex_and_non_finite_cells`), `tests/wilson/test_noise_stationary.py` (`test_complex_psd_is_rejected`)

**Acceptance Criteria:**
- [ ] The 9 integer slots (`forward_bands` `k0`; `inverse_bands` `k0` and `n_bins`; `atom_spectrum` `m` and `n`; `grid.carrier_phase` `m`; `grid.edge_columns` `lead` and `trail`; `empirical_variance` `n_realizations`) raise `TypeError("{name} must be an integer, got {type}")` for `2.0`, `np.float64(2)`, `True`, `np.True_`, `"2"` and `None`. An `np.int64` gives the same result as the `int`. Their range `ValueError`s keep their text.
- [ ] The bands of `band_atoms`, `forward_bands` and `inverse_bands` go through `index_array` in `_check_interior_bands`: a bool, str, complex or object array raises `TypeError("bands must be an integer index array, got dtype {dtype}")`; `[2.5, 3.0]` and `[2.0, nan]` raise `ValueError("bands must hold integer indices, got 2.5")` and `ValueError("bands must hold integer indices, got nan")` (the first was truncated silently, the second raised NumPy's own error); `[2.0, 3.0]` gives the same result as `[2, 3]`.
- [ ] `eq45_plan`'s `t_max` raises `TypeError("t_max must be a real number, got {type}")` for a str, a bool, complex, a `Decimal` and a 0-d array, and is stored as a Python `float`. `keep`'s message reads `keep must be a boolean mask over the packed cells, got dtype int64`.
- [ ] `include_logdet` of the six loglike functions raises `TypeError("include_logdet must be a bool, got {type}")` unless it is a `bool` or `np.bool_`.
- [ ] `cells` (`inner_product`), `backend` (`get_backend`) and `impl` (both `Backend.cholesky`) raise `TypeError("{name} must be a string, got {type}")` for a non-string. An unknown string keeps its `ValueError` text. `get_backend` still takes `None` and a `Backend`.
- [ ] Spec §3.5: `sanitize_psd`, `_reject_complex_psd`, `inverse`/`inverse_many`'s `wmn` and `inverse_bands`' `w` raise `TypeError("{name} must be a real numeric array, got dtype {dtype}")` for a complex, bool, str or object array. Integer and float arrays pass. The two `WindowError` checks in `windows.py` are unchanged. The Raises sections and doctests say `TypeError`.
- [ ] Every `raise TypeError` in `src/domainia/wilson` has the `", got"` form (Step 11's snippet prints nothing), and `grep -rn "iscomplexobj" src/domainia/wilson` prints nothing.
- [ ] Under `jax.jit` the dtype checks read a tracer's dtype: the `jax`-marked test passes, and so do the existing jit tests.
- [ ] `tests/wilson/test_wrong_kinds.py`: `254 passed` (NumPy lane: `249 passed, 5 skipped`). `src/domainia/wilson` still has 72 doctest items.
- [ ] The full suite and the NumPy lane pass, with no warnings summary. The float64 bit gate prints `0 of 1504 arrays differ`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py tests/wilson/test_grid_bands.py tests/wilson/test_noise_stationary.py` → `349 passed, 1 skipped`; then the bit gate (Step 14) → `0 of 1504 arrays differ`

**Steps:**

- [ ] **Step 1: Grep and probe the sites in `src/domainia/wilson`**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "int(k0)\|np.asarray(bands, dtype=int)\|int(n_realizations)\|float(T if t_max\|keep.dtype != np.bool_\|iscomplexobj" src/domainia/wilson/*.py`
Expected: 10 hits.

```text
src/domainia/wilson/evolutionary.py:477:    if keep.dtype != np.bool_:   # an index array (e.g. np.arange(N)) would silently drop cell 0
src/domainia/wilson/evolutionary.py:481:    t_max = float(T if t_max is None else t_max)
src/domainia/wilson/grid.py:206:    if np.iscomplexobj(arr):
src/domainia/wilson/noise.py:443:    if int(n_realizations) < 1:
src/domainia/wilson/stationary.py:253:    if np.iscomplexobj(values):
src/domainia/wilson/transform.py:707:    if be.xp.iscomplexobj(wmn):
src/domainia/wilson/transform.py:771:    if be.xp.iscomplexobj(wmn):
src/domainia/wilson/transform.py:1048:    bands = np.asarray(bands, dtype=int).ravel()
src/domainia/wilson/transform.py:1104:    ell = (bands[:, None] - 1) * (Nt // 2) + np.arange(Nt)[None, :] - int(k0)
src/domainia/wilson/transform.py:1199:    if be.xp.iscomplexobj(w):
```

Every hit changes in Steps 6-8, except `_band_windows`' `int(k0)` (`transform.py:1104`): `forward_bands` and `inverse_bands` now check `k0` before they call it, so it stays. The other integer sites have no conversion to grep for: `atom_spectrum` compares `m` and `n` and looks them up in a dict, `carrier_phase` tests `m % 2`, and `edge_columns` passes `lead` and `trail` to `range()`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "include_logdet: bool" src/domainia/wilson/inner_product.py`
Expected: 6 hits, at lines 662 (`loglike_whittle_many`), 733 (`loglike_whittle`), 792 (`loglike_band_averaged`), 838 (`loglike_evolutionary_whittle`), 887 (`loglike_gaussian_many`) and 965 (`loglike_gaussian`). `loglike_band_averaged` and `loglike_evolutionary_whittle` pass the flag straight on to `loglike_whittle`, so the check goes into the other four (Step 9) and covers all six.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -c "; got" src/domainia/wilson/*.py | sort`
Expected: 13 lines, one per file, with counts that sum to 43: `backend.py` 6, `evolutionary.py` 1, `grid.py` 2, `inner_product.py` 9, `matrix.py` 2, `noise.py` 3, `stationary.py` 10, `transform.py` 5, `windows.py` 5, and 0 in `__init__.py`, `_checks.py`, `filter.py` and `special.py`. Only four are inside a `raise TypeError`, all in `windows.py`: lines 141 (`window name must be a non-empty string`), 146 (`must be callable`), 151 (the second line of `must be hashable`) and 286 (the second line of `require_window`'s non-window message). Line 179 is `Meyer`'s `ValueError`. Step 10 converts them. Every other hit is a `ValueError` and keeps its text, including `cells must be 'active' or 'interior'; got`, `impl must be 'numpy' or 'scipy'; got` and `backend must be 'numpy', 'jax' or a Backend; got`, which stay the membership errors after the new kind checks.

Write the snippet `$TMPDIR/wilson_typeerror_gate.py` (scratch, outside the repo). It prints every `raise TypeError` in `src/domainia/wilson` whose source text lacks `", got"` or has `"; got"`:

```python
import ast
import pathlib

for p in sorted(pathlib.Path("src/domainia/wilson").glob("*.py")):
    src = p.read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "TypeError"):
            text = ast.get_source_segment(src, node)
            if "; got" in text or ", got" not in text:
                print(f"{p}:{node.lineno}")
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/wilson_typeerror_gate.py`
Expected: 6 lines.

```text
src/domainia/wilson/evolutionary.py:478
src/domainia/wilson/windows.py:141
src/domainia/wilson/windows.py:146
src/domainia/wilson/windows.py:282
src/domainia/wilson/windows.py:285
src/domainia/wilson/windows.py:150
```

`evolutionary.py:478` (`keep must be a boolean mask over the packed cells`) and `windows.py:282` (`window must be a Window instance, not the class {obj.__qualname__}: pass an instance, e.g. Meyer() or domainia.wilson.MEYER`) have no `"got"` at all; Steps 8 and 10 give them the form.

What the sites do today (probed on the tree after Task 1):
- `forward_bands` with `k0="0"`, `k0=0.0` or `k0=False` runs (`int(k0)`); `inverse_bands` takes `k0=True`. `inverse_bands`' `n_bins=True` gets the misleading `ValueError` `the bins [0, 1) do not cover the band windows`, and `n_bins=65.0` NumPy's `'float' object cannot be interpreted as an integer`.
- `np.asarray(bands, dtype=int)` truncates `[2.5]` to band 2, takes `[True]` as band 1 and parses `["2"]`.
- `atom_spectrum(2.0, 3.0, 8, 16, 1.0)` and `atom_spectrum(True, True, 8, 16, 1.0)` run. `carrier_phase(2.5, 4)` and `carrier_phase(True, 4)` run. `edge_columns(16, True, True)` runs; `lead=2.0` raises Python's `range()` error.
- `empirical_variance` with `n_realizations=True` runs one realization; `2.0` raises Python's `range()` error.
- `eq45_plan(8, 16, 1.0, t_max="300")` and `t_max=np.array(300.0)` run.
- Every `include_logdet` takes `"no"`, `0`, `1`, `0.5` and `None` as a flag (`"no"` includes the log-determinant).
- `cells`, `backend` and `impl` report a non-string as the membership `ValueError` (`cells must be 'active' or 'interior'; got 1`).
- `inverse`, `inverse_many` and `inverse_bands` take a bool grid silently, and raise NumPy's `ufunc 'isfinite' not supported` for a str or object grid. `sanitize_psd` casts a bool, str or object PSD to float silently. Complex raises `ValueError` at all five sites.

- [ ] **Step 2: Grep the tests that pin these sites**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "real-valued\|must be real\|wmn must" tests --include='*.py' | grep -v "tests/wilson/oracle"`
Expected: 6 hits.
- `tests/wilson/test_grid_bands.py:300` (`pytest.raises(ValueError, match="w must be real")`) and `tests/wilson/test_noise_stationary.py:236` (`pytest.raises(ValueError, match="real-valued")`): the spec's two pins. Step 4 moves both to `TypeError`.
- `tests/test_translate.py:199`, `:201`, `:251`, `:270`: `translate`'s own DC/Nyquist-bin checks (`bin 0 (DC) must be real`), not wilson's. They stay.

No test passes a complex `wmn` to `inverse`, `inverse_many` or `inverse_time(_many)`, or a complex array to `WDMToTime`: the spec's "`wmn`-complex tests the grep finds" are none. Step 3 adds them.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "cells must be\|impl must be\|backend must be\|boolean mask\|must be callable\|non-empty string\|hashable\|pass an instance\|window must be" tests --include='*.py' | grep -v "tests/wilson/oracle"`
Expected: 20 hits. None needs an edit:
- `tests/wilson/test_inner_product_fixes.py:92` (`cells="edges"`), `tests/wilson/test_backend.py:51` (`backend="torch"`), `tests/test_dom_cov_api.py:130` (`chol_impl="lapack"`, Task 5's): unknown strings, which stay `ValueError` with the same text.
- `tests/wilson/test_evolutionary_kernels.py:186` (`match="boolean mask"`): the new `keep` message still contains it.
- `tests/wilson/test_windows.py:302` (`hashable.*module-level function or a frozen dataclass`) and `:308` (`Window.*pass an instance, e\.g\. Meyer\(\)`): the new messages keep both phrases in that order; `:301` is the test's name. `tests/wilson/test_window_threading.py:358` (`match="window must be"`): the new non-window message still starts so.
- `tests/test_dom_windows.py:55`, `:58`, `:80`, `:81`, `:82`: `WDMDomain`'s own hashable check (`domains.py:380`, Task 3's), not `windows.py`'s.
- `tests/test_dom_evolutionary_cov.py:101`, `tests/test_dom_variance_generic.py:77` (`cells must be a CellSet or a WDMDomain; got Grid`): `domains.py`'s message, Task 3's.
- `tests/test_translate_coordinates.py:159`: `translate`'s channel names (Task 6's).
- `tests/wilson/test_checks.py:106`, `:123`, `:125`: Task 1's helper tests, unchanged.
- `tests/wilson/test_window_threading.py:579`: a comment.

A bare `pytest.raises(ValueError)` around a site would not show in a grep. In the scratch prototype, the full suite with Steps 6-10 applied and no test edited failed exactly 6 tests: `test_inverse_bands_rejects_complex_and_non_finite_cells[numpy]` and `[jax]`, and the four `test_complex_psd_is_rejected` cases (the NumPy lane: 5, without the `[jax]` leg). So Step 4 is the whole pin list.

- [ ] **Step 3: Append the failing tests to `tests/wilson/test_wrong_kinds.py`**

The imports gain `re`:

```python
from __future__ import annotations

from decimal import Decimal
```
→
```python
from __future__ import annotations

import re
from decimal import Decimal
```

Then append this block at the end of the file, after `test_eq45_plan_P_range_and_numpy_int` (the file's last test, from Task 1's tree), with two blank lines before the first comment line:

```python
# ---------------------------------------------------------------------------
# The other wilson sites (spec 2026-11-08 §2.3-§2.5, §3.5)
# ---------------------------------------------------------------------------

_T, _IP = wilson.transform, wilson.inner_product
_Z = np.fft.rfft(np.random.default_rng(0).standard_normal(128))     # rfft bins of an 8 x 16 grid
_BANDS = np.array([2, 3])
_W_BANDS = _T.forward_bands(_Z, 0, _BANDS, 8, 16, 1.0)
_W = wilson.forward_time(np.random.default_rng(1).standard_normal(128), 8, 16, 1.0)
_VAR = wilson.whittle_variance(8, 16, 1.0, np.ones_like)
_SN = np.ones(65)


def _exact(message):
    """A ``match=`` pattern for exactly ``message``."""
    return f"^{re.escape(message)}$"


#: Integer slots: the call taking the slot's value, and a value it accepts.
_INT_SLOTS = {
    "forward_bands k0": (lambda v: _T.forward_bands(_Z, v, _BANDS, 8, 16, 1.0), 0),
    "inverse_bands k0": (lambda v: _T.inverse_bands(_W_BANDS, v, 65, _BANDS, 8, 16, 1.0), 0),
    "inverse_bands n_bins": (lambda v: _T.inverse_bands(_W_BANDS, 0, v, _BANDS, 8, 16, 1.0), 65),
    "atom_spectrum m": (lambda v: _T.atom_spectrum(v, 3, 8, 16, 1.0), 2),
    "atom_spectrum n": (lambda v: _T.atom_spectrum(2, v, 8, 16, 1.0), 3),
    "carrier_phase m": (lambda v: wilson.grid.carrier_phase(v, 4), 1),
    "edge_columns lead": (lambda v: wilson.grid.edge_columns(16, v, 1), 2),
    "edge_columns trail": (lambda v: wilson.grid.edge_columns(16, 2, v), 1),
    "empirical_variance n_realizations": (lambda v: wilson.empirical_variance(8, 16, 1.0, np.ones_like, v,
                                                                              seed=0), 2),
}


@pytest.mark.parametrize(("value", "kind"), [
    (2.0, "float"), (np.float64(2), "float64"), (True, "bool"), (np.True_, "bool"), ("2", "str"),
    (None, "NoneType"),
], ids=["float", "np.float64", "bool", "np.bool_", "str", "None"])
@pytest.mark.parametrize("slot", list(_INT_SLOTS))
def test_integer_slots_of_the_wrong_kind_raise_type_error(slot, value, kind):
    """``int()``, ``range()`` or a parity test took these silently (``k0="0"`` and
    ``carrier_phase(2.5, 4)`` ran), or ``True`` passed as 1."""
    call, _ = _INT_SLOTS[slot]
    with pytest.raises(TypeError, match=_exact(f"{slot.split()[-1]} must be an integer, got {kind}")):
        call(value)


@pytest.mark.parametrize("slot", list(_INT_SLOTS))
def test_integer_slots_take_a_numpy_integer(slot):
    call, good = _INT_SLOTS[slot]
    assert np.array_equal(np.asarray(call(np.int64(good))), np.asarray(call(good)))


@pytest.mark.parametrize(("call", "match"), [
    (lambda: _T.forward_bands(_Z, 50, _BANDS, 8, 16, 1.0), r"^the bins \[50, 115\) do not cover"),
    (lambda: _T.inverse_bands(_W_BANDS, 0, 20, _BANDS, 8, 16, 1.0), r"^the bins \[0, 20\) do not cover"),
    (lambda: _T.atom_spectrum(2, 16, 8, 16, 1.0), r"^n must lie in 0\.\.15"),
    (lambda: wilson.empirical_variance(8, 16, 1.0, np.ones_like, 0), "^n_realizations must be >= 1, got 0"),
], ids=["forward_bands k0", "inverse_bands n_bins", "atom_spectrum n", "empirical_variance"])
def test_integer_slots_out_of_range_stay_value_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


#: The three kernels that take a band list, through ``_check_interior_bands``.
_BAND_SLOTS = {
    "band_atoms": lambda b: _T.band_atoms(b, 8, 16, 1.0),
    "forward_bands": lambda b: _T.forward_bands(_Z, 0, b, 8, 16, 1.0),
    "inverse_bands": lambda b: _T.inverse_bands(_W_BANDS, 0, 65, b, 8, 16, 1.0),
}


@pytest.mark.parametrize(("bands", "dtype"), [
    ([True, True], "bool"), (["2", "3"], "<U1"), ([2 + 0j, 3 + 0j], "complex128"),
    (np.array([2, 3], object), "object"),
], ids=["bool", "str", "complex", "object"])
@pytest.mark.parametrize("entry", list(_BAND_SLOTS))
def test_bands_of_the_wrong_dtype_raise_type_error(entry, bands, dtype):
    """``np.asarray(bands, dtype=int)`` took ``[True, True]`` as band 1 and parsed ``["2", "3"]``."""
    with pytest.raises(TypeError, match=_exact(f"bands must be an integer index array, got dtype {dtype}")):
        _BAND_SLOTS[entry](bands)


@pytest.mark.parametrize(("bands", "first"), [([2.5, 3.0], "2.5"), ([2.0, np.nan], "nan")],
                         ids=["2.5", "nan"])
@pytest.mark.parametrize("entry", list(_BAND_SLOTS))
def test_bands_of_non_integral_floats_raise_value_error(entry, bands, first):
    """``[2.5, 3.0]`` used to be truncated to bands 2 and 3 silently."""
    with pytest.raises(ValueError, match=_exact(f"bands must hold integer indices, got {first}")):
        _BAND_SLOTS[entry](bands)


@pytest.mark.parametrize("entry", list(_BAND_SLOTS))
def test_bands_take_integer_and_integral_float_arrays(entry):
    want = np.asarray(_BAND_SLOTS[entry](_BANDS))
    for bands in ([2, 3], np.array([2, 3], np.int64), np.array([2.0, 3.0]), [np.float64(2), np.float64(3)]):
        assert np.array_equal(np.asarray(_BAND_SLOTS[entry](bands)), want)
    with pytest.raises(ValueError, match=r"^bands must be interior bands 1\.\.7; got \[0, 3\]$"):
        _BAND_SLOTS[entry]([0, 3])


@pytest.mark.parametrize(("t_max", "kind"), [
    ("128", "str"), (True, "bool"), (128 + 0j, "complex"), (Decimal("128"), "Decimal"),
    (np.array(128.0), "ndarray"),
], ids=["str", "bool", "complex", "Decimal", "0-d array"])
def test_eq45_plan_t_max_of_the_wrong_kind_raises_type_error(t_max, kind):
    """``float()`` parsed ``t_max="128"`` and took a 0-d array."""
    with pytest.raises(TypeError, match=_exact(f"t_max must be a real number, got {kind}")):
        wilson.evolutionary.eq45_plan(8, 16, 1.0, t_max=t_max)


def test_eq45_plan_t_max_takes_numpy_reals_and_is_stored_as_a_float():
    ref = wilson.evolutionary.eq45_plan(8, 16, 1.0)
    for t_max in (128, np.float64(128.0), np.int64(128), Fraction(128)):
        plan = wilson.evolutionary.eq45_plan(8, 16, 1.0, t_max=t_max)
        assert type(plan.t_max) is float and plan.t_max == ref.t_max == 128.0
        assert np.array_equal(plan.Tp, ref.Tp)


def test_eq45_plan_keep_message_names_the_dtype():
    with pytest.raises(TypeError, match=_exact("keep must be a boolean mask over the packed cells, "
                                               "got dtype int64")):
        wilson.evolutionary.eq45_plan(8, 16, 1.0, np.ones(128, np.int64))


#: The loglike functions that take ``include_logdet``.
_LOGLIKE = {
    "loglike_whittle": lambda flag: _IP.loglike_whittle(_W, 0 * _W, _VAR, 8, include_logdet=flag),
    "loglike_whittle_many": lambda flag: _IP.loglike_whittle_many(_W, np.zeros((2, 9, 16)), _VAR, 8,
                                                                  include_logdet=flag),
    "loglike_band_averaged": lambda flag: _IP.loglike_band_averaged(_W, 0 * _W, _VAR, 8,
                                                                    include_logdet=flag),
    "loglike_evolutionary_whittle": lambda flag: _IP.loglike_evolutionary_whittle(_W, 0 * _W, _VAR, 8,
                                                                                  include_logdet=flag),
    "loglike_gaussian": lambda flag: _IP.loglike_gaussian(_W, 0 * _W, 8, 16, 1.0, _SN, include_logdet=flag),
    "loglike_gaussian_many": lambda flag: _IP.loglike_gaussian_many(_W, np.zeros((2, 9, 16)), 8, 16, 1.0,
                                                                    _SN, include_logdet=flag),
}


@pytest.mark.parametrize(("flag", "kind"), [
    ("no", "str"), (0, "int"), (1, "int"), (np.int64(1), "int64"), (0.5, "float"), (None, "NoneType"),
], ids=["str", "0", "1", "np.int64", "float", "None"])
@pytest.mark.parametrize("entry", list(_LOGLIKE))
def test_include_logdet_of_the_wrong_kind_raises_type_error(entry, flag, kind):
    """A truthy ``"no"`` used to include the log-determinant."""
    with pytest.raises(TypeError, match=_exact(f"include_logdet must be a bool, got {kind}")):
        _LOGLIKE[entry](flag)


@pytest.mark.parametrize("entry", list(_LOGLIKE))
def test_include_logdet_takes_a_numpy_bool(entry):
    for flag in (True, False):
        assert np.array_equal(_LOGLIKE[entry](np.bool_(flag)), _LOGLIKE[entry](flag))
    assert not np.array_equal(_LOGLIKE[entry](True), _LOGLIKE[entry](False))


_CELLS = {
    "whittle": lambda c: _IP.whittle(_W, _W, _VAR, 8, cells=c),
    "whittle_many": lambda c: _IP.whittle_many(_W[None], _W[None], _VAR, 8, cells=c),
    "loglike_whittle": lambda c: _IP.loglike_whittle(_W, 0 * _W, _VAR, 8, cells=c),
    "loglike_whittle_many": lambda c: _IP.loglike_whittle_many(_W, np.zeros((2, 9, 16)), _VAR, 8, cells=c),
}


@pytest.mark.parametrize(("cells", "kind"), [(1, "int"), (None, "NoneType"), (b"active", "bytes"),
                                             (("active",), "tuple")], ids=["int", "None", "bytes", "tuple"])
@pytest.mark.parametrize("entry", list(_CELLS))
def test_cells_of_the_wrong_kind_raise_type_error(entry, cells, kind):
    with pytest.raises(TypeError, match=_exact(f"cells must be a string, got {kind}")):
        _CELLS[entry](cells)


@pytest.mark.parametrize("entry", list(_CELLS))
def test_cells_keep_their_choices(entry):
    for cells in ("active", "interior"):
        assert np.array_equal(_CELLS[entry](np.str_(cells)), _CELLS[entry](cells))
    with pytest.raises(ValueError, match=_exact("cells must be 'active' or 'interior'; got 'edges'")):
        _CELLS[entry]("edges")


@pytest.mark.parametrize(("value", "kind"), [(1, "int"), (b"numpy", "bytes"), (True, "bool"),
                                             (np, "module")], ids=["int", "bytes", "bool", "module"])
def test_get_backend_backend_of_the_wrong_kind_raises_type_error(value, kind):
    with pytest.raises(TypeError, match=_exact(f"backend must be a string, got {kind}")):
        wilson.backend.get_backend(np.zeros(2), backend=value)


def test_get_backend_keeps_its_choices():
    get_backend, NUMPY = wilson.backend.get_backend, wilson.backend.NUMPY
    for backend in (None, "numpy", np.str_("numpy"), NUMPY):
        assert get_backend(np.zeros(2), backend=backend) is NUMPY
    message = "backend must be 'numpy', 'jax' or a Backend; got 'torch'"
    with pytest.raises(ValueError, match=_exact(message)):
        get_backend(backend="torch")


@pytest.mark.parametrize(("impl", "kind"), [(1, "int"), (None, "NoneType"), (b"numpy", "bytes")],
                         ids=["int", "None", "bytes"])
def test_cholesky_impl_of_the_wrong_kind_raises_type_error(backend, impl, kind):
    with pytest.raises(TypeError, match=_exact(f"impl must be a string, got {kind}")):
        backend.cholesky(backend.asarray(np.eye(2)), impl=impl)


def test_cholesky_keeps_its_choices(backend):
    A = backend.asarray(np.array([[4.0, 2.0], [2.0, 3.0]]))
    for impl in ("numpy", "scipy"):
        L, ok = backend.cholesky(A, impl=np.str_(impl))
        assert bool(ok) and np.allclose(np.asarray(L) @ np.asarray(L).T, np.asarray(A))
    with pytest.raises(ValueError, match=_exact("impl must be 'numpy' or 'scipy'; got 'lapack'")):
        backend.cholesky(A, impl="lapack")


#: The real-array checks of spec §3.5: the call, an accepted float64 array, and the name in
#: the message.
_REAL_ARRAYS = {
    "inverse wmn": (lambda a: wilson.inverse(a, 8, 16, 1.0), _W, "wmn"),
    "inverse_many wmn": (lambda a: wilson.inverse_many(a, 8, 16, 1.0), _W[None], "wmn"),
    "inverse_time wmn": (lambda a: wilson.inverse_time(a, 8, 16, 1.0), _W, "wmn"),
    "inverse_time_many wmn": (lambda a: wilson.inverse_time_many(a, 8, 16, 1.0), _W[None], "wmn"),
    "inverse_bands w": (lambda a: _T.inverse_bands(a, 0, 65, _BANDS, 8, 16, 1.0), _W_BANDS, "w"),
    "sanitize_psd": (lambda a: wilson.grid.sanitize_psd(a, "demo"), np.ones(5), "demo: PSD"),
}


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64, np.bool_, np.str_, object],
                         ids=["complex128", "complex64", "bool", "str", "object"])
@pytest.mark.parametrize("slot", list(_REAL_ARRAYS))
def test_real_arrays_of_the_wrong_dtype_raise_type_error(slot, dtype):
    """Complex used to be a ``ValueError``; a bool grid ran, and a bool, str or object PSD was cast."""
    call, good, name = _REAL_ARRAYS[slot]
    bad = good.astype(dtype)
    message = f"{name} must be a real numeric array, got dtype {bad.dtype}"
    with pytest.raises(TypeError, match=_exact(message)):
        call(bad)


@pytest.mark.parametrize("slot", list(_REAL_ARRAYS))
def test_real_arrays_take_integer_and_float_dtypes(slot):
    call, good, _ = _REAL_ARRAYS[slot]
    ints = np.round(10 * good).astype(np.int64)
    assert np.array_equal(call(ints), call(ints.astype(np.float64)))


def test_the_stationary_psd_check_uses_the_sanitize_psd_message():
    psd = np.ones(65, complex)
    with pytest.raises(TypeError, match=_exact("stationary noise input: PSD must be a real numeric array, "
                                               "got dtype complex128")):
        wilson.stationary.build_K_full_active(8, 16, 1.0, psd)
    with pytest.raises(TypeError, match=_exact("stationary noise input: PSD must be a real numeric array, "
                                               "got dtype bool")):
        wilson.stationary.build_K_full_active(8, 16, 1.0, np.ones(65, bool))


class _UnhashableRamp:
    __hash__ = None

    def __call__(self, x):
        return x


@pytest.mark.parametrize(("make", "message"), [
    (lambda: wilson.RampWindow(np.sqrt, 3), "window name must be a non-empty string, got 3"),
    (lambda: wilson.RampWindow("x", "r"), "ramp must be callable, got str"),
    (lambda: wilson.RampWindow(_UnhashableRamp(), "r"),
     "ramp must be hashable (a window is a cache key and static under jax.jit), got an unhashable "
     "_UnhashableRamp: use a module-level function or a frozen dataclass instance"),
    (lambda: wilson.windows.require_window(wilson.Meyer),
     "window must be a Window instance, got the class Meyer: pass an instance, e.g. Meyer() or "
     "domainia.wilson.MEYER"),
    (lambda: wilson.windows.require_window("meyer"),
     "window must be a domainia.wilson Window (Meyer, RampWindow, SampledWindow), got str"),
], ids=["name", "callable", "hashable", "class", "non-window"])
def test_the_window_kind_messages_take_the_got_form(make, message):
    with pytest.raises(TypeError, match=_exact(message)):
        make()


@pytest.mark.jax
def test_the_real_array_checks_read_a_jit_tracers_dtype(jax_be):
    """Under ``jax.jit`` the checks see tracers: a real grid passes, a complex one raises
    ``TypeError`` while tracing (JAX adds a note, so the message is compared whole)."""
    import jax
    import jax.numpy as jnp

    inv = jax.jit(lambda w: wilson.inverse(w, 8, 16, 1.0))
    inv_bands = jax.jit(lambda w: _T.inverse_bands(w, 0, 65, _BANDS, 8, 16, 1.0))
    fwd_bands = jax.jit(lambda z: _T.forward_bands(z, np.int64(0), _BANDS, 8, 16, 1.0))
    want_bands = _T.inverse_bands(_W_BANDS, 0, 65, _BANDS, 8, 16, 1.0)
    for got, want in ((inv(jnp.asarray(_W)), wilson.inverse(_W, 8, 16, 1.0)),
                      (inv_bands(jnp.asarray(_W_BANDS)), want_bands),
                      (fwd_bands(jnp.asarray(_Z)), _W_BANDS)):
        assert isinstance(got, jax.Array) and np.allclose(np.asarray(got), want, rtol=0, atol=1e-12)
    for jitted, arg, name in ((inv, _W, "wmn"), (inv_bands, _W_BANDS, "w")):
        with pytest.raises(TypeError) as info:
            jitted(jnp.asarray(arg) + 0j)
        assert str(info.value) == f"{name} must be a real numeric array, got dtype complex128"
```

Notes on the tests:
- Every `match=` is anchored (`_exact`), so a Python or NumPy `TypeError` with another message does not pass for the new check.
- Each slot kind pins its accepted NumPy kind: `np.int64` for the integer slots, integral `np.float64` entries for the band lists, `np.float64`/`np.int64` for `t_max`, `np.bool_` for `include_logdet`, `np.str_` for the string options, and `int64`/`float64` arrays for the real arrays.
- `test_cholesky_*` use the `backend` fixture (`tests/wilson/conftest.py`), so they run on both backends; the JAX leg skips in the NumPy lane. The last test is `jax`-marked and uses `jax_be`. Inside `jax.jit` the complex case raises while tracing, and JAX adds a note to the exception, so it compares `str(info.value)` (as Task 1's `test_checks.py` does).
- `_UnhashableRamp` sets `__hash__ = None`, so `hash()` raises `TypeError` and `_check_callable` reports it.

- [ ] **Step 4: Move the two pins to `TypeError`**

`tests/wilson/test_grid_bands.py`, in `test_inverse_bands_rejects_complex_and_non_finite_cells`:

```python
    w = np.ones((bands.size, Nt))
    with pytest.raises(ValueError, match="w must be real"):
        inverse_bands(backend.asarray(w + 1j), 0, 40, bands, Nf, Nt, dt)
```
→
```python
    w = np.ones((bands.size, Nt))
    with pytest.raises(TypeError, match="w must be a real numeric array, got dtype complex128"):
        inverse_bands(backend.asarray(w + 1j), 0, 40, bands, Nf, Nt, dt)
```

`tests/wilson/test_noise_stationary.py`, `test_complex_psd_is_rejected`: the docstring

```python
    """A PSD is real: a complex one (even with a zero imaginary part) raises ValueError
    in every input form, instead of being truncated to its real part (1D and callables
    used to warn ComplexWarning and go on; the 2D form took np.real silently)."""
```
→
```python
    """A PSD is real: a complex one (even with a zero imaginary part) raises TypeError
    in every input form, instead of being truncated to its real part (1D and callables
    used to warn ComplexWarning and go on; the 2D form took np.real silently)."""
```

and the check

```python
            with pytest.raises(ValueError, match="real-valued"):
```
→
```python
            with pytest.raises(TypeError, match="PSD must be a real numeric array, got dtype complex128"):
```

The match leaves out the `context` prefix (`stationary noise input: `); `test_wrong_kinds.py` pins the whole message.

- [ ] **Step 5: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py tests/wilson/test_grid_bands.py tests/wilson/test_noise_stationary.py`
Expected: `183 failed, 166 passed, 1 skipped` (the skip is `test_grid_bands.py`'s oracle test).
- `test_wrong_kinds.py`: 177 failed, 77 passed.
  - `test_integer_slots_of_the_wrong_kind_raise_type_error`: all 54. 26 are `Failed: DID NOT RAISE <class 'TypeError'>`. 22 raise Python's or NumPy's own `TypeError` (`'float' object cannot be interpreted as an integer`, `int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`, `'<=' not supported between instances of 'int' and 'NoneType'`), which the anchored `match=` refuses: pytest prints `AssertionError` with the actual message (`-`) against the expected one (`+`). 6 raise a `ValueError` (`the bins [0, 1) do not cover the band windows [8, 32)`, `(None, 3) is not an active cell of the (8, 16) grid`).
  - `test_bands_of_the_wrong_dtype_raise_type_error`: 12 (bool, str, object: `DID NOT RAISE`; complex: `int() argument must be a string, a bytes-like object or a real number, not 'complex'`, refused by the match). `test_bands_of_non_integral_floats_raise_value_error`: 6 (`2.5`: `DID NOT RAISE <class 'ValueError'>`; `nan`: NumPy's `cannot convert float NaN to integer`, refused by the match).
  - `test_eq45_plan_t_max_of_the_wrong_kind_raises_type_error`: 5 (str, `Decimal`, 0-d array: `DID NOT RAISE`; bool: `ValueError: pair mid-times must lie inside [0, t_max]`; complex: `float() argument must be a string or a real number, not 'complex'`, refused by the match). `test_eq45_plan_keep_message_names_the_dtype`: 1 (the old message has no `, got dtype int64`).
  - `test_include_logdet_of_the_wrong_kind_raises_type_error`: 36 (`DID NOT RAISE`).
  - `test_cells_of_the_wrong_kind_raise_type_error`: 16, `test_get_backend_backend_of_the_wrong_kind_raises_type_error`: 4, `test_cholesky_impl_of_the_wrong_kind_raises_type_error`: 6 (both backends). Each gets the membership `ValueError` (`cells must be 'active' or 'interior'; got 1`).
  - `test_real_arrays_of_the_wrong_dtype_raise_type_error`: 30. Complex gets the old `ValueError` (`wmn must be real; a complex grid would have its imaginary part silently discarded`, `w must be real; the adjoint identity holds only for real cells, ...` and `demo: PSD must be real-valued, got a complex array`). Bool is `DID NOT RAISE` everywhere. str and object are `DID NOT RAISE` for `sanitize_psd`, and elsewhere NumPy's `TypeError: ufunc 'isfinite' not supported for the input types`, which the match refuses.
  - `test_the_stationary_psd_check_uses_the_sanitize_psd_message`, `test_the_real_array_checks_read_a_jit_tracers_dtype`: 1 each (the old `ValueError`s). `test_the_window_kind_messages_take_the_got_form`: 5 (the old `"; got"` and `not the class` texts).
  - The 77 passes: the 41 tests of Task 1's tree, and the 36 new pass-through tests (`np.int64` slots, the in-range `ValueError`s, the integral-float bands, `t_max`'s NumPy reals, `np.bool_`, `np.str_`, the `int64` arrays), which already hold.
- `test_grid_bands.py`: `test_inverse_bands_rejects_complex_and_non_finite_cells[numpy]` and `[jax]` (`ValueError: w must be real; the adjoint identity holds only for real cells, so a complex w would give a complex result that is not the adjoint of any cells`).
- `test_noise_stationary.py`: the four `test_complex_psd_is_rejected` cases (`ValueError: stationary noise input: PSD must be real-valued, got a complex array`).

- [ ] **Step 6: `transform.py`: integers, bands and the real grids**

The import (`require_int` here is `_checks`' own name; `grid.py` keeps its `_require_int` alias for its own use):

```python
from .backend import NUMPY, Backend, get_backend, is_tracer
from .grid import (
```
→
```python
from ._checks import REAL_KINDS, index_array, require_dtype, require_int
from .backend import NUMPY, Backend, get_backend, is_tracer
from .grid import (
```

`inverse`, the Raises section and the examples (one doctest line pair added):

```python
    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if ``wmn`` is complex or wrongly shaped, or
        if it holds a ``nan`` or ``inf`` (see :func:`domainia.wilson.validate_finite`).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> bool(np.allclose(wilson.inverse(w, Nf, Nt, dt), np.fft.rfft(x)))
    True
    """
```
→
```python
    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, ``dt`` is not a real number, or ``wmn``
        is not a real numeric array: a complex grid would have its imaginary part
        silently discarded, and a bool, str or object grid is refused too.
    ValueError
        If the grid is invalid, if ``wmn`` is wrongly shaped, or
        if it holds a ``nan`` or ``inf`` (see :func:`domainia.wilson.validate_finite`).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> bool(np.allclose(wilson.inverse(w, Nf, Nt, dt), np.fft.rfft(x)))
    True
    >>> wilson.inverse(w + 0j, Nf, Nt, dt)
    Traceback (most recent call last):
    TypeError: wmn must be a real numeric array, got dtype complex128
    """
```

`inverse`'s check (the explanation moves into a comment and the Raises section; the helper's message has a fixed form):

```python
    wmn = be.asarray(wmn)
    if be.xp.iscomplexobj(wmn):
        raise ValueError(
            "wmn must be real; a complex grid would have its imaginary part "
            "silently discarded"
        )
    if wmn.shape != (Nf + 1, Nt):
```
→
```python
    wmn = be.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.shape != (Nf + 1, Nt):
```

`inverse_many`'s check (the same block as `inverse`'s, anchored here by the shape check after it):

```python
    wmn = be.asarray(wmn)
    if be.xp.iscomplexobj(wmn):
        raise ValueError(
            "wmn must be real; a complex grid would have its imaginary part "
            "silently discarded"
        )
    if wmn.ndim != 3 or wmn.shape[1:] != (Nf + 1, Nt):
```
→
```python
    wmn = be.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.ndim != 3 or wmn.shape[1:] != (Nf + 1, Nt):
```

`atom_spectrum` (the checks do not rebind `m` and `n`; the callers pass `int(...)` already, so the computed values are unchanged):

```python
    require_window(window)
    validate_grid(Nf, Nt)
    validate_dt(dt)
    if not 0 <= n < Nt:
```
→
```python
    require_window(window)
    validate_grid(Nf, Nt)
    validate_dt(dt)
    require_int("m", m)
    require_int("n", n)
    if not 0 <= n < Nt:
```

`_check_interior_bands` (`index_array` returns `int64`, which is what `dtype=int` gave on this platform):

```python
def _check_interior_bands(bands: NDArray[np.integer], Nf: int) -> NDArray[np.integer]:
    bands = np.asarray(bands, dtype=int).ravel()
```
→
```python
def _check_interior_bands(bands: NDArray[np.integer], Nf: int) -> NDArray[np.integer]:
    bands = index_array("bands", bands).ravel()
```

`forward_bands`:

```python
    validate_grid(Nf, Nt)
    validate_dt(dt)
    be = be or get_backend(z)
    bands = _check_interior_bands(bands, Nf)
```
→
```python
    validate_grid(Nf, Nt)
    validate_dt(dt)
    require_int("k0", k0)
    be = be or get_backend(z)
    bands = _check_interior_bands(bands, Nf)
```

`inverse_bands` (`n_bins` too: it is a count in the same signature; today `True` gets a misleading coverage `ValueError`):

```python
    validate_grid(Nf, Nt)
    validate_dt(dt)
    be = be or get_backend(w)
    bands = _check_interior_bands(bands, Nf)
    w = be.asarray(w)
    if be.xp.iscomplexobj(w):
        raise ValueError(
            "w must be real; the adjoint identity holds only for real cells, so a "
            "complex w would give a complex result that is not the adjoint of any cells"
        )
    if w.shape[-2:] != (bands.size, Nt):
```
→
```python
    validate_grid(Nf, Nt)
    validate_dt(dt)
    require_int("k0", k0)
    require_int("n_bins", n_bins)
    be = be or get_backend(w)
    bands = _check_interior_bands(bands, Nf)
    w = be.asarray(w)
    # The adjoint identity holds only for real cells: a complex w would give a complex
    # result that is not the adjoint of any cells.
    require_dtype("w", w, REAL_KINDS, "a real numeric array")
    if w.shape[-2:] != (bands.size, Nt):
```

`require_dtype` reads `.dtype` without converting, so it works on a JAX tracer: each check stays where the `iscomplexobj` was, after `be.asarray`. `require_int` runs on `k0` and `n_bins`, which are static Python or NumPy ints under `jax.jit` (`FourierToWDM` passes `self.src.k0`).

- [ ] **Step 7: `grid.py` and `stationary.py`: the PSD checks, `carrier_phase`, `edge_columns`**

`grid.py`, a new import line above Task 1's (Task 1's line stays as it is):

```python
from ._checks import require_int as _require_int, validate_dt, validate_finite
```
→
```python
from ._checks import REAL_KINDS, require_dtype
from ._checks import require_int as _require_int, validate_dt, validate_finite
```

`sanitize_psd`'s description:

```python
    ``nan`` is *not* treated that way: it has no physical reading, so it raises.
    Complex input raises too rather than being silently truncated to its real
    part.
```
→
```python
    ``nan`` is *not* treated that way: it has no physical reading, so it raises.
    Complex input raises ``TypeError`` rather than being silently truncated to its
    real part, and so does a bool, str or object array.
```

`sanitize_psd`'s Raises section, examples and check. The name in the message is `"{context}: PSD"`, so the message keeps the module's `{context}: PSD` prefix:

```python
    Raises
    ------
    ValueError
        If any entry is negative, ``nan``, or complex.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.sanitize_psd([1e-3, np.inf], 'demo')   # inf -> no power
    array([0.001, 0.   ])
    >>> wilson.grid.sanitize_psd([-1.0], 'demo')
    Traceback (most recent call last):
    ValueError: demo: PSD has negative value(s); a power spectral density must be non-negative
    """
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        raise ValueError(f"{context}: PSD must be real-valued, got a complex array")
```
→
```python
    Raises
    ------
    TypeError
        If ``values`` is not a real numeric array (a complex, bool, str or object dtype).
    ValueError
        If any entry is negative or ``nan``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.sanitize_psd([1e-3, np.inf], 'demo')   # inf -> no power
    array([0.001, 0.   ])
    >>> wilson.grid.sanitize_psd([-1.0], 'demo')
    Traceback (most recent call last):
    ValueError: demo: PSD has negative value(s); a power spectral density must be non-negative
    >>> wilson.grid.sanitize_psd([1e-3 + 0j], 'demo')
    Traceback (most recent call last):
    TypeError: demo: PSD must be a real numeric array, got dtype complex128
    """
    arr = np.asarray(values)
    require_dtype(f"{context}: PSD", arr, REAL_KINDS, "a real numeric array")
```

`carrier_phase` (one doctest line pair added):

```python
    >>> wilson.grid.carrier_phase(1, 4)
    array([ 1., -1.,  1., -1.])
    """
    if m % 2 == 0:
```
→
```python
    >>> wilson.grid.carrier_phase(1, 4)
    array([ 1., -1.,  1., -1.])
    >>> wilson.grid.carrier_phase(1.0, 4)
    Traceback (most recent call last):
    TypeError: m must be an integer, got float
    """
    _require_int("m", m)
    if m % 2 == 0:
```

`edge_columns` (one doctest line pair added):

```python
    >>> wilson.grid.edge_columns(256, 8, 7)
    (0, 1, 2, 3, 4, 5, 6, 7, 249, 250, 251, 252, 253, 254, 255)
    """
    return tuple(range(lead)) + tuple(range(Nt - trail, Nt))
```
→
```python
    >>> wilson.grid.edge_columns(256, 8, 7)
    (0, 1, 2, 3, 4, 5, 6, 7, 249, 250, 251, 252, 253, 254, 255)
    >>> wilson.grid.edge_columns(256, True, 7)
    Traceback (most recent call last):
    TypeError: lead must be an integer, got bool
    """
    _require_int("lead", lead)
    _require_int("trail", trail)
    return tuple(range(lead)) + tuple(range(Nt - trail, Nt))
```

`stationary.py`, the import:

```python
from .backend import Backend, get_backend, is_tracer
from .grid import n_samples, sanitize_psd, validate_dt, validate_grid
```
→
```python
from ._checks import REAL_KINDS, require_dtype
from .backend import Backend, get_backend, is_tracer
from .grid import n_samples, sanitize_psd, validate_dt, validate_grid
```

`_normalize_noise_to_sigma2`'s Raises section:

```python
    Raises
    ------
    ValueError
        If a 2D matrix input is not diagonal (i.e., the noise is not
        stationary), if the input shape is not recognized, or if the PSD is
        complex in any form (even with a zero imaginary part): it is
        rejected before any cast, rather than truncated to its real part
        (with only a ``ComplexWarning`` for callables and 1D arrays, silently
        for 2D ones), as :func:`domainia.wilson.grid.sanitize_psd` does.
    """
```
→
```python
    Raises
    ------
    TypeError
        If the PSD is not a real numeric array in any form: complex (even with a
        zero imaginary part), bool, str or object.  It is rejected before any
        cast, rather than truncated to its real part (with only a
        ``ComplexWarning`` for callables and 1D arrays, silently for 2D ones), as
        :func:`domainia.wilson.grid.sanitize_psd` does.
    ValueError
        If a 2D matrix input is not diagonal (i.e., the noise is not
        stationary), or if the input shape is not recognized.
    """
```

`_reject_complex_psd` keeps its name and its two call sites. Its body is one `require_dtype` call with `sanitize_psd`'s name and phrase, so both give the same message:

```python
def _reject_complex_psd(values: NDArray[np.generic], context: str) -> None:
    """``ValueError`` for a complex PSD, with :func:`domainia.wilson.grid.sanitize_psd`'s message.

    Checked on the raw input, before the ``float`` casts / ``np.real`` above
    would discard the imaginary part.
    """
    if np.iscomplexobj(values):
        raise ValueError(f"{context}: PSD must be real-valued, got a complex array")
```
→
```python
def _reject_complex_psd(values: NDArray[np.generic], context: str) -> None:
    """``TypeError`` for a PSD that is not a real numeric array (complex, bool, str or
    object), with :func:`domainia.wilson.grid.sanitize_psd`'s message.

    Checked on the raw input, before the ``float`` casts / ``np.real`` above
    would discard the imaginary part.
    """
    require_dtype(f"{context}: PSD", values, REAL_KINDS, "a real numeric array")
```

- [ ] **Step 8: `noise.py` and `evolutionary.py`**

`noise.empirical_variance` (`noise.py` imports `require_int` since Task 1; the `ValueError` text stays):

```python
    if int(n_realizations) < 1:
```
→
```python
    if require_int("n_realizations", n_realizations) < 1:
```

`evolutionary.py`, the import:

```python
from .backend import NUMPY, Backend, get_backend, pytree_dataclass
from .grid import (
```
→
```python
from ._checks import require_dtype, require_real
from .backend import NUMPY, Backend, get_backend, pytree_dataclass
from .grid import (
```

`eq45_plan`'s Raises section:

```python
        If ``Nf``, ``Nt`` or ``P`` is not an integer (``bool`` included), ``dt`` is not a
        real number, or ``keep`` is not a boolean mask.
```
→
```python
        If ``Nf``, ``Nt`` or ``P`` is not an integer (``bool`` included), ``dt`` or
        ``t_max`` is not a real number, or ``keep`` is not a boolean mask.
```

`eq45_plan`'s `keep` check (`keep.dtype != np.bool_` is the same test as kind `"b"`; the message gains `", got dtype {keep.dtype}"`, and the comment moves above the call):

```python
    if keep.dtype != np.bool_:   # an index array (e.g. np.arange(N)) would silently drop cell 0
        raise TypeError("keep must be a boolean mask over the packed cells")
```
→
```python
    # An index array (e.g. np.arange(N)) would silently drop cell 0.
    require_dtype("keep", keep, "b", "a boolean mask over the packed cells")
```

and its `t_max` (`T = N * dt` is already a Python `float`, and `require_real` returns `float(t_max)`, so the stored value is the same):

```python
    t_max = float(T if t_max is None else t_max)
```
→
```python
    t_max = T if t_max is None else require_real("t_max", t_max)
```

- [ ] **Step 9: `inner_product.py` and `backend.py`: the flag and the string options**

`inner_product.py`, the import:

```python
from .grid import active_mask, n_samples, sanitize_psd, validate_finite
```
→
```python
from ._checks import require_bool, require_str
from .grid import active_mask, n_samples, sanitize_psd, validate_finite
```

`_cell_mask`, which every `cells=` reaches (through `_check_grids`) before anything reads it:

```python
    :func:`domainia.wilson.grid.active_mask` -- every active cell, DC and Nyquist included.
    """
    if cells == "interior":
```
→
```python
    :func:`domainia.wilson.grid.active_mask` -- every active cell, DC and Nyquist included.
    """
    require_str("cells", cells)
    if cells == "interior":
```

`loglike_whittle_many`, the first line of the body:

```python
    ((3,), True)
    """
    w_d, w_h = np.asarray(w_d), np.asarray(w_h)
```
→
```python
    ((3,), True)
    """
    include_logdet = require_bool("include_logdet", include_logdet)
    w_d, w_h = np.asarray(w_d), np.asarray(w_h)
```

`loglike_whittle` (it serves `loglike_band_averaged` and `loglike_evolutionary_whittle` too):

```python
    True
    """
    residual = _residual(w_d, w_h)
```
→
```python
    True
    """
    include_logdet = require_bool("include_logdet", include_logdet)
    residual = _residual(w_d, w_h)
```

`loglike_gaussian_many`:

```python
    N = n_samples(Nf, Nt)
    Sn_array = _check_psd_grid(Sn_array, N, "loglike_gaussian_many")
```
→
```python
    include_logdet = require_bool("include_logdet", include_logdet)
    N = n_samples(Nf, Nt)
    Sn_array = _check_psd_grid(Sn_array, N, "loglike_gaussian_many")
```

`loglike_gaussian`:

```python
    N = n_samples(Nf, Nt)
    Sn_array = _check_psd_grid(Sn_array, N, "loglike_gaussian")
```
→
```python
    include_logdet = require_bool("include_logdet", include_logdet)
    N = n_samples(Nf, Nt)
    Sn_array = _check_psd_grid(Sn_array, N, "loglike_gaussian")
```

`backend.py` imports only the standard library and NumPy; `_checks` does too, so the new import makes no cycle:

```python
import numpy as np

__all__ = [
    "NUMPY",
```
→
```python
import numpy as np

from ._checks import require_str

__all__ = [
    "NUMPY",
```

`_NumpyBackend.cholesky`:

```python
        positive definite gives a NaN-filled ``L`` and ``ok=False``."""
        A = np.asarray(A)
```
→
```python
        positive definite gives a NaN-filled ``L`` and ``ok=False``."""
        require_str("impl", impl)
        A = np.asarray(A)
```

`_JaxBackend.cholesky` (`impl` is a Python string under `jax.jit`, so the check runs at trace time):

```python
    def cholesky(self, A: Any, *, impl: Literal["numpy", "scipy"]) -> tuple[Any, Any]:
        if impl not in ("numpy", "scipy"):
```
→
```python
    def cholesky(self, A: Any, *, impl: Literal["numpy", "scipy"]) -> tuple[Any, Any]:
        require_str("impl", impl)
        if impl not in ("numpy", "scipy"):
```

`get_backend`: a `Backend` returns first and `None` still means "from the arrays"; anything else must be a string, and an unknown string keeps its `ValueError`:

```python
    if isinstance(backend, Backend):
        return backend
    if backend == "numpy":
```
→
```python
    if isinstance(backend, Backend):
        return backend
    if backend is not None:
        require_str("backend", backend)
    if backend == "numpy":
```

- [ ] **Step 10: `windows.py`: the four kind messages**

The import:

```python
from ._checks import require_int
```
→
```python
from ._checks import require_callable, require_instance, require_int
```

`_check_name`'s message, and `_check_callable` (its callable check is `require_callable`, whose message is the same text with `", got"`; the hashable message is reflowed so that `", got"` sits on one source line):

```python
        raise TypeError(f"window name must be a non-empty string; got {name!r}")


def _check_callable(what: str, fn: Any) -> None:
    if not callable(fn):
        raise TypeError(f"{what} must be callable; got {type(fn).__name__}")
    try:
        hash(fn)
    except TypeError:
        raise TypeError(f"{what} must be hashable (a window is a cache key and static under jax.jit): "
                        f"use a module-level function or a frozen dataclass instance; got an "
                        f"unhashable {type(fn).__name__}") from None
```
→
```python
        raise TypeError(f"window name must be a non-empty string, got {name!r}")


def _check_callable(what: str, fn: Any) -> None:
    require_callable(what, fn)
    try:
        hash(fn)
    except TypeError:
        raise TypeError(f"{what} must be hashable (a window is a cache key and static under "
                        f"jax.jit), got an unhashable {type(fn).__name__}: use a module-level "
                        f"function or a frozen dataclass instance") from None
```

`require_window` (the class message gets `", got the class {obj.__qualname__}"`; the instance check is `require_instance`):

```python
    if isinstance(obj, type):
        raise TypeError(f"window must be a Window instance, not the class {obj.__qualname__}: "
                        f"pass an instance, e.g. Meyer() or domainia.wilson.MEYER")
    if not isinstance(obj, Window):
        raise TypeError(f"window must be a domainia.wilson Window (Meyer, RampWindow, "
                        f"SampledWindow); got {type(obj).__name__}")
    return obj
```
→
```python
    if isinstance(obj, type):
        raise TypeError(f"window must be a Window instance, got the class {obj.__qualname__}: "
                        f"pass an instance, e.g. Meyer() or domainia.wilson.MEYER")
    return require_instance("window", obj, Window,
                            "a domainia.wilson Window (Meyer, RampWindow, SampledWindow)")
```

`_check_name` still refuses an empty name with `TypeError`, as `domains` and `translate` do for "non-empty strings". The two `WindowError` checks (`_as_samples`, `RampWindow.__post_init__`) are not touched (spec §3.5).

- [ ] **Step 11: Run the tests to see them pass, and the gates**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py tests/wilson/test_grid_bands.py tests/wilson/test_noise_stationary.py`
Expected: `349 passed, 1 skipped` (`test_wrong_kinds.py`: 254), no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py tests/wilson/test_grid_bands.py tests/wilson/test_noise_stationary.py`
Expected: `332 passed, 18 skipped` (`test_wrong_kinds.py`: 249 passed, 5 skipped: the JAX `cholesky` legs and the jit test).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/wilson_typeerror_gate.py`
Expected: no output.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "iscomplexobj" src/domainia/wilson; grep -n "; got" src/domainia/wilson/windows.py`
Expected: one line, from the second grep (the first prints nothing): `178:            raise ValueError(f"Meyer d must be >= 1; got {self.d}")` (a `ValueError`; Task 1's `test_checks.py` pins it).

- [ ] **Step 12: The wilson suite, its doctests and the package tests**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson src/domainia/wilson tests/test_package.py`
Expected: passes, no warnings summary. This includes the new doctest lines of `inverse`, `sanitize_psd`, `carrier_phase` and `edge_columns`. They were added to docstrings that already had examples, so `src/domainia/wilson` still has 72 doctest items (`grid.py` 15).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q --collect-only src/domainia/wilson | tail -1`
Expected: `72 tests collected` (and a time).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_maps_fourier_wdm.py tests/test_dom_restored_coverage.py tests/wilson/test_transform_shim.py tests/wilson/test_single_precision.py`
Expected: all pass. The `jax.jit` runs that reach the changed checks are here: `FourierToWDM`'s `impl="fft"` path (`forward_bands`, in `test_jax_parity_with_the_map_as_a_jit_argument`), `TimeToWDM`'s adjoint (`inverse_time_many`, in `test_wdm_maps_under_jit_with_the_map_as_argument`) and the jit tests of `test_transform_shim.py`. `test_single_precision.py` pins the float32 runs of `inverse`, `inverse_many`, `forward_bands` and `inverse_bands`: a float32 grid is real, so `require_dtype` passes it.

- [ ] **Step 13: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, no warnings summary.

- [ ] **Step 14: The float64 bit gate**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz`
Expected: `0 of 1504 arrays differ`. The checks only raise; no accepted value is converted on a new path (`index_array`'s `int64` is `dtype=int`'s, and `t_max` and `n_realizations` keep their values).

- [ ] **Step 15: Commit** (sandbox disabled for both git commands)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/wilson/transform.py src/domainia/wilson/grid.py src/domainia/wilson/stationary.py src/domainia/wilson/noise.py src/domainia/wilson/evolutionary.py src/domainia/wilson/inner_product.py src/domainia/wilson/backend.py src/domainia/wilson/windows.py tests/wilson/test_wrong_kinds.py tests/wilson/test_grid_bands.py tests/wilson/test_noise_stationary.py
git commit -m "domainia.wilson: the remaining wrong kinds raise TypeError

The band kernels' k0 (and inverse_bands' n_bins), atom_spectrum's m and n,
carrier_phase's m, edge_columns' lead and trail and empirical_variance's
n_realizations use require_int; the band lists use index_array, so [2.5]
is a ValueError instead of band 2. eq45_plan's t_max uses require_real,
include_logdet require_bool, and cells, backend and impl require_str
before their ValueError. sanitize_psd, the stationary PSD check, inverse's
and inverse_many's wmn and inverse_bands' w refuse a complex, bool, str or
object dtype with TypeError (spec 2026-11-08 §2.3-§2.5, §3.5). Every
TypeError message in wilson says \", got\".

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `cd /Users/aaron/Documents/lisa/domainia && git status --short`
Expected: no line for any file of this task.

---

### Task 3: Wrong kinds in domains.py (spec §1.5, §2.3–§2.6)

**Goal:** `domainia.domains` follows the rule: a wrong kind raises `TypeError` with a `", got"` message, and a wrong value raises `ValueError`. The `CellSet` class methods check their `grid` and `sub` with `require_instance` before they read them. `dt`, `t0` and `sigma0` go through `require_real` and are stored as Python `float` (spec §1.5). `units` is checked with `require_str`. `CellSet`'s `m`/`n` and `CellSet.bands`' band list use `index_array`, and `CellSet.from_mask` refuses a non-bool `keep` with `require_dtype`. `ChannelAxis` refuses a bare string instead of splitting it.

**Assumes the tree after Task 2.** Task 1 changed `domains._int` to call `require_int` and added the line `from domainia.wilson._checks import require_int`. Task 2 touches only `src/domainia/wilson/` and its tests. So the rest of `domains.py`, and the three test files edited below, are as at 779550d.

**Files:**
- Modify: `src/domainia/domains.py` (the `_checks` import, `_positive_float`, `TimeDomain.__post_init__`, the `units` check of `FourierDomain.__post_init__`, `_cell_index`, `CellSet.__post_init__`, `CellSet.index_of`, `CellSet.all_active`, `CellSet.bands`, `CellSet.from_mask`, `WDMDomain.__post_init__`, `_cells_and_window`, `ChannelAxis`)
- Create: `tests/test_wrong_kinds_domains.py`
- Modify: `tests/test_dom_domains.py` (`test_validation`), `tests/test_dom_evolutionary_cov.py` and `tests/test_dom_variance_generic.py` (the anchored `_cells_and_window` pin in each)

**Acceptance Criteria:**
- [ ] The 6 entry points that raised a bare `AttributeError` for a wrong-kind argument raise `TypeError` before they read it: `CellSet.all_active`, `CellSet.bands`, `CellSet.from_mask` and `WDMDomain.full` (`"grid must be a domainia.wilson.Grid, got {type}"`), and `CellSet.index_of` and `CellSet.complement` (`"sub must be a CellSet, got {type}"`). `WDMDomain.full` gets its check from `all_active`, and `complement` from `index_of`.
- [ ] `dt` (`TimeDomain`, `FourierDomain`, `FourierDomain.full`), `t0` and `sigma0` raise `TypeError("{name} must be a real number, got {type}")` for `bool`, `np.bool_`, `str`, complex, `Decimal` and arrays of any shape, 0-d included, and for `None` except at `sigma0`. There `None` is the documented default, and `units="sigma0"` without it keeps its `ValueError` ("needs sigma0"). Range and finiteness keep their `ValueError` text. Accepted values (`np.float64`, `np.int64`, `np.float32`, `Fraction`, `int`) are stored as Python `float`, so `rfft_scale` is a Python `float` too.
- [ ] A non-string `units` raises `TypeError("units must be a string, got {type}")`. An unknown string keeps its `ValueError`.
- [ ] `CellSet` `m`/`n` and `CellSet.bands`' band list raise `TypeError("{name} must be an integer index array, got dtype {dtype}")` for a bool, str, complex or object array, and `ValueError("{name} must hold integer indices, got {first}")` for a float that is not integral or not finite. Integer and integral-float arrays pass, as `tests/test_dom_domains.py:148` pins. `CellSet.bands` no longer truncates `1.5` to band 1 or takes `True` and `"2"`. The caller's index arrays stay writeable.
- [ ] `CellSet.from_mask` raises `TypeError("keep must be a boolean mask, got dtype {dtype}")` for a `keep` that is not bool, and keeps its `ValueError` for a wrong shape.
- [ ] `ChannelAxis("AE")` (and `ChannelAxis(np.str_("AE"))`) raises `TypeError("names must be a tuple of channel names, got str")` instead of splitting it into `("A", "E")`.
- [ ] No `raise TypeError` in `src/domainia/domains.py` has `"; got"`. The five that had it now say `", got"`. `tests/test_dom_evolutionary_cov.py:101` and `tests/test_dom_variance_generic.py:77` pin the new `_cells_and_window` text.
- [ ] `tests/test_wrong_kinds_domains.py`: `190 passed` on both lanes. The tests of the maps, data and covariance pytrees pass unchanged. The full suite and the NumPy lane pass.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_domains.py tests/test_dom_domains.py tests/test_dom_evolutionary_cov.py tests/test_dom_variance_generic.py` → `226 passed`

**Steps:**

- [ ] **Step 1: Grep the sites in `src/domainia/domains.py`**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "= float(\|np.asarray(\|tuple(self.names)\|raise TypeError\|not in FOURIER_UNITS" src/domainia/domains.py`
Expected: 13 hits.
- `v = float(value)` in `_positive_float` (used for both `dt`s and `sigma0`), and `t0 = float(self.t0)` in `TimeDomain`. `float()` takes `True` as 1.0, `"0.5"` as 0.5, a `Decimal`, 0-d and size-1 arrays (the last with a NumPy `DeprecationWarning`), and drops the imaginary part of `np.complex64(1)` with a `ComplexWarning`. Step 6 puts `require_real` in their place.
- `if self.units not in FOURIER_UNITS:` in `FourierDomain`. A non-string `units` (`None`, `3`, `b"rfft"`, `True`) is reported as the `ValueError` `units must be one of (...); got None`. Step 6 adds `require_str` before it.
- `raw = np.asarray(value)` in `_cell_index` (`CellSet` `m`/`n`). It reports a bool, str, complex or object array as a `ValueError` (`m must hold integer cell indices; got dtype bool`). Step 6 makes it a wrapper of `index_array`.
- `b = np.asarray(bands, dtype=np.int64)` in `CellSet.bands`. The cast truncates `1.5` to band 1, takes `True` as band 1 and `"2"` as band 2, and gives Python's `int()` message for complex and `None`. Step 6 uses `index_array`.
- `keep = np.asarray(keep)` in `CellSet.from_mask`. Its one `ValueError` covers a wrong dtype and a wrong shape (`keep must be a boolean mask of shape (128,)`). Step 6 splits it: `require_dtype`, then the shape.
- `cols = np.asarray(wilson.grid.edge_columns(...), dtype=np.int64)` in `CellSet._edge`. `edge_columns` returns integer columns, so this cast is not a check. It stays.
- Five `raise TypeError`s with `"; got"` (the unhashable window's is on the next line): `CellSet`'s `grid`, `WDMDomain`'s `cells`, the unhashable window, `_cells_and_window`'s `cells` and the channel names. Step 6 gives them the `", got"` form; the first, second and fourth through `require_instance`.
- `names = tuple(self.names)` in `ChannelAxis`. `tuple("AE")` is `("A", "E")`, so `ChannelAxis("AE")` builds a two-channel axis. Step 6 refuses a `str` before it.

Six entry points raise a bare `AttributeError` for a wrong-kind argument (probed with `None`, `3`, `"x"`, a `CellSet` and a `TimeDomain`, 28 cases): `CellSet.all_active`, `CellSet.bands` and `WDMDomain.full` (`'NoneType' object has no attribute 'Nf'`), `CellSet.from_mask` (`... no attribute 'N'`), and `CellSet.index_of` and `CellSet.complement` (`... no attribute 'grid'`). `WDMDomain.full` calls `CellSet.all_active(grid)` first, and `complement` calls `index_of(sub)` first, so four new checks cover all six. Every other entry point already raises `TypeError` for these values: `CellSet(grid, ...)` and `WDMDomain(cells)` with their own check, the `window` arguments through `require_window`, and the integers through `domains._int` (Task 1).

- [ ] **Step 2: Grep the tests that pin these sites**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn 'integer cell indices\|must be a domainia.wilson.Grid\|cells must be a CellSet\|match="hashable"\|channel names must\|keep must be a boolean\|units must be one of\|match="integer"' tests --include='*.py'`
Expected: 8 hits.
- `tests/test_dom_evolutionary_cov.py:101` and `tests/test_dom_variance_generic.py:77`: `match=r"^cells must be a CellSet or a WDMDomain; got Grid$"`, anchored. Step 4 changes `; got` to `, got` in both.
- `tests/test_dom_domains.py:146`: `pytest.raises(ValueError, match="integer")` in a loop whose last case, `((True,), (0,))`, is a bool array. Step 4 moves that case to its own `TypeError` check. The non-integral and NaN cases stay `ValueError`, and the integral-float assertion at `:148` stays as it is.
- `tests/test_translate_coordinates.py:159`: `match="channel names must be non-empty strings"` (`from_coordinates(R, d, ("A", 1))`, through `ChannelAxis`). It is a prefix match, and the new text `channel names must be non-empty strings, got ('A', 1)` still matches. No edit.
- `tests/test_dom_windows.py:81`: `match="hashable"`. It still matches. No edit.
- `tests/wilson/test_checks.py:114` and `:203`: Task 1's tests of `require_instance` and `index_array` themselves. No edit.
- `tests/wilson/test_wrong_kinds.py:241`: Task 2's `test_eq45_plan_keep_message_names_the_dtype` (`keep must be a boolean mask over the packed cells, got dtype int64`). It pins `wilson.evolutionary.eq45_plan`, not `domains.py`. No edit.

`tests/test_dom_domains.py:27`, `:62` and `:185` (a `pytest.raises(TypeError)` with no `match=`) are left to Task 13. No test passes a wrong kind to a domain expecting success. With Step 6 applied and no test edits, the full suite fails exactly 4 tests: `tests/test_dom_domains.py::test_validation` and the three cases of the two anchored pins (`test_dom_evolutionary_cov.py::test_cells_must_be_a_cell_set_or_a_wdm_domain`, `test_dom_variance_generic.py::test_cells_must_be_a_cell_set_or_a_wdm_domain[BandAveragedVariance]` and `[MidpointVariance]`). So these three files are the whole pin list.

- [ ] **Step 3: Write the failing tests `tests/test_wrong_kinds_domains.py`**

```python
"""``domainia.domains`` refuses a wrong kind with ``TypeError`` and a wrong value with ``ValueError``
(spec 2026-11-08 §1.5, §2.3-§2.6): grids, cell sets, real scalars, ``units``, integers, index
arrays, the ``from_mask`` mask and channel names."""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import (CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain,
                              _cells_and_window)

GRID = wilson.Grid(8, 16, 0.5)
CELLS = CellSet.all_active(GRID)
FULL = WDMDomain(CELLS)
TIME = TimeDomain(GRID.N, GRID.dt)


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


#: Each slot that takes a ``wilson.Grid``, as a function of the value under test.
GRID_SLOTS = {
    "CellSet": lambda v: CellSet(v, (1,), (0,)),
    "CellSet.all_active": lambda v: CellSet.all_active(v),
    "CellSet.bands": lambda v: CellSet.bands(v, (1, 2)),
    "CellSet.from_mask": lambda v: CellSet.from_mask(v, np.ones(GRID.N, bool)),
    "WDMDomain.full": lambda v: WDMDomain.full(v),
}


@pytest.mark.parametrize("bad", [None, (8, 16, 0.5), CELLS, TIME],
                         ids=["None", "tuple", "CellSet", "TimeDomain"])
@pytest.mark.parametrize("slot", GRID_SLOTS)
def test_a_grid_of_the_wrong_kind_raises_type_error(slot, bad):
    """The class methods read ``grid.Nf`` or ``grid.N`` first: a bare ``AttributeError`` before.
    ``WDMDomain.full`` gets its check from ``CellSet.all_active``."""
    message = f"grid must be a domainia.wilson.Grid, got {type(bad).__name__}"
    with pytest.raises(TypeError, match=_exact(message)):
        GRID_SLOTS[slot](bad)


SUB_SLOTS = {
    "CellSet.index_of": lambda v: CELLS.index_of(v),
    "CellSet.complement": lambda v: CELLS.complement(v),
}


@pytest.mark.parametrize("bad", [None, FULL, GRID, (1, 2)],
                         ids=["None", "WDMDomain", "Grid", "tuple"])
@pytest.mark.parametrize("slot", SUB_SLOTS)
def test_a_sub_of_the_wrong_kind_raises_type_error(slot, bad):
    """``index_of`` read ``sub.grid`` first: a bare ``AttributeError`` before.  ``complement``
    gets its check from ``index_of``."""
    with pytest.raises(TypeError, match=_exact(f"sub must be a CellSet, got {type(bad).__name__}")):
        SUB_SLOTS[slot](bad)


@pytest.mark.parametrize("bad", [None, GRID, TIME], ids=["None", "Grid", "TimeDomain"])
def test_cells_of_the_wrong_kind_take_the_got_form(bad):
    kind = type(bad).__name__
    with pytest.raises(TypeError, match=_exact(f"cells must be a CellSet, got {kind}")):
        WDMDomain(bad)
    with pytest.raises(TypeError, match=_exact(f"cells must be a CellSet or a WDMDomain, got {kind}")):
        _cells_and_window(bad)


class _UnhashableWindow:
    """A Window by the protocol that cannot be hashed."""

    name = "unhashable"
    __hash__ = None  # type: ignore[assignment]

    def sample(self, Nf, Nt, dt):
        return wilson.MEYER.sample(Nf, Nt, dt)


def test_an_unhashable_window_takes_the_got_form():
    with pytest.raises(TypeError, match=r"^window must be hashable \(.*\), "
                                        r"got an unhashable _UnhashableWindow$"):
        WDMDomain(CELLS, window=_UnhashableWindow())


#: Each real-scalar slot, as a function of the value under test.
REAL_SLOTS = {
    "TimeDomain.dt": lambda v: TimeDomain(128, v),
    "TimeDomain.t0": lambda v: TimeDomain(128, 0.5, v),
    "FourierDomain.dt": lambda v: FourierDomain(128, v, 0, 10, "rfft"),
    "FourierDomain.sigma0": lambda v: FourierDomain(128, 0.5, 0, 10, "sigma0", sigma0=v),
    "FourierDomain.full.dt": lambda v: FourierDomain.full(128, v),
}


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (1.0 + 0j, "complex"),
    (np.complex64(1.0), "complex64"), (Decimal("0.5"), "Decimal"), (np.array(0.5), "ndarray"),
    (np.array([0.5]), "ndarray"),
], ids=["True", "np.True_", "str", "complex", "complex64", "Decimal", "0-d", "size-1"])
@pytest.mark.parametrize("slot", REAL_SLOTS)
def test_a_real_of_the_wrong_kind_raises_type_error(slot, bad, kind):
    """``float()`` took ``True`` as 1.0, ``"0.5"`` as 0.5, a ``Decimal`` and size-1 arrays,
    and dropped the imaginary part of a NumPy complex scalar with a ``ComplexWarning``."""
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be a real number, got {kind}")):
        REAL_SLOTS[slot](bad)


@pytest.mark.parametrize("slot", [s for s in REAL_SLOTS if s != "FourierDomain.sigma0"])
def test_none_is_a_wrong_kind_where_it_is_not_a_default(slot):
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be a real number, got NoneType")):
        REAL_SLOTS[slot](None)


def test_a_missing_sigma0_stays_a_value_error():
    """``sigma0=None`` is the documented default: with ``units="sigma0"`` it is a missing value."""
    with pytest.raises(ValueError, match="units='sigma0' needs sigma0"):
        REAL_SLOTS["FourierDomain.sigma0"](None)


@pytest.mark.parametrize("v", [np.float64(0.5), np.int64(2), np.float32(0.5), Fraction(1, 2), 2],
                         ids=["float64", "int64", "float32", "Fraction", "int"])
def test_real_scalars_are_taken_and_stored_as_float(v):
    """Spec §1.5: the domains store ``dt``, ``t0`` and ``sigma0`` as Python floats, so
    ``rfft_scale`` stays a Python float too."""
    t = TimeDomain(128, v, v)
    assert (type(t.dt), type(t.t0)) == (float, float) and (t.dt, t.t0) == (float(v), float(v))
    f = FourierDomain(128, v, 0, 10, "dt_rfft")
    assert type(f.dt) is float and type(f.rfft_scale) is float
    assert f == FourierDomain(128, float(v), 0, 10, "dt_rfft")
    s = FourierDomain(128, 0.5, 0, 10, "sigma0", sigma0=v)
    assert type(s.sigma0) is float and type(s.rfft_scale) is float and s.sigma0 == float(v)
    assert type(FourierDomain.full(128, v).dt) is float


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
@pytest.mark.parametrize("slot", ["TimeDomain.dt", "FourierDomain.dt", "FourierDomain.sigma0",
                                  "FourierDomain.full.dt"])
def test_reals_out_of_range_stay_value_errors(slot, bad):
    name = slot.split(".")[-1]
    with pytest.raises(ValueError, match=f"^{name} must be finite and > 0; got"):
        REAL_SLOTS[slot](bad)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_a_non_finite_t0_stays_a_value_error(bad):
    with pytest.raises(ValueError, match="^t0 must be finite; got"):
        TimeDomain(128, 0.5, bad)


@pytest.mark.parametrize(("bad", "kind"), [
    (None, "NoneType"), (3, "int"), (b"rfft", "bytes"), (True, "bool"), (["rfft"], "list"),
], ids=["None", "int", "bytes", "bool", "list"])
def test_units_of_the_wrong_kind_raise_type_error(bad, kind):
    """A non-string ``units`` was reported as ``units must be one of (...); got None``."""
    with pytest.raises(TypeError, match=_exact(f"units must be a string, got {kind}")):
        FourierDomain(128, 0.5, 0, 10, bad)


def test_an_unknown_units_string_stays_a_value_error():
    with pytest.raises(ValueError, match=r"^units must be one of \('rfft', 'dt_rfft', 'sigma0'\); "
                                         r"got 'hz'$"):
        FourierDomain(128, 0.5, 0, 10, "hz")
    assert FourierDomain(128, 0.5, 0, 10, np.str_("rfft")) == FourierDomain(128, 0.5, 0, 10, "rfft")


#: Each integer slot (``domains._int``, Task 1), as a function of the value under test.
INT_SLOTS = {
    "TimeDomain.N": lambda v: TimeDomain(v, 0.5),
    "FourierDomain.N": lambda v: FourierDomain(v, 0.5, 0, 10, "rfft"),
    "FourierDomain.k0": lambda v: FourierDomain(128, 0.5, v, 10, "rfft"),
    "FourierDomain.n": lambda v: FourierDomain(128, 0.5, 0, v, "rfft"),
    "FourierDomain.full.N": lambda v: FourierDomain.full(v, 0.5),
    "CellSet.drop_columns.lead": lambda v: CELLS.drop_columns(v, 1),
    "CellSet.drop_columns.trail": lambda v: CELLS.drop_columns(1, v),
    "CellSet.columns.lead": lambda v: CELLS.columns(v, 1),
    "CellSet.columns.trail": lambda v: CELLS.columns(1, v),
}


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), (2.0, "float"), (np.float64(2.0), "float64"), ("2", "str"), (None, "NoneType"),
])
@pytest.mark.parametrize("slot", INT_SLOTS)
def test_an_integer_of_the_wrong_kind_raises_type_error(slot, bad, kind):
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be an integer, got {kind}")):
        INT_SLOTS[slot](bad)


def test_numpy_integers_are_taken_and_stored_as_int():
    t = TimeDomain(np.int64(128), 0.5)
    f = FourierDomain(np.int64(128), 0.5, np.int64(0), np.int64(10), "rfft")
    assert type(t.N) is int and (type(f.N), type(f.k0), type(f.n)) == (int, int, int)
    assert f == FourierDomain(128, 0.5, 0, 10, "rfft") and type(FourierDomain.full(np.int64(128), 0.5).N) is int
    assert CELLS.drop_columns(np.int64(2), np.int64(1)) == CELLS.drop_columns(2, 1)
    assert CELLS.columns(np.int64(2), np.int64(1)) == CELLS.columns(2, 1)


#: Each index-array slot, as a function of a one-entry value.
INDEX_SLOTS = {
    "CellSet.m": lambda v: CellSet(GRID, v, [0]),
    "CellSet.n": lambda v: CellSet(GRID, [1], v),
    "CellSet.bands.bands": lambda v: CellSet.bands(GRID, v),
}


@pytest.mark.parametrize(("bad", "dtype"), [
    ([True], "bool"), (["2"], "<U1"), ([2 + 0j], "complex128"), ([None], "object"),
], ids=["bool", "str", "complex", "None"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_an_index_array_of_the_wrong_dtype_raises_type_error(slot, bad, dtype):
    """``CellSet`` reported these as ``ValueError``; ``CellSet.bands`` took ``[True]`` as band 1
    and ``["2"]`` as band 2."""
    name = slot.split(".")[-1]
    with pytest.raises(TypeError, match=_exact(f"{name} must be an integer index array, got dtype {dtype}")):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize(("bad", "first"), [([1.5], "1.5"), ([np.nan], "nan")], ids=["1.5", "nan"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_a_non_integral_index_raises_value_error(slot, bad, first):
    """``CellSet.bands`` truncated ``1.5`` to band 1."""
    name = slot.split(".")[-1]
    with pytest.raises(ValueError, match=_exact(f"{name} must hold integer indices, got {first}")):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_integer_and_integral_float_index_arrays_are_taken(slot):
    want = INDEX_SLOTS[slot]([2])
    for v in (np.array([2], np.int64), np.array([2], np.int32), np.array([2.0]), (np.float64(2.0),)):
        assert INDEX_SLOTS[slot](v) == want


def test_the_caller_index_arrays_stay_writeable():
    m, n = np.array([1, 2], np.int64), np.array([0, 0], np.int64)
    c = CellSet(GRID, m, n)
    assert m.flags.writeable and n.flags.writeable and not c.m_array.flags.writeable
    assert not np.shares_memory(c.m_array, m)


@pytest.mark.parametrize(("bad", "dtype"), [
    (np.ones(GRID.N, np.int64), "int64"), (np.arange(GRID.N), "int64"), (np.ones(GRID.N), "float64"),
    (np.ones(GRID.N, complex), "complex128"), (np.array(["1"] * GRID.N), "<U1"), (None, "object"),
], ids=["ones-int", "arange", "float", "complex", "str", "None"])
def test_a_mask_of_the_wrong_dtype_raises_type_error(bad, dtype):
    """``keep`` is a value array: its dtype is its kind.  An index array such as
    ``np.arange(N)`` is the likely mistake; it was a ``ValueError`` before."""
    with pytest.raises(TypeError, match=_exact(f"keep must be a boolean mask, got dtype {dtype}")):
        CellSet.from_mask(GRID, bad)


@pytest.mark.parametrize("bad", [np.ones(GRID.N - 1, bool), np.ones((GRID.Nf, GRID.Nt), bool)],
                         ids=["short", "2-d"])
def test_a_mask_of_the_wrong_shape_stays_a_value_error(bad):
    with pytest.raises(ValueError, match=_exact(f"keep must be a boolean mask of shape ({GRID.N},)")):
        CellSet.from_mask(GRID, bad)


def test_boolean_masks_are_taken():
    keep = CELLS.drop_columns(2, 1).keep_mask()
    want = CELLS.drop_columns(2, 1)
    assert CellSet.from_mask(GRID, keep) == want
    assert CellSet.from_mask(GRID, keep.tolist()) == want
    assert CellSet.from_mask(GRID, np.asarray(keep, np.bool_)) == want


@pytest.mark.parametrize(("bad", "kind"), [("AE", "str"), (np.str_("AE"), "str_")],
                         ids=["str", "np.str_"])
def test_channel_axis_refuses_a_bare_string(bad, kind):
    """``ChannelAxis("AE")`` used to split the string into the names ``("A", "E")``."""
    with pytest.raises(TypeError, match=_exact(f"names must be a tuple of channel names, got {kind}")):
        ChannelAxis(bad)


@pytest.mark.parametrize("bad", [("A", 3), ("A", ""), (b"A",)], ids=["int", "empty", "bytes"])
def test_channel_names_that_are_not_non_empty_strings_take_the_got_form(bad):
    with pytest.raises(TypeError, match=_exact(f"channel names must be non-empty strings, got {bad!r}")):
        ChannelAxis(bad)


def test_tuples_and_lists_of_names_are_taken():
    assert ChannelAxis(("A", "E")).names == ("A", "E") == ChannelAxis(["A", "E"]).names
    assert ChannelAxis().size == 0
    with pytest.raises(ValueError, match="unique"):
        ChannelAxis(("A", "A"))
```

`_cells_and_window` is private, and the file calls it directly: its consumers (`EvolutionaryCovariance`, `BandAveragedVariance`, `MidpointVariance`) are pinned in their own test files (Step 4). The integer slots are Task 1's `domains._int`, and their cases pass already; they are here so that every slot kind of `domains.py` has its row.

- [ ] **Step 4: Move the pinned cases**

`tests/test_dom_domains.py`, in `test_validation`:

```python
    # non-integral, non-finite or boolean indices are refused, never truncated (2.9999999 -> 2)
    for m_bad, n_bad in (((2.9999999,), (0,)), ((1,), (0.5,)), ((np.nan,), (0,)), ((True,), (0,))):
        with pytest.raises(ValueError, match="integer"):
            CellSet(g, m_bad, n_bad)
```
→
```python
    # non-integral or non-finite indices are refused, never truncated (2.9999999 -> 2)
    for m_bad, n_bad in (((2.9999999,), (0,)), ((1,), (0.5,)), ((np.nan,), (0,))):
        with pytest.raises(ValueError, match="integer"):
            CellSet(g, m_bad, n_bad)
    # a boolean index array is a wrong kind
    with pytest.raises(TypeError, match="^m must be an integer index array, got dtype bool$"):
        CellSet(g, (True,), (0,))
```

`tests/test_dom_evolutionary_cov.py`, in `test_cells_must_be_a_cell_set_or_a_wdm_domain`, and `tests/test_dom_variance_generic.py`, in `test_cells_must_be_a_cell_set_or_a_wdm_domain` (the same line in both files):

```python
    with pytest.raises(TypeError, match=r"^cells must be a CellSet or a WDMDomain; got Grid$"):
```
→
```python
    with pytest.raises(TypeError, match=r"^cells must be a CellSet or a WDMDomain, got Grid$"):
```

- [ ] **Step 5: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_domains.py tests/test_dom_domains.py tests/test_dom_evolutionary_cov.py tests/test_dom_variance_generic.py 2>&1 | tail -1`
Expected: `114 failed, 112 passed, 10 warnings`. The 10 warnings come from `test_a_real_of_the_wrong_kind_raises_type_error`: 5 `ComplexWarning`s (`float(np.complex64(1.0))`) and 5 `DeprecationWarning`s (`float()` of a size-1 array).
- `test_wrong_kinds_domains.py`: 110 failed, 80 passed.
  - `test_a_grid_of_the_wrong_kind_raises_type_error`: all 20. The class methods raise `AttributeError` (for `[CellSet.all_active-None]`: `'NoneType' object has no attribute 'Nf'`). The 4 `CellSet` cases raise the old text `grid must be a domainia.wilson.Grid; got NoneType`, which does not match.
  - `test_a_sub_of_the_wrong_kind_raises_type_error`: all 8, `AttributeError` (`'NoneType' object has no attribute 'grid'`).
  - `test_cells_of_the_wrong_kind_take_the_got_form` (3) and `test_an_unhashable_window_takes_the_got_form` (1): the old `"; got"` texts do not match.
  - `test_a_real_of_the_wrong_kind_raises_type_error`: all 40. `Failed: DID NOT RAISE <class 'TypeError'>` for `True`, `np.True_`, `"0.5"`, `np.complex64`, `Decimal` and both arrays; Python's `float() argument must be a string or a real number, not 'complex'` for `1+0j`. `test_none_is_a_wrong_kind_where_it_is_not_a_default` (4): Python's `float()` message.
  - `test_units_of_the_wrong_kind_raise_type_error` (5): `ValueError: units must be one of ('rfft', 'dt_rfft', 'sigma0'); got None`.
  - `test_an_index_array_of_the_wrong_dtype_raises_type_error` (12): `CellSet` raises `ValueError: m must hold integer cell indices; got dtype bool`; `CellSet.bands` gives `DID NOT RAISE` for bool and str, and Python's `int()` message for complex and `None`.
  - `test_a_non_integral_index_raises_value_error` (6): `CellSet`'s old text (`m must hold integer cell indices; got np.float64(1.5)`) does not match; `CellSet.bands` gives `DID NOT RAISE` for `1.5` and `cannot convert float NaN to integer` for NaN.
  - `test_a_mask_of_the_wrong_dtype_raises_type_error` (6): `ValueError: keep must be a boolean mask of shape (128,)`.
  - `test_channel_axis_refuses_a_bare_string` (2): `DID NOT RAISE`. `test_channel_names_that_are_not_non_empty_strings_take_the_got_form` (3): the old `"; got"` text.
  - The 80 that pass: the 45 integer cases and `test_numpy_integers_are_taken_and_stored_as_int` (Task 1), `test_real_scalars_are_taken_and_stored_as_float` (5: `float()` already returned a `float`), the 16 + 3 range cases, `test_a_missing_sigma0_stays_a_value_error`, `test_an_unknown_units_string_stays_a_value_error`, the 3 `test_integer_and_integral_float_index_arrays_are_taken`, `test_the_caller_index_arrays_stay_writeable`, the 2 `test_a_mask_of_the_wrong_shape_stays_a_value_error`, `test_boolean_masks_are_taken` and `test_tuples_and_lists_of_names_are_taken`.
- `tests/test_dom_domains.py::test_validation` (`ValueError: m must hold integer cell indices; got dtype bool`), and the anchored pins in `tests/test_dom_evolutionary_cov.py` (1) and `tests/test_dom_variance_generic.py` (2): 4 failed.

- [ ] **Step 6: `src/domainia/domains.py`**

The import, after `from domainia import wilson`:

```python
from domainia.wilson._checks import require_int
```
→
```python
from domainia.wilson._checks import (index_array, require_dtype, require_instance, require_int,
                                     require_real, require_str)
```

`_positive_float`, whole. It serves `TimeDomain`'s and `FourierDomain`'s `dt` and `FourierDomain`'s `sigma0`. `require_real` returns a Python `float`, so the stored values stay Python floats (spec §1.5):

```python
def _positive_float(name: str, value: object) -> float:
    v = float(value)  # type: ignore[arg-type]
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    return v
```
→
```python
def _positive_float(name: str, value: object) -> float:
    """``value`` as a Python ``float`` (so ``rfft_scale`` is one too): ``TypeError`` unless it
    is a real number (``require_real``: no ``bool``, ``str``, ``None``, complex or array), then
    ``ValueError`` unless it is finite and > 0."""
    v = require_real(name, value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    return v
```

`TimeDomain.__post_init__`, whole. The default of `t0` is `0.0`, not `None`, so `None` is a wrong kind:

```python
    def __post_init__(self) -> None:
        object.__setattr__(self, "N", _int("N", self.N, minimum=1))
        object.__setattr__(self, "dt", _positive_float("dt", self.dt))
        t0 = float(self.t0)
        if not math.isfinite(t0):
            raise ValueError(f"t0 must be finite; got {self.t0!r}")
        object.__setattr__(self, "t0", t0)
```
→
```python
    def __post_init__(self) -> None:
        object.__setattr__(self, "N", _int("N", self.N, minimum=1))
        object.__setattr__(self, "dt", _positive_float("dt", self.dt))
        t0 = require_real("t0", self.t0)
        if not math.isfinite(t0):
            raise ValueError(f"t0 must be finite; got {self.t0!r}")
        object.__setattr__(self, "t0", t0)
```

`FourierDomain.__post_init__`, the `units` check. The kind check goes before the membership test, whose `ValueError` text stays:

```python
            raise ValueError(f"bins [{k0}, {k0 + n}) exceed the {N // 2 + 1} rfft bins of N={N}")
        if self.units not in FOURIER_UNITS:
```
→
```python
            raise ValueError(f"bins [{k0}, {k0 + n}) exceed the {N // 2 + 1} rfft bins of N={N}")
        require_str("units", self.units)
        if self.units not in FOURIER_UNITS:
```

`_cell_index`, whole. It stays a wrapper because `index_array` returns the caller's own array for `int64` input (`astype(..., copy=False)`), and `CellSet.__post_init__` then makes its arrays read-only: `np.array(...)` copies, so the caller's array stays writeable (`test_identity_is_value_based_and_cheap` in `tests/test_dom_domains.py`). An empty array now has its dtype checked too (the old check skipped `size == 0`); `()` and `[]` are float64 and still pass:

```python
def _cell_index(name: str, value: object) -> IntArray:
    """``value`` as a fresh flat int64 array (a copy: the caller's array is never touched).

    Integer dtypes pass; finite integral floats (``3.0``) are accepted; anything else
    (``2.9999999``, NaN, booleans) raises instead of being truncated by the int64 cast.
    """
    raw = np.asarray(value)
    if raw.size and raw.dtype.kind not in "iu":
        if raw.dtype.kind != "f":
            raise ValueError(f"{name} must hold integer cell indices; got dtype {raw.dtype}")
        bad = ~np.isfinite(raw) | (raw != np.trunc(raw))
        if np.any(bad):
            first = raw.reshape(-1)[np.flatnonzero(bad.reshape(-1))[0]]
            raise ValueError(f"{name} must hold integer cell indices; got {first!r}")
    return np.array(raw, dtype=np.int64).reshape(-1)
```
→
```python
def _cell_index(name: str, value: object) -> IntArray:
    """``value`` as a fresh flat int64 array (a copy: the caller's array is never touched).

    Through ``index_array``: integer dtypes pass and finite integral floats (``3.0``) are
    accepted; a float that is not integral or not finite (``2.9999999``, NaN) raises
    ``ValueError`` instead of being truncated by the int64 cast; a bool, str, object or
    complex array raises ``TypeError``.
    """
    return np.array(index_array(name, value), dtype=np.int64).reshape(-1)
```

`CellSet.__post_init__`, its head:

```python
    def __post_init__(self) -> None:
        if not isinstance(self.grid, wilson.Grid):
            raise TypeError(f"grid must be a domainia.wilson.Grid; got {type(self.grid).__name__}")
```
→
```python
    def __post_init__(self) -> None:
        require_instance("grid", self.grid, wilson.Grid, "a domainia.wilson.Grid")
```

`CellSet.index_of`, its head. `complement` calls `index_of(sub)` first, so it gets the check too:

```python
    def index_of(self, sub: CellSet) -> IntArray:
        """Positions of ``sub``'s cells within ``self`` (``sub`` must be a subset)."""
        if sub.grid != self.grid:
```
→
```python
    def index_of(self, sub: CellSet) -> IntArray:
        """Positions of ``sub``'s cells within ``self`` (``sub`` must be a subset)."""
        require_instance("sub", sub, CellSet, "a CellSet")
        if sub.grid != self.grid:
```

`CellSet.all_active`. `WDMDomain.full` calls it first, so it gets the check too:

```python
    def all_active(cls, grid: wilson.Grid) -> CellSet:
        """Every active cell: ``wilson.grid.packed_cells(Nf, Nt)``."""
        m, n = packed_cells(grid.Nf, grid.Nt)
```
→
```python
    def all_active(cls, grid: wilson.Grid) -> CellSet:
        """Every active cell: ``wilson.grid.packed_cells(Nf, Nt)``."""
        require_instance("grid", grid, wilson.Grid, "a domainia.wilson.Grid")
        m, n = packed_cells(grid.Nf, grid.Nt)
```

`CellSet.bands`, its head. The two `ValueError`s below keep their text:

```python
    def bands(cls, grid: wilson.Grid, bands: tuple[int, ...] | list[int]) -> CellSet:
        """Every active cell of the bands ``bands`` (strictly ascending)."""
        b = np.asarray(bands, dtype=np.int64).reshape(-1)
```
→
```python
    def bands(cls, grid: wilson.Grid, bands: tuple[int, ...] | list[int]) -> CellSet:
        """Every active cell of the bands ``bands`` (strictly ascending)."""
        require_instance("grid", grid, wilson.Grid, "a domainia.wilson.Grid")
        b = index_array("bands", bands).reshape(-1)
```

`CellSet.from_mask`, whole. `grid` is checked first, because the shape message reads `grid.N`. `None` now raises `TypeError` (object dtype), not `ValueError`:

```python
    def from_mask(cls, grid: wilson.Grid, keep: BoolArray) -> CellSet:
        """The cells where ``keep`` (a boolean mask over the ``N`` packed cells) is True."""
        keep = np.asarray(keep)
        if keep.dtype != np.bool_ or keep.shape != (grid.N,):
            raise ValueError(f"keep must be a boolean mask of shape ({grid.N},)")
        m, n = packed_cells(grid.Nf, grid.Nt)
        return cls(grid, tuple(m[keep].tolist()), tuple(n[keep].tolist()))
```
→
```python
    def from_mask(cls, grid: wilson.Grid, keep: BoolArray) -> CellSet:
        """The cells where ``keep`` (a boolean mask over the ``N`` packed cells) is True.

        ``keep`` of another dtype (an index array such as ``np.arange(N)``) raises
        ``TypeError``; a mask of the wrong shape raises ``ValueError``."""
        require_instance("grid", grid, wilson.Grid, "a domainia.wilson.Grid")
        keep = np.asarray(keep)
        require_dtype("keep", keep, "b", "a boolean mask")
        if keep.shape != (grid.N,):
            raise ValueError(f"keep must be a boolean mask of shape ({grid.N},)")
        m, n = packed_cells(grid.Nf, grid.Nt)
        return cls(grid, tuple(m[keep].tolist()), tuple(n[keep].tolist()))
```

`WDMDomain.__post_init__`, whole. The window checks stay; the hashable message takes the `", got"` form:

```python
    def __post_init__(self) -> None:
        if not isinstance(self.cells, CellSet):
            raise TypeError(f"cells must be a CellSet; got {type(self.cells).__name__}")
        require_window(self.window)
        try:
            hash(self.window)
        except TypeError:
            raise TypeError(f"window must be hashable (it is part of the domain's identity and "
                            f"static under jax.jit); got an unhashable "
                            f"{type(self.window).__name__}") from None
```
→
```python
    def __post_init__(self) -> None:
        require_instance("cells", self.cells, CellSet, "a CellSet")
        require_window(self.window)
        try:
            hash(self.window)
        except TypeError:
            raise TypeError(f"window must be hashable (it is part of the domain's identity and "
                            f"static under jax.jit), got an unhashable "
                            f"{type(self.window).__name__}") from None
```

`_cells_and_window`, its end. The `WDMDomain` branch above returns first, so the expected kind names both:

```python
        return cells.cells, cells.window
    if not isinstance(cells, CellSet):
        raise TypeError(f"cells must be a CellSet or a WDMDomain; got {type(cells).__name__}")
    return cells, require_window(default if window is None else window)
```
→
```python
        return cells.cells, cells.window
    require_instance("cells", cells, CellSet, "a CellSet or a WDMDomain")
    return cells, require_window(default if window is None else window)
```

`ChannelAxis`, from the docstring to the element check. `np.str_` is a `str` subclass, so it is refused too. A list or an array of names is still taken, and a non-iterable keeps Python's own `TypeError` from `tuple()`. The element check keeps its exception, which `translate.py`'s `_names` docstring documents; only its punctuation changes:

```python
    """Named channels; ``ChannelAxis()`` means one channel and no channel axis."""

    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not all(isinstance(s, str) and s for s in names):
            raise TypeError(f"channel names must be non-empty strings; got {self.names!r}")
```
→
```python
    """Named channels; ``ChannelAxis()`` means one channel and no channel axis.

    ``names`` is a tuple (or a list) of non-empty strings (``TypeError`` otherwise), all
    different (``ValueError``).  A bare string is refused, not split into one name per
    character: ``ChannelAxis("AE")`` raises ``TypeError``; write ``ChannelAxis(("A", "E"))``.
    """

    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.names, str):
            raise TypeError(f"names must be a tuple of channel names, "
                            f"got {type(self.names).__name__}")
        names = tuple(self.names)
        if not all(isinstance(s, str) and s for s in names):
            raise TypeError(f"channel names must be non-empty strings, got {self.names!r}")
```

- [ ] **Step 7: Grep the result**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "= float(\|np.asarray(bands\|integer cell indices\|isinstance(self.grid\|isinstance(self.cells\|isinstance(cells, CellSet)" src/domainia/domains.py`
Expected: no output.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "raise TypeError" src/domainia/domains.py`
Expected: 3 hits, each with `", got"` on its line or the next: the unhashable window (`WDMDomain`), the bare string and the channel names (`ChannelAxis`). The other kind checks raise from `require_instance`, `require_real`, `require_str`, `require_dtype` and `index_array`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" src/domainia/domains.py`
Expected: 8 hits, all `ValueError`, with their text unchanged: `_int`'s `must be >= {minimum}`, `_positive_float`'s `must be finite and > 0`, `t0 must be finite`, `units must be one of`, `sigma0 is set only with units='sigma0'`, the two `bands must ...` messages, and `channel names must be unique`. So Task 9's gate already holds for `domains.py`.

- [ ] **Step 8: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_domains.py tests/test_dom_domains.py tests/test_dom_evolutionary_cov.py tests/test_dom_variance_generic.py 2>&1 | tail -1`
Expected: `226 passed`, no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_domains.py 2>&1 | tail -1`
Expected: `190 passed`.

- [ ] **Step 9: The pytree tests, the full suite and the NumPy lane**

`CellSet` and `WDMDomain` are never pytrees themselves: they are static meta of the maps, the data classes and the variances, and a `jax.tree_util` rebuild passes them through unchanged. So the checks of Step 6 never see a placeholder leaf (spec §2.7). These files rebuild maps and data under `tree_map` and `jit`, and `tests/test_translate_coordinates.py` holds the `ChannelAxis` prefix pin:

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_maps_base.py tests/test_dom_built_for.py tests/test_dom_data.py tests/test_dom_cov_api.py tests/test_dom_windows.py tests/test_translate_coordinates.py 2>&1 | tail -1`
Expected: passes.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: all pass (3 skipped, the oracle tests), 0 errors, no warnings summary. A failure here is a pin that Step 2 did not find: if it expects the old kind behaviour, move it to `TypeError` as Step 4 does, and record it in the errata.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: all pass, 0 errors, no warnings summary.

The float64 bit gate is not run: this task does not touch `wilson/transform.py` or `maps/`, and the domains store the same Python floats and ints as before for every valid input.

- [ ] **Step 10: Commit** (sandbox disabled for both git commands)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/domains.py tests/test_wrong_kinds_domains.py tests/test_dom_domains.py tests/test_dom_evolutionary_cov.py tests/test_dom_variance_generic.py
git commit -m "domainia.domains: wrong kinds raise TypeError

The CellSet class methods check grid and sub with require_instance before
they read them (a bare AttributeError before; WDMDomain.full and
complement inherit the check). dt, t0 and sigma0 go through require_real
and stay Python floats (float() took True, \"0.5\", a Decimal and size-1
arrays). units is checked with require_str. CellSet m/n and the bands of
CellSet.bands use index_array: a bool, str, complex or object array is a
TypeError, and bands no longer truncates 1.5. from_mask refuses a keep
that is not bool with TypeError and keeps its shape ValueError.
ChannelAxis refuses a bare string instead of splitting it. Every
TypeError in domains.py takes the \", got\" form, and the two anchored
_cells_and_window pins follow (spec 2026-11-08 §1.5, §2.3-§2.6).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `cd /Users/aaron/Documents/lisa/domainia && git status --short`
Expected: no line for any file of this task.

---

### Task 4: Wrong kinds in domainia.maps (spec §2.3–§2.7)

**Goal:** `domainia.maps` follows the rule: a wrong kind raises `TypeError` with a `", got"` message, and a wrong value raises `ValueError`. Every map constructor checks its domains, `Compose` checks its two maps and `FourierToWDM.after` its inner map, all with `require_instance`. The counts use `require_int`, and the bin lists and the five `rows` sites use `index_array`. `TimeToFourier`'s `fmin` is a real number stored as a Python `float`. `FourierToWDM`'s `restrict` is a `bool` and its `impl` a string, both checked before the `built_for` key is formed. The checks read only static fields and map arguments, so `jax.tree_util` rebuilds keep working (spec §2.7).

**Assumes the tree after Task 3.** Task 1 added `src/domainia/wilson/_checks.py`. Tasks 1–3 do not touch `src/domainia/maps/` or the maps tests, so the old texts below are as at 779550d.

**Files:**
- Modify: `src/domainia/maps/base.py` (the `_checks` import, `domain_size`, `LinearMap.rows`, `PushedCovariance.__post_init__` and `.rows`, `Identity` (new `__post_init__`), `Compose.__post_init__`, `Restrict.__post_init__`, `BinSelect.__post_init__`)
- Modify: `src/domainia/maps/time.py` (imports; the `__post_init__` of `TimeToWDM`, `WDMToTime`, `TimeToFourier`, `CropTime`, `ZeroPadTime`; `TimeToWDM.rows`, `TimeToFourier.rows`)
- Modify: `src/domainia/maps/fourier.py` (imports; `pad_maps`, `crop_maps`; the `__post_init__` of `FourierZeroPad`, `FourierCrop`, `FourierToWDM`; `FourierToWDM.rows`, `FourierToWDM.after`)
- Create: `tests/test_wrong_kinds_maps.py`
- Modify: `tests/test_dom_maps_dirichlet.py` (one comment that names `operator.index`)

**Acceptance Criteria:**
- [ ] The 11 constructors that take domains (`TimeToWDM`, `WDMToTime`, `TimeToFourier`, `CropTime`, `ZeroPadTime`, `FourierZeroPad`, `FourierCrop`, `FourierToWDM`, `Restrict`, `BinSelect`, `Identity`) raise `TypeError("{slot} must be {expected}, got {type}")` for each of their 18 `src`/`dst` slots, before reading the domain. No `AttributeError`, no silent build (`ZeroPadTime`, `Identity`). `TimeToFourier`'s derived `dst` is checked once it is filled.
- [ ] `Compose` raises `TypeError("outer must be a LinearMap, got …")` (and the same for `inner`). `FourierToWDM.after` raises `TypeError("inner must be a FourierZeroPad or a FourierCrop, got …")` before its `ValueError`s.
- [ ] The 11 integer slots (`CropTime` `start`/`n`, `ZeroPadTime` `n_extra`, `FourierCrop` `start`, `BinSelect`'s `(k0, n)`, `pad_maps` `N`/`Np`, `crop_maps` `N`/`start`/`Nc`) raise `TypeError("{name} must be an integer, got {type}")` for `True`, `2.0`, `np.float64(2.0)`, `"2"` and `None`. NumPy integers pass and are stored as `int`.
- [ ] The 9 index-array slots (`pad_maps`/`crop_maps` `k_out` and `j_in`; `rows` of `LinearMap`, `TimeToWDM`, `TimeToFourier`, `FourierToWDM`, `PushedCovariance`) raise `TypeError` for a bool, str, complex or object array, and `ValueError` for `1.5` or `nan`. Integer and integral-float arrays pass.
- [ ] `fmin` raises `TypeError` for `bool`, `str`, `None`, complex and a 0-d array, and is stored as `float` (so the `built_for` key holds a `float`). `restrict` raises `TypeError` unless a `bool`/`np.bool_`, and is stored as `bool`. `impl` raises `TypeError` unless a `str`; an unknown string keeps its `ValueError` text.
- [ ] `jax.tree_util.tree_map(lambda x: None, m)` and `tree_map(lambda x: object(), m)` still work on one map of each of the 12 classes and on an `after()` map (`jax`-marked test).
- [ ] No `raise TypeError` in `src/domainia/maps` has `"; got"`. `domain_size` and `PushedCovariance`'s covariance check say `", got"`.
- [ ] `tests/test_wrong_kinds_maps.py`: `181 passed` (NumPy lane: `179 passed, 2 skipped`). The full suite and the NumPy lane pass. The float64 bit gate prints `0 of 1504 arrays differ`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_maps.py` → `181 passed`; then the bit gate (Step 10) → `0 of 1504 arrays differ`

**Steps:**

- [ ] **Step 1: Grep the sites in `src/domainia/maps`**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "^import operator\|operator\.index\|int(v) for v\|asarray(idx, dtype=np.int64)\|asarray(k_out, np.int64)\|asarray(j_in, np.int64)" src/domainia/maps/*.py`
Expected: 17 hits.
- `time.py`: `import operator`; `operator.index` in `CropTime` (`start`, `n`) and `ZeroPadTime` (`n_extra`); `np.asarray(idx, dtype=np.int64)` in `TimeToWDM.rows` and `TimeToFourier.rows`.
- `base.py`: `np.asarray(idx, dtype=np.int64)` in `LinearMap.rows` and `PushedCovariance.rows`; `int(v) for v in self.dst` in `BinSelect`.
- `fourier.py`: `import operator`; `operator.index` in `FourierCrop`; `np.asarray(k_out, np.int64)` and two `np.asarray(j_in, np.int64)` on the `for j in` line, in `pad_maps` and again in `crop_maps`; `np.asarray(idx, dtype=np.int64)` in `FourierToWDM.rows`; and `j = np.asarray(j_in, np.int64)` in `_count_self_conjugate_once`.

All but the last change in Steps 5–7. `_count_self_conjugate_once` is private, and its two callers now pass it the checked `j_in`, so it stays.

What these sites do today (probed at 779550d):
- `operator.index` takes `True` as 1. `BinSelect`'s `int()` takes `True`, `2.7` (as 2) and `"2"`. `pad_maps` and `crop_maps` take `True` and `2.0` for every count.
- `np.asarray(…, np.int64)` truncates `[1.5]` to `[1]` and takes `[True]` and `["1"]` silently.
- `TimeToFourier` stores `fmin` raw: `fmin="0.2"` builds, with the `built_for` key `(src, "0.2")`; `np.array(0.2)` becomes an unhashable static field; `True` means 1 Hz.
- `FourierToWDM` takes any truthy `restrict` (`"no"` turns the truncation on) and reports a non-string `impl` as the `ValueError` `impl must be 'dense' or 'fft'; got None`.
- A wrong-kind domain raises a bare `AttributeError` in 8 constructors, Python's own `dataclasses.replace()` `TypeError` in `BinSelect`, and nothing at all in `ZeroPadTime` and `Identity`. `Compose` accepts any two objects. `after()` raises `AttributeError` (`'Identity' object has no attribute 'A'`).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "raise TypeError" src/domainia/maps/*.py`
Expected: 9 hits, none with `"; got"`.
- Two are argument-kind messages without the `", got"` form. Step 5 rewrites both: `domain_size` (`not a domain: {dom!r}`) and `PushedCovariance.__post_init__` (`cannot push a {type}: a covariance needs a matvec …`).
- `PushedCovariance needs a LinearMap, got {type}` already has the form and stays.
- The other six refuse an operation and name no argument: `widely linear` (`LinearMap.push`, `_WidelyLinear.rows`), `has no matvec` (`PushedCovariance._matvec`), `cannot push through a {kind!r} map`, `matvec of a pushed covariance is defined for real maps only`, and `Compose.push is defined for real maps only`. They are not kind messages, tests pin them by `match=`, and they stay.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" src/domainia/maps/*.py`
Expected: 6 hits, all `ValueError`: `_last_axis` (`base.py`), `_flat` and `ZeroPadTime`'s `n_extra must be >= 0` (`time.py`), and `_geometry`'s `interior bands`, `impl must be 'dense' or 'fft'` and `push_rows`'s `M' must be` (`fourier.py`). They keep their text. So Task 9's gate already holds for `maps/`.

- [ ] **Step 2: Grep the tests that pin these sites**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "operator.index\|not a domain\|cannot push a\|matvec\.\*dense" tests --include='*.py'`
Expected: 4 hits.
- `tests/test_dom_maps_dirichlet.py:132`: a comment, `(CropTime's operator.index rule)`. It goes stale, so Step 8 rewords it.
- `tests/test_dom_maps_base.py:183` and `:185`: `pytest.raises(TypeError, match=r"matvec.*dense")` on `PushedCovariance`'s covariance check. The new text keeps `matvec` before `dense`, so no edit.
- `tests/test_operators_covariance.py:199`: docstring prose ("cannot push a cell covariance to time"), not a pin. No edit.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "pytest.raises(TypeError):" tests/test_dom_maps_*.py`
Expected: 2 hits, `tests/test_dom_maps_fourier_time.py:55` (`CropTime(td, 10.0, 40)`) and `tests/test_dom_maps_dirichlet.py:134` (`FourierCrop(SRC, M.dst, 24.0)`). Both stay `TypeError`, now from `require_int`. No edit. The `type(...start) is int` pin next to the second still holds.

No test passes `True`, a float, a string or a wrong-kind domain to these sites expecting success, and no test pins a `ValueError` for one. `tests/test_dom_built_for.py` keys `TimeToFourier` as `(TIME, 0.2)` and `FourierToWDM` as `(REC, BANDS, "dense", False)`, which the normalised values still equal. With Steps 5–8 applied, the full suite has no failure in a maps test.

- [ ] **Step 3: Write the failing tests `tests/test_wrong_kinds_maps.py`**

```python
"""``domainia.maps`` refuses a wrong kind with ``TypeError`` and a wrong value with ``ValueError``
(spec 2026-11-08 §2.3-§2.7): domains, counts, index arrays, ``fmin``, ``restrict``, ``impl``,
``after``'s inner map and ``Compose``'s parts."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest
from dom_db_inputs import DenseCov, jax_or_skip

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import (BinSelect, Compose, CropTime, FourierCrop, FourierToWDM,
                           FourierZeroPad, Identity, Restrict, TimeToFourier, TimeToWDM,
                           WDMToTime, ZeroPadTime, crop_maps, domain_size, pad_maps)

GRID = wilson.Grid(8, 16, 0.5)
TIME = TimeDomain(GRID.N, GRID.dt)
FULL = WDMDomain.full(GRID)
CUT = WDMDomain(FULL.cells.drop_columns(2, 1))
BANDS = WDMDomain(CellSet.bands(GRID, tuple(range(1, 8))))
REC = FourierDomain(GRID.N, GRID.dt, 0, GRID.N // 2 + 1, "rfft")     # every bin of the record
WIN = FourierDomain(GRID.N, GRID.dt, 0, 64, "rfft")                  # the band windows of BANDS
STORE = FourierDomain(GRID.N, GRID.dt, 1, 63, "rfft")
LONG = FourierDomain(160, GRID.dt, 0, 81, "rfft")                    # a 160-sample record
CELLS = CellSet.all_active(GRID)                                     # not a domain

#: A good build of each map constructor; a test overrides one domain slot.
BUILDS = {
    "TimeToWDM": lambda src=TIME, dst=FULL: TimeToWDM(src, dst),
    "WDMToTime": lambda src=FULL, dst=TIME: WDMToTime(src, dst),
    "TimeToFourier": lambda src=TIME: TimeToFourier(src, 0.2),
    "CropTime": lambda src=TIME: CropTime(src, 8, 64),
    "ZeroPadTime": lambda src=TIME: ZeroPadTime(src, 32),
    "FourierZeroPad": lambda src=STORE, dst=LONG: FourierZeroPad(src, dst),
    "FourierCrop": lambda src=LONG, dst=WIN: FourierCrop(src, dst, 16),
    "FourierToWDM": lambda src=WIN, dst=BANDS: FourierToWDM(src, dst),
    "Restrict": lambda src=FULL, dst=CUT: Restrict(src, dst),
    "BinSelect": lambda src=REC, dst=(3, 10): BinSelect(src, dst),
    "Identity": lambda src=TIME: Identity(src),
}
ANY_DOMAIN = "a TimeDomain, a FourierDomain or a WDMDomain"
#: (map, slot, the expected kind, a value of another kind than expected).
DOMAIN_SLOTS = [
    ("TimeToWDM", "src", "a TimeDomain", FULL),
    ("TimeToWDM", "dst", "a WDMDomain", TIME),
    ("WDMToTime", "src", "a WDMDomain", TIME),
    ("WDMToTime", "dst", "a TimeDomain", FULL),
    ("TimeToFourier", "src", "a TimeDomain", FULL),
    ("CropTime", "src", "a TimeDomain", FULL),
    ("ZeroPadTime", "src", "a TimeDomain", FULL),
    ("FourierZeroPad", "src", "a FourierDomain", TIME),
    ("FourierZeroPad", "dst", "a FourierDomain", TIME),
    ("FourierCrop", "src", "a FourierDomain", TIME),
    ("FourierCrop", "dst", "a FourierDomain", TIME),
    ("FourierToWDM", "src", "a FourierDomain", TIME),
    ("FourierToWDM", "dst", "a WDMDomain", TIME),
    ("Restrict", "src", "a WDMDomain", CELLS),
    ("Restrict", "dst", "a WDMDomain", CELLS),
    ("BinSelect", "src", "a FourierDomain", TIME),
    ("BinSelect", "dst", "a FourierDomain or a (k0, n) tuple", [3, 10]),
    ("Identity", "src", ANY_DOMAIN, CELLS),
]


@pytest.mark.parametrize("which", ["None", "other kind"])
@pytest.mark.parametrize(("name", "slot", "expected", "other"), DOMAIN_SLOTS,
                         ids=[f"{n}.{s}" for n, s, _, _ in DOMAIN_SLOTS])
def test_a_domain_of_the_wrong_kind_raises_type_error(name, slot, expected, other, which):
    """Each map constructor checks its domains before it reads them: ``TypeError`` naming the
    slot and the expected class, not a bare ``AttributeError`` (or a silent build, for
    ``ZeroPadTime`` and ``Identity``)."""
    bad = None if which == "None" else other
    message = f"{slot} must be {expected}, got {type(bad).__name__}"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        BUILDS[name](**{slot: bad})


def test_the_other_kind_checks_take_the_got_form():
    with pytest.raises(TypeError, match=f"^dom must be {ANY_DOMAIN}, got CellSet$"):
        domain_size(CELLS)
    F = TimeToFourier(TIME, 0.2)
    with pytest.raises(TypeError, match="^dst must be a FourierDomain, got int$"):
        dataclasses.replace(F, dst=3)        # dst is derived: checked once it is filled
    with pytest.raises(TypeError, match=r"^PushedCovariance needs a covariance with a matvec "
                                        r"\(SupportsMatvec\) or a dense \(SupportsDense\) method, "
                                        r"got ndarray$"):
        Identity(TIME).push(np.eye(GRID.N))


#: Each integer slot, as a function of the value under test.
INT_SLOTS = {
    "CropTime.start": lambda v: CropTime(TIME, v, 64),
    "CropTime.n": lambda v: CropTime(TIME, 8, v),
    "ZeroPadTime.n_extra": lambda v: ZeroPadTime(TIME, v),
    "FourierCrop.start": lambda v: FourierCrop(LONG, WIN, v),
    "BinSelect.k0": lambda v: BinSelect(REC, (v, 10)),
    "BinSelect.n": lambda v: BinSelect(REC, (3, v)),
    "pad_maps.N": lambda v: pad_maps(v, 160, [1, 2], [1, 2]),
    "pad_maps.Np": lambda v: pad_maps(128, v, [1, 2], [1, 2]),
    "crop_maps.N": lambda v: crop_maps(v, 16, 128, [1, 2], [1, 2]),
    "crop_maps.start": lambda v: crop_maps(160, v, 128, [1, 2], [1, 2]),
    "crop_maps.Nc": lambda v: crop_maps(160, 16, v, [1, 2], [1, 2]),
}


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), (2.0, "float"), (np.float64(2.0), "float64"), ("2", "str"), (None, "NoneType"),
])
@pytest.mark.parametrize("slot", INT_SLOTS)
def test_an_integer_of_the_wrong_kind_raises_type_error(slot, bad, kind):
    """``operator.index`` took ``True``; ``int()`` (``BinSelect``) and the arithmetic of the
    Dirichlet kernels took ``True``, ``2.0`` and (``BinSelect``) ``"2"``."""
    name = slot.split(".")[1]
    with pytest.raises(TypeError, match=f"^{name} must be an integer, got {kind}$"):
        INT_SLOTS[slot](bad)


def test_numpy_integers_are_taken_and_stored_as_int():
    C = CropTime(TIME, np.int64(8), np.int32(64))
    assert (type(C.start), type(C.n)) == (int, int) and C.dst == CropTime(TIME, 8, 64).dst
    assert type(ZeroPadTime(TIME, np.int64(32)).n_extra) is int
    crop = FourierCrop(LONG, WIN, np.int64(16))
    assert type(crop.start) is int and crop.built_for == (LONG, WIN, 16)
    assert BinSelect(REC, (np.int64(3), np.int64(10))).dst == BinSelect(REC, (3, 10)).dst
    pairs = ((pad_maps(np.int64(128), np.int64(160), [1, 2], [1, 2]),
              pad_maps(128, 160, [1, 2], [1, 2])),
             (crop_maps(np.int64(160), np.int64(16), np.int64(128), [1, 2], [1, 2]),
              crop_maps(160, 16, 128, [1, 2], [1, 2])))
    for got, want in pairs:
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_counts_out_of_range_stay_value_errors():
    with pytest.raises(ValueError, match="not inside"):
        CropTime(TIME, -1, 64)
    with pytest.raises(ValueError, match="n_extra must be >= 0"):
        ZeroPadTime(TIME, -1)
    with pytest.raises(ValueError, match="not inside"):
        BinSelect(STORE, (0, 10))


#: Each index-array slot: the kernels' bin lists and the ``rows`` of every map that casts them.
INDEX_SLOTS = {
    "pad_maps.k_out": lambda v: pad_maps(128, 160, v, [1, 2]),
    "pad_maps.j_in": lambda v: pad_maps(128, 160, [1, 2], v),
    "crop_maps.k_out": lambda v: crop_maps(160, 16, 128, v, [1, 2]),
    "crop_maps.j_in": lambda v: crop_maps(160, 16, 128, [1, 2], v),
    "LinearMap.rows.idx": lambda v: Restrict(FULL, CUT).rows(v),
    "TimeToWDM.rows.idx": lambda v: TimeToWDM(TIME, FULL).rows(v),
    "TimeToFourier.rows.idx": lambda v: TimeToFourier(TIME, 0.2).rows(v),
    "FourierToWDM.rows.idx": lambda v: FourierToWDM(WIN, BANDS).rows(v),
    # a "cells" push reads push_rows, not map.rows: this reaches PushedCovariance.rows alone
    "PushedCovariance.rows.idx":
        lambda v: FourierToWDM(WIN, BANDS).push(DenseCov(np.eye(64))).rows(v),
}


@pytest.mark.parametrize(("bad", "dtype"), [
    ([True, False], "bool"), (["1"], "<U1"), ([1 + 0j], "complex128"), (None, "object"),
], ids=["bool", "str", "complex", "None"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_an_index_array_of_the_wrong_dtype_raises_type_error(slot, bad, dtype):
    name = slot.split(".")[-1]
    message = f"^{name} must be an integer index array, got dtype {dtype}$"
    with pytest.raises(TypeError, match=message):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize(("bad", "first"), [([1.5], "1.5"), ([1.0, np.nan], "nan")],
                         ids=["1.5", "nan"])
@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_a_non_integral_index_raises_value_error(slot, bad, first):
    """``np.asarray(..., dtype=np.int64)`` truncated ``1.5`` to ``1`` silently."""
    name = slot.split(".")[-1]
    with pytest.raises(ValueError, match=f"^{name} must hold integer indices, got {first}$"):
        INDEX_SLOTS[slot](bad)


@pytest.mark.parametrize("slot", INDEX_SLOTS)
def test_integer_and_integral_float_index_arrays_are_taken(slot):
    want = INDEX_SLOTS[slot]([1, 2])
    for v in (np.array([1, 2], dtype=np.int32), np.array([1.0, 2.0])):
        got = INDEX_SLOTS[slot](v)
        for g, w in zip(got if isinstance(got, tuple) else (got,),
                        want if isinstance(want, tuple) else (want,)):
            np.testing.assert_array_equal(g, w)


@pytest.mark.parametrize(("bad", "kind"), [
    (True, "bool"), ("0.2", "str"), (None, "NoneType"), (0.2 + 0j, "complex"),
    (np.array(0.2), "ndarray"),
])
def test_fmin_of_the_wrong_kind_raises_type_error(bad, kind):
    """``fmin`` was stored raw: ``"0.2"`` gave the built_for key ``(src, "0.2")``,
    ``np.array(0.2)`` an unhashable static field, and ``True`` meant 1 Hz."""
    with pytest.raises(TypeError, match=f"^fmin must be a real number, got {kind}$"):
        TimeToFourier(TIME, bad)


def test_fmin_is_stored_as_a_float():
    for v in (0.2, np.float64(0.2), np.float32(0.25), 0, np.int64(0)):
        F = TimeToFourier(TIME, v)
        assert type(F.fmin) is float and F.built_for == (TIME, float(v))
        assert F.dst == TimeToFourier(TIME, float(v)).dst
    with pytest.raises(ValueError, match="leaves no interior Fourier frequencies"):
        TimeToFourier(TIME, 5.0)


@pytest.mark.parametrize(("bad", "kind"), [
    ("no", "str"), (1, "int"), (0.5, "float"), (None, "NoneType"),
])
def test_restrict_of_the_wrong_kind_raises_type_error(bad, kind):
    """A truthy ``"no"`` used to switch the atom truncation on."""
    with pytest.raises(TypeError, match=f"^restrict must be a bool, got {kind}$"):
        FourierToWDM(WIN, BANDS, restrict=bad)


def test_restrict_is_stored_as_a_bool_before_the_key_is_formed():
    m = FourierToWDM(WIN, BANDS, restrict=np.True_)
    assert m.restrict is True and m.built_for == (WIN, BANDS, "dense", True)


@pytest.mark.parametrize(("bad", "kind"), [
    (None, "NoneType"), (3, "int"), (True, "bool"), (b"dense", "bytes"),
])
def test_impl_of_the_wrong_kind_raises_type_error(bad, kind):
    with pytest.raises(TypeError, match=f"^impl must be a string, got {kind}$"):
        FourierToWDM(WIN, BANDS, bad)


def test_an_unknown_impl_stays_a_value_error():
    with pytest.raises(ValueError, match="impl must be 'dense' or 'fft'; got 'sparse'"):
        FourierToWDM(WIN, BANDS, "sparse")


@pytest.mark.parametrize("bad", [None, Identity(WIN), BinSelect(REC, WIN)],
                         ids=["None", "Identity", "BinSelect"])
def test_after_needs_a_zero_pad_or_a_crop(bad):
    with pytest.raises(TypeError, match=f"^inner must be a FourierZeroPad or a FourierCrop, "
                                        f"got {type(bad).__name__}$"):
        FourierToWDM(WIN, BANDS).after(bad)


@pytest.mark.parametrize(("outer", "inner", "slot", "kind"), [
    (None, Restrict(FULL, CUT), "outer", "NoneType"),
    (Restrict(FULL, CUT), FULL, "inner", "WDMDomain"),
    (3, "x", "outer", "int"),
])
def test_compose_needs_two_linear_maps(outer, inner, slot, kind):
    """``Compose`` used to accept any two objects and fail at first use."""
    with pytest.raises(TypeError, match=f"^{slot} must be a LinearMap, got {kind}$"):
        Compose(outer, inner)


def _every_map():
    """One map of each of the 12 classes, and an ``after()`` map."""
    W = FourierToWDM(WIN, BANDS)
    maps = [BUILDS[name]() for name in BUILDS]
    maps += [Compose(Restrict(FULL, CUT), TimeToWDM(TIME, FULL)),
             W.after(FourierCrop(LONG, WIN, 16))]
    return maps


@pytest.mark.jax
@pytest.mark.parametrize("leaf", [lambda x: None, lambda x: object()], ids=["None", "object"])
def test_jax_rebuilds_with_placeholder_leaves_still_work(leaf):
    """``jax.tree_util`` rebuilds a map through its constructor, with placeholders for the
    arrays it carries (spec §2.7): the checks read only static fields and map arguments."""
    jax, _ = jax_or_skip()
    maps = _every_map()
    assert len({type(m) for m in maps}) == 12
    for m in maps:
        r = jax.tree_util.tree_map(leaf, m)
        assert type(r) is type(m) and (r.src, r.dst) == (m.src, m.dst)
        assert getattr(r, "built_for", None) == getattr(m, "built_for", None)
```

The `PushedCovariance.rows` slot goes through a `"cells"` push: its rows read `push_rows`, not `map.rows`, so that case reaches the `PushedCovariance.rows` check alone.

- [ ] **Step 4: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_maps.py 2>&1 | tail -1`
Expected: `167 failed, 14 passed`.
- `test_a_domain_of_the_wrong_kind_raises_type_error`: all 36. Most raise `AttributeError` (for `[TimeToWDM.src-None]`: `AttributeError: 'NoneType' object has no attribute 'N'`). `ZeroPadTime` and `Identity` give `Failed: DID NOT RAISE <class 'TypeError'>`. `BinSelect.src` raises `replace() should be called on dataclass instances`, which does not match.
- `test_an_integer_of_the_wrong_kind_raises_type_error`: all 55. `True` gives `DID NOT RAISE` everywhere, and so does every `BinSelect`, `pad_maps` and `crop_maps` case but `None`. The `operator.index` sites raise Python's `'float' object cannot be interpreted as an integer`, which does not match.
- `test_an_index_array_of_the_wrong_dtype_raises_type_error` (36) and `test_a_non_integral_index_raises_value_error` (18): all fail, with `DID NOT RAISE` or Python's `int()` message.
- `test_fmin_of_the_wrong_kind_raises_type_error` (5), `test_restrict_of_the_wrong_kind_raises_type_error` (4), `test_impl_of_the_wrong_kind_raises_type_error` (4, `ValueError: impl must be 'dense' or 'fft'; got None`), `test_after_needs_a_zero_pad_or_a_crop` (3), `test_compose_needs_two_linear_maps` (3): all fail.
- `test_the_other_kind_checks_take_the_got_form` (the old `not a domain: CellSet(...)` text), `test_fmin_is_stored_as_a_float` (`assert (<class 'numpy.float64'> is float)`) and `test_restrict_is_stored_as_a_bool_before_the_key_is_formed` (`assert (np.True_ is True)`) fail.
- The 14 that pass: `test_numpy_integers_are_taken_and_stored_as_int`, `test_counts_out_of_range_stay_value_errors`, the 9 `test_integer_and_integral_float_index_arrays_are_taken`, `test_an_unknown_impl_stays_a_value_error`, and the 2 `test_jax_rebuilds_with_placeholder_leaves_still_work` (rebuilds work today; the test guards them through Steps 5–7).

- [ ] **Step 5: `maps/base.py` (`src/domainia/maps/base.py`)**

The import, after `import numpy as np`:

```python
from domainia.wilson.backend import NUMPY, Backend, get_backend, pytree_dataclass, require_x64
```
→
```python
from domainia.wilson._checks import index_array, require_instance, require_int
from domainia.wilson.backend import NUMPY, Backend, get_backend, pytree_dataclass, require_x64
```

`domain_size`, its last line:

```python
    raise TypeError(f"not a domain: {dom!r}")
```
→
```python
    raise TypeError(f"dom must be a TimeDomain, a FourierDomain or a WDMDomain, "
                    f"got {type(dom).__name__}")
```

`LinearMap.rows`:

```python
        """Host dense rows ``M[idx, :]`` (``(k, n_src)``), from the adjoint of unit vectors."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
```
→
```python
        """Host dense rows ``M[idx, :]`` (``(k, n_src)``), from the adjoint of unit vectors."""
        idx = np.atleast_1d(index_array("idx", idx))
```

`PushedCovariance.__post_init__`, the covariance check. The text names the class first, like the `LinearMap` check above it:

```python
            raise TypeError(f"cannot push a {type(self.cov).__name__}: a covariance needs a matvec "
                            "(SupportsMatvec) or a dense (SupportsDense) method")
```
→
```python
            raise TypeError("PushedCovariance needs a covariance with a matvec (SupportsMatvec) or "
                            f"a dense (SupportsDense) method, got {type(self.cov).__name__}")
```

`PushedCovariance.rows`:

```python
    def rows(self, idx: Any, be: Backend | None = None) -> Any:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
```
→
```python
    def rows(self, idx: Any, be: Backend | None = None) -> Any:
        idx = np.atleast_1d(index_array("idx", idx))
```

`Identity` gains a `__post_init__`. Its pytree meta is `("src",)`, so a rebuild passes the real domain:

```python
    src: Any

    @property
    def dst(self) -> Any:
        return self.src
```
→
```python
    src: Any

    def __post_init__(self) -> None:
        require_instance("src", self.src, (TimeDomain, FourierDomain, WDMDomain),
                         "a TimeDomain, a FourierDomain or a WDMDomain")

    @property
    def dst(self) -> Any:
        return self.src
```

`Compose.__post_init__`, whole. The two checks replace the `isinstance` guard, and the domain checks now always run. `outer` and `inner` are pytree data, but they are maps themselves, so `tree_map` rebuilds them first and hands `Compose` real maps:

```python
    def __post_init__(self) -> None:
        if isinstance(self.outer, LinearMap) and isinstance(self.inner, LinearMap):
            a, b = self.inner.dst, self.outer.src
            if isinstance(a, WDMDomain) and isinstance(b, WDMDomain) and a.window != b.window:
                raise ValueError(f"cannot compose: inner.dst and outer.src use different windows "
                                 f"({a.window.name} vs {b.window.name})")
            if b != a:
                raise ValueError(f"cannot compose: {a!r} != {b!r}")
```
→
```python
    def __post_init__(self) -> None:
        require_instance("outer", self.outer, LinearMap, "a LinearMap")
        require_instance("inner", self.inner, LinearMap, "a LinearMap")
        a, b = self.inner.dst, self.outer.src
        if isinstance(a, WDMDomain) and isinstance(b, WDMDomain) and a.window != b.window:
            raise ValueError(f"cannot compose: inner.dst and outer.src use different windows "
                             f"({a.window.name} vs {b.window.name})")
        if b != a:
            raise ValueError(f"cannot compose: {a!r} != {b!r}")
```

`Restrict.__post_init__`, its first line. The checks go before the window and grid checks:

```python
    def __post_init__(self) -> None:
        if self.src.window != self.dst.window:
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, WDMDomain, "a WDMDomain")
        require_instance("dst", self.dst, WDMDomain, "a WDMDomain")
        if self.src.window != self.dst.window:
```

`BinSelect.__post_init__`, its head. `src` is checked first, because the tuple branch calls `dataclasses.replace(self.src, …)`. `dst` is checked after the tuple branch: a rebuild passes the stored `FourierDomain`. A tuple of the wrong length stays Python's unpacking `ValueError`:

```python
    def __post_init__(self) -> None:
        if isinstance(self.dst, tuple):
            k0, n = (int(v) for v in self.dst)
            object.__setattr__(self, "dst", dataclasses.replace(self.src, k0=k0, n=n))
        d, s = self.dst, self.src
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, FourierDomain, "a FourierDomain")
        if isinstance(self.dst, tuple):
            k0, n = self.dst
            object.__setattr__(self, "dst", dataclasses.replace(
                self.src, k0=require_int("k0", k0), n=require_int("n", n)))
        require_instance("dst", self.dst, FourierDomain, "a FourierDomain or a (k0, n) tuple")
        d, s = self.dst, self.src
```

- [ ] **Step 6: `maps/time.py` (`src/domainia/maps/time.py`)**

The imports. `operator` has no other use:

```python
import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from domainia import wilson
from domainia.wilson.backend import pytree_dataclass
```
→
```python
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from domainia import wilson
from domainia.wilson._checks import index_array, require_instance, require_int, require_real
from domainia.wilson.backend import pytree_dataclass
```

`TimeToWDM.__post_init__`, its head:

```python
    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.dst), given=self.m_idx is not _BUILD)
        if self.m_idx is _BUILD:
            _check_grid(self.src, self.dst)
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, TimeDomain, "a TimeDomain")
        require_instance("dst", self.dst, WDMDomain, "a WDMDomain")
        _built_for(self, (self.src, self.dst), given=self.m_idx is not _BUILD)
        if self.m_idx is _BUILD:
            _check_grid(self.src, self.dst)
```

`TimeToWDM.rows`:

```python
    def rows(self, idx):
        g = self.grid
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
```
→
```python
    def rows(self, idx):
        g = self.grid
        idx = np.atleast_1d(index_array("idx", idx))
```

`WDMToTime.__post_init__`, its head:

```python
    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.dst), given=self.m_idx is not _BUILD)
        if self.m_idx is _BUILD:
            g = _check_grid(self.dst, self.src)
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, WDMDomain, "a WDMDomain")
        require_instance("dst", self.dst, TimeDomain, "a TimeDomain")
        _built_for(self, (self.src, self.dst), given=self.m_idx is not _BUILD)
        if self.m_idx is _BUILD:
            g = _check_grid(self.dst, self.src)
```

`TimeToFourier.__post_init__`, whole. `fmin` is stored as a `float` before the key is formed. `dst` is a derived field that defaults to `_BUILD`, so it is checked after it is filled. `_keep_bins`' own `float(fmin)` stays (now a no-op):

```python
    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.fmin), given=self.dst is not _BUILD)
        if self.dst is _BUILD:
            k0, n = _keep_bins(self.src.N, self.src.dt, self.fmin)
            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n, "rfft"))
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, TimeDomain, "a TimeDomain")
        object.__setattr__(self, "fmin", require_real("fmin", self.fmin))
        _built_for(self, (self.src, self.fmin), given=self.dst is not _BUILD)
        if self.dst is _BUILD:
            k0, n = _keep_bins(self.src.N, self.src.dt, self.fmin)
            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n, "rfft"))
        require_instance("dst", self.dst, FourierDomain, "a FourierDomain")
```

`TimeToFourier.rows`, after its docstring:

```python
        (unreduced, ``k j`` reaches 1.3e8 at ``N = 16384`` and the phase error 6e-12)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
```
→
```python
        (unreduced, ``k j`` reaches 1.3e8 at ``N = 16384`` and the phase error 6e-12)."""
        idx = np.atleast_1d(index_array("idx", idx))
```

`CropTime.__post_init__`, its head:

```python
    def __post_init__(self) -> None:
        object.__setattr__(self, "start", operator.index(self.start))
        object.__setattr__(self, "n", operator.index(self.n))
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, TimeDomain, "a TimeDomain")
        object.__setattr__(self, "start", require_int("start", self.start))
        object.__setattr__(self, "n", require_int("n", self.n))
```

`ZeroPadTime.__post_init__`, its head:

```python
    def __post_init__(self) -> None:
        object.__setattr__(self, "n_extra", operator.index(self.n_extra))
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, TimeDomain, "a TimeDomain")
        object.__setattr__(self, "n_extra", require_int("n_extra", self.n_extra))
```

- [ ] **Step 7: `maps/fourier.py` (`src/domainia/maps/fourier.py`)**

The imports. `operator` has no other use:

```python
import operator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from domainia import wilson
from domainia.wilson.backend import pytree_dataclass
```
→
```python
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from domainia import wilson
from domainia.wilson._checks import (index_array, require_bool, require_instance, require_int,
                                     require_str)
from domainia.wilson.backend import pytree_dataclass
```

`pad_maps`, after its docstring. `j_in` is checked once and reused, and `_count_self_conjugate_once` receives the checked array:

```python
    ``z'`` is the exact rfft of the padded real record in ``z``'s units."""
    k = np.asarray(k_out, np.int64)[:, None]
    out = []
    for j in (np.asarray(j_in, np.int64)[None, :], -np.asarray(j_in, np.int64)[None, :]):
```
→
```python
    ``z'`` is the exact rfft of the padded real record in ``z``'s units."""
    N, Np = require_int("N", N), require_int("Np", Np)
    k = index_array("k_out", k_out)[:, None]
    j_in = index_array("j_in", j_in)
    out = []
    for j in (j_in[None, :], -j_in[None, :]):
```

`crop_maps`, after its docstring:

```python
    ``z^c`` is the exact rfft of the cropped real record in ``z``'s units."""
    k = np.asarray(k_out, np.int64)[:, None]
    out = []
    for j in (np.asarray(j_in, np.int64)[None, :], -np.asarray(j_in, np.int64)[None, :]):
```
→
```python
    ``z^c`` is the exact rfft of the cropped real record in ``z``'s units."""
    N, start, Nc = require_int("N", N), require_int("start", start), require_int("Nc", Nc)
    k = index_array("k_out", k_out)[:, None]
    j_in = index_array("j_in", j_in)
    out = []
    for j in (j_in[None, :], -j_in[None, :]):
```

`FourierZeroPad.__post_init__`, its head:

```python
    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.dst), given=self.A is not _BUILD)
        if self.A is _BUILD:
            s, d = self.src, self.dst
            _require_same_units("FourierZeroPad", s, d)
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, FourierDomain, "a FourierDomain")
        require_instance("dst", self.dst, FourierDomain, "a FourierDomain")
        _built_for(self, (self.src, self.dst), given=self.A is not _BUILD)
        if self.A is _BUILD:
            s, d = self.src, self.dst
            _require_same_units("FourierZeroPad", s, d)
```

`FourierCrop.__post_init__`, its head:

```python
    def __post_init__(self) -> None:
        object.__setattr__(self, "start", operator.index(self.start))
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, FourierDomain, "a FourierDomain")
        require_instance("dst", self.dst, FourierDomain, "a FourierDomain")
        object.__setattr__(self, "start", require_int("start", self.start))
```

`FourierToWDM.__post_init__`, its head. `impl` and `restrict` are checked before the key is formed, so a wrong kind is a `TypeError` on the constants-given path too. The membership `ValueError` below stays where it is:

```python
    def __post_init__(self) -> None:
        _built_for(self, (self.src, self.dst, self.impl, self.restrict),
                   given=self.C is not _BUILD)
```
→
```python
    def __post_init__(self) -> None:
        require_instance("src", self.src, FourierDomain, "a FourierDomain")
        require_instance("dst", self.dst, WDMDomain, "a WDMDomain")
        require_str("impl", self.impl)
        object.__setattr__(self, "restrict", require_bool("restrict", self.restrict))
        _built_for(self, (self.src, self.dst, self.impl, self.restrict),
                   given=self.C is not _BUILD)
```

`FourierToWDM.rows`:

```python
    def rows(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        C = np.asarray(self.C)
```
→
```python
    def rows(self, idx):
        idx = np.atleast_1d(index_array("idx", idx))
        C = np.asarray(self.C)
```

`FourierToWDM.after`, after its docstring. The check runs before `inner.dst` is read, so a real pad or crop with the wrong domains keeps its `ValueError` (`the band window`):

```python
        the map itself (its ``C`` equals the atoms to roundoff)."""
        if self.impl != "dense" or inner.dst != self.src or self.src != self.window:
```
→
```python
        the map itself (its ``C`` equals the atoms to roundoff)."""
        require_instance("inner", inner, (FourierZeroPad, FourierCrop),
                         "a FourierZeroPad or a FourierCrop")
        if self.impl != "dense" or inner.dst != self.src or self.src != self.window:
```

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "operator\.index\|^import operator\|int(v) for v\|asarray(idx, dtype=np.int64)\|asarray(k_out\|asarray(j_in, np.int64)\[" src/domainia/maps/*.py`
Expected: no output.

- [ ] **Step 8: The stale comment (`tests/test_dom_maps_dirichlet.py`)**

In `test_widely_linear_adjoint_and_no_push`:

```python
    if cls == "crop":           # start is a sample index (CropTime's operator.index rule)
```
→
```python
    if cls == "crop":           # start is a sample index (CropTime's integer rule)
```

- [ ] **Step 9: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_maps.py 2>&1 | tail -1`
Expected: `181 passed`, no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_maps.py 2>&1 | tail -1`
Expected: `179 passed, 2 skipped` (the two `jax` rebuild cases).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_maps_base.py tests/test_dom_maps_dirichlet.py tests/test_dom_maps_fourier_time.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_wdm.py tests/test_dom_built_for.py tests/test_dom_windows.py 2>&1 | tail -1`
Expected: `145 passed`, the same count as before Step 5. A failure here is a pin that Step 2 did not find: if it expects the old kind behaviour, move it to `TypeError` as the new file does, and record it in the errata.

- [ ] **Step 10: Full suite, NumPy lane and the float64 bit gate**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes (the oracle tests skip), 0 errors, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, 0 errors, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz`
Expected: `0 of 1504 arrays differ`. `index_array` returns the same `int64` arrays that `np.asarray(…, np.int64)` did for valid input, `require_int` returns the Python ints the domains already store, and `_keep_bins` already called `float(fmin)`. If an array differs, compare the `num`, `den` and `n_nu` passed to `_geometric` by `pad_maps`/`crop_maps` before and after Step 7.

- [ ] **Step 11: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/maps/base.py src/domainia/maps/time.py src/domainia/maps/fourier.py tests/test_wrong_kinds_maps.py tests/test_dom_maps_dirichlet.py && git commit -m "domainia.maps: wrong kinds raise TypeError

Every map constructor checks its domains with require_instance before it
reads them (a bare AttributeError, or a silent build for ZeroPadTime and
Identity, before); Compose checks its two maps and FourierToWDM.after its
inner pad or crop. The counts use require_int (operator.index and int()
took True, int() took 2.7 and \"2\"). The bin lists of pad_maps/crop_maps
and the five rows sites use index_array (1.5 was truncated). fmin is a
real number stored as float, restrict a bool and impl a string, all
checked before the built_for key is formed. domain_size and the
PushedCovariance covariance check take the \", got\" form. The checks read
static fields and map arguments only, so jax.tree_util rebuilds still work.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Wrong kinds in domainia.covariance (spec §2.3, §2.5, §2.6, §2.9)

**Goal:** `domainia.covariance` follows the rule: a wrong kind raises `TypeError`, a wrong value raises `ValueError`. The counts stop going through `int()`, the flags through `bool()`, and the string options are checked for kind before their membership `ValueError`. `ChannelStack` refuses a channel that is not a `Covariance` (an `AttributeError` today). `EvolutionaryCovariance` checks `P` and `t_max` at construction, for both builders. The shape of `b` (spec §2.8) is Task 7's, in the same files, and is not touched here.

**Assumes the tree after Task 4.** Tasks 1–4 do not touch `src/domainia/covariance/`, so its six files are as at 779550d and the line numbers below hold. Task 3 converted `domains._cells_and_window`'s kind message to `", got"` and moved its pin, `tests/test_dom_evolutionary_cov.py:101`, with it.

**Files:**
- Modify: `src/domainia/covariance/base.py` (the imports, `quad_form`, `DiagFactor.quad`, `Covariance.factor`)
- Modify: `src/domainia/covariance/diagonal.py` (the imports, `DiagonalCovariance.__init__`)
- Modify: `src/domainia/covariance/dense.py` (the imports, `dense_factor`, `DenseCovariance.__init__`)
- Modify: `src/domainia/covariance/blocktri.py` (the imports, the `BlockTridiagonalCovariance` docstring and `__init__`)
- Modify: `src/domainia/covariance/stack.py` (the imports, `ChannelStack.__init__`)
- Modify: `src/domainia/covariance/evolutionary.py` (the imports, the `t_max`/`P` parameter text, `EvolutionaryCovariance.__init__`)
- Create: `tests/test_wrong_kinds_covariance.py`

**Acceptance Criteria:**
- [ ] Integers (`require_int`): `DiagonalCovariance` `n` and `channel`, `DenseCovariance` `n`, `ChannelStack` `params_per_channel`, each element of `BlockTridiagonalCovariance`'s `sizes` (named `sizes[m]`), and `EvolutionaryCovariance` `P`. A float (`2.0` included), `bool`, `np.bool_`, `str`, a 0-d array, a NumPy float and `None` (except `channel`, whose default is `None`) raise `TypeError("{name} must be an integer, got {type}")`. `np.int64` passes and is stored as `int`; an integer ndarray of sizes passes. `params_per_channel < 0` keeps its `ValueError`.
- [ ] Flags (`require_bool`): `DenseCovariance` and `dense_factor` `symmetrize`, `EvolutionaryCovariance` `symmetrize` for both assemblies, `Covariance.factor`'s `jit` (`None` or a bool, on every backend). `"no"`, `1`, `0`, `0.5`, a 0-d array and `None` (except `jit`) raise `TypeError("{name} must be a bool, got {type}")`. `np.True_`/`np.False_` pass and are stored as `bool`.
- [ ] Strings (`require_str`, then the existing membership `ValueError`, its text unchanged): `chol_impl` (`DenseCovariance`, `dense_factor`, `EvolutionaryCovariance`), `layout` (`BlockTridiagonalCovariance`, and `EvolutionaryCovariance` when not `None`), `builder`, `assembly`, and `form` (`quad_form` and every factor's `quad`). A non-string raises `TypeError("{name} must be a string, got {type}")`. `np.str_` passes. `EvolutionaryCovariance(layout=0)`, `layout=False` and `layout=""` no longer read as `"ragged"`: the first two raise `TypeError`, `""` the `BlockTridiagonalCovariance` `ValueError`.
- [ ] `ChannelStack(covs, k)` raises `TypeError("covs[c] must be a Covariance, got {type}")` for a channel that is not a `Covariance`.
- [ ] `EvolutionaryCovariance` raises at construction, for `builder="interp"` and `builder="direct"`: `TypeError` for a `P` that is not an integer, `ValueError("P must be >= 1, got {P}")` for `P < 1`, and `TypeError("t_max must be a real number, got {type}")` for a `t_max` that is not a real number (`t_max="300"` is accepted today). An accepted `t_max` is stored as a Python `float`. The range of `t_max` stays the plan's lazy check (`pair mid-times`).
- [ ] A non-callable `psd` raises `TypeError("psd must be callable, got {type}")`. After this task the only `raise TypeError` in `src/domainia/covariance` is `Covariance._multiply`'s capability message, kept as it is (pinned by `tests/test_dom_cov_multiply.py:130`).
- [ ] `tests/test_wrong_kinds_covariance.py`: `197 failed, 9 passed` before the source change, `206 passed` after.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_covariance.py` → `206 passed`; then the full suite and the NumPy lane → all pass

**Steps:**

- [ ] **Step 1: Grep the sites and the tests that pin them**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "int(\|bool(\|float(\| or \"ragged\"\|callable(" src/domainia/covariance/*.py`
Expected: 14 hits.
- The sites this task changes: `dense.py:40` (`self.n = int(n)`), `dense.py:42` (`self.symmetrize = bool(symmetrize)`), `blocktri.py:147` (`tuple(int(s) for s in sizes)`), `base.py:382` (`bool(jit)`), `diagonal.py:40` (`int(channel)`), `diagonal.py:41` (`self.n = int(n)`), `stack.py:65` (`int(params_per_channel)`), `evolutionary.py:129` (`if not callable(psd):`), `evolutionary.py:152` (`layout or "ragged"`).
- Left as they are: `blocktri.py:148` (`int(sum(self.sizes))`, a sum of checked ints), `diagonal.py:42` (`bool(getattr(variance, "jittable", True))`, the variance's own attribute), `evolutionary.py:146` (`int(np.sum(...))`, a computed count), `base.py:153` and `:159` (`as_evaluator` dispatch on `callable(obj)`).

The sites with no conversion to grep for, read from the six `__init__`s and the factors' `quad`: `DenseCovariance` and `EvolutionaryCovariance` `chol_impl`, `BlockTridiagonalCovariance` `layout`, `EvolutionaryCovariance` `builder`, `assembly`, `layout`, `symmetrize` (read only for `assembly="dense"`, `:155`), `P` and `t_max` (passed on unread), `dense_factor`'s `symmetrize` (`if symmetrize:`) and `chol_impl`, and `form` in `quad_form` (`base.py:167-172`) and `DiagFactor.quad` (`base.py:222`, which compares `form == "elementwise"` before it reaches `quad_form`).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "TypeError" src/domainia/covariance/*.py`
Expected: 3 hits. `base.py:333` (docstring) and `base.py:335` (`Covariance._multiply`'s `"... has no multiply: it implements neither _multiply nor _matrix"`): a capability error, not a kind message, pinned anchored by `tests/test_dom_cov_multiply.py:130`; it stays. `evolutionary.py:130` (`"builder ... needs a callable PSD psd(f, t, nu); ... is not callable"`): a kind message without `", got"`; Step 6 replaces it with `require_callable`. No `TypeError` in `src/domainia/covariance` has the `"; got"` form, so Task 9's gate finds nothing here.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "callable\"\|unknown builder\|chol_impl must be\|layout must be\|layout applies\|assembly must be\|has no quadratic form\|non-negative\|needs at least one channel\|has no multiply\|pair mid-times\|P must be\|needs assembly" tests --include='*.py' | grep -v "tests/wilson/"`
Expected: 10 hits, all kept unchanged:
- `tests/test_dom_cov_stack.py:26` (`match="non-negative"`): `params_per_channel=-1` stays `ValueError`.
- `tests/test_dom_evolutionary_cov.py:84` (`unknown builder`), `:86` (`TypeError`, `match="callable"`: the new text `"psd must be callable, got object"` still matches), `:90` (`builder 'direct' needs assembly='dense'`), `:92` (`pair mid-times`, raised lazily by `.blocks(NU)`: the `t_max` range is still the plan's).
- `tests/test_dom_cov_api.py:124` (`layout applies to assembly='blocktri' only`) and `:130` (`chol_impl must be 'numpy' or 'scipy'`): valid strings, unchanged messages.
- `tests/test_dom_cov_multiply.py:130` (`has no multiply`), and `:60`, `:102` (the `"diagonal-callable"` fixture name, not a message).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" tests/test_dom_evolutionary_cov.py`
Expected: no output. `:101` pins the kind message of `domains._cells_and_window` (`EvolutionaryCovariance(PSD, cells.grid)`), raised in `domains.py`; Task 3 converted it to `", got"` and moved this pin with it, so the file needs no change here.

A probe while drafting called every constructor, `factor`, `multiply`, `blocks` and every factor's `quad` of the six files with wrong kinds (207 calls). It found the sites above and these `AttributeError`s: `ChannelStack`'s channels (`'int' object has no attribute 'n'`, 4 calls), a `be` that is not a `Backend` (`'str' object has no attribute 'name'` in `factor`, `multiply`, `constants`, `blocks`; not a domain, `CellSet`, map or covariance, so §2.6 does not list it: left), and a list `b` in `DiagFactor.solve` (`'list' object has no attribute 'ndim'`: `b` is Task 7's, spec §2.8). `EvolutionaryCovariance`'s `cells` and `window` already raise `TypeError`, in `domains.py` and `wilson/windows.py`. The probe confirmed spec §2.4: `EvolutionaryCovariance(PSD, cells, t_max="300")` constructs, and its `.blocks(NU)` succeeds (`float("300")`); `P=24.0`, `True` and `"24"` construct, and only `builder="interp"`'s first `.blocks()` refuses them.

- [ ] **Step 2: Write the failing tests `tests/test_wrong_kinds_covariance.py`**

```python
"""``domainia.covariance`` refuses a wrong kind with ``TypeError`` and a wrong value with
``ValueError`` (spec 2026-11-08 §2.3, §2.5, §2.6, §2.9)."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import (BlockTridiagonalCovariance, ChannelStack, DenseCovariance,
                                 DiagonalCovariance, EvolutionaryCovariance, dense_factor, quad_form)
from domainia.domains import CellSet
from domainia.spectra import GenericPSD
from domainia.variance import BandAveragedVariance
from domainia.wilson.backend import NUMPY

NF, NT, DT = 8, 16, 1 / 16
T_REC = NF * NT * DT
CELLS = CellSet.all_active(wilson.Grid(NF, NT, DT))
NU = np.array([-1.7, 0.9])
PSD = GenericPSD(lambda f, t, nu: 1.0 + 0.0 * f * t)
V = np.array([1.0, 2.0, 4.0, 8.0])
S = np.diag(V)
Y = np.ones(4)
DIRECT = {"builder": "direct", "assembly": "dense"}


def _blocks(nu, be):
    """Two bands of two cells, ragged ``(diag, up)``."""
    return [be.asarray(np.eye(2)), be.asarray(2.0 * np.eye(2))], [be.asarray(np.zeros((2, 2)))]


def _ev(**kw):
    return EvolutionaryCovariance(PSD, CELLS, **kw)


def _cases(slots, wrong):
    """``(make, name, value, kind)`` for every slot and wrong value.  A slot whose third entry
    is True has ``None`` as its documented default, so ``None`` is not a wrong kind there."""
    return [pytest.param(make, name, value, kind, id=f"{sid}-{value!r}")
            for sid, (make, name, none_is_default) in slots.items()
            for value, kind in wrong if not (value is None and none_is_default)]


# --- integers (spec §2.3, §2.9) -------------------------------------------------------------

INT_SLOTS = {
    "DiagonalCovariance.n": (lambda v: DiagonalCovariance(V, v), "n", False),
    "DiagonalCovariance.channel": (
        lambda v: DiagonalCovariance(BandAveragedVariance(PSD, CELLS), CELLS.size, channel=v), "channel", True),
    "DenseCovariance.n": (lambda v: DenseCovariance(S, v, chol_impl="numpy"), "n", False),
    "ChannelStack.params_per_channel": (
        lambda v: ChannelStack([DiagonalCovariance(V, 4)], v), "params_per_channel", False),
    "BlockTridiagonalCovariance.sizes": (lambda v: BlockTridiagonalCovariance(_blocks, (2, v)), r"sizes\[1\]", False),
    "EvolutionaryCovariance.P": (lambda v: _ev(P=v), "P", False),
    "EvolutionaryCovariance.P-direct": (lambda v: _ev(P=v, **DIRECT), "P", False),
}
WRONG_INTS = [(2.0, "float"), (2.7, "float"), (True, "bool"), (np.True_, "bool"), ("2", "str"),
              (None, "NoneType"), (np.array(2), "ndarray"), (np.float64(2), "float64")]


@pytest.mark.parametrize(("make", "name", "value", "kind"), _cases(INT_SLOTS, WRONG_INTS))
def test_counts_of_the_wrong_kind_raise_type_error(make, name, value, kind):
    """``int()`` used to truncate ``2.7`` to 2 and take ``True``, ``"2"`` and a 0-d array;
    ``P`` was read only by ``builder="interp"``, at the first factorisation."""
    with pytest.raises(TypeError, match=f"^{name} must be an integer, got {kind}$"):
        make(value)


def test_numpy_integer_counts_pass_and_are_stored_as_int():
    assert type(DiagonalCovariance(V, np.int64(4)).n) is int
    assert type(DenseCovariance(S, np.int64(4), chol_impl="numpy").n) is int
    assert type(ChannelStack([DiagonalCovariance(V, 4)], np.int64(2)).k) is int
    d = DiagonalCovariance(BandAveragedVariance(PSD, CELLS), CELLS.size, channel=np.int64(1))
    assert d.variance.kwargs == (("channel", 1),) and type(d.variance.kwargs[0][1]) is int
    b = BlockTridiagonalCovariance(_blocks, np.array([2, 2]))
    assert b.sizes == (2, 2) and all(type(s) is int for s in b.sizes) and b.n == 4
    ev = _ev(P=np.int64(4))
    assert ev.block_evaluator.P == 4 and type(ev.block_evaluator.P) is int
    assert len(ev.blocks(NU)[0]) == NF + 1


@pytest.mark.parametrize(("sizes", "kind"), [((2.0, 2.0), "float"), (np.array([2.0, 2.0]), "float64"),
                                             ((True, 2), "bool"), ("22", "str")])
def test_sizes_are_checked_element_by_element(sizes, kind):
    with pytest.raises(TypeError, match=rf"^sizes\[0\] must be an integer, got {kind}$"):
        BlockTridiagonalCovariance(_blocks, sizes)


def test_count_ranges_stay_value_errors():
    """``P < 1`` is now refused at construction, for both builders, with the plan's message."""
    with pytest.raises(ValueError, match="params_per_channel must be non-negative; got -1"):
        ChannelStack([DiagonalCovariance(V, 4)], -1)
    for kw in ({}, DIRECT):
        for P in (0, -1):
            with pytest.raises(ValueError, match=f"^P must be >= 1, got {P}$"):
                _ev(P=P, **kw)


# --- flags (spec §2.5) ----------------------------------------------------------------------

FLAG_SLOTS = {
    "DenseCovariance.symmetrize": (
        lambda v: DenseCovariance(S, 4, chol_impl="numpy", symmetrize=v), "symmetrize", False),
    "dense_factor.symmetrize": (
        lambda v: dense_factor(S, chol_impl="numpy", symmetrize=v, be=NUMPY), "symmetrize", False),
    "EvolutionaryCovariance.symmetrize-blocktri": (lambda v: _ev(symmetrize=v), "symmetrize", False),
    "EvolutionaryCovariance.symmetrize-dense": (lambda v: _ev(assembly="dense", symmetrize=v), "symmetrize", False),
    "Covariance.factor.jit": (lambda v: DiagonalCovariance(V, 4).factor(NU, jit=v), "jit", True),
}
WRONG_FLAGS = [("no", "str"), (1, "int"), (0, "int"), (0.5, "float"), (None, "NoneType"),
               (np.array(True), "ndarray")]


@pytest.mark.parametrize(("make", "name", "value", "kind"), _cases(FLAG_SLOTS, WRONG_FLAGS))
def test_flags_of_the_wrong_kind_raise_type_error(make, name, value, kind):
    """``bool()`` used to read ``"no"`` as True; ``EvolutionaryCovariance`` read ``symmetrize``
    only for ``assembly="dense"``, and ``factor`` read ``jit`` only on JAX."""
    with pytest.raises(TypeError, match=f"^{name} must be a bool, got {kind}$"):
        make(value)


def test_numpy_bool_flags_pass_and_are_stored_as_bool():
    assert DenseCovariance(S, 4, chol_impl="numpy", symmetrize=np.False_).symmetrize is False
    assert _ev(assembly="dense", symmetrize=np.False_).impl.symmetrize is False
    F = dense_factor(S, chol_impl="numpy", symmetrize=np.True_, be=NUMPY)
    assert float(F.logdet) == pytest.approx(np.log(64.0), rel=1e-15)
    for jit in (None, np.False_, True):
        F = DiagonalCovariance(V, 4).factor(NU, jit=jit)
        assert float(F.logdet) == pytest.approx(np.log(64.0), rel=1e-15)


# --- string options (spec §2.5) -------------------------------------------------------------

DIAG_F = DiagonalCovariance(V, 4).factor(NU)
CHOL_F = DenseCovariance(S, 4, chol_impl="numpy").factor(NU)
BLOCK_F = BlockTridiagonalCovariance(_blocks, (2, 2)).factor(NU)
STACK_F = ChannelStack([DiagonalCovariance(V, 4), DiagonalCovariance(V, 4)], 0).factor(np.zeros(0))

STR_SLOTS = {
    "DenseCovariance.chol_impl": (lambda v: DenseCovariance(S, 4, chol_impl=v), "chol_impl", False),
    "dense_factor.chol_impl": (
        lambda v: dense_factor(S, chol_impl=v, symmetrize=True, be=NUMPY), "chol_impl", False),
    "EvolutionaryCovariance.chol_impl": (lambda v: _ev(chol_impl=v), "chol_impl", False),
    "BlockTridiagonalCovariance.layout": (
        lambda v: BlockTridiagonalCovariance(_blocks, (2, 2), layout=v), "layout", False),
    "EvolutionaryCovariance.layout": (lambda v: _ev(layout=v), "layout", True),
    "EvolutionaryCovariance.builder": (lambda v: _ev(builder=v), "builder", False),
    "EvolutionaryCovariance.assembly": (lambda v: _ev(assembly=v), "assembly", False),
    "DiagFactor.quad.form": (lambda v: DIAG_F.quad(Y, form=v), "form", False),
    "CholFactor.quad.form": (lambda v: CHOL_F.quad(Y, form=v), "form", False),
    "BlockTriCovFactor.quad.form": (lambda v: BLOCK_F.quad(Y, form=v), "form", False),
    "StackFactor.quad.form": (lambda v: STACK_F.quad(np.ones((2, 4)), form=v), "form", False),
    "quad_form.form": (lambda v: quad_form(DIAG_F, Y, v, NUMPY), "form", False),
}
WRONG_STRS = [(3, "int"), (0, "int"), (False, "bool"), (None, "NoneType"), (b"solve", "bytes"),
              (["solve"], "list"), (np.array("elementwise"), "ndarray")]


@pytest.mark.parametrize(("make", "name", "value", "kind"), _cases(STR_SLOTS, WRONG_STRS))
def test_string_options_of_the_wrong_kind_raise_type_error(make, name, value, kind):
    """A wrong kind used to reach the membership ``ValueError``, or pass:
    ``EvolutionaryCovariance(layout=0)`` was ``"ragged"``, and ``DiagFactor.quad`` took
    ``np.array("elementwise")`` as ``"elementwise"``."""
    with pytest.raises(TypeError, match=f"^{name} must be a string, got {kind}$"):
        make(value)


def test_numpy_str_options_pass():
    assert DenseCovariance(S, 4, chol_impl=np.str_("scipy")).chol_impl == "scipy"
    assert BlockTridiagonalCovariance(_blocks, (2, 2), layout=np.str_("padded")).jittable
    assert _ev(builder=np.str_("direct"), assembly=np.str_("dense")).n == CELLS.size
    assert float(DIAG_F.quad(Y, form=np.str_("elementwise"))) == pytest.approx(1.875, rel=1e-15)
    assert float(CHOL_F.quad(Y, form=np.str_("whiten"))) == pytest.approx(1.875, rel=1e-15)


@pytest.mark.parametrize(("make", "match"), [
    (lambda: DenseCovariance(S, 4, chol_impl="lapack"), "chol_impl must be 'numpy' or 'scipy'; got 'lapack'"),
    (lambda: BlockTridiagonalCovariance(_blocks, (2, 2), layout="dense"),
     "layout must be 'ragged' or 'padded'; got 'dense'"),
    (lambda: _ev(layout=""), "layout must be 'ragged' or 'padded'; got ''"),
    (lambda: _ev(builder="bessel"), "unknown builder 'bessel'"),
    (lambda: _ev(assembly="banded"), "assembly must be 'blocktri' or 'dense'"),
    (lambda: DIAG_F.quad(Y, form="bogus"), "DiagFactor has no quadratic form 'bogus'"),
    (lambda: CHOL_F.quad(Y, form="elementwise"), "CholFactor has no quadratic form 'elementwise'"),
], ids=["chol_impl", "layout", "layout-empty", "builder", "assembly", "form-diag", "form-chol"])
def test_unknown_strings_stay_value_errors(make, match):
    """``layout=""`` used to be read as ``"ragged"`` by ``EvolutionaryCovariance``."""
    with pytest.raises(ValueError, match=match):
        make()


# --- covariances and callables (spec §2.6) --------------------------------------------------

@pytest.mark.parametrize(("covs", "index", "kind"), [
    ([1], 0, "int"), ("ab", 0, "str"), ([None], 0, "NoneType"), ([S], 0, "ndarray"),
    ([DiagonalCovariance(V, 4), object()], 1, "object"),
])
def test_channel_stack_channels_must_be_covariances(covs, index, kind):
    """Each used to fail with ``AttributeError: ... has no attribute 'n'``."""
    with pytest.raises(TypeError, match=rf"^covs\[{index}\] must be a Covariance, got {kind}$"):
        ChannelStack(covs, 1)


def test_channel_stack_takes_any_covariances_and_still_needs_one():
    st = ChannelStack((DiagonalCovariance(V, 4), DenseCovariance(S, 4, chol_impl="numpy")), 1)
    assert st.n == 8
    with pytest.raises(ValueError, match="ChannelStack needs at least one channel"):
        ChannelStack([], 1)


def test_the_psd_must_be_callable():
    with pytest.raises(TypeError, match="^psd must be callable, got object$"):
        EvolutionaryCovariance(object(), CELLS)


# --- t_max (spec §2.4, §2.9) ----------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, DIRECT], ids=["interp", "direct"])
@pytest.mark.parametrize(("t_max", "kind"), [
    ("300", "str"), (True, "bool"), (np.True_, "bool"), (1 + 0j, "complex"), (np.complex128(8.0), "complex128"),
    (Decimal("8"), "Decimal"), (np.array(8.0), "ndarray"), ([8.0], "list"),
])
def test_t_max_of_the_wrong_kind_raises_type_error_at_construction(t_max, kind, kw):
    """``t_max="300"`` used to be accepted, and parsed by ``float()`` at the first factorisation."""
    with pytest.raises(TypeError, match=f"^t_max must be a real number, got {kind}$"):
        _ev(t_max=t_max, **kw)


def test_real_t_max_passes_as_a_float_and_its_range_stays_lazy():
    ref = _ev().blocks(NU)
    for t_max in (np.float64(T_REC), np.int64(8)):
        ev = _ev(t_max=t_max)
        assert type(ev.block_evaluator.t_max) is float
        got = ev.blocks(NU)
        assert all(np.array_equal(a, b) for a, b in zip(got[0], ref[0], strict=True))
    late = _ev(t_max=0.5 * T_REC)                 # constructs: the range is the plan's check
    with pytest.raises(ValueError, match="pair mid-times"):
        late.blocks(NU)
```

`_cases` leaves out `None` where it is the documented default (`channel`, `jit`, `EvolutionaryCovariance`'s `layout`). The test ids are the slot and the `repr` of the value.

- [ ] **Step 3: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_covariance.py 2>&1 | tail -1`
Expected: `197 failed, 9 passed`.
- `test_counts_of_the_wrong_kind_raise_type_error`: 55 failed. Most with `Failed: DID NOT RAISE <class 'TypeError'>` (`int()` takes `2.7`, `True`, `"2"`, a 0-d array; `P` is not read at construction). The `None` cases of `n` (both classes), `params_per_channel` and `sizes` raise Python's `int() argument must be ...`, which does not match.
- `test_flags_of_the_wrong_kind_raise_type_error`: 29 failed, `DID NOT RAISE` (`bool()` takes them all; `jit` is not read on NumPy).
- `test_string_options_of_the_wrong_kind_raise_type_error`: 83 failed. Most reach the membership `ValueError` (for example `ValueError: chol_impl must be 'numpy' or 'scipy'; got 3`, `DiagFactor has no quadratic form 3`); `dense_factor`'s `chol_impl` fails in the backend's `impl` check; `EvolutionaryCovariance(layout=0)` and `layout=False`, and `DiagFactor.quad`/`StackFactor.quad` with `form=np.array("elementwise")`, do not raise.
- `test_channel_stack_channels_must_be_covariances`: 5 failed, `AttributeError: 'int' object has no attribute 'n'` (and the like).
- `test_t_max_of_the_wrong_kind_raises_type_error_at_construction`: 16 failed, `DID NOT RAISE`.
- `test_sizes_are_checked_element_by_element`: 4 failed, `DID NOT RAISE`.
- One each: `test_numpy_integer_counts_pass_and_are_stored_as_int` (`assert (np.int64(4) == 4 and <class 'numpy.int64'> is int)`: `P` is stored raw), `test_count_ranges_stay_value_errors` (`DID NOT RAISE <class 'ValueError'>`: `P=0` constructs), `test_the_psd_must_be_callable` (the old message), `test_unknown_strings_stay_value_errors[layout-empty]` (`DID NOT RAISE`), `test_real_t_max_passes_as_a_float_and_its_range_stays_lazy` (`assert <class 'numpy.float64'> is float`).
- The 9 that pass: `test_numpy_bool_flags_pass_and_are_stored_as_bool`, `test_numpy_str_options_pass`, `test_channel_stack_takes_any_covariances_and_still_needs_one`, and 6 of the 7 `test_unknown_strings_stay_value_errors` cases.

- [ ] **Step 4: `covariance/base.py`: `form` and `jit`**

The imports. Replace

```python
import numpy as np
from domainia.wilson.backend import NUMPY, Backend, get_backend, pytree_dataclass, require_x64
```

with

```python
import numpy as np
from domainia.wilson._checks import require_bool, require_str
from domainia.wilson.backend import NUMPY, Backend, get_backend, pytree_dataclass, require_x64
```

`quad_form`, its docstring and first line. Replace

```python
    """``b^T C^-1 b`` in the operation order ``form`` (module docstring); ``C`` the factor's
    constants (passed through to ``solve`` / ``whiten``)."""
    if form == "solve":
```

with

```python
    """``b^T C^-1 b`` in the operation order ``form`` (module docstring); ``C`` the factor's
    constants (passed through to ``solve`` / ``whiten``).  A ``form`` that is not a string
    raises ``TypeError``; a string the factor has no order for raises ``ValueError``."""
    require_str("form", form)
    if form == "solve":
```

`DiagFactor.quad` compares `form` before it calls `quad_form`, so it checks the kind there. Replace

```python
        be = be or get_backend(self.v, b)
        if form == "elementwise":
```

with

```python
        be = be or get_backend(self.v, b)
        if require_str("form", form) == "elementwise":
```

`CholFactor.quad` and `BlockTriCovFactor.quad` call `quad_form` directly, and `StackFactor.quad` calls each channel factor's `quad`, so they need no change.

`Covariance.factor`: `jit` is checked first, on every backend. Replace

```python
        NumPy, ``jit=False`` or a non-jittable covariance: eager, in the same order (the
        reference path).
        """
        be = be or get_backend(nu)
```

with

```python
        NumPy, ``jit=False`` or a non-jittable covariance: eager, in the same order (the
        reference path).  ``jit`` is ``None`` or a bool on every backend (``TypeError``
        otherwise).
        """
        jit = None if jit is None else require_bool("jit", jit)
        be = be or get_backend(nu)
```

and

```python
        if be.name == "jax" and (self.jittable if jit is None else bool(jit)):
```

with

```python
        if be.name == "jax" and (self.jittable if jit is None else jit):
```

- [ ] **Step 5: `diagonal.py`, `dense.py`, `blocktri.py`, `stack.py`**

`diagonal.py`, the imports. Replace

```python
from domainia.wilson.backend import Backend
```

with

```python
from domainia.wilson._checks import require_int
from domainia.wilson.backend import Backend
```

`DiagonalCovariance.__init__`: `n` and `channel` are checked before the variance is read. Replace

```python
    def __init__(self, variance: Any, n: int, *, channel: int | None = None) -> None:
        cells = getattr(variance, "cells", None)
        self.cells = cells if isinstance(cells, CellSet) else None
        window = getattr(variance, "window", None)
        self._window = window if isinstance(window, Window) else None
        self.variance = as_evaluator(variance, **({} if channel is None else {"channel": int(channel)}))
        self.n = int(n)
```

with

```python
    def __init__(self, variance: Any, n: int, *, channel: int | None = None) -> None:
        n = require_int("n", n)
        channel = None if channel is None else require_int("channel", channel)
        cells = getattr(variance, "cells", None)
        self.cells = cells if isinstance(cells, CellSet) else None
        window = getattr(variance, "window", None)
        self._window = window if isinstance(window, Window) else None
        self.variance = as_evaluator(variance, **({} if channel is None else {"channel": channel}))
        self.n = n
```

`dense.py`, the imports. Replace

```python
from domainia.wilson.backend import Backend
```

with

```python
from domainia.wilson._checks import require_bool, require_int, require_str
from domainia.wilson.backend import Backend
```

`dense_factor`. `chol_impl` gets its kind check here, under its own name. Its membership stays the backend's `ValueError` (`Backend.cholesky`'s `impl`), as today. Replace

```python
    ``symmetrize=True``) or ``cho_factor(S)`` (``"scipy"``, ``symmetrize=False``).
    """
    if symmetrize:
        S = 0.5 * (S + S.T)
```

with

```python
    ``symmetrize=True``) or ``cho_factor(S)`` (``"scipy"``, ``symmetrize=False``).
    ``symmetrize`` must be a bool and ``chol_impl`` a string (``TypeError`` otherwise).
    """
    require_str("chol_impl", chol_impl)
    if require_bool("symmetrize", symmetrize):
        S = 0.5 * (S + S.T)
```

`DenseCovariance.__init__`. Replace

```python
        if chol_impl not in ("numpy", "scipy"):
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        self.matrix = as_evaluator(matrix)
        self.n = int(n)
        self.chol_impl = chol_impl
        self.symmetrize = bool(symmetrize)
```

with

```python
        if require_str("chol_impl", chol_impl) not in ("numpy", "scipy"):
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        self.matrix = as_evaluator(matrix)
        self.n = require_int("n", n)
        self.chol_impl = chol_impl
        self.symmetrize = require_bool("symmetrize", symmetrize)
```

`blocktri.py`, the imports. Replace

```python
import numpy as np
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass
```

with

```python
import numpy as np
from domainia.wilson._checks import require_int, require_str
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass
```

The end of the `BlockTridiagonalCovariance` docstring and the start of `__init__`. Each size gets the integer rule, named by its index. Replace

```python
    ``fn(nu, be)``) returning ``(diag, up)`` ragged or padded (:func:`to_layout`);
    ``sizes`` the cells per band.  ``layout="ragged"`` runs eagerly on both backends.
    """

    def __init__(self, blocks: Any, sizes: tuple[int, ...], *, layout: Layout = "ragged") -> None:
        if layout not in ("ragged", "padded"):
            raise ValueError(f"layout must be 'ragged' or 'padded'; got {layout!r}")
        self.blocks = as_evaluator(blocks)
        self.sizes = tuple(int(s) for s in sizes)
```

with

```python
    ``fn(nu, be)``) returning ``(diag, up)`` ragged or padded (:func:`to_layout`);
    ``sizes`` the cells per band, a sequence of integers (an integer ndarray included).
    ``layout="ragged"`` runs eagerly on both backends.
    """

    def __init__(self, blocks: Any, sizes: tuple[int, ...], *, layout: Layout = "ragged") -> None:
        if require_str("layout", layout) not in ("ragged", "padded"):
            raise ValueError(f"layout must be 'ragged' or 'padded'; got {layout!r}")
        self.blocks = as_evaluator(blocks)
        self.sizes = tuple(require_int(f"sizes[{m}]", s) for m, s in enumerate(sizes))
```

`stack.py`, the imports. Replace

```python
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass
```

with

```python
from domainia.wilson._checks import require_instance, require_int
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass
```

`ChannelStack.__init__`. Each channel must be a `Covariance` (`PushedCovariance` is not one, and has none of the methods `ChannelStack` calls). Replace

```python
        if not self.covs:
            raise ValueError("ChannelStack needs at least one channel")
        self.k = int(params_per_channel)
```

with

```python
        if not self.covs:
            raise ValueError("ChannelStack needs at least one channel")
        for c, cov in enumerate(self.covs):
            require_instance(f"covs[{c}]", cov, Covariance, "a Covariance")
        self.k = require_int("params_per_channel", params_per_channel)
```

The `"non-negative; got"` line below it is a `ValueError` and stays.

- [ ] **Step 6: `evolutionary.py`: every option, and `P` and `t_max` at construction**

The imports. Replace

```python
from domainia.wilson.backend import Backend, get_backend
```

with

```python
from domainia.wilson._checks import require_bool, require_callable, require_int, require_real, require_str
from domainia.wilson.backend import Backend, get_backend
```

The `t_max` and `P` entries of the `EvolutionaryCovariance` docstring. Replace

```python
    t_max
        End of the Chebyshev time domain (default: the cells' grid ``T``).
    P
        Chebyshev order of ``interp`` (24).
```

with

```python
    t_max
        End of the Chebyshev time domain (default: the cells' grid ``T``).  A real number,
        checked at construction; whether the kept pair mid-times lie inside
        ``[0, t_max]`` is the plan's check, at the first factorisation.
    P
        Chebyshev order of ``interp`` (24): an integer ``>= 1``, checked at construction
        for every builder.
```

The checks at the top of `__init__`, down to the first assignment. Replace

```python
        if builder not in BUILDERS:
            raise ValueError(f"unknown builder {builder!r}; expected one of {BUILDERS}")
        if not callable(psd):
            raise TypeError(f"builder {builder!r} needs a callable PSD psd(f, t, nu); "
                            f"{type(psd).__name__} is not callable")
        cells, window = _cells_and_window(cells, window)
        if assembly not in ("blocktri", "dense"):
            raise ValueError(f"assembly must be 'blocktri' or 'dense'; got {assembly!r}")
        if builder == "direct" and assembly != "dense":
            raise ValueError("builder 'direct' needs assembly='dense'")
        if layout is not None and assembly == "dense":
            raise ValueError(f"layout applies to assembly='blocktri' only; got layout={layout!r} "
                             "with assembly='dense'")
        if chol_impl not in ("numpy", "scipy"):
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        self.psd, self.cells, self.builder, self.assembly = psd, cells, builder, assembly
```

with

```python
        if require_str("builder", builder) not in BUILDERS:
            raise ValueError(f"unknown builder {builder!r}; expected one of {BUILDERS}")
        require_callable("psd", psd)
        cells, window = _cells_and_window(cells, window)
        if require_str("assembly", assembly) not in ("blocktri", "dense"):
            raise ValueError(f"assembly must be 'blocktri' or 'dense'; got {assembly!r}")
        if builder == "direct" and assembly != "dense":
            raise ValueError("builder 'direct' needs assembly='dense'")
        if layout is not None:
            require_str("layout", layout)
            if assembly == "dense":
                raise ValueError(f"layout applies to assembly='blocktri' only; got layout={layout!r} "
                                 "with assembly='dense'")
        if require_str("chol_impl", chol_impl) not in ("numpy", "scipy"):
            raise ValueError(f"chol_impl must be 'numpy' or 'scipy'; got {chol_impl!r}")
        symmetrize = require_bool("symmetrize", symmetrize)
        P = require_int("P", P)
        if P < 1:
            raise ValueError(f"P must be >= 1, got {P}")
        t_max = None if t_max is None else require_real("t_max", t_max)
        self.psd, self.cells, self.builder, self.assembly = psd, cells, builder, assembly
```

`P` gets `eq45_plan`'s own check and message (`wilson/evolutionary.py`, `"P must be >= 1, got {P}"`), so `eq45_plan` later receives a checked `int`. `t_max` is converted to a Python `float` here. Its range (the kept pair mid-times inside `[0, t_max]`) stays the plan's check, at the first `.blocks()` or `.factor()` (`tests/test_dom_evolutionary_cov.py:92`).

The blocktri assembly tests `layout is None` instead of its truth, so that a `layout` of `""` reaches `BlockTridiagonalCovariance`'s `ValueError` instead of reading as `"ragged"`. Replace

```python
            self.impl: Covariance = BlockTridiagonalCovariance(blocks, sizes, layout=layout or "ragged")
```

with

```python
            self.impl: Covariance = BlockTridiagonalCovariance(blocks, sizes,
                                                               layout="ragged" if layout is None else layout)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "[^_]int(\|[^_]bool(\| or \"ragged\"\|is not callable" src/domainia/covariance/*.py; grep -n "raise TypeError" src/domainia/covariance/*.py`
Expected: four lines. `blocktri.py:150` (`self.n = int(sum(self.sizes))`), `diagonal.py:45` (`bool(getattr(variance, "jittable", True))`) and `evolutionary.py:155` (`int(np.sum(...))`), from the first grep; then `base.py:338` (`has no multiply`), from the second.

- [ ] **Step 7: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_covariance.py 2>&1 | tail -1`
Expected: `206 passed`, no warnings.

- [ ] **Step 8: The covariance tests**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_covariance.py tests/test_dom_evolutionary_cov.py tests/test_dom_cov_api.py tests/test_dom_cov_stack.py tests/test_dom_cov_blocktri.py tests/test_dom_cov_base.py tests/test_dom_cov_multiply.py 2>&1 | tail -1`
Expected: all pass, 0 errors. A failure here is a wrong-kind pin Step 1 did not find: a test that passes a float count, a non-bool flag or a non-string option. Fix the test's argument to the right kind, and record it in the commit message.

- [ ] **Step 9: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`
Expected: all pass (3 skipped: the `DOMAINIA_ORACLE_ROOT` skips), 0 errors, no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`
Expected: all pass, 0 errors. No existing test needs a change.

The float64 bit gate is not run here (Conventions: Tasks 2, 4, 10, 11 and 15). This task adds argument checks only: `P` and `t_max` reach `eq45_plan` as the same `int` and `float` values it computed before.

- [ ] **Step 10: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/covariance/base.py src/domainia/covariance/diagonal.py src/domainia/covariance/dense.py src/domainia/covariance/blocktri.py src/domainia/covariance/stack.py src/domainia/covariance/evolutionary.py tests/test_wrong_kinds_covariance.py
git commit -m "domainia.covariance: wrong kinds raise TypeError

The counts (DiagonalCovariance n and channel, DenseCovariance n,
ChannelStack params_per_channel, each BlockTridiagonalCovariance size)
must be integers instead of passing through int(). The symmetrize flags
and factor's jit must be bools, and chol_impl, layout, builder, assembly
and the factors' form must be strings before their membership checks.
ChannelStack's channels must be Covariances (an AttributeError before).
EvolutionaryCovariance checks P and t_max at construction for both
builders (t_max=\"300\" was accepted), its layout=0 or \"\" no longer
reads as \"ragged\", and a non-callable psd gets the \", got\" message.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Wrong kinds in spectra, data, translate, variance, linalg and testing (spec §2.3–§2.6, §2.9, §3.5)

**Goal:** The last six modules follow the rule: a wrong kind raises `TypeError` with a `", got"` message, and a wrong value raises `ValueError`. `from_coordinates` refuses a complex `R` with `TypeError` (§3.5). `to_coordinates` refuses a Fourier coordinate that overflows float64, instead of returning `inf` (§2.9). `domainia.testing` follows the rule like the rest (§2.9). `variance.py` needs no edit: its own check (`_require_generic`) already raises `TypeError`, and its `cells` check is `domains._cells_and_window` (Task 3).

**Assumes the tree after Task 5.** No earlier task edits `spectra.py`, `data.py`, `translate.py`, `variance.py`, `linalg.py` or `testing.py`, so the line numbers below are those of 779550d (Steps 8 and 15 give the numbers after this task's edits). Task 1's `domainia.wilson._checks` gives every helper used here. Two earlier changes meet this task:
* Task 3 converts `domains._cells_and_window`'s message to `"cells must be a CellSet or a WDMDomain, got {type}"`. Its suite needs the two anchored pins converted with it, `tests/test_dom_variance_generic.py:77` and `tests/test_dom_evolutionary_cov.py:101`. Step 1 checks the message and its pins. This task does not edit them.
* `eq45_plan` already checks its `P`, and Task 2 checks its `t_max` (§2.4). `testing.eq45_dense` passes both through and does not check them again.

The task has two red/green cycles, one commit each:
* Cycle A (Steps 1 to 9): `spectra.py`, `data.py`, `translate.py`, and the four existing pins they move.
* Cycle B (Steps 10 to 17): `linalg.py` and `testing.py`.

**Sites.** Each row is one slot; "check" is the `_checks` helper it calls.

| module | slot | check | `TypeError` message |
|---|---|---|---|
| `spectra` | `TabulatedSpectrum.domain` | `require_instance` | `TabulatedSpectrum.domain must be a FourierDomain, got {type}` |
| | `TabulatedSpectrum.values` | `require_dtype(..., REAL_KINDS, ...)` | `TabulatedSpectrum.values must be a real numeric array, got dtype {dtype}` |
| | `TabulatedSpectrum.on`'s `domain` | `require_instance` | `domain must be a FourierDomain, got {type}` |
| | `TabulatedSpectrum.at`'s `bins` | `index_array` | `bins must be an integer index array, got dtype {dtype}` (2.5, nan, inf: `ValueError` `bins must hold integer indices, got {first}`) |
| | `GenericPSD.fn` | `require_callable` | `GenericPSD.fn must be callable, got {type}` |
| `data` | `channels` of `TimeSeries`, `FourierBins`, `WDMCells` | `require_instance` | `{Class}.channels must be a ChannelAxis, got {type}` |
| | their `domain` (checked today) | `require_instance` | `{Class}.domain must be a {Domain}, got {type}` |
| `translate` | `from_coordinates`' `R` | `require_dtype(..., REAL_KINDS, ...)` | `R must be an array of real coordinates, got dtype {dtype}` |
| | nine existing kind messages | text only | `"; got"` → `", got"` |
| `linalg` | each size of `BandLayout.from_sizes` | `require_int` | `sizes[{m}] must be an integer, got {type}` |
| | `BandLayout.from_cells`' `cells` | `require_instance` | `cells must be a CellSet, got {type}` |
| | `blocktri_factor`'s `layout` | `require_str` | `layout must be a string, got {type}` |
| | `band_layout` of `blocktri_factor`, `blocktri_solve`, `blocktri_whiten` | `require_instance` | `band_layout must be a BandLayout, got {type}` |
| | `f` of `blocktri_solve`, `blocktri_whiten`, `blocktri_solve_packed`, `blocktri_whiten_packed` | `require_instance` | `f must be a BlockTriFactor, got {type}` |
| | `f` of `cblock_forward`, `cblock_backward` | `require_instance` | `f must be a CBlockFactor, got {type}` |
| | `cblock_factor`'s `b` | `require_int` | `b must be an integer, got {type}` |
| | `pcg`'s `tol`, `true_rtol` (`None` is its default) | `require_real` | `tol must be a real number, got {type}` |
| | `pcg`'s `maxit` | `require_int` | `maxit must be an integer, got {type}` |
| | `pcg`'s `on_fail` | `require_str` | `on_fail must be a string, got {type}` |
| `testing` | `cells` of `dense_W` (so of `wdm_congruence`) and `eq45_dense` | `require_instance` | `cells must be a CellSet, got {type}` |
| | `eq45_dense`'s `Sn` | `require_callable` | `Sn must be callable, got {type}` |
| | `eq54_covariance`'s `margin` (`dense_lambda_generic` passes its own through) | `require_int` | `margin must be an integer, got {type}` |
| | `eq54_covariance`'s `s_I`, `s_G_src` | `require_dtype(..., REAL_KINDS, ...)` | `s_I must be a real numeric array, got dtype {dtype}` |
| | four `PointwiseOperator` messages | text only | `"; got"` → `", got"` |

And §2.9: `to_coordinates` raises `ValueError("channel 'E': sqrt(2) times the imaginary part of rfft bin 3 overflows float64 (a part above 1.27e+308); rescale the data first")` (channel, part and bin as found).

**How the sites were found.** In the scratch prototype, a probe called every public entry point of the six modules with a wrong kind in each slot: an `object()`, a `Grid` or a `TimeDomain` for a domain or object; a tuple, a str, `np.str_`, `np.float64` or `None` for `channels`; `2.0`, `2.5`, `True`, `"2"` or `None` for an integer; `True`, `"1e-6"`, `None`, complex, `Decimal` or a 0-d array for a real; `1` or `None` for an option; complex, bool, str or object arrays. It changed 103 outcomes. 27 bare `AttributeError`s, 32 silent acceptances (3 with only a `ComplexWarning`), 9 `ValueError`s, 2 `IndexError`s and 2 NumPy `UFuncTypeError`s became the `TypeError`s above; 2 `ValueError`s (`bins` of 2.5 and nan) stayed `ValueError`s with `index_array`'s text; and 29 `TypeError`s changed their text (the `"; got"` form, or Python's own message for `pcg`'s `tol`/`maxit` and `eq54_covariance`'s `margin`).

**Left as they are** (record, not scope): the `be=` keyword, which every map, covariance and kernel takes, so a `be="numpy"` raises a bare `AttributeError` package-wide; array-likes given as lists to the kernels (`blocktri_solve(f, [...])`, `hhat`'s `basis_L`), which are not wrong kinds; the value arrays the spec does not name (`wdm_congruence`'s `Sigma`, `PointwiseOperator`'s variances and values, `hhat`'s `lags`, `blocktri_dense`'s blocks, the input of `smoothstep` and `wide_profile`); `pcg`'s `apply_A`/`apply_M`, which already raise Python's `TypeError` when not callable; and the five `TypeError` messages with no "got" clause (`variance.py:45`, `linalg.py:262`, `translate.py:129`, `:347`, `:473`), which Task 9's gate does not flag. `translate.py:473` is pinned with `^...$` at `tests/test_translate_operator.py:445`.

**Files:**
- Modify: `src/domainia/spectra.py` (imports; `TabulatedSpectrum`'s docstring, `__post_init__`, `on`, `at`; `GenericPSD.__post_init__`, new)
- Modify: `src/domainia/data.py` (imports; module docstring; the three `__post_init__`)
- Modify: `src/domainia/translate.py` (imports; nine kind messages; `to_coordinates`; `from_coordinates`)
- Modify: `src/domainia/linalg.py` (imports; `BandLayout.from_sizes`, `BandLayout.from_cells`, `blocktri_factor`, `_layout_of`, `blocktri_whiten`, `blocktri_solve`, `blocktri_whiten_packed`, `blocktri_solve_packed`, `cblock_factor`, `cblock_forward`, `cblock_backward`, `pcg`)
- Modify: `src/domainia/testing.py` (imports; module docstring; `dense_W`, `eq45_dense`, `eq54_covariance`; four `PointwiseOperator` messages)
- Create: `tests/test_wrong_kinds_misc.py`
- Modify: `tests/test_dom_spectra.py` (`test_tabulated_spectrum_slices_by_absolute_bin_and_refuses_outside`), `tests/test_translate_coordinates.py` (`test_argument_refusals`), `tests/test_translate_operator.py` (the `object-dtype` case and `test_refusals`)

**Acceptance Criteria:**
- [ ] Every slot in the Sites table raises `TypeError` with its message for a wrong kind: `bool`, a float or a str where an integer is taken; `bool`, a str, complex, `None` or a 0-d array where a real is taken; a non-string option; a wrong object; a complex, bool, str or object dtype where real numbers are taken. `np.int64` passes every integer slot, and `np.int64` and `np.float64` pass every real slot and real array.
- [ ] Wrong values stay `ValueError` with their text: an unknown `on_fail` or `layout` string, a non-integral or non-finite bin, an `s_G_src` of the wrong length, complex values on a time or WDM domain in `transform`.
- [ ] `to_coordinates` raises `ValueError` naming the channel, the part and the bin when `sqrt(2)` times a real or imaginary part overflows (1.3e308); at 1.2e308 it returns finite coordinates.
- [ ] `grep -n "; got"` over the six files lists only lines of `raise ValueError(...)` calls (13 lines, Step 15).
- [ ] `tests/test_wrong_kinds_misc.py`: 111 passed. The full suite and the NumPy lane pass, with no new warnings.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_misc.py` → `111 passed`; then the full suite and the NumPy lane → all pass.

**Steps:**

#### Cycle A: `spectra.py`, `data.py`, `translate.py`

- [ ] **Step 1: Grep the sites and the tests that pin them**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" src/domainia/spectra.py src/domainia/data.py src/domainia/translate.py src/domainia/variance.py src/domainia/linalg.py src/domainia/testing.py`
Expected: 31 lines, none in `variance.py`. Sixteen are on `TypeError` messages, and this task converts all of them:
- `data.py:59`, `:74`, `:90`: the three `domain` checks (Step 6);
- `translate.py:68` (`_record`), `:115`, `:125`, `:134` (`_stack`), `:262` (`_channel_names`), `:350` (`from_coordinates`), `:442` (`TranslatedOperator.__post_init__`), `:462` (`_to_native`), `:511` (`translate_operator`) (Step 7);
- `testing.py:177`, `:180`, `:187`, `:211`: `PointwiseOperator` (Step 14).

The other 15 are on `ValueError` messages and keep their text, except two that go away: `spectra.py:70` (`at`; `index_array` replaces it, Step 5) and `translate.py:352` (`from_coordinates`' complex check, now a `TypeError`, Step 7).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "; got" tests --include='*.py' | grep -v tests/wilson/oracle`
Expected: 16 hits. Two pin `translate.py` `TypeError` messages, and Step 3 converts them: `tests/test_translate_operator.py:457` (`"channel 'A': values must be numeric; got dtype object"`) and `:526` (`...WDMDomain\.full\); got str$`). The other 14 pin `ValueError` messages of other modules (or quote one in a docstring) and stay: `tests/test_wrong_kinds_covariance.py:94`, `:180`, `:182`, `:183`, `tests/test_wrong_kinds_domains.py:147`, `:153`, `:161`, `tests/test_wrong_kinds_maps.py:237`, `tests/wilson/test_checks.py:262`, `:265`, `tests/wilson/test_wrong_kinds.py:218`, `:298`, `:313`, `:330`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "a CellSet or a WDMDomain" src tests`
Expected: five hits. `src/domainia/domains.py` (`_cells_and_window`: `require_instance("cells", cells, CellSet, "a CellSet or a WDMDomain")`) and `src/domainia/wilson/_checks.py` (an example in `require_instance`'s docstring); and three pins of the `, got` text, `tests/test_dom_evolutionary_cov.py:101`, `tests/test_dom_variance_generic.py:77` and `tests/test_wrong_kinds_domains.py:70`. Task 3 converted them.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn 'match="integers"\|match="complex"\|match="real coordinates"' tests --include='*.py'`
Expected: three hits.
- `tests/test_dom_spectra.py:26` (`ValueError, match="integers"`, around `s.at([100.5])`): `index_array`'s message is `bins must hold integer indices, got 100.5`, which "integers" does not match. Step 3 pins the new text.
- `tests/test_translate_coordinates.py:168` (`ValueError, match="complex"`, around `from_coordinates(R + 0j, ...)`): becomes `TypeError` (spec §3.5). Step 3.
- `tests/test_translate_coordinates.py:172` (`TypeError, match="real coordinates"`, an object `R`): the new message `R must be an array of real coordinates, got dtype object` still matches. No change.

The `"complex"` refusals of `tests/test_translate.py:193-194` (`complex-time`, `complex-wdm`) are `transform`'s complex-on-time/WDM `ValueError`, which stays (spec §3.5). They sit in a dict, so the grep does not show them; they pass unchanged.

A bare `pytest.raises(ValueError)` around a site that becomes a `TypeError` would not show in these greps. In the scratch prototype, the full suite and the NumPy lane with the source edits of both cycles applied (Steps 5 to 7 and 13 to 14) and no test edits each failed exactly the four tests that Step 3 changes, so these are the whole pin list.

- [ ] **Step 2: Write the failing tests `tests/test_wrong_kinds_misc.py`**

Create the file with this content (cycle B appends to it in Step 11):

```python
"""``domainia.spectra``, ``domainia.data``, ``domainia.translate``, ``domainia.variance``,
``domainia.linalg`` and ``domainia.testing`` refuse a wrong kind with ``TypeError`` and a wrong
value with ``ValueError``, as ``domainia.wilson`` does."""

from __future__ import annotations

import re

import numpy as np
import pytest

from domainia import wilson
from domainia.data import FourierBins, TimeSeries, WDMCells
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from domainia.spectra import GenericPSD, TabulatedSpectrum
from domainia.translate import (TranslatedOperator, from_coordinates, log_coordinate_scale,
                                to_coordinates, transform, translate_operator)
from domainia.variance import BandAveragedVariance, MidpointVariance

GRID = wilson.Grid(8, 16, 0.5)
CELLS = CellSet.all_active(GRID)
TIME = TimeDomain(128, 0.5)
FOURIER = FourierDomain.full(128, 0.5)
WDM = WDMDomain.full(GRID)
BINS = FourierDomain(128, 0.5, 2, 10, "dt_rfft")               # absolute bins 2 .. 11
SPECTRUM = TabulatedSpectrum(BINS, np.arange(10.0)[None])


def exactly(message):
    return f"^{re.escape(message)}$"


# --- domainia.spectra ---------------------------------------------------------------------


@pytest.mark.parametrize(("domain", "kind"), [(TIME, "TimeDomain"), (GRID, "Grid"), (None, "NoneType")])
def test_a_spectrum_lives_on_a_fourier_domain(domain, kind):
    """Both used to raise a bare ``AttributeError`` (``domain.n``, ``domain.N``)."""
    message = f"TabulatedSpectrum.domain must be a FourierDomain, got {kind}"
    with pytest.raises(TypeError, match=exactly(message)):
        TabulatedSpectrum(domain, np.ones((1, 10)))
    with pytest.raises(TypeError, match=exactly(f"domain must be a FourierDomain, got {kind}")):
        SPECTRUM.on(domain)


@pytest.mark.parametrize(("values", "dtype"), [
    (np.ones((1, 10)) + 0j, "complex128"), (np.ones((1, 10), bool), "bool"),
    (np.full((1, 10), "1"), "<U1"), (np.ones((1, 10)).astype(object), "object"),
])
def test_spectrum_values_must_be_real_numbers(values, dtype):
    """A complex table used to lose its imaginary part (only a ``ComplexWarning``); a bool, str
    or object table was converted to float."""
    message = f"TabulatedSpectrum.values must be a real numeric array, got dtype {dtype}"
    with pytest.raises(TypeError, match=exactly(message)):
        TabulatedSpectrum(BINS, values)


def test_spectrum_values_of_numpy_integers_and_floats_pass():
    for values in (np.arange(10, dtype=np.int64)[None], np.arange(10, dtype=np.float64)[None]):
        s = TabulatedSpectrum(BINS, values)
        assert s.values.dtype == np.float64 and np.array_equal(s.values, SPECTRUM.values)


@pytest.mark.parametrize(("bins", "dtype"), [
    ([True], "bool"), ([2 + 0j], "complex128"), (["2"], "<U1"), (np.array([2], dtype=object), "object"),
])
def test_spectrum_bins_of_the_wrong_kind_raise_type_error(bins, dtype):
    """They used to raise ``ValueError("bins must be integers; ...")``."""
    with pytest.raises(TypeError, match=exactly(f"bins must be an integer index array, got dtype {dtype}")):
        SPECTRUM.at(bins)


def test_spectrum_bins_that_are_not_integers_stay_value_errors():
    for bins, first in (([2.5], "2.5"), ([3.0, np.nan], "nan"), ([np.inf], "inf")):
        with pytest.raises(ValueError, match=exactly(f"bins must hold integer indices, got {first}")):
            SPECTRUM.at(bins)
    want = SPECTRUM.values[:, [0, 9]]
    assert np.array_equal(SPECTRUM.at(np.array([2, 11], dtype=np.int64)), want)
    assert np.array_equal(SPECTRUM.at(np.array([2.0, 11.0], dtype=np.float64)), want)


@pytest.mark.parametrize(("fn", "kind"), [(1.0, "float"), (None, "NoneType"), ("psd", "str")])
def test_a_generic_psd_needs_a_callable(fn, kind):
    """It used to accept anything and fail at the first evaluation."""
    with pytest.raises(TypeError, match=exactly(f"GenericPSD.fn must be callable, got {kind}")):
        GenericPSD(fn)


# --- domainia.variance (already TypeError: pinned here) -----------------------------------


@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
def test_the_variances_take_only_a_generic_psd(cls):
    with pytest.raises(TypeError, match="wrap the spectrum in domainia.spectra.GenericPSD"):
        cls(lambda f, t, nu: f, CELLS)


# --- domainia.data ------------------------------------------------------------------------

DATA = [(TimeSeries, TIME, np.zeros((1, 128))), (FourierBins, BINS, np.zeros((1, 10), complex)),
        (WDMCells, WDM, np.zeros((1, 128)))]


@pytest.mark.parametrize(("cls", "domain", "values"), DATA, ids=["TimeSeries", "FourierBins", "WDMCells"])
@pytest.mark.parametrize(("channels", "kind"), [
    (("A",), "tuple"), ("A", "str"), (np.str_("A"), "str_"), (np.float64(1.0), "float64"),
    (None, "NoneType"),
])
def test_data_channels_must_be_a_channel_axis(cls, domain, values, channels, kind):
    """A tuple, a str or ``None`` used to raise a bare ``AttributeError`` (``channels.size``),
    and a NumPy scalar was taken as a one-name axis (spec §2.6)."""
    message = f"{cls.__name__}.channels must be a ChannelAxis, got {kind}"
    with pytest.raises(TypeError, match=exactly(message)):
        cls(values, domain, channels)
    assert cls(values, domain, ChannelAxis(("A",))).channels.names == ("A",)


@pytest.mark.parametrize(("cls", "kind"), [(TimeSeries, "TimeDomain"), (FourierBins, "FourierDomain"),
                                           (WDMCells, "WDMDomain")])
def test_data_domain_messages_say_got(cls, kind):
    with pytest.raises(TypeError, match=exactly(f"{cls.__name__}.domain must be a {kind}, got Grid")):
        cls(np.zeros(128), GRID)


# --- domainia.translate -------------------------------------------------------------------

Y = {"A": np.ones(128)}
FULL_RECORD = "must be a full-record domain (TimeDomain, FourierDomain.full or WDMDomain.full), got Grid"


@pytest.mark.parametrize(("call", "message"), [
    (lambda: transform([np.ones(128)], TIME, WDM),
     "channel_data must be a mapping {channel: array}, got list"),
    (lambda: transform({1: np.ones(128)}, TIME, WDM), "channel names must be strings, got 1"),
    (lambda: transform({"A": np.ones(128).astype(object)}, TIME, WDM),
     "channel 'A': values must be numeric, got dtype object"),
    (lambda: transform(Y, GRID, WDM), f"source {FULL_RECORD}"),
    (lambda: log_coordinate_scale(GRID), f"domain {FULL_RECORD}"),
    (lambda: to_coordinates(Y, TIME, "A"), "channel_names must be a tuple of channel names, got str"),
    (lambda: translate_operator(object(), WDM),
     "op must implement domainia.translate.Operator, got object"),
    (lambda: TranslatedOperator(object(), WDM),
     "native must be a native domainia.translate.Operator (translate_operator re-targets a "
     "TranslatedOperator), got object"),
], ids=["mapping", "name", "numeric", "source", "domain", "channel_names", "op", "native"])
def test_translate_kind_messages_say_got(call, message):
    with pytest.raises(TypeError, match=exactly(message)):
        call()


@pytest.mark.parametrize(("R", "dtype"), [
    (np.ones((1, 128)) + 0j, "complex128"), (np.ones((1, 128), bool), "bool"),
    (np.full((1, 128), "1"), "<U1"), (np.ones((1, 128)).astype(object), "object"),
])
def test_from_coordinates_refuses_coordinates_that_are_not_real(R, dtype):
    """Complex coordinates used to raise ``ValueError`` (spec §3.5): ``R`` is never complex."""
    message = f"R must be an array of real coordinates, got dtype {dtype}"
    with pytest.raises(TypeError, match=exactly(message)):
        from_coordinates(R, TIME, ("A",))


def test_from_coordinates_takes_numpy_integers_and_floats():
    for R in (np.ones((1, 128), dtype=np.int64), np.ones((1, 128), dtype=np.float64)):
        out = from_coordinates(R, TIME, ("A",))
        assert out["A"].dtype == np.float64 and np.array_equal(out["A"], np.ones(128))


def test_complex_values_on_a_time_domain_stay_a_value_error():
    """Complex values are valid on a ``FourierDomain``: the domain makes them wrong (spec §3.5)."""
    with pytest.raises(ValueError, match="complex values on a real TimeDomain"):
        transform({"A": np.ones(128) + 0j}, TIME, WDM)


@pytest.mark.parametrize("part", ["real", "imaginary"])
def test_to_coordinates_refuses_an_overflowing_sqrt2_scaling(part):
    """``sqrt(2) * 1.3e308`` overflows float64; it used to come back as ``inf`` (spec §2.9).
    NumPy warns about the overflow before the check runs, hence ``np.errstate``."""
    unit = 1.0 if part == "real" else 1j
    X = np.zeros(65, complex)
    X[3] = 1.3e308 * unit
    data = {"A": np.zeros(65, complex), "E": X}
    message = f"^channel 'E': sqrt\\(2\\) times the {part} part of rfft bin 3 overflows float64"
    with np.errstate(over="ignore"), pytest.raises(ValueError, match=message):
        to_coordinates(data, FOURIER, ("E", "A"))
    X[3] = 1.2e308 * unit
    R = to_coordinates(data, FOURIER, ("E", "A"))
    assert np.all(np.isfinite(R)) and np.max(np.abs(R[0])) == np.sqrt(2.0) * 1.2e308
```

- [ ] **Step 3: Move the four pinned cases**

`tests/test_dom_spectra.py`, in `test_tabulated_spectrum_slices_by_absolute_bin_and_refuses_outside`:

```python
    with pytest.raises(ValueError, match="integers"):
        s.at([100.5])                                                            # never truncated
```
→
```python
    with pytest.raises(ValueError, match="bins must hold integer indices, got 100.5"):
        s.at([100.5])                                                            # never truncated
```

`tests/test_translate_coordinates.py`, in `test_argument_refusals`:

```python
    with pytest.raises(ValueError, match="complex"):
        from_coordinates(R + 0j, d, ("A", "E"))
```
→
```python
    with pytest.raises(TypeError, match="R must be an array of real coordinates, got dtype complex128"):
        from_coordinates(R + 0j, d, ("A", "E"))
```

`tests/test_translate_operator.py`, the `object-dtype` case of `test_every_transform_refusal_of_a_native_output_is_blamed_on_the_native`:

```python
     "channel 'A': values must be numeric; got dtype object"),
```
→
```python
     "channel 'A': values must be numeric, got dtype object"),
```

and in `test_refusals` (the `native.domain` pin; the line above it stays):

```python
                                            r"FourierDomain\.full or WDMDomain\.full\); got str$"):
```
→
```python
                                            r"FourierDomain\.full or WDMDomain\.full\), got str$"):
```

- [ ] **Step 4: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_misc.py tests/test_dom_spectra.py tests/test_translate_coordinates.py tests/test_translate_operator.py`
Expected: `51 failed, 146 passed, 1 warning` (the warning is the `ComplexWarning` of `spectra.py:37`, from the complex `values` case):
- `test_wrong_kinds_misc.py`: 47 failed, 5 passed. The `TypeError` cases get no error at all (`GenericPSD`, `np.str_`/`np.float64` channels, the bool/str/object `values`), a bare `AttributeError` (`'tuple' object has no attribute 'size'`, `'TimeDomain' object has no attribute 'n'`), a `ValueError` (`bins must be integers; got bool values`, `R must hold real coordinates; got complex values`), or the old `"; got"` text. The two overflow cases fail with `DID NOT RAISE <class 'ValueError'>`. The five that pass are the NumPy-type and complex-on-time pins and the two variance pins.
- `test_dom_spectra.py::test_tabulated_spectrum_slices_by_absolute_bin_and_refuses_outside`: `bins must be integers; got float64 values` does not match the new text.
- `test_translate_coordinates.py::test_argument_refusals`: `ValueError: R must hold real coordinates; got complex values` instead of `TypeError`.
- `test_translate_operator.py::test_every_transform_refusal_of_a_native_output_is_blamed_on_the_native[object-dtype]` and `::test_refusals`: the old `"; got"` text.

- [ ] **Step 5: `src/domainia/spectra.py`**

The imports:

```python
import numpy as np
from domainia.wilson.backend import Backend
```
→
```python
import numpy as np
from domainia.wilson._checks import (REAL_KINDS, index_array, require_callable, require_dtype,
                                     require_instance)
from domainia.wilson.backend import Backend
```

`TabulatedSpectrum`'s docstring:

```python
    ``values`` is a read-only float64 copy (the caller's array is never aliased), as
    ``CellSet`` keeps its indices.
```
→
```python
    ``values`` is a read-only float64 copy (the caller's array is never aliased), as
    ``CellSet`` keeps its indices.  A ``domain`` that is not a ``FourierDomain``, or ``values``
    of a complex, bool, str or object dtype, raises ``TypeError``.
```

`TabulatedSpectrum.__post_init__`, the kind checks before the float copy (the cast is what drops an imaginary part with only a `ComplexWarning`):

```python
    def __post_init__(self) -> None:
        v = np.array(self.values, dtype=float)                 # a copy, values unchanged
        if v.ndim != 2 or v.shape[1] != self.domain.n:
            raise ValueError(f"values must be (channels, {self.domain.n}); got {v.shape}")
```
→
```python
    def __post_init__(self) -> None:
        d = require_instance("TabulatedSpectrum.domain", self.domain, FourierDomain, "a FourierDomain")
        require_dtype("TabulatedSpectrum.values", self.values, REAL_KINDS, "a real numeric array")
        v = np.array(self.values, dtype=float)                 # a copy, values unchanged
        if v.ndim != 2 or v.shape[1] != d.n:
            raise ValueError(f"values must be (channels, {d.n}); got {v.shape}")
```

`TabulatedSpectrum.on`:

```python
        ``domain.units`` and ``domain.sigma0`` are not read (a PSD has no Fourier units)."""
        self._check(domain.N, domain.dt, domain.k0, domain.k0 + domain.n)
```
→
```python
        ``domain.units`` and ``domain.sigma0`` are not read (a PSD has no Fourier units)."""
        require_instance("domain", domain, FourierDomain, "a FourierDomain")
        self._check(domain.N, domain.dt, domain.k0, domain.k0 + domain.n)
```

`TabulatedSpectrum.at`, from its docstring's second paragraph to the cast:

```python
        Integer bins only: a non-integral float (``99.9999``) raises instead of being
        truncated by the int64 cast.
        """
        b = np.asarray(bins)
        if b.dtype.kind not in "iu" and not (b.dtype.kind == "f" and np.all(np.isfinite(b))
                                             and np.array_equal(b, np.rint(b))):
            raise ValueError(f"bins must be integers; got {b.dtype} values")
        bins = b.astype(np.int64)
```
→
```python
        Integer bins only: integral floats (``[100.0]``) are accepted, a non-integral or
        non-finite float (``99.9999``) raises ``ValueError`` instead of being truncated by the
        int64 cast, and a bool, complex, str or object array raises ``TypeError``.
        """
        bins = index_array("bins", bins)
```

(the two lines after it, `if bins.size:` and the `return`, stay.) `index_array` returns an int64 array, so `bins - self.domain.k0` is unchanged.

`GenericPSD` gains a `__post_init__`. `GenericPSD` is a plain frozen dataclass, not a pytree, so no JAX rebuild reaches it:

```python
    fn: Callable[[Any, Any, Any], Any]

    def __call__(self, f: Any, t: Any, nu: Any, be: Backend | None = None) -> Any:
```
→
```python
    fn: Callable[[Any, Any, Any], Any]

    def __post_init__(self) -> None:
        require_callable("GenericPSD.fn", self.fn)

    def __call__(self, f: Any, t: Any, nu: Any, be: Backend | None = None) -> Any:
```

- [ ] **Step 6: `src/domainia/data.py`**

The three classes are pytrees whose `domain` and `channels` are static meta (spec §2.7): a JAX rebuild passes the same objects back, so the checks are safe there.

The imports:

```python
from domainia.wilson.backend import is_tracer, pytree_dataclass
```
→
```python
from domainia.wilson._checks import require_instance
from domainia.wilson.backend import is_tracer, pytree_dataclass
```

The module docstring:

```python
``C = len(channels.names)``; ``N`` is ``domain.size``.  The shape is checked when
```
→
```python
``C = len(channels.names)``; ``N`` is ``domain.size``.  ``domain`` must be the class's own
domain kind and ``channels`` a :class:`~domainia.domains.ChannelAxis` (``TypeError``
otherwise; a tuple or a str of names is refused).  The shape is checked when
```

`TimeSeries.__post_init__`:

```python
        if not isinstance(self.domain, TimeDomain):
            raise TypeError(f"TimeSeries.domain must be a TimeDomain; got {type(self.domain).__name__}")
        _check(self.values, self.domain.N, self.channels, "TimeSeries")
```
→
```python
        require_instance("TimeSeries.domain", self.domain, TimeDomain, "a TimeDomain")
        require_instance("TimeSeries.channels", self.channels, ChannelAxis, "a ChannelAxis")
        _check(self.values, self.domain.N, self.channels, "TimeSeries")
```

`FourierBins.__post_init__`:

```python
        if not isinstance(self.domain, FourierDomain):
            raise TypeError(f"FourierBins.domain must be a FourierDomain; got "
                            f"{type(self.domain).__name__}")
        _check(self.values, self.domain.n, self.channels, "FourierBins")
```
→
```python
        require_instance("FourierBins.domain", self.domain, FourierDomain, "a FourierDomain")
        require_instance("FourierBins.channels", self.channels, ChannelAxis, "a ChannelAxis")
        _check(self.values, self.domain.n, self.channels, "FourierBins")
```

`WDMCells.__post_init__`:

```python
        if not isinstance(self.domain, WDMDomain):
            raise TypeError(f"WDMCells.domain must be a WDMDomain; got {type(self.domain).__name__}")
        _check(self.values, self.domain.size, self.channels, "WDMCells")
```
→
```python
        require_instance("WDMCells.domain", self.domain, WDMDomain, "a WDMDomain")
        require_instance("WDMCells.channels", self.channels, ChannelAxis, "a ChannelAxis")
        _check(self.values, self.domain.size, self.channels, "WDMCells")
```

Task 3 makes `ChannelAxis(names)` itself refuse a bare str; that is not this task's. This check refuses anything that is not already a `ChannelAxis`.

- [ ] **Step 7: `src/domainia/translate.py`**

The imports:

```python
from domainia import wilson
from domainia.domains import ChannelAxis, FourierDomain, TimeDomain, WDMDomain
```
→
```python
from domainia import wilson
from domainia.domains import ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from domainia.wilson._checks import REAL_KINDS, require_dtype
```

`_record`:

```python
    raise TypeError(f"{role} must be a full-record domain (TimeDomain, FourierDomain.full or "
                    f"WDMDomain.full); got {type(domain).__name__}")
```
→
```python
    raise TypeError(f"{role} must be a full-record domain (TimeDomain, FourierDomain.full or "
                    f"WDMDomain.full), got {type(domain).__name__}")
```

`_stack`, three messages:

```python
        raise TypeError(f"channel_data must be a mapping {{channel: array}}; got "
                        f"{type(channel_data).__name__}")
```
→
```python
        raise TypeError(f"channel_data must be a mapping {{channel: array}}, got "
                        f"{type(channel_data).__name__}")
```

```python
            raise TypeError(f"channel names must be strings; got {name!r}")
        if not name:
```
→
```python
            raise TypeError(f"channel names must be strings, got {name!r}")
        if not name:
```

```python
            raise TypeError(f"channel {name!r}: values must be numeric; got dtype {a.dtype}")
```
→
```python
            raise TypeError(f"channel {name!r}: values must be numeric, got dtype {a.dtype}")
```

`_channel_names`:

```python
        raise TypeError(f"channel_names must be a tuple of channel names; got "
                        f"{type(channel_names).__name__}")
```
→
```python
        raise TypeError(f"channel_names must be a tuple of channel names, got "
                        f"{type(channel_names).__name__}")
```

`to_coordinates`, from the end of its docstring to the end of the function. `_stack` already refused non-finite input, so a non-finite coordinate can only come from the `sqrt(2)` scaling in `_real_coordinates`. The check follows `_finish`'s pattern: it does not change `np.errstate`, so NumPy's overflow `RuntimeWarning` still comes first. `names = want` makes the row names follow `channel_names` as the rows do:

```python
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
```
→
```python
    must be a full-record domain (``TypeError`` for a non-domain, ``ValueError`` for a
    partial one, as :func:`log_coordinate_scale`).  A Fourier coordinate that overflows
    float64 (``sqrt(2)`` times a real or imaginary part above about 1.27e308) raises
    ``ValueError`` instead of coming back as ``inf``.
    """
    _record(domain, "domain")
    want = None if channel_names is None else _channel_names(channel_names)
    names, X = _stack(channel_data, domain)
    if want is not None:
        if set(names) != set(want):
            raise ValueError(f"channel_data has channels {names}; channel_names are {want}")
        X = X[[names.index(k) for k in want]]
        names = want
    R = _real_coordinates(X, domain)
    bad = ~np.isfinite(R)
    if bad.any():
        # X is finite (_stack checked it), so only the Fourier sqrt(2) scaling can overflow.
        row, col = (int(i) for i in np.argwhere(bad)[0])
        h = (domain.N - 1) // 2
        k, part = (col, "real") if col <= h else (col - h, "imaginary")
        limit = np.finfo(np.float64).max / np.sqrt(2.0)
        raise ValueError(f"channel {names[row]!r}: sqrt(2) times the {part} part of rfft bin {k} "
                         f"overflows float64 (a part above {limit:.3g}); rescale the data first")
    return R
```

`from_coordinates`, from its "Raises" sentence to the dtype check (spec §3.5: `R` is never complex, so a complex dtype is a wrong kind). The shape check and everything after it stay:

```python
    Raises ``TypeError`` for a masked array, non-numeric values, or ``channel_names`` that
    is not a tuple or list of non-empty strings; ``ValueError`` for duplicate or empty
    ``channel_names``, complex or non-finite values, or a shape other than
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
```
→
```python
    Raises ``TypeError`` for a masked array, values that are not real numbers (a complex,
    bool, str or object dtype: ``R`` is never complex), or ``channel_names`` that is not a
    tuple or list of non-empty strings; ``ValueError`` for duplicate or empty
    ``channel_names``, non-finite values, or a shape other than ``(len(channel_names), N)``;
    ``domain`` as :func:`to_coordinates`.
    """
    N, _ = _record(domain, "domain")
    names = _channel_names(channel_names)
    if isinstance(R, np.ma.MaskedArray):
        raise TypeError("R: masked arrays are refused (numpy.asarray drops the mask)")
    a = np.asarray(R)
    require_dtype("R", a, REAL_KINDS, "an array of real coordinates")
```

`TranslatedOperator.__post_init__`:

```python
            raise TypeError(f"native must be a native domainia.translate.Operator (translate_operator "
                            f"re-targets a TranslatedOperator); got {type(self.native).__name__}")
```
→
```python
            raise TypeError(f"native must be a native domainia.translate.Operator (translate_operator "
                            f"re-targets a TranslatedOperator), got {type(self.native).__name__}")
```

`TranslatedOperator._to_native`:

```python
            raise TypeError(f"values must be a mapping {{channel: array}}; got {type(values).__name__}")
```
→
```python
            raise TypeError(f"values must be a mapping {{channel: array}}, got {type(values).__name__}")
```

`translate_operator`:

```python
        raise TypeError(f"op must implement domainia.translate.Operator; got {type(op).__name__}")
```
→
```python
        raise TypeError(f"op must implement domainia.translate.Operator, got {type(op).__name__}")
```

The complex-on-time/WDM check in `_stack` (`"channel {name!r}: complex values on a real {kind}"`) stays a `ValueError` (spec §3.5).

- [ ] **Step 8: Run the tests to see them pass, then the full suite and the NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_misc.py tests/test_dom_spectra.py tests/test_translate_coordinates.py tests/test_translate_operator.py`
Expected: `197 passed`, no warnings (52 of them in `test_wrong_kinds_misc.py`).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" src/domainia/spectra.py src/domainia/data.py src/domainia/translate.py`
Expected: six lines, all `ValueError`s: `spectra.py:44` (`values must be (channels, ...)`), `data.py:35`, `:37`, `:40` (`_check`), `translate.py:60`, `:66` (`_record`'s partial-domain refusals).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q`
Expected: all pass, 3 skipped (the `DOMAINIA_ORACLE_ROOT` skips), 0 errors, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q`
Expected: all pass, 0 errors, no warnings summary.

- [ ] **Step 9: Commit cycle A** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/spectra.py src/domainia/data.py src/domainia/translate.py tests/test_wrong_kinds_misc.py tests/test_dom_spectra.py tests/test_translate_coordinates.py tests/test_translate_operator.py
git commit -m "spectra, data, translate: wrong kinds raise TypeError

TabulatedSpectrum checks its domain (and .on's), refuses complex, bool,
str and object values, and reads bins through index_array. GenericPSD
needs a callable. The data classes need a ChannelAxis. from_coordinates
refuses a complex R with TypeError (spec 3.5), and to_coordinates
refuses a sqrt(2) scaling that overflows float64 instead of returning
inf (spec 2.9). Every TypeError message takes the ', got' form.
tests/test_wrong_kinds_misc.py is new.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

#### Cycle B: `linalg.py`, `testing.py`

- [ ] **Step 10: Grep the sites and the tests that pin them**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "int(b)\|int(s) for s in sizes\|float(true_rtol)\|on_fail not in\|cells.band_sizes()\|g = cells.grid\|dtype=float)" src/domainia/linalg.py src/domainia/testing.py`
Expected: twelve hits.
- `linalg.py:109` (`from_sizes`), `:127` (`from_cells`), `:444` (`cblock_factor`), `:524` (`pcg`'s `on_fail`), `:531` (`pcg`'s `true_rtol`): Step 13.
- `testing.py:65` (`dense_W`), `:80` (`eq45_dense`), `:117`, `:119` (`eq54_covariance`): Step 14.
- `testing.py:73` (`wdm_congruence`'s `Sigma`), `:142`, `:150` (`smoothstep`, `wide_profile`): left as they are (see "Left as they are").

`pcg` uses `tol` and `maxit` raw (`5.0 * tol`, `s[6] < maxit`), so the grep cannot show them. `blocktri_factor`'s `layout`, the `band_layout` arguments and the factor kernels' `f` had no check at all.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "pytest.raises" tests/test_dom_linalg_blocktri.py tests/test_dom_linalg_factors.py tests/test_dom_linalg_pcg.py tests/test_dom_linalg_jax.py tests/test_dom_testing.py`
Expected: 13 hits, all a right kind with a wrong value, so all stay: `ValueError` for `positive definite`, `band_layout` (`band_layout=None` on the padded layout), `upper blocks`, `layout` (`layout="dense"`, a string), `whole blocks` (`cblock_factor(np.eye(9, dtype=complex), 4)`), `on_fail` (`on_fail="ignore"`, a string), `(R, N)`, `eagerly only` (twice) and `G + 2 margin`; `RuntimeError` for the three PCG failures. The `PointwiseOperator` refusals of `tests/test_translate_operator.py:535-547` match message prefixes (`"TimeDomain or WDMDomain"`, `"mapping"`, `"strings"`) that still match. The prototype run described in Step 1 (both cycles' source edits, no test edits) found no existing test that cycle B breaks.

- [ ] **Step 11: Add the cycle B tests**

In `tests/test_wrong_kinds_misc.py`, the imports:

```python
from domainia import wilson
```
→
```python
from domainia import linalg as la
from domainia import testing as T
from domainia import wilson
```

and append to the end of the file:

```python
# --- domainia.linalg and domainia.testing -------------------------------------------------

M4 = np.eye(4, dtype=complex)
B2 = np.ones((2, 4))
G, MARGIN = 4, 2
S_I, S_G = np.ones(G), np.ones(G + 2 * MARGIN)
H = np.zeros(2 * (G + MARGIN) + 1, complex)
H[G + MARGIN] = 1.0                                           # lag 0


def _pcg(**kw):
    """PCG on ``A = 2 I`` with ``M = A^-1``: one iteration, exact."""
    args = {"tol": 1e-10, "maxit": 10} | kw
    return la.pcg(lambda X: 2.0 * X, lambda X: 0.5 * X, B2, **args)


INTEGER_SLOTS = {
    "cblock_factor b": ("b", lambda v: la.cblock_factor(M4, v)),
    "pcg maxit": ("maxit", lambda v: _pcg(maxit=v)),
    "BandLayout.from_sizes": ("sizes[0]", lambda v: la.BandLayout.from_sizes((v, 3))),
    "eq54_covariance margin": ("margin", lambda v: T.eq54_covariance(S_I, S_G, H, margin=v)),
    "dense_lambda_generic margin": ("margin", lambda v: T.dense_lambda_generic(np.eye(G), S_I, S_G, H,
                                                                                margin=v)),
}


@pytest.mark.parametrize("slot", INTEGER_SLOTS)
@pytest.mark.parametrize(("value", "kind"), [
    (2.0, "float"), (np.float64(2.0), "float64"), (True, "bool"), ("2", "str"), (None, "NoneType"),
])
def test_integer_slots_refuse_other_kinds(slot, value, kind):
    """``int()`` used to truncate ``2.5`` and take ``True`` as 1; ``pcg`` compared a float
    ``maxit``; ``margin=True`` passed as 1 and ``margin=1.0`` raised ``IndexError``."""
    name, call = INTEGER_SLOTS[slot]
    with pytest.raises(TypeError, match=exactly(f"{name} must be an integer, got {kind}")):
        call(value)


@pytest.mark.parametrize("slot", INTEGER_SLOTS)
def test_integer_slots_take_numpy_integers(slot):
    _, call = INTEGER_SLOTS[slot]
    got, want = call(np.int64(2)), call(2)
    if isinstance(got, np.ndarray):
        assert np.array_equal(got, want)
    elif isinstance(got, la.BandLayout):
        assert (got.nb, got.B, got.n) == (want.nb, want.B, want.n) and np.array_equal(got.back, want.back)
    elif isinstance(got, la.CBlockFactor):
        assert got.b == 2 and type(got.b) is int and np.array_equal(got.Ls, want.Ls)
    else:
        assert np.array_equal(got.x, want.x) and bool(np.all(got.converged))


@pytest.mark.parametrize(("name", "value", "kind"), [
    (name, value, kind) for name in ("tol", "true_rtol")
    for value, kind in ((True, "bool"), ("1e-6", "str"), (1e-6j, "complex"), (np.array(1e-6), "ndarray"))
] + [("tol", None, "NoneType")])
def test_pcg_tolerances_must_be_real_numbers(name, value, kind):
    """``float()`` used to parse ``true_rtol="1e-6"`` and take ``True`` as 1.0; ``tol=True``
    and a 0-d array passed, and a str, complex or ``None`` ``tol`` failed deep inside with
    Python's own message (``true_rtol=None`` is the documented default)."""
    with pytest.raises(TypeError, match=exactly(f"{name} must be a real number, got {kind}")):
        _pcg(**{name: value})


def test_pcg_tolerances_take_numpy_integers_and_floats():
    res = _pcg(tol=np.float64(1e-10), maxit=np.int64(10), true_rtol=np.int64(1))
    assert np.array_equal(res.x, 0.5 * B2) and bool(np.all(res.converged))
    assert np.array_equal(_pcg(tol=np.int64(0)).x, res.x)


STRING_SLOTS = {
    "pcg on_fail": ("on_fail", lambda v: _pcg(on_fail=v)),
    "blocktri_factor layout": ("layout", lambda v: la.blocktri_factor([np.eye(2)], [], layout=v)),
}


@pytest.mark.parametrize("slot", STRING_SLOTS)
@pytest.mark.parametrize(("value", "kind"), [(1, "int"), (None, "NoneType"), (b"nan", "bytes")])
def test_string_options_refuse_other_kinds(slot, value, kind):
    """They used to raise the unknown-choice ``ValueError``, which still answers an unknown
    string."""
    name, call = STRING_SLOTS[slot]
    with pytest.raises(TypeError, match=exactly(f"{name} must be a string, got {kind}")):
        call(value)
    with pytest.raises(ValueError, match=f"{name} must be "):
        call("dense")


def test_the_layouts_and_factors_are_checked():
    """Each used to raise a bare ``AttributeError``."""
    diag, up = [np.eye(2), np.eye(2)], [np.zeros((2, 2))]
    layout = la.BandLayout.from_sizes([2, 2])
    ragged = la.blocktri_factor(diag, up, layout="ragged")
    padded = la.blocktri_factor(diag, up, layout="padded", band_layout=layout)
    b = np.ones(4)
    for cells, kind in ((GRID, "Grid"), (WDM, "WDMDomain")):
        with pytest.raises(TypeError, match=exactly(f"cells must be a CellSet, got {kind}")):
            la.BandLayout.from_cells(cells)
    assert la.BandLayout.from_cells(CELLS).sizes == tuple(s for _, s in CELLS.band_sizes())
    message = exactly("band_layout must be a BandLayout, got object")
    with pytest.raises(TypeError, match=message):
        la.blocktri_factor(diag, up, layout="padded", band_layout=object())
    for fn in (la.blocktri_solve, la.blocktri_whiten):
        with pytest.raises(TypeError, match=message):
            fn(padded, b, band_layout=object())
        with pytest.raises(TypeError, match=exactly("f must be a BlockTriFactor, got CBlockFactor")):
            fn(la.cblock_factor(M4, 2), b)
        np.testing.assert_array_equal(fn(padded, b, band_layout=layout), fn(ragged, b))
    for fn in (la.blocktri_solve_packed, la.blocktri_whiten_packed):
        with pytest.raises(TypeError, match=exactly("f must be a BlockTriFactor, got object")):
            fn(object(), np.ones((2, 2, 1)))
    for fn in (la.cblock_forward, la.cblock_backward):
        with pytest.raises(TypeError, match=exactly("f must be a CBlockFactor, got BlockTriFactor")):
            fn(ragged, b)


@pytest.mark.parametrize(("cells", "kind"), [(GRID, "Grid"), (WDM, "WDMDomain"), (None, "NoneType")])
def test_the_dense_oracles_take_a_cell_set(cells, kind):
    """``dense_W``, ``wdm_congruence`` and ``eq45_dense`` used to raise a bare
    ``AttributeError`` (``cells.grid``)."""
    message = exactly(f"cells must be a CellSet, got {kind}")
    with pytest.raises(TypeError, match=message):
        T.dense_W(cells)
    with pytest.raises(TypeError, match=message):
        T.wdm_congruence(np.eye(128), cells)
    with pytest.raises(TypeError, match=message):
        T.eq45_dense(cells, lambda f, t: 1.0 + 0.0 * f * t)


def test_eq45_dense_needs_a_callable_spectrum():
    with pytest.raises(TypeError, match=exactly("Sn must be callable, got float")):
        T.eq45_dense(CELLS, 1.0)


@pytest.mark.parametrize("name", ["s_I", "s_G_src"])
@pytest.mark.parametrize(("dtype", "shown"), [(complex, "complex128"), (bool, "bool"), (object, "object")])
def test_eq54_spectra_must_be_real_numbers(name, dtype, shown):
    """A complex ``s_I`` or ``s_G_src`` used to lose its imaginary part (only a
    ``ComplexWarning``)."""
    args = {"s_I": S_I, "s_G_src": S_G}
    args[name] = args[name].astype(dtype)
    with pytest.raises(TypeError, match=exactly(f"{name} must be a real numeric array, got dtype {shown}")):
        T.eq54_covariance(args["s_I"], args["s_G_src"], H, margin=MARGIN)


def test_eq54_spectra_take_numpy_integers_and_floats():
    want = T.eq54_covariance(S_I, S_G, H, margin=MARGIN)
    for dtype in (np.int64, np.float64):
        got = T.eq54_covariance(S_I.astype(dtype), S_G.astype(dtype), H, margin=MARGIN)
        assert np.array_equal(got, want)


def test_the_pointwise_test_operator_messages_say_got():
    message = "PointwiseOperator needs a TimeDomain or WDMDomain, got Grid"
    with pytest.raises(TypeError, match=exactly(message)):
        T.PointwiseOperator(GRID, {"A": np.ones(128)})
    with pytest.raises(TypeError, match=exactly("variances must be a {channel: array} mapping, got list")):
        T.PointwiseOperator(TIME, [np.ones(128)])
    with pytest.raises(TypeError, match=exactly("channel names must be strings, got 1")):
        T.PointwiseOperator(TIME, {1: np.ones(128)})
    with pytest.raises(TypeError, match=exactly("values must be a mapping, got list")):
        T.PointwiseOperator(TIME, {"A": np.ones(128)}).apply([np.ones(128)])
```

- [ ] **Step 12: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_misc.py`
Expected: `52 failed, 59 passed, 2 warnings` (the warnings are the `ComplexWarning`s of `testing.py:117` and `:119`, from the complex `s_I` and `s_G_src` cases). The 52 failures are all cycle B tests:
- 22 `DID NOT RAISE <class 'TypeError'>`: `2.0`, `np.float64(2.0)` and `True` for `b`, `maxit` and `sizes[0]`, and `"2"` for `b` and `sizes[0]` (`int()` parsed it); `True` and a 0-d array for both tolerances, and `true_rtol="1e-6"`; every complex, bool and object `s_I`/`s_G_src` (the complex ones with only the warning).
- 6 the unknown-choice `ValueError` (`on_fail must be 'raise' or 'nan'; got 1`, `layout must be 'ragged' or 'padded'; got None`, ...).
- 2 `ValueError: s_G_src needs G + 2 margin = 6 entries; got 8` (`margin=True` read as 1) and 4 `IndexError: arrays used as indices must be of integer (or boolean) type` (`margin=2.0` and `np.float64(2.0)`).
- 4 bare `AttributeError`s: `'Grid' object has no attribute 'band_sizes'` (`from_cells`, which stops `test_the_layouts_and_factors_are_checked`), and `... has no attribute 'grid'` for the three `cells` of `test_the_dense_oracles_take_a_cell_set`.
- 14 `TypeError`s with Python's own or the old text, which `match=` refuses: `int() argument must be ... not 'NoneType'`, `can't multiply sequence by non-int of type 'float'`, `'float' object is not callable`, `PointwiseOperator needs a TimeDomain or WDMDomain; got Grid`, ...

The 7 cycle B tests that pass are the NumPy-type pins.

- [ ] **Step 13: `src/domainia/linalg.py`**

The imports:

```python
from domainia.wilson.backend import Backend, get_backend, is_tracer, jax_backend, pytree_dataclass
```
→
```python
from domainia.wilson._checks import require_instance, require_int, require_real, require_str
from domainia.wilson.backend import Backend, get_backend, is_tracer, jax_backend, pytree_dataclass
```

`BandLayout.from_sizes`, the integer rule on each element (spec §2.3; an integer ndarray still passes, its elements are NumPy integers):

```python
        """The layout of bands with ``sizes[m]`` cells each (concatenated in order)."""
        sizes = [int(s) for s in sizes]
```
→
```python
        """The layout of bands with ``sizes[m]`` cells each (concatenated in order).  Each
        size must be an integer (``TypeError`` otherwise, ``2.0`` and ``True`` included)."""
        sizes = [require_int(f"sizes[{m}]", s) for m, s in enumerate(sizes)]
```

`BandLayout.from_cells`:

```python
        """One slot row per band present in a band-ordered :class:`~domainia.domains.CellSet`."""
        return cls.from_sizes([s for _, s in cells.band_sizes()])
```
→
```python
        """One slot row per band present in a band-ordered :class:`~domainia.domains.CellSet`."""
        require_instance("cells", cells, CellSet, "a CellSet")
        return cls.from_sizes([s for _, s in cells.band_sizes()])
```

`blocktri_factor`, the end of its docstring and its first line:

```python
    (in the order of :meth:`BandLayout.add_identity_dummies`).
    """
    if layout == "ragged":
```
→
```python
    (in the order of :meth:`BandLayout.add_identity_dummies`).

    A ``layout`` that is not a string, or a ``band_layout`` that is not a
    :class:`BandLayout`, raises ``TypeError``; an unknown ``layout`` raises ``ValueError``.
    """
    require_str("layout", layout)
    if layout == "ragged":
```

and its padded branch (`band_layout=None` keeps its `ValueError`):

```python
        if band_layout is None:
            raise ValueError("layout='padded' needs band_layout")
        if isinstance(diag, (list, tuple)):
```
→
```python
        if band_layout is None:
            raise ValueError("layout='padded' needs band_layout")
        require_instance("band_layout", band_layout, BandLayout, "a BandLayout")
        if isinstance(diag, (list, tuple)):
```

`_layout_of` (`blocktri_whiten` and `blocktri_solve` call it for the padded layout):

```python
    lay = band_layout if band_layout is not None else f.band_layout
    if lay is None:
        raise ValueError("a padded factor without its band_layout: pass band_layout=")
    return lay
```
→
```python
    lay = band_layout if band_layout is not None else f.band_layout
    if lay is None:
        raise ValueError("a padded factor without its band_layout: pass band_layout=")
    return require_instance("band_layout", lay, BandLayout, "a BandLayout")
```

`blocktri_whiten`, the end of its docstring and its first line:

```python
    the factor's own (``None`` when a covariance stored it without).
    """
    if f.layout == "ragged":
```
→
```python
    the factor's own (``None`` when a covariance stored it without).
    """
    require_instance("f", f, BlockTriFactor, "a BlockTriFactor")
    if f.layout == "ragged":
```

`blocktri_solve`, likewise:

```python
    as :func:`blocktri_whiten`.
    """
    if f.layout == "ragged":
```
→
```python
    as :func:`blocktri_whiten`.
    """
    require_instance("f", f, BlockTriFactor, "a BlockTriFactor")
    if f.layout == "ragged":
```

`blocktri_whiten_packed`:

```python
    the forward-only quadratic form is ``sum(z * z)``."""
    if f.layout != "padded":
```
→
```python
    the forward-only quadratic form is ``sum(z * z)``."""
    require_instance("f", f, BlockTriFactor, "a BlockTriFactor")
    if f.layout != "padded":
```

`blocktri_solve_packed`:

```python
    :func:`blocktri_whiten_packed`, then a reverse ``be.scan`` of the back substitution."""
    if f.layout != "padded":
```
→
```python
    :func:`blocktri_whiten_packed`, then a reverse ``be.scan`` of the back substitution."""
    require_instance("f", f, BlockTriFactor, "a BlockTriFactor")
    if f.layout != "padded":
```

Under `jax.jit` a factor or layout passed as an argument is rebuilt as the same class with traced leaves, so these `isinstance` checks hold there (`tests/test_dom_linalg_jax.py` covers it).

`cblock_factor`, from the end of its docstring to `n = M.shape[0]` (the integer check moves before the backend is chosen; `CBlockFactor.b` is static meta and stays a Python `int`):

```python
    the blocks outside the tridiagonal are not read."""
    be = _be(be, M)
    xp = be.xp
    M = be.asarray(M)
    b = int(b)
    n = M.shape[0]
```
→
```python
    the blocks outside the tridiagonal are not read.  ``b`` must be an integer
    (``TypeError`` otherwise)."""
    b = require_int("b", b)
    be = _be(be, M)
    xp = be.xp
    M = be.asarray(M)
    n = M.shape[0]
```

`cblock_forward`:

```python
    """``L^-1 h``, by block forward substitution."""
    be = _be(be, f.Ls, h)
```
→
```python
    """``L^-1 h``, by block forward substitution."""
    require_instance("f", f, CBlockFactor, "a CBlockFactor")
    be = _be(be, f.Ls, h)
```

`cblock_backward`:

```python
    """``L^-H u``, by block back substitution."""
    be = _be(be, f.Ls, u)
```
→
```python
    """``L^-H u``, by block back substitution."""
    require_instance("f", f, CBlockFactor, "a CBlockFactor")
    be = _be(be, f.Ls, u)
```

`pcg`, from the end of its docstring to the `rtol` line. Every kind is refused before `B` is touched. `tol` becomes a Python `float`, which changes nothing for the callers (all pass Python floats), and keeps `rn <= tol * bn` bit for bit:

```python
    non-positive or non-finite curvature ``p.Ap`` fails (the early-exit loop had no such
    check).
    """
    if on_fail not in ("raise", "nan"):
        raise ValueError(f"on_fail must be 'raise' or 'nan'; got {on_fail!r}")
    be = _be(be, B)
    xp = be.xp
    B = be.asarray(B)
    if B.ndim != 2:
        raise ValueError(f"B must be (R, N); got shape {B.shape}")
    rtol = max(5.0 * tol, 1e-12) if true_rtol is None else float(true_rtol)
```
→
```python
    non-positive or non-finite curvature ``p.Ap`` fails (the early-exit loop had no such
    check).

    A wrong kind raises ``TypeError``: ``tol`` and ``true_rtol`` must be real numbers
    (``true_rtol`` may be ``None``), ``maxit`` an integer and ``on_fail`` a string (an
    unknown one raises ``ValueError``).
    """
    require_str("on_fail", on_fail)
    if on_fail not in ("raise", "nan"):
        raise ValueError(f"on_fail must be 'raise' or 'nan'; got {on_fail!r}")
    tol = require_real("tol", tol)
    maxit = require_int("maxit", maxit)
    true_rtol = None if true_rtol is None else require_real("true_rtol", true_rtol)
    be = _be(be, B)
    xp = be.xp
    B = be.asarray(B)
    if B.ndim != 2:
        raise ValueError(f"B must be (R, N); got shape {B.shape}")
    rtol = max(5.0 * tol, 1e-12) if true_rtol is None else true_rtol
```

- [ ] **Step 14: `src/domainia/testing.py`**

The imports:

```python
from domainia import wilson
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
```
→
```python
from domainia import wilson
from domainia.wilson._checks import (REAL_KINDS, require_callable, require_dtype, require_instance,
                                     require_int)
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
```

The module docstring, after its paragraph on `window=`:

```python
:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.
```
→
```python
:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.

The oracles follow the package's rule: a wrong kind (``cells`` that is not a ``CellSet``, a
float ``margin``, a complex ``s_I``) raises ``TypeError``, a wrong value ``ValueError``.
```

`dense_W` (`wdm_congruence` calls it first, so it refuses a wrong `cells` the same way):

```python
    Wilson basis of ``window``."""
    g = cells.grid
    return wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=window)[cells.packed_positions()]
```
→
```python
    Wilson basis of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    return wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=window)[cells.packed_positions()]
```

`eq45_dense`:

```python
    with the atoms of ``window``."""
    g = cells.grid
    plan = eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max, window=window)
```
→
```python
    with the atoms of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    require_callable("Sn", Sn)
    plan = eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max, window=window)
```

`eq54_covariance`, from the end of its docstring to the first cast (`dense_lambda_generic` passes its `margin`, `s_I` and `s_G_src` straight to it). `eq54_covariance`'s docstring is a raw string, so `\hat h` has one backslash in the file:

```python
    ``s_G_src`` has ``G + 2 margin`` entries; ``h`` is :math:`\hat h` at the lags
    ``-(G + margin) .. G + margin``.
    """
    s_I = np.asarray(s_I, dtype=float)
```
→
```python
    ``s_G_src`` has ``G + 2 margin`` entries; ``h`` is :math:`\hat h` at the lags
    ``-(G + margin) .. G + margin``.  ``margin`` must be an integer, and ``s_I`` and
    ``s_G_src`` real numeric arrays (``TypeError`` otherwise).
    """
    margin = require_int("margin", margin)
    require_dtype("s_I", s_I, REAL_KINDS, "a real numeric array")
    require_dtype("s_G_src", s_G_src, REAL_KINDS, "a real numeric array")
    s_I = np.asarray(s_I, dtype=float)
```

`PointwiseOperator.__post_init__`, two messages:

```python
            raise TypeError(f"PointwiseOperator needs a TimeDomain or WDMDomain; got "
                            f"{type(self.domain).__name__}")
        if not isinstance(self.variances, Mapping):
            raise TypeError(f"variances must be a {{channel: array}} mapping; got "
                            f"{type(self.variances).__name__}")
```
→
```python
            raise TypeError(f"PointwiseOperator needs a TimeDomain or WDMDomain, got "
                            f"{type(self.domain).__name__}")
        if not isinstance(self.variances, Mapping):
            raise TypeError(f"variances must be a {{channel: array}} mapping, got "
                            f"{type(self.variances).__name__}")
```

and the channel-name message:

```python
                raise TypeError(f"channel names must be strings; got {name!r}")
```
→
```python
                raise TypeError(f"channel names must be strings, got {name!r}")
```

`PointwiseOperator._values`:

```python
            raise TypeError(f"values must be a mapping; got {type(values).__name__}")
```
→
```python
            raise TypeError(f"values must be a mapping, got {type(values).__name__}")
```

- [ ] **Step 15: Run the tests to see them pass, and the `"; got"` check**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_misc.py tests/test_dom_linalg_blocktri.py tests/test_dom_linalg_factors.py tests/test_dom_linalg_pcg.py tests/test_dom_linalg_jax.py tests/test_dom_testing.py tests/test_dom_restored_coverage.py tests/test_translate_operator.py`
Expected: `252 passed`, no warnings (111 of them in `test_wrong_kinds_misc.py`).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" src/domainia/spectra.py src/domainia/data.py src/domainia/translate.py src/domainia/variance.py src/domainia/linalg.py src/domainia/testing.py`
Expected: 13 lines, every one inside a `raise ValueError(...)`:
- `spectra.py:44`: `raise ValueError(f"values must be (channels, {d.n}); got {v.shape}")`
- `data.py:35`: `raise ValueError(f"{what}: values need at least {need} axes; got shape {shape}")`
- `data.py:37`: `raise ValueError(f"{what}: last axis must be the domain size {size}; got shape {shape}")`
- `data.py:40`: `f"{channels.names}; got shape {shape}")`
- `translate.py:60`: `f"bin, dt_rfft units); got {domain!r}")`
- `translate.py:66`: `f"active cell); got {domain.cells.size} of {g.N} cells")`
- `linalg.py:113`: `raise ValueError(f"sizes must be non-negative with at least one band of cells; got {sizes}")`
- `linalg.py:271`: `raise ValueError(f"need {len(diag) - 1} upper blocks; got {len(up)}")`
- `linalg.py:288`: `raise ValueError(f"layout must be 'ragged' or 'padded'; got {layout!r}")`
- `linalg.py:545`: `raise ValueError(f"on_fail must be 'raise' or 'nan'; got {on_fail!r}")`
- `linalg.py:553`: `raise ValueError(f"B must be (R, N); got shape {B.shape}")`
- `testing.py:131`: `raise ValueError(f"s_G_src needs G + 2 margin = {G + 2 * margin} entries; got {s_G_src.size}")`
- `testing.py:134`: `raise ValueError(f"h needs the {2 * (G + margin) + 1} lags -(G+margin)..G+margin; got {h.size}")`

- [ ] **Step 16: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q`
Expected: all pass, 3 skipped (the `DOMAINIA_ORACLE_ROOT` skips), 0 errors, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q`
Expected: all pass, 0 errors, no warnings summary.

- [ ] **Step 17: Commit cycle B** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/linalg.py src/domainia/testing.py tests/test_wrong_kinds_misc.py
git commit -m "linalg, testing: wrong kinds raise TypeError

cblock_factor's b, pcg's maxit, BandLayout.from_sizes' sizes and
eq54_covariance's margin take integers only (int() truncated, True
passed as 1, margin=1.0 raised IndexError). pcg's tol and true_rtol
take real numbers, its on_fail and blocktri_factor's layout strings.
The layouts, factors and cell sets the kernels and dense oracles take
are checked with isinstance, eq45_dense's Sn must be callable, and
eq54_covariance refuses complex s_I and s_G_src. PointwiseOperator's
kind messages take the ', got' form.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The shape of b (spec §2.8)

**Goal:** `Covariance.multiply` and every factor's `solve`, `whiten` and `quad` compare the leading axes of `b` with the shape they expect, and raise `ValueError` on a mismatch. Three silent wrong answers go: a `(1,)` `b` broadcast by `DiagonalCovariance`, an `(n+1,)` `b` cut to its first `n` entries by `BlockTridiagonalCovariance`, and a third row dropped by a two-channel `ChannelStack`. They go on the multiply path and on the likelihood path. `(n, k)` column batches keep working.

**Assumes the tree after Task 6.** Task 5 edited the six files of `src/domainia/covariance/` (the imports and the kind checks), and the old texts below are its result. Task 6 does not touch `src/domainia/covariance/` or `tests/test_dom_cov_multiply.py`, and no earlier task edits that test file. Task 6 adds kind checks at the top of `linalg.blocktri_solve` and `blocktri_whiten` (their `f` and `band_layout`); this task calls them as they are. The line numbers below are those after Task 5 (= after Task 6).

**Files:**
- Modify: `src/domainia/covariance/base.py` (the module docstring; new `_check_leading`; `DiagFactor`: new `_b_shape`, `solve`, `whiten`, `quad`; `CholFactor`: new `_b_shape`, `solve`, `whiten`; the `Covariance` docstring, new `Covariance._b_shape`, `Covariance.multiply`)
- Modify: `src/domainia/covariance/blocktri.py` (the `.base` import; new `BlockTriCovFactor._b_shape`, `BlockTriCovFactor._apply`)
- Modify: `src/domainia/covariance/stack.py` (the imports; `StackFactor`: new `_b_shape` and `_rows`, `quad`, `solve`, `whiten`; new `ChannelStack._b_shape`)
- Modify: `tests/test_dom_cov_multiply.py` (the docstring, the imports, new tests at the end)
- Not changed: `diagonal.py`, `dense.py` and `evolutionary.py`. They inherit `Covariance._b_shape` (`(n,)`), and their factors are the classes above. `CholFactor.quad` and `BlockTriCovFactor.quad` reach the check through `quad_form`, so they are not edited either.

**Acceptance Criteria:**
- [ ] `covariance.base._check_leading(b, expected, what)` raises `ValueError(f"{what}: b must have leading shape {expected}, got shape {b.shape}")` unless `b.shape[:len(expected)] == expected`. It is private: `__all__` and the `Factor` protocol are unchanged.
- [ ] `Covariance._b_shape()` is `(n,)`. `ChannelStack._b_shape()` is `(len(covs), n_c)` when its channels share one leading shape `(n_c,)`. Otherwise it raises `ValueError`, for example `"ChannelStack: b has one row per channel, so the channels must share one size; their row shapes are (128,), (5,)"`. Construction and `factor` do not call it: a mixed-size stack still builds and factorises, and `tests/test_operators_covariance.py::test_the_arguments_are_validated` still gets `CovarianceOperator`'s own message.
- [ ] `Covariance.multiply` reads `b` with `be.asarray`, then checks it against `_b_shape()` with `what = "<class name>.multiply"`, before it calls `_multiply`.
- [ ] Each factor derives its own leading shape: `DiagFactor` `(v.shape[0],)`, `CholFactor` `(L.shape[0],)`, `BlockTriCovFactor` the sum of its ragged blocks' sizes or the padded constants' `layout.n` (`_b_shape(C)`), `StackFactor` `(len(channels),)`, with each row `b[c]` then checked by its channel's factor.
- [ ] Every factor's `solve` and `whiten` reads `b` with `be.asarray` (so a list is accepted, as `multiply` and `CholFactor` already did) and checks it. `DiagFactor.quad`'s `"elementwise"` form and `StackFactor.quad` check in `quad`. The `"solve"` and `"whiten"` forms are checked in `solve` and `whiten`, which `quad_form` calls. A `form` of the wrong kind still raises `TypeError` before the shape check.
- [ ] A ragged `BlockTriCovFactor` whose factorisation failed still raises `"the factorisation failed ..."`, not a shape error.
- [ ] `(n, k)` batches, and `(channels, n_c, k)` for a `ChannelStack`, still work in `multiply`, `solve` and `whiten`. The checks also run on JAX, eagerly and under `jax.jit`.
- [ ] `tests/test_dom_cov_multiply.py`: `30 failed, 26 passed` before the source change and `56 passed` after (NumPy lane: `29 failed, 26 passed, 1 skipped`, then `55 passed, 1 skipped`).
- [ ] The full suite and the NumPy lane pass.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_cov_multiply.py` → `56 passed`; then the full suite and the NumPy lane → all pass

**Steps:**

- [ ] **Step 1: Probe the silent modes and grep the sites**

Write this probe to `$TMPDIR/probe_b_shape.py` (outside the repo):

```python
"""Probe the shape of b (spec 2026-11-08 §2.8): one line per call, the result's shape or the error."""
import numpy as np

from domainia.covariance import BlockTridiagonalCovariance, ChannelStack, DiagonalCovariance

rng = np.random.default_rng(0)
V = 0.5 + rng.random(6)
SIZES = (3, 4, 0, 2, 5)
D = [np.eye(s) * (2.0 + m) for m, s in enumerate(SIZES)]
U = [np.zeros((SIZES[m], SIZES[m + 1])) for m in range(len(SIZES) - 1)]


def blocks(nu, be):
    return [be.asarray(d) for d in D], [be.asarray(u) for u in U]


MODES = [
    ("diagonal (1,)", DiagonalCovariance(V, 6), np.zeros(0), np.ones(1), ("solve", "whiten", "elementwise")),
    ("blocktri-ragged (n+1,)", BlockTridiagonalCovariance(blocks, SIZES), np.zeros(0), np.ones(15),
     ("solve", "whiten")),
    ("blocktri-padded (n+1,)", BlockTridiagonalCovariance(blocks, SIZES, layout="padded"), np.zeros(0),
     np.ones(15), ("solve", "whiten")),
    ("stack (3, n)", ChannelStack([DiagonalCovariance(V, 6), DiagonalCovariance(2 * V, 6)], 0), np.zeros(0),
     np.ones((3, 6)), ("solve", "whiten", "elementwise")),
]


def show(label, fn):
    try:
        print(f"{label}: ok, shape {np.shape(fn())}")
    except Exception as e:  # noqa: BLE001
        print(f"{label}: {type(e).__name__}: {str(e)[:96]}")


for name, cov, nu, b, forms in MODES:
    F = cov.factor(nu)
    show(f"{name} multiply", lambda: cov.multiply(nu, b))
    show(f"{name} solve", lambda: F.solve(b))
    show(f"{name} whiten", lambda: F.whiten(b))
    for form in forms:
        show(f"{name} quad {form}", lambda: F.quad(b, form=form))
mixed = ChannelStack([DiagonalCovariance(V, 6), DiagonalCovariance(V[:5], 5)], 0)
show("mixed stack multiply (2, 6)", lambda: mixed.multiply(np.zeros(0), np.ones((2, 6))))
show("mixed stack quad [(6,), (5,)]", lambda: mixed.factor(np.zeros(0)).quad([np.ones(6), np.ones(5)]))
show("diagonal solve list", lambda: DiagonalCovariance(V, 6).factor(np.zeros(0)).solve([1.0] * 6))
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=src ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/probe_b_shape.py`
Expected:

```
diagonal (1,) multiply: ok, shape (6,)
diagonal (1,) solve: ok, shape (6,)
diagonal (1,) whiten: ok, shape (6,)
diagonal (1,) quad solve: ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,
diagonal (1,) quad whiten: ok, shape ()
diagonal (1,) quad elementwise: ok, shape ()
blocktri-ragged (n+1,) multiply: ok, shape (14,)
blocktri-ragged (n+1,) solve: ok, shape (14,)
blocktri-ragged (n+1,) whiten: ok, shape (14,)
blocktri-ragged (n+1,) quad solve: ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,
blocktri-ragged (n+1,) quad whiten: ok, shape ()
blocktri-padded (n+1,) multiply: ok, shape (14,)
blocktri-padded (n+1,) solve: ok, shape (14,)
blocktri-padded (n+1,) whiten: ok, shape (14,)
blocktri-padded (n+1,) quad solve: ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,
blocktri-padded (n+1,) quad whiten: ok, shape ()
stack (3, n) multiply: ok, shape (2, 6)
stack (3, n) solve: ok, shape (2, 6)
stack (3, n) whiten: ok, shape (2, 6)
stack (3, n) quad solve: ok, shape ()
stack (3, n) quad whiten: ok, shape ()
stack (3, n) quad elementwise: ok, shape ()
mixed stack multiply (2, 6): ValueError: operands could not be broadcast together with shapes (5,) (6,) 
mixed stack quad [(6,), (5,)]: ok, shape ()
diagonal solve list: AttributeError: 'list' object has no attribute 'ndim'
```

What it shows:
- The three modes of spec §2.8 are silent today on every call that prints `ok`: `multiply`, `solve`, `whiten` and the `"whiten"` (and, on the diagonal, `"elementwise"`) form of `quad`, on the diagonal, on both block layouts, and on the stack (all six calls there).
- `quad(form="solve")` on the diagonal and the two block layouts is not silent, but its error is NumPy's: `b @ solve(b)` meets the two lengths in `matmul`. After this task the check raises first.
- A mixed-size stack's `multiply` fails today only because NumPy cannot broadcast `(5,)` with `(6,)`. Its factor's `quad` on a list of rows of different lengths works today, channel by channel. Spec §2.8 refuses a mixed-size stack in `StackFactor`'s methods, so after this task that call raises `ValueError` too (NumPy's, from `np.asarray` of the ragged list).
- `diagonal solve list`: `DiagFactor.solve` reads `b.ndim` of the raw argument, and so does `linalg.blocktri_solve` for `BlockTriCovFactor`, while `CholFactor.solve` and `multiply` accept a list. The check needs `b.shape`, so every factor method now reads `b` with `be.asarray` first, as `multiply` already does.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "\.ndim\|def solve\|def whiten\|def quad\|def multiply\|def _multiply" src/domainia/covariance/*.py`
Expected: 25 hits, and what to do with each:
- `base.py:82`, `:84`, `:86`: the `Factor` protocol's stubs. Unchanged: the protocol does not gain `_b_shape`.
- `base.py:165`: `quad_form`. Unchanged: its `"solve"` and `"whiten"` forms reach the check through the factor's `solve` and `whiten`.
- `base.py:215`, `:216`, `:218`, `:221`, `:223`: `DiagFactor.solve`, `whiten` and `quad` (Step 4).
- `base.py:258`, `:265`: `CholFactor.solve` and `whiten` (Step 4). `base.py:269`: `CholFactor.quad`, through `quad_form`, unchanged.
- `base.py:342`: `Covariance.multiply` (Step 4).
- `base.py:334`, `diagonal.py:67`, `blocktri.py:171`, `stack.py:116`, `evolutionary.py:188`: the `_multiply` hooks, and `diagonal.py:69` (`b.ndim` inside one). Unchanged: `multiply` checks and converts `b` before it calls them.
- `blocktri.py:115`, `:118`: `BlockTriCovFactor.solve` and `whiten`, which both go through `_apply` (Step 5). `blocktri.py:121`: `quad`, through `quad_form`, unchanged.
- `stack.py:40`, `:47`, `:52`: `StackFactor.quad`, `solve` and `whiten` (Step 6).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "_b_shape\|_check_leading\|leading shape" src tests`
Expected: no output. The names and the message are new.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "\.multiply(\|_factor\.solve(\|_factor\.quad(" src/domainia/operators.py`
Expected: three hits, `operators.py:267`, `:270`, `:276` (`CovarianceOperator.apply`, `solve`, `quadratic_form`). Unchanged: `_b` gives `(K s,)` for a single covariance and `(K, s)` for a `ChannelStack`, after `__post_init__` has checked that `cov.n == K s`, or that every channel has `n == s`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "ChannelStack(" tests --include='*.py' | grep -v "tests/wilson/"`
Expected: 22 hits. The only stack whose channels differ in size is `tests/test_operators_covariance.py:232` (`[ev, EvolutionaryCovariance(PSD, CUT)]`). `CovarianceOperator` refuses it with its own message before any `b` is formed, so it stays as it is. Step 7 runs it.

A test that passes a mis-shaped `b` on purpose would not show in a grep. No existing test is known to rely on the old broadcasting; Step 9 confirms it.

- [ ] **Step 2: Write the failing tests in `tests/test_dom_cov_multiply.py`**

The module docstring and the first imports:

```python
"""Covariance.multiply: ``Sigma(nu) b`` in the layout of the factor's ``solve``, for every
covariance class, and ChannelStack's dense form."""

from __future__ import annotations

import numpy as np
```
→
```python
"""Covariance.multiply: ``Sigma(nu) b`` in the layout of the factor's ``solve``, for every
covariance class, and ChannelStack's dense form.  ``multiply`` and the factors' ``solve``,
``whiten`` and ``quad`` check the leading shape of ``b`` (spec 2026-11-08 §2.8)."""

from __future__ import annotations

import re

import numpy as np
```

The import of `domainia.covariance`:

```python
    EvolutionaryCovariance,
)
```
→
```python
    EvolutionaryCovariance,
    StackFactor,
)
```

At the end of the file (the last lines of `test_a_covariance_with_neither_multiply_nor_a_dense_form_raises_type_error` are shown, kept as they are):

```python
    with pytest.raises(TypeError, match=r"^_FactorOnly has no multiply: it implements neither "
                                        r"_multiply nor _matrix"):
        _FactorOnly().multiply(np.zeros(1), np.ones(3))
```
→
```python
    with pytest.raises(TypeError, match=r"^_FactorOnly has no multiply: it implements neither "
                                        r"_multiply nor _matrix"):
        _FactorOnly().multiply(np.zeros(1), np.ones(3))


# --- the leading shape of b (spec 2026-11-08 §2.8) ------------------------------------------

N = CELLS.size
NU2 = np.array([0.3, 0.0, -1.1, 0.6])


def _two_channels():
    return ChannelStack([COVS["diagonal-callable"](), COVS["diagonal-array"]()], 2)


#: The three silent modes: ``(cov, nu, b, multiply's leading shape, the factor's)``.  Before
#: §2.8 a ``DiagonalCovariance`` broadcast a ``(1,)`` ``b``, a ``BlockTridiagonalCovariance``
#: cut an ``(n + 1,)`` one to its first ``n`` entries (both layouts), and a two-channel
#: ``ChannelStack`` dropped a third row, all without an error.
SILENT = {
    "diagonal-(1,)": (COVS["diagonal-array"], NU, np.ones(1), (N,), (N,)),
    "blocktri-ragged-(n+1,)": (COVS["blocktri-ragged"], NU, np.ones(15), (14,), (14,)),
    "blocktri-padded-(n+1,)": (COVS["blocktri-padded"], NU, np.ones(15), (14,), (14,)),
    "stack-(3,n)": (_two_channels, NU2, np.ones((3, N)), (2, N), (2,)),
}
#: The factor calls, and the method whose check fires: the ``"solve"`` and ``"whiten"``
#: forms of ``quad`` reach the check through ``solve`` and ``whiten``; ``DiagFactor``'s
#: ``"elementwise"`` form and ``StackFactor.quad`` check in ``quad``.
CALLS = {
    "solve": (lambda F, b: F.solve(b), "solve"),
    "whiten": (lambda F, b: F.whiten(b), "whiten"),
    "quad-solve": (lambda F, b: F.quad(b, form="solve"), "solve"),
    "quad-whiten": (lambda F, b: F.quad(b, form="whiten"), "whiten"),
    "quad-elementwise": (lambda F, b: F.quad(b, form="elementwise"), "quad"),
}
FACTOR_CASES = [(mode, call) for mode in SILENT for call in CALLS
                if call != "quad-elementwise" or not mode.startswith("blocktri")]


def _refusal(what, expected, b, end="$"):
    """The check's message.  Under ``jax.jit`` JAX appends a traceback note, so ``end=""``."""
    return (rf"^{re.escape(what)}: b must have leading shape {re.escape(str(expected))}, "
            rf"got shape {re.escape(str(b.shape))}{end}")


@pytest.mark.parametrize("mode", SILENT)
def test_multiply_refuses_b_of_the_wrong_leading_shape(mode):
    make, nu, b, want, _ = SILENT[mode]
    cov = make()
    with pytest.raises(ValueError, match=_refusal(f"{type(cov).__name__}.multiply", want, b)):
        cov.multiply(nu, b)


@pytest.mark.parametrize(("mode", "call"), FACTOR_CASES)
def test_the_factor_refuses_b_of_the_wrong_leading_shape(mode, call):
    make, nu, b, _, want = SILENT[mode]
    F = make().factor(nu)
    fn, method = CALLS[call]
    if isinstance(F, StackFactor) and call.startswith("quad"):
        method = "quad"                                  # StackFactor.quad checks before its channels
    with pytest.raises(ValueError, match=_refusal(f"{type(F).__name__}.{method}", want, b)):
        fn(F, b)


def test_a_channel_stack_whose_channels_differ_in_size_is_refused_when_used():
    """It builds and factorises (``CovarianceOperator`` reports its own size mismatch), but
    ``b`` has one row per channel, so ``multiply`` refuses it, and so do the factor's methods
    (the channel whose size differs from the row's)."""
    st = ChannelStack([DiagonalCovariance(_V, N), DiagonalCovariance(_V[:5], 5)], 0)
    F = st.factor(np.zeros(0))
    assert st.n == N + 5 and bool(F.ok)
    for b in (np.ones((2, N)), np.ones((2, 5))):
        with pytest.raises(ValueError, match=r"^ChannelStack: b has one row per channel, so the channels "
                                             r"must share one size; their row shapes are \(128,\), \(5,\)$"):
            st.multiply(np.zeros(0), b)
    b = np.ones((2, N))
    for fn, method in CALLS.values():
        with pytest.raises(ValueError, match=_refusal(f"DiagFactor.{method}", (5,), b[1])):
            fn(F, b)


@pytest.mark.parametrize("name", COVS)
def test_an_n_by_k_batch_of_columns_still_works(name):
    """Only the leading axes are compared: ``(n, k)`` is ``k`` columns, each as ``(n,)``."""
    cov = COVS[name]()
    B = _rng.standard_normal((cov.n, 3))
    got = cov.multiply(NU, B)
    want = cov.dense(NU) @ B
    assert got.shape == (cov.n, 3)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    F = cov.factor(NU)
    for fn in (F.solve, F.whiten):
        out = fn(B)
        assert out.shape == (cov.n, 3)
        for j in range(3):
            np.testing.assert_allclose(out[:, j], fn(B[:, j]), rtol=0, atol=1e-13 * np.max(np.abs(out)))


def test_a_channel_stack_takes_a_batch_per_channel():
    st = _two_channels()
    B = _rng.standard_normal((2, N, 3))
    got = st.multiply(NU2, B)
    assert got.shape == (2, N, 3)
    np.testing.assert_allclose(got[:, :, 1], st.multiply(NU2, B[:, :, 1]), rtol=0,
                               atol=1e-15 * np.max(np.abs(got)))
    F = st.factor(NU2)
    np.testing.assert_allclose(F.solve(got), B, rtol=0, atol=1e-13 * np.max(np.abs(B)))
    assert F.whiten(B).shape == (2, N, 3)


@pytest.mark.parametrize("name", COVS)
def test_a_list_b_is_read_as_an_array(name):
    """``multiply`` and every factor method read ``b`` with ``be.asarray`` before the check
    (``DiagFactor`` and ``BlockTriCovFactor`` used to fail on ``b.ndim``)."""
    cov = COVS[name]()
    b = _rng.standard_normal(cov.n)
    F = cov.factor(NU)
    for fn in (lambda x: cov.multiply(NU, x), F.solve, F.whiten):
        want = fn(b)
        np.testing.assert_allclose(fn(list(b)), want, rtol=0, atol=1e-14 * np.max(np.abs(want)))
    assert float(F.quad(list(b))) == pytest.approx(float(F.quad(b)), rel=1e-14)


def test_a_failed_ragged_factor_still_reports_the_failure():
    """A ragged factor that failed stops at its first bad block, so its blocks do not add up
    to ``n``: its methods raise the failure, not a shape error."""
    cov = BlockTridiagonalCovariance(lambda nu, be: ([-be.asarray(D) for D in _DIAG],
                                                     [be.asarray(U) for U in _UP]), SIZES)
    F = cov.factor(NU)
    assert not bool(F.ok)
    for call in ("solve", "whiten", "quad-solve", "quad-whiten"):
        with pytest.raises(ValueError, match="^the factorisation failed"):
            CALLS[call][0](F, np.ones(cov.n))


@pytest.mark.jax
def test_the_check_runs_on_jax_and_under_jit(jax_be):
    """``multiply`` on JAX checks as on NumPy.  Shapes are static under ``jax.jit``, so the
    factors' checks run while tracing."""
    import jax
    import jax.numpy as jnp

    diag = COVS["diagonal-array"]()
    nu = jnp.asarray(NU)
    np.testing.assert_allclose(np.asarray(diag.multiply(nu, jnp.ones(N))), _V, rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match=_refusal("DiagonalCovariance.multiply", (N,), np.ones(1))):
        diag.multiply(nu, jnp.ones(1))
    F = diag.factor(nu)
    solve = jax.jit(lambda F, b: F.solve(b))
    np.testing.assert_allclose(np.asarray(solve(F, jnp.ones(N))), 1 / _V, rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match=_refusal("DiagFactor.solve", (N,), np.ones(1), end="")):
        solve(F, jnp.ones(1))
    with pytest.raises(ValueError, match=_refusal("DiagFactor.quad", (N,), np.ones(1), end="")):
        jax.jit(lambda F, b: F.quad(b, form="elementwise"))(F, jnp.ones(1))
    padded = COVS["blocktri-padded"]()
    C, G = padded.constants(jax_be), padded.factor(nu)
    whiten = jax.jit(lambda G, C, b: G.whiten(b, C=C))
    assert whiten(G, C, jnp.ones(14)).shape == (14,)
    with pytest.raises(ValueError, match=_refusal("BlockTriCovFactor.whiten", (14,), np.ones(15), end="")):
        whiten(G, C, jnp.ones(15))
    S = _two_channels().factor(jnp.asarray(NU2))
    with pytest.raises(ValueError, match=_refusal("StackFactor.quad", (2,), np.ones((3, N)), end="")):
        jax.jit(lambda S, b: S.quad(b))(S, jnp.ones((3, N)))
```

`SILENT` holds the three modes of spec §2.8, with both block layouts. Each is called through `multiply` and through the factor's `solve`, `whiten` and `quad` (forms `"solve"`, `"whiten"` and, where the factor has it, `"elementwise"`). `test_a_list_b_is_read_as_an_array` pins the decision of Step 1. `test_a_failed_ragged_factor_still_reports_the_failure` guards the one place where a factor's own shape is not its `n` (Step 5). Under `jax.jit`, JAX appends a traceback note to the message, so those patterns drop the `$` anchor (`end=""`).

- [ ] **Step 3: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_cov_multiply.py --tb=line`
Expected: `30 failed, 26 passed`:
- `test_multiply_refuses_b_of_the_wrong_leading_shape`: all 4 fail with `Failed: DID NOT RAISE <class 'ValueError'>`.
- `test_the_factor_refuses_b_of_the_wrong_leading_shape`: all 18 fail. 15 with `DID NOT RAISE`. The three `quad-solve` cases on `diagonal-(1,)`, `blocktri-ragged-(n+1,)` and `blocktri-padded-(n+1,)` fail with an `AssertionError`: NumPy's `matmul: Input operand 1 has a mismatch in its core dimension 0, ...` does not match the pattern.
- `test_a_channel_stack_whose_channels_differ_in_size_is_refused_when_used`: an `AssertionError`, since NumPy's `operands could not be broadcast together with shapes (5,) (128,)` does not match.
- `test_a_list_b_is_read_as_an_array`: 6 fail with `AttributeError: 'list' object has no attribute 'ndim'`. These are `diagonal-array` and `diagonal-callable` (in `DiagFactor.solve`), and `blocktri-ragged`, `blocktri-padded`, `evolutionary-blocktri` and `evolutionary-padded` (in `linalg.blocktri_solve`). The three `CholFactor` cases (`dense`, `evolutionary-dense`, `evolutionary-direct`) pass.
- `test_the_check_runs_on_jax_and_under_jit`: `DID NOT RAISE`, because the eager JAX `multiply` broadcasts the `(1,)` `b`.
- Passing: the 12 tests the file had, the 10 batch cases (`test_an_n_by_k_batch_of_columns_still_works` and `test_a_channel_stack_takes_a_batch_per_channel`), the 3 `CholFactor` list cases and `test_a_failed_ragged_factor_still_reports_the_failure`. They pin what must not change.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_cov_multiply.py --tb=no`
Expected: `29 failed, 26 passed, 1 skipped` (the JAX test skips).

- [ ] **Step 4: `base.py`: `_check_leading`, the diagonal and Cholesky factors, and `multiply`**

The module docstring, after the list of the factor's members:

```python
``quad(b, form=...)`` and ``n_u`` (0 for every covariance here).
```
→
```python
``quad(b, form=...)`` and ``n_u`` (0 for every covariance here).  ``b`` is read with
``be.asarray`` (a list is accepted), and its leading axes must be the factor's own
``(n,)``, or ``(channels,)`` for a ``StackFactor`` (each row then checked by its
channel's factor); ``solve`` and ``whiten`` also take an ``(n, k)`` column batch.  Any
other leading shape raises ``ValueError``, as in :meth:`Covariance.multiply`.
```

`_check_leading`, after `factor_constants` (its last lines are shown, kept as they are):

```python
        raise ValueError(f"{type(F).__name__}.{what} needs its covariance's constants: pass "
                         "C=cov.constants(be) (cov.factor(nu) attaches them)")
    return C
```
→
```python
        raise ValueError(f"{type(F).__name__}.{what} needs its covariance's constants: pass "
                         "C=cov.constants(be) (cov.factor(nu) attaches them)")
    return C


def _check_leading(b: Any, expected: tuple[int, ...], what: str) -> None:
    """Raise ``ValueError`` unless ``b``'s leading axes are ``expected``.  Trailing axes (an
    ``(n, k)`` column batch) are not compared.  Without the check a ``(1,)`` ``b`` would
    broadcast, a longer one be cut and an extra channel row be dropped, silently.  Shapes
    are static under ``jax.jit``, so it runs while tracing."""
    if tuple(b.shape[:len(expected)]) != tuple(expected):
        raise ValueError(f"{what}: b must have leading shape {expected}, got shape {b.shape}")
```

`DiagFactor`, from `solve` to the end of `quad`:

```python
    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        return b / (self.v if b.ndim == 1 else self.v[:, None])

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.v, b)
        s = be.xp.sqrt(self.v)
        return b / (s if b.ndim == 1 else s[:, None])

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        be = be or get_backend(self.v, b)
        if require_str("form", form) == "elementwise":
            return be.xp.sum(b ** 2 / self.v)
        return quad_form(self, b, form, be)
```
→
```python
    def _b_shape(self) -> tuple[int, ...]:
        """``(n,)``, from the variances."""
        return (self.v.shape[0],)

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.v, b)
        b = be.asarray(b)
        _check_leading(b, self._b_shape(), "DiagFactor.solve")
        return b / (self.v if b.ndim == 1 else self.v[:, None])

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.v, b)
        b = be.asarray(b)
        _check_leading(b, self._b_shape(), "DiagFactor.whiten")
        s = be.xp.sqrt(self.v)
        return b / (s if b.ndim == 1 else s[:, None])

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        be = be or get_backend(self.v, b)
        if require_str("form", form) == "elementwise":
            b = be.asarray(b)
            _check_leading(b, self._b_shape(), "DiagFactor.quad")
            return be.xp.sum(b ** 2 / self.v)
        return quad_form(self, b, form, be)
```

`DiagFactor.solve` did not resolve a backend before. It now does so as `whiten` and `quad` do (`get_backend(self.v, b)`), to read `b` with `be.asarray`. On an ndarray `b`, `np.asarray` returns the same array, so the arithmetic is unchanged.

`CholFactor`, `solve` and `whiten` (`quad` stays):

```python
    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.L, b)
        if self.impl == "scipy":
            return be.cho_solve(self.L, b)
        u = be.solve_triangular(self.L, b, lower=True)
        return be.solve_triangular(self.L, u, lower=True, trans=1)

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.L, b)
        return be.solve_triangular(self.L, b, lower=True)
```
→
```python
    def _b_shape(self) -> tuple[int, ...]:
        """``(n,)``, from ``L``."""
        return (self.L.shape[0],)

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.L, b)
        b = be.asarray(b)
        _check_leading(b, self._b_shape(), "CholFactor.solve")
        if self.impl == "scipy":
            return be.cho_solve(self.L, b)
        u = be.solve_triangular(self.L, b, lower=True)
        return be.solve_triangular(self.L, u, lower=True, trans=1)

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.L, b)
        b = be.asarray(b)
        _check_leading(b, self._b_shape(), "CholFactor.whiten")
        return be.solve_triangular(self.L, b, lower=True)
```

The `Covariance` class docstring, its last sentences:

```python
    :meth:`dense` is its host value.  :meth:`multiply` is ``_multiply(C, nu, b, be)``,
    which defaults to ``_matrix(C, nu, be) @ b``; a class with a cheaper product overrides
    it.  ``jittable = False`` marks a strategy that runs eagerly on both backends (the
    ragged layout, ``on_fail="raise"``).
    """
```
→
```python
    :meth:`dense` is its host value.  :meth:`multiply` is ``_multiply(C, nu, b, be)``,
    which defaults to ``_matrix(C, nu, be) @ b``; a class with a cheaper product overrides
    it.  :meth:`multiply` first checks ``b``'s leading axes against :meth:`_b_shape`.
    ``jittable = False`` marks a strategy that runs eagerly on both backends (the
    ragged layout, ``on_fail="raise"``).
    """
```

`Covariance.multiply`, with the new `_b_shape` before it:

```python
    def multiply(self, nu: Any, b: Any, be: Backend | None = None) -> Any:
        """``Sigma(nu) b``, eagerly.  ``b`` has the layout the factor's ``solve`` takes:
        ``(n,)``, or ``(channels, n_c)`` for :class:`~domainia.covariance.stack.ChannelStack`.

        NumPy is the tested backend; a JAX ``be`` is passed through.  The name is not
        ``matvec``: ``PushedCovariance`` duck-types ``matvec(X, be)``, which has a
        fixed-noise-state signature.
        """
        be = be or get_backend(nu, b)
        return self._multiply(self.constants(be), be.asarray(nu), be.asarray(b), be)
```
→
```python
    def _b_shape(self) -> tuple[int, ...]:
        """The leading shape of the ``b`` that :meth:`multiply` takes: ``(n,)``.
        ``ChannelStack`` overrides it."""
        return (self.n,)

    def multiply(self, nu: Any, b: Any, be: Backend | None = None) -> Any:
        """``Sigma(nu) b``, eagerly.  ``b`` has the layout the factor's ``solve`` takes: its
        leading axes are :meth:`_b_shape`, ``(n,)``, or ``(channels, n_c)`` for
        :class:`~domainia.covariance.stack.ChannelStack`, and any trailing axes are a batch
        of columns (``(n, k)``).  Any other leading shape raises ``ValueError``.

        NumPy is the tested backend; a JAX ``be`` is passed through.  The name is not
        ``matvec``: ``PushedCovariance`` duck-types ``matvec(X, be)``, which has a
        fixed-noise-state signature.
        """
        be = be or get_backend(nu, b)
        b = be.asarray(b)
        _check_leading(b, self._b_shape(), f"{type(self).__name__}.multiply")
        return self._multiply(self.constants(be), be.asarray(nu), b, be)
```

- [ ] **Step 5: `blocktri.py`: `BlockTriCovFactor`**

The import from `.base`:

```python
from .base import Covariance, QuadForm, as_evaluator, factor_constants, quad_form
```
→
```python
from .base import Covariance, QuadForm, _check_leading, as_evaluator, factor_constants, quad_form
```

`BlockTriCovFactor._apply`, with the new `_b_shape` before it:

```python
    def _apply(self, fn: Any, b: Any, C: Any, be: Backend, what: str) -> Any:
        # domainia.linalg takes natural, band-major vectors on both layouts: the padded
        # layout packs, runs the ``*_packed`` kernels and unpacks inside (the dummy entries
        # are exactly 0).
        if self.layout == "ragged":
            return fn(self.bt, b, be=be)
        return fn(self.bt, b, band_layout=factor_constants(self, C, what)["layout"], be=be)
```
→
```python
    def _b_shape(self, C: Any = None) -> tuple[int, ...]:
        """``(n,)``: the sum of the ragged blocks' sizes, or the padded layout's ``n`` (static
        meta of the ``BandLayout`` in the constants ``C``; its ``Ls`` are ``(nb, B, B)``)."""
        if self.layout == "ragged":
            return (sum(L.shape[0] for L in self.bt.Ls),)
        return (factor_constants(self, C, "_b_shape")["layout"].n,)

    def _apply(self, fn: Any, b: Any, C: Any, be: Backend, what: str) -> Any:
        # domainia.linalg takes natural, band-major vectors on both layouts: the padded
        # layout packs, runs the ``*_packed`` kernels and unpacks inside (the dummy entries
        # are exactly 0).
        b = be.asarray(b)
        if self.layout == "ragged":
            # A failed ragged factor stops at its first bad block, so its blocks do not add
            # up to n: skip the check and let ``fn`` raise the failure.
            if bool(self.ok):
                _check_leading(b, self._b_shape(), f"BlockTriCovFactor.{what}")
            return fn(self.bt, b, be=be)
        C = factor_constants(self, C, what)
        _check_leading(b, self._b_shape(C), f"BlockTriCovFactor.{what}")
        return fn(self.bt, b, band_layout=C["layout"], be=be)
```

`solve` and `whiten` both call `_apply`, and `quad` calls them through `quad_form`, so all three are checked. The padded factor's `Ls` are `(nb, B, B)`, which does not give `n`, so its shape comes from the constants' `BandLayout` (`n` is static meta, also under `jit`). That is why this `_b_shape` takes `C`, unlike the other factors'. A ragged factor whose Cholesky failed stops at its first bad block (`linalg._ragged_factor`), so its blocks add up to less than `n`. `_apply` then skips the check, and `blocktri_solve`/`blocktri_whiten` raise `"the factorisation failed ..."` as before (`ok` is always a concrete bool on the ragged layout, which runs eagerly only).

- [ ] **Step 6: `stack.py`: `StackFactor` and `ChannelStack`**

The imports:

```python
from domainia.wilson._checks import require_instance, require_int
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass
from domainia.wilson.windows import Window

from .base import Covariance
```
→
```python
from domainia.wilson._checks import require_instance, require_int, require_str
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass
from domainia.wilson.windows import Window

from .base import Covariance, _check_leading
```

`StackFactor`, from `quad` to the end of `whiten`, with `_b_shape` and `_rows` before them:

```python
    def quad(self, b: Any, *, C: Any = None, form: str = "solve", be: Backend | None = None) -> Any:
        """``sum_c quad_c(b[c])`` from ``0.0`` in channel order; ``b`` is ``(C, n)``."""
        total = 0.0
        for c, (f, Cc) in enumerate(zip(self.channels, self._sub(C), strict=True)):
            total = total + f.quad(b[c], C=Cc, form=form, be=be)
        return total

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(b)
        return be.xp.stack([f.solve(b[c], C=Cc, be=be)
                            for c, (f, Cc) in enumerate(zip(self.channels, self._sub(C), strict=True))])

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(b)
        return be.xp.stack([f.whiten(b[c], C=Cc, be=be)
                            for c, (f, Cc) in enumerate(zip(self.channels, self._sub(C), strict=True))])
```
→
```python
    def _b_shape(self) -> tuple[int, ...]:
        """``(channels,)``: one row of ``b`` per channel.  Each row ``b[c]`` is then checked by
        its channel's factor, so a stack whose channels differ in size refuses every ``b``."""
        return (len(self.channels),)

    def _rows(self, b: Any, be: Backend | None, what: str) -> Any:
        """``b`` as an array, its first axis checked against the channels."""
        b = (be or get_backend(b)).asarray(b)
        _check_leading(b, self._b_shape(), f"StackFactor.{what}")
        return b

    def quad(self, b: Any, *, C: Any = None, form: str = "solve", be: Backend | None = None) -> Any:
        """``sum_c quad_c(b[c])`` from ``0.0`` in channel order; ``b`` is ``(C, n)``."""
        require_str("form", form)
        b = self._rows(b, be, "quad")
        total = 0.0
        for c, (f, Cc) in enumerate(zip(self.channels, self._sub(C), strict=True)):
            total = total + f.quad(b[c], C=Cc, form=form, be=be)
        return total

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(b)
        b = self._rows(b, be, "solve")
        return be.xp.stack([f.solve(b[c], C=Cc, be=be)
                            for c, (f, Cc) in enumerate(zip(self.channels, self._sub(C), strict=True))])

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(b)
        b = self._rows(b, be, "whiten")
        return be.xp.stack([f.whiten(b[c], C=Cc, be=be)
                            for c, (f, Cc) in enumerate(zip(self.channels, self._sub(C), strict=True))])
```

`StackFactor.quad` loops over its channels' `quad` and never calls `solve`, so it checks on its own. It calls `require_str("form", form)` first, so that a `form` of the wrong kind stays a `TypeError` ahead of the shape check, as in `DiagFactor.quad` (the channels check `form` again, unchanged). `_rows` resolves its backend from `b` alone and passes the caller's `be` on unchanged, so each channel still picks its own backend, as before.

`ChannelStack._b_shape`, before `_multiply` (its docstring is shown, kept as it is):

```python
    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        """Each channel's ``Sigma_c(nu_c) b[c]``, stacked: ``b`` is ``(channels, n_c)``, so the
        channels must share ``n_c`` (as :meth:`StackFactor.solve`)."""
```
→
```python
    def _b_shape(self) -> tuple[int, ...]:
        """``(channels, n_c)``: one row per channel, of the channels' common leading shape
        (``(n_c,)``).  A stack whose channels differ in size can be built and factorised,
        but has no such layout: ``ValueError``."""
        rows = [cov._b_shape() for cov in self.covs]
        if any(r != rows[0] for r in rows):
            raise ValueError("ChannelStack: b has one row per channel, so the channels must share one "
                             f"size; their row shapes are {', '.join(str(r) for r in rows)}")
        return (len(self.covs), *rows[0])

    def _multiply(self, C: Any, nu: Any, b: Any, be: Backend) -> Any:
        """Each channel's ``Sigma_c(nu_c) b[c]``, stacked: ``b`` is ``(channels, n_c)``, so the
        channels must share ``n_c`` (as :meth:`StackFactor.solve`)."""
```

`ChannelStack._b_shape` compares its channels' own `_b_shape()`, which is `(n_c,)` for every covariance but a nested `ChannelStack`. A nested stack then gets the layout its `_multiply` already reads, `(channels, inner channels, n_c)`.

- [ ] **Step 7: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_cov_multiply.py`
Expected: `56 passed`, no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_cov_multiply.py`
Expected: `55 passed, 1 skipped`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_cov_multiply.py tests/test_operators_covariance.py tests/test_dom_cov_api.py tests/test_wrong_kinds_covariance.py tests/test_dom_cov_stack.py tests/test_dom_cov_blocktri.py tests/test_dom_cov_base.py tests/test_dom_evolutionary_cov.py`
Expected: all pass, 0 errors. This includes `test_operators_covariance.py::test_the_arguments_are_validated` (the mixed-size stack at `:232` gets `CovarianceOperator`'s message), `test_dom_cov_api.py::test_factors_are_pytrees_that_stack` (under `vmap`, a stacked factor's `v` and `L` have their unbatched shape `(3,)`, so the check passes) and `test_factor_classes_follow_the_protocol` (a bare factor fails for want of state, not with `NotImplementedError`).

- [ ] **Step 8: Run the probe again**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=src ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/probe_b_shape.py`
Expected:

```
diagonal (1,) multiply: ValueError: DiagonalCovariance.multiply: b must have leading shape (6,), got shape (1,)
diagonal (1,) solve: ValueError: DiagFactor.solve: b must have leading shape (6,), got shape (1,)
diagonal (1,) whiten: ValueError: DiagFactor.whiten: b must have leading shape (6,), got shape (1,)
diagonal (1,) quad solve: ValueError: DiagFactor.solve: b must have leading shape (6,), got shape (1,)
diagonal (1,) quad whiten: ValueError: DiagFactor.whiten: b must have leading shape (6,), got shape (1,)
diagonal (1,) quad elementwise: ValueError: DiagFactor.quad: b must have leading shape (6,), got shape (1,)
blocktri-ragged (n+1,) multiply: ValueError: BlockTridiagonalCovariance.multiply: b must have leading shape (14,), got shape (15,)
blocktri-ragged (n+1,) solve: ValueError: BlockTriCovFactor.solve: b must have leading shape (14,), got shape (15,)
blocktri-ragged (n+1,) whiten: ValueError: BlockTriCovFactor.whiten: b must have leading shape (14,), got shape (15,)
blocktri-ragged (n+1,) quad solve: ValueError: BlockTriCovFactor.solve: b must have leading shape (14,), got shape (15,)
blocktri-ragged (n+1,) quad whiten: ValueError: BlockTriCovFactor.whiten: b must have leading shape (14,), got shape (15,)
blocktri-padded (n+1,) multiply: ValueError: BlockTridiagonalCovariance.multiply: b must have leading shape (14,), got shape (15,)
blocktri-padded (n+1,) solve: ValueError: BlockTriCovFactor.solve: b must have leading shape (14,), got shape (15,)
blocktri-padded (n+1,) whiten: ValueError: BlockTriCovFactor.whiten: b must have leading shape (14,), got shape (15,)
blocktri-padded (n+1,) quad solve: ValueError: BlockTriCovFactor.solve: b must have leading shape (14,), got shape (15,)
blocktri-padded (n+1,) quad whiten: ValueError: BlockTriCovFactor.whiten: b must have leading shape (14,), got shape (15,)
stack (3, n) multiply: ValueError: ChannelStack.multiply: b must have leading shape (2, 6), got shape (3, 6)
stack (3, n) solve: ValueError: StackFactor.solve: b must have leading shape (2,), got shape (3, 6)
stack (3, n) whiten: ValueError: StackFactor.whiten: b must have leading shape (2,), got shape (3, 6)
stack (3, n) quad solve: ValueError: StackFactor.quad: b must have leading shape (2,), got shape (3, 6)
stack (3, n) quad whiten: ValueError: StackFactor.quad: b must have leading shape (2,), got shape (3, 6)
stack (3, n) quad elementwise: ValueError: StackFactor.quad: b must have leading shape (2,), got shape (3, 6)
mixed stack multiply (2, 6): ValueError: ChannelStack: b has one row per channel, so the channels must share one size; their row shapes a
mixed stack quad [(6,), (5,)]: ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1
diagonal solve list: ok, shape (6,)
```

Every silent call of Step 1 now raises the check's `ValueError`. The `"solve"` and `"whiten"` forms of `quad` name `solve` and `whiten`, the methods whose check fires. The list `b` now works.

- [ ] **Step 9: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`
Expected: all pass, 0 errors, no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`
Expected: all pass, 0 errors, no warnings.

A failure here, outside `tests/test_dom_cov_multiply.py`, is a test that passes a `b` of the wrong leading shape, which Step 1 could not grep. Read it: if the test meant that shape, give it the right one; if it pins the old broadcasting, move it to a `ValueError` case. Record it in the plan's Execution errata. No existing test is expected to need a change.

The float64 bit gate is not run here (Conventions: Tasks 2, 4, 10, 11 and 15). This task adds shape checks in `domainia.covariance` and touches neither `wilson/transform.py` nor `maps/`.

- [ ] **Step 10: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/covariance/base.py src/domainia/covariance/blocktri.py src/domainia/covariance/stack.py tests/test_dom_cov_multiply.py
git commit -m "domainia.covariance: check the leading shape of b

Covariance.multiply and every factor's solve, whiten and quad compare
the leading axes of b with the shape they expect (_b_shape) and raise
ValueError on a mismatch. A (1,) b was broadcast by DiagonalCovariance,
an (n+1,) b cut to n by BlockTridiagonalCovariance, and a third row
dropped by a two-channel ChannelStack, all silently. (n, k) column
batches still work. A ChannelStack whose channels differ in size still
builds and factorises; multiply and its factor's methods refuse it. The
factors read b with be.asarray first, so a list b works everywhere, as
it already did in multiply.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: DenseOperator: one rtol, the fail-open fixes (spec §3)

**Goal:** `DenseOperator` keeps one `rtol` for both jobs, with the §3.1 guards. It fails closed where it failed open (§3.2): `tol = top * (n * eps)` and `S = 0.5 * C + 0.5 * C.T` cannot overflow, and a non-finite eigenvalue from `eigh` and the zero matrix raise `ValueError`. `op.rtol` stores the relative tolerance used (§3.3). The symmetry factor follows the input dtype, and the refusal prints it (§3.4). Every `TypeError` in `operators.py` that names what it got uses the `", got"` form (§2.2), through the `_checks` helpers.

**Assumes the tree after Task 7.** No earlier task edits `src/domainia/operators.py`, `tests/test_operators_dense.py` or `tests/test_operators_covariance.py` (Step 1 checks this). Task 1 added `src/domainia/wilson/_checks.py`, which this task imports. `DenseOperator` first calls `translate._record` and `translate._channel_names`, whose messages Task 6 put in the `", got"` form (it owns `translate.py`). Their pins in `test_the_matrix_is_validated` (`"channel_names must be a tuple"`, `"^domain must be a full-record domain"`) match leading words only, so they still hold, and this task does not edit them.

**Files:**
- Modify: `src/domainia/operators.py` (the `_checks` import; `DenseOperator`'s docstring, its `rtol` field, and `__post_init__` up to `keep = w > tol`; the `cov`, `support` and `nu` checks of `CovarianceOperator.__post_init__`; the `TypeError` of `operator_matrix`)
- Modify: `tests/test_operators_dense.py` (`import dataclasses`, `EPS64`/`EPS32`, an `rng` argument for `_psd`, the `operator_matrix` pin, `test_a_float32_rank_deficient_matrix_uses_float32_eps`, ten new tests at the end)
- Modify: `tests/test_operators_covariance.py` (three `match=` patterns in `test_the_arguments_are_validated` now pin the `", got"` form)

**Acceptance Criteria:**
- [ ] `rtol` is checked with `require_real` (`TypeError` `"rtol must be a real number, got {type}"`). The `"rtol must be finite and >= 0; got ..."` `ValueError` is unchanged. `rtol >= 1` raises `ValueError` `"rtol must be < 1; got ..."`. `rtol=None` stays the default.
- [ ] Suppose the PSD check fails, `rtol` was given, and `λ_min >= -(top * (n * eps))` (`eps` by the input dtype). Then the message says the eigenvalue is within round-off of zero and that the default `rtol=None` accepts it. Otherwise the message is the plain `"matrix is not positive semi-definite: ..."`.
- [ ] `tol = top * rtol`, with `rtol = n * eps` for `rtol=None`, so the default is `tol = top * (n * eps)`. `S = 0.5 * C + 0.5 * C.T`. The non-finite eigenvalue check runs on `w` right after `eigh`, before `top` and `tol` are formed. `top == 0.0` (the zero matrix) raises `ValueError`.
- [ ] `op.rtol` is a Python `float` and never `None`. It is `n * eps32` for float32 input with `rtol=None`, `n * eps64` for other input with `rtol=None`, or the given `rtol`. The docstring says so, and so does the field annotation (`rtol: float`). `dataclasses.replace(op)` on a float32 rank-deficient operator gives the same `matrix` and `degrees_of_freedom`.
- [ ] The symmetry factor is `f = 1e-12 * (eps / eps64)`: `1e-12`, or about 5.4e-4 for float32 input. The docstring gives both. The refusal prints the factor applied: `(more than 1e-12)` or `(more than 0.00054)`.
- [ ] `matrix` and `nu` are checked with `require_dtype(..., REAL_KINDS, "a real numeric array")`, and `cov` and `support` with `require_instance`. `operator_matrix`'s `op` message takes `", got"` by hand. `grep -n "; got" src/domainia/operators.py` shows `ValueError` lines only.
- [ ] Every §3.6 case is a test in `tests/test_operators_dense.py`. The file has 44 passed, and `tests/test_operators_covariance.py` has 27 passed.
- [ ] The full suite and the NumPy lane pass, with no warnings summary.

**Verify:** `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_operators_dense.py tests/test_operators_covariance.py` → `71 passed`

**Steps:**

- [ ] **Step 1: Confirm the start, and grep the patterns this task changes**

Run: `cd /Users/aaron/Documents/lisa/domainia && git log --oneline 779550d..HEAD -- src/domainia/operators.py tests/test_operators_dense.py tests/test_operators_covariance.py && test -f src/domainia/wilson/_checks.py && echo ok`
Expected: `ok` alone. No commit since 779550d touched the three files. If a commit is listed, stop and compare its diff with the old texts in Steps 2 and 4.

The line numbers below are those at 779550d.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "\.rtol\|rtol is None" src tests`
Expected: four hits.
- `src/domainia/operators.py:100` (`rtol = self.rtol`) and `:110` (`tol = top * n * eps if rtol is None else rtol * top`). Both are replaced in Step 4.
- Two lines of `pcg` in `src/domainia/linalg.py` (`true_rtol = None if true_rtol is None else require_real("true_rtol", true_rtol)` and `rtol = max(5.0 * tol, 1e-12) if true_rtol is None else true_rtol`, from Task 6). They are unrelated.

No test reads `op.rtol`, so no test pins `op.rtol is None`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "not positive semi-definite\|not symmetric\|more than 1e-12" src tests`
Expected: seven hits.
- `operators.py:57` (docstring), `:98-99` and `:112` (the raises). All are replaced in Step 4.
- `tests/test_operators_dense.py:139` (`match="not symmetric"`) and `:141` (`match="not positive semi-definite"`), in `test_the_matrix_is_validated`. They are substrings of the new messages and stay.
- `tests/test_dom_linalg_blocktri.py:131`. This is a comment.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" src/domainia/operators.py`
Expected: eight hits.
- Six `TypeError`s, all converted in Step 4: `matrix` (`:88`), `rtol` (`:103`), `cov` (`:202`), `support` (`:204`), `nu` (`:231`) and `op` (`:294`).
- Two `ValueError`s, which keep their text: `rtol must be finite and >= 0; got` (`:106`) and `every ChannelStack channel needs n = support.size = {s}; got` (`:213`).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "real numeric\|must be a domainia.covariance\|support must be a WDMDomain\|must implement domainia\|rtol must\|match=\"rtol\"\|match=\"Operator\"" tests`
Expected: 27 hits. None of them fails after this task.
- `tests/test_operators_dense.py:143`, `:145`, `:155`, `:157` match a substring that the new message keeps. They stay.
- Step 2 tightens `tests/test_operators_dense.py:164` (`match="Operator"`, `operator_matrix(C)`) and `tests/test_operators_covariance.py:216`, `:218`, `:224` (`cov`, `support`, `nu`) to anchored `", got"` patterns, so that the conversion is pinned.
- `tests/test_translate_operator.py:500` (`match="Operator"`) is `translate_operator`'s own refusal in `translate.py`. It is not in this file.
- The other 18 hits pin other modules' messages, already in the `", got"` form from Tasks 1, 2 and 6: `tests/wilson/test_checks.py` (9), `tests/wilson/test_wrong_kinds.py` (4), `tests/test_wrong_kinds_misc.py` (3; its `"op must implement domainia.translate.Operator, got object"` is `translate_operator`'s), `tests/wilson/test_noise_stationary.py` (1) and `tests/wilson/test_grid_bands.py` (1). They are unrelated.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "(C + C.T) / 2\|0.5 \* (C + C.T)" src tests`
Expected:
- `operators.py:54` (docstring) and `:107` (`S = 0.5 * (C + C.T)`). Both are replaced in Step 4.
- `tests/test_operators_dense.py:102` asserts that `DenseOperator(C, d, NAMES).matrix` equals `(C + C.T) / 2` bit for bit. It still holds: halving is exact for normal floats, so `0.5 * C + 0.5 * C.T` rounds the same sum.
- The other hits (`test_operators_dense.py:172`, `test_operators_covariance.py:150`, `test_dom_linalg_blocktri.py:92`, `test_dom_restored_coverage.py:113-114`, `:141`, `test_translate_operator.py:225`, `:320`) symmetrise the tests' own reference matrices. They are unrelated.

- [ ] **Step 2: Write the failing tests**

In `tests/test_operators_dense.py`, replace

```python
from __future__ import annotations

import numpy as np
import pytest

```

with

```python
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

```

then replace

```python
_rng = np.random.default_rng(53)
```

with

```python
_rng = np.random.default_rng(53)
EPS64, EPS32 = float(np.finfo(np.float64).eps), float(np.finfo(np.float32).eps)
```

then replace the head of `_psd` (its last line, `return B @ B.T + ...`, stays)

```python
def _psd(n, rank):
    """A random symmetric PSD ``(n, n)`` matrix of rank ``rank`` whose two channel blocks are
    correlated (no block structure)."""
    B = _rng.standard_normal((n, rank)) / np.sqrt(n)
```

with

```python
def _psd(n, rank, rng=_rng):
    """A random symmetric PSD ``(n, n)`` matrix of rank ``rank`` whose two channel blocks are
    correlated (no block structure)."""
    B = rng.standard_normal((n, rank)) / np.sqrt(n)
```

then, at the end of `test_the_matrix_is_validated`, replace

```python
    with pytest.raises(TypeError, match="Operator"):
        operator_matrix(C)
```

with

```python
    with pytest.raises(TypeError, match=r"^op must implement domainia\.translate\.Operator, got ndarray$"):
        operator_matrix(C)
```

then replace the whole function `test_a_float32_rank_deficient_matrix_uses_float32_eps`

```python
def test_a_float32_rank_deficient_matrix_uses_float32_eps():
    n = N
    B = _rng.standard_normal((n, n // 2)).astype(np.float32)
    C = B @ B.T
    C = ((C + C.T) / 2).astype(np.float32)                           # exactly symmetric
    assert DenseOperator(C, TIME, ("A",)).degrees_of_freedom == n // 2
```

with

```python
def test_a_float32_rank_deficient_matrix_uses_float32_eps():
    """... and stores it: ``op.rtol`` is ``K N eps32``, so a rebuild from the float64
    ``op.matrix`` (``dataclasses.replace``, or the fields passed back) keeps the same
    directions.  Before, the rebuild used float64's eps and refused the matrix."""
    n = N
    B = _rng.standard_normal((n, n // 2)).astype(np.float32)
    C = B @ B.T
    C = ((C + C.T) / 2).astype(np.float32)                           # exactly symmetric
    op = DenseOperator(C, TIME, ("A",))
    assert op.degrees_of_freedom == n // 2 and op.rtol == n * EPS32
    for again in (dataclasses.replace(op),
                  DenseOperator(op.matrix, op.domain, op.channel_names, rtol=op.rtol)):
        np.testing.assert_array_equal(again.matrix, op.matrix)
        assert again.degrees_of_freedom == op.degrees_of_freedom and again.rtol == op.rtol
```

and replace the last two lines of the file (the end of `test_a_float16_matrix_keeps_float64_eps`)

```python
    C = ((Q * w) @ Q.T).astype(np.float16)
    assert DenseOperator(C, TIME, NAMES).degrees_of_freedom == n
```

with

```python
    C = ((Q * w) @ Q.T).astype(np.float16)
    assert DenseOperator(C, TIME, NAMES).degrees_of_freedom == n


def _diag(last):
    """``diag(1, ..., 1, last)`` on ``TIME``: ``eigh`` returns the entries exactly, and
    ``max |lambda| = 1``."""
    return np.diag(np.r_[np.ones(N - 1), last])


def test_the_default_tolerance_is_exact_at_its_boundary():
    """``tol = max|lambda| * (K N eps)``, the code's own expression: ``-tol`` is accepted and
    ``-1.01 tol`` refused; ``+tol`` is dropped and ``+1.01 tol`` kept."""
    tol = 1.0 * (N * EPS64)
    assert DenseOperator(_diag(-1.0 * tol), TIME, ("A",)).degrees_of_freedom == N - 1
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite"):
        DenseOperator(_diag(-1.01 * tol), TIME, ("A",))
    assert DenseOperator(_diag(1.0 * tol), TIME, ("A",)).degrees_of_freedom == N - 1
    assert DenseOperator(_diag(1.01 * tol), TIME, ("A",)).degrees_of_freedom == N


def test_an_rtol_below_round_off_says_so():
    """``rtol=0`` on a rank-deficient ``B B^T``: round-off makes a zero eigenvalue negative,
    and the message says the default accepts it.  The seed is fixed: in about 1 draw in 1200
    (17 of 20000 seeds) round-off leaves every zero eigenvalue >= 0.  ``-I`` with ``rtol=0``
    gets the plain message."""
    n = 2 * N
    C = _psd(n, n - 7, np.random.default_rng(17))
    w = np.linalg.eigh(0.5 * C + 0.5 * C.T)[0]
    assert -(np.max(np.abs(w)) * (n * EPS64)) <= w[0] < 0.0          # the premise: round-off
    assert DenseOperator(C, TIME, NAMES).degrees_of_freedom == n - 7
    with pytest.raises(ValueError, match=r"^matrix is not positive semi-definite at rtol=0\.0: .* "
                                         r"within round-off of zero .* the default rtol=None accepts it$"):
        DenseOperator(C, TIME, NAMES, rtol=0)
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite") as info:
        DenseOperator(-np.eye(N), TIME, ("A",), rtol=0)
    assert "round-off" not in str(info.value)


def test_rtol_forgives_negative_eigenvalues_down_to_minus_rtol_max_lambda():
    C = _diag(-1e-3)
    op = DenseOperator(C, TIME, ("A",), rtol=1e-2)
    assert op.degrees_of_freedom == N - 1 and op.rtol == 1e-2
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite: "):
        DenseOperator(C, TIME, ("A",))


@pytest.mark.parametrize("rtol", [1, 1.0, 5])
def test_an_rtol_of_one_or_more_is_refused(rtol):
    """``max |lambda| >= |lambda_min|``, so from ``rtol = 1`` on no eigenvalue is below
    ``-tol``: ``-I`` was accepted, with no direction kept."""
    with pytest.raises(ValueError, match=r"^rtol must be < 1; got "):
        DenseOperator(-np.eye(N), TIME, ("A",), rtol=rtol)


@pytest.mark.parametrize(("rtol", "kind"), [
    ("1e-6", "str"), (True, "bool"), (np.True_, "bool"), (1e-6 + 0j, "complex"),
    (np.array(1e-6), "ndarray"),
])
def test_rtol_of_the_wrong_kind_raises_type_error(rtol, kind):
    with pytest.raises(TypeError, match=f"^rtol must be a real number, got {kind}$"):
        DenseOperator(np.eye(N), TIME, ("A",), rtol=rtol)


@pytest.mark.parametrize(("matrix", "dtype"), [
    (np.eye(N) + 0j, "complex128"), (np.eye(N, dtype=np.complex64), "complex64"),
    (np.eye(N).astype(object), "object"), (np.eye(N) > 0, "bool"),
])
def test_a_matrix_of_the_wrong_dtype_raises_type_error(matrix, dtype):
    with pytest.raises(TypeError, match=f"^matrix must be a real numeric array, got dtype {dtype}$"):
        DenseOperator(matrix, TIME, ("A",))


def test_rtol_reads_back_the_tolerance_used():
    """``op.rtol`` is never ``None``: ``K N eps`` for ``rtol=None`` (float32's eps for a float32
    matrix), else the given ``rtol``, as a Python float."""
    C = _psd(N, N)
    op = DenseOperator(C, TIME, ("A",))
    assert type(op.rtol) is float and op.rtol == N * EPS64
    assert DenseOperator(C.astype(np.float32), TIME, ("A",)).rtol == N * EPS32
    assert DenseOperator(np.eye(N, dtype=np.int64), TIME, ("A",)).rtol == N * EPS64
    for given, want in ((np.float64(1e-6), 1e-6), (np.int64(0), 0.0)):
        op = DenseOperator(C, TIME, ("A",), rtol=given)
        assert type(op.rtol) is float and op.rtol == want


def test_huge_matrices_fail_closed():
    """``max|lambda| * K N * eps`` overflowed to inf for ``-1.2e307 I`` (accepted, nothing kept);
    ``(C + C^T) / 2`` overflowed for ``diag(1e308)``; ``eigh`` returns inf for a finite matrix
    whose largest eigenvalue (here 2e308) is beyond float64."""
    time16 = TimeDomain(16, DT)
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite: "):
        DenseOperator(-1.2e307 * np.eye(16), time16, ("A",))
    op = DenseOperator(np.diag(np.full(16, 1e308)), time16, ("A",))
    assert op.degrees_of_freedom == 16
    np.testing.assert_array_equal(op.matrix, 1e308 * np.eye(16))
    with pytest.raises(ValueError, match="^matrix: numpy.linalg.eigh returned a non-finite eigenvalue"):
        DenseOperator(np.full((4, 4), 5e307), TimeDomain(4, DT), ("A",))


def test_the_zero_matrix_is_refused():
    with pytest.raises(ValueError, match="^matrix is zero"):
        DenseOperator(np.zeros((N, N)), TIME, ("A",))


def test_the_symmetry_factor_follows_the_dtype():
    """``max |C - C^T| <= f max |C|`` with ``f = 1e-12``, or ``1e-12 eps32 / eps64`` (about
    5.4e-4) for a float32 matrix: a float32 product ``B D B^T`` nobody symmetrised (here
    ``D`` spans 8 decades) passes, a float64 asymmetry of 1e-11 does not, and the message
    prints the factor applied."""
    rng = np.random.default_rng(11)
    B = rng.standard_normal((N, N)).astype(np.float32)
    D = np.diag(np.logspace(0, -8, N)).astype(np.float32)
    C = B @ D @ B.T
    assert 1e-12 < np.max(np.abs(C - C.T)) / np.max(np.abs(C)) < 1e-6   # the premise
    assert DenseOperator(C, TIME, ("A",)).degrees_of_freedom >= N // 2   # 18 to 21 in 500 seeds
    C = _psd(N, N, rng)
    up = np.max(np.abs(C)) * np.triu(np.ones_like(C), 1)
    with pytest.raises(ValueError, match=r"^matrix is not symmetric: max \|C - C\^T\| is 1e-11 of "
                                         r"max \|C\| \(more than 1e-12\)$"):
        DenseOperator(C + 1e-11 * up, TIME, ("A",))
    with pytest.raises(ValueError, match=r"\(more than 0\.00054\)$"):
        DenseOperator((C + 1e-2 * up).astype(np.float32), TIME, ("A",))
```

Notes on the new tests:
- `test_an_rtol_below_round_off_says_so` draws its matrix from its own generator (seed 17), not from the module's `_rng`, whose state depends on which tests ran before. Over 20000 seeds of `_psd(64, 57)` with `rtol=0`, 19983 were refused with the round-off message, none with the plain message, and 17 were accepted: round-off left every zero eigenvalue `>= 0`. The test asserts that premise first (`-(n eps top) <= λ_min < 0`, computed with `eigh` on `0.5 * C + 0.5 * C.T` as the code does). On a platform where seed 17 falls in that tail, the test fails on the premise, not on the message.
- In `test_the_symmetry_factor_follows_the_dtype`, the float32 product at seed 11 has asymmetry 5.9e-8 relative. Over 300 seeds at `N` = 32, 64 and 128 with `D` over 4 or 8 decades it was 1.6e-8 to 1.5e-7, against the factor 5.4e-4. (With `D = I` the product came out exactly symmetric.) Its smallest eigenvalue stayed at or above `-0.0094 · n eps32 top`, so the PSD check passes with a wide margin. It keeps 18 to 21 directions over 500 seeds at `N = 32`, so the test asserts `>= N // 2`.

In `tests/test_operators_covariance.py`, in `test_the_arguments_are_validated`, replace

```python
    with pytest.raises(TypeError, match="cov must be a domainia.covariance.Covariance"):
        CovarianceOperator(np.eye(N), NU, WDM, ("A",))
    with pytest.raises(TypeError, match="support must be a WDMDomain"):
        CovarianceOperator(ev, NU, ALL, ("A",))
```

with

```python
    with pytest.raises(TypeError, match=r"^cov must be a domainia\.covariance\.Covariance, got ndarray$"):
        CovarianceOperator(np.eye(N), NU, WDM, ("A",))
    with pytest.raises(TypeError, match="^support must be a WDMDomain, got CellSet$"):
        CovarianceOperator(ev, NU, ALL, ("A",))
```

then replace

```python
    with pytest.raises(TypeError, match="nu must be a real numeric array"):
        CovarianceOperator(ev, "nu", WDM, ("A",))
```

with

```python
    with pytest.raises(TypeError, match="^nu must be a real numeric array, got dtype <U2$"):
        CovarianceOperator(ev, "nu", WDM, ("A",))
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q --tb=line tests/test_operators_dense.py tests/test_operators_covariance.py 2>&1 | tail -45`
Expected: `20 failed, 51 passed`, with no warnings summary. In `tests/test_operators_dense.py` 19 tests fail and 25 pass:
- `test_the_matrix_is_validated`: the message is `op must implement domainia.translate.Operator; got ndarray`.
- `test_a_float32_rank_deficient_matrix_uses_float32_eps`: `assert (16 == (32 // 2) and None == (32 * 1.1920928955078125e-07))`. `op.rtol` is `None`.
- `test_an_rtol_below_round_off_says_so`: `Regex pattern did not match`. This is the plain message, `... below -tol = -0.0`.
- `test_an_rtol_of_one_or_more_is_refused[1]`, `[1.0]`, `[5]`: `DID NOT RAISE`.
- `test_rtol_of_the_wrong_kind_raises_type_error` (all 5): the old message `rtol must be a real number or None; got ...`.
- `test_a_matrix_of_the_wrong_dtype_raises_type_error` (all 4): the old message `...; got dtype ...`.
- `test_rtol_reads_back_the_tolerance_used`: `assert (<class 'NoneType'> is float)`.
- `test_huge_matrices_fail_closed`: `DID NOT RAISE`. `-1.2e307 I` is accepted.
- `test_the_zero_matrix_is_refused`: `DID NOT RAISE`.
- `test_the_symmetry_factor_follows_the_dtype`: `ValueError: matrix is not symmetric: max |C - C^T| is 5.9e-08 of max |C| (more than 1e-12)`.

In `tests/test_operators_covariance.py`, `test_the_arguments_are_validated` fails: `- Covariance; got ndarray` / `+ Covariance, got ndarray`.

`test_the_default_tolerance_is_exact_at_its_boundary` and `test_rtol_forgives_negative_eigenvalues_down_to_minus_rtol_max_lambda` already pass. They pin behaviour that this task must keep. The `RuntimeWarning: overflow encountered in add` of the old `0.5 * (C + C.T)` does not show, because `test_huge_matrices_fail_closed` stops at its first `pytest.raises`.

- [ ] **Step 4: `operators.py`**

Replace

```python
from domainia.wilson.backend import NUMPY
```

with

```python
from domainia.wilson._checks import REAL_KINDS, require_dtype, require_instance, require_real
from domainia.wilson.backend import NUMPY
```

In `DenseOperator`'s docstring, replace the first two paragraphs after the summary line

```python
    ``matrix`` is ``(K N, K N)`` on the stacked real coordinates of ``domain`` (module
    docstring), ``K = len(channel_names)``, so channels may be correlated.  It must be
    finite and symmetric to 1e-12 relative (``max |C - C^T| <= 1e-12 max |C|``); the
    operator then stores ``(C + C^T) / 2`` as ``matrix``, a read-only float64 copy.

    ``numpy.linalg.eigh`` decomposes it once.  An eigenvalue below ``-tol`` raises
    ``ValueError`` (``C`` is not positive semi-definite); the eigenpairs with
    ``lambda > tol`` are kept (``V``, ``Lambda``).  ``tol = max |lambda| K N eps`` (the
    default of ``numpy.linalg.matrix_rank``; ``eps`` is float32's for a float32 ``matrix`` and
    float64's otherwise), or ``rtol max |lambda|`` when ``rtol`` is
    given.
```

with

```python
    ``matrix`` is ``(K N, K N)`` on the stacked real coordinates of ``domain`` (module
    docstring), ``K = len(channel_names)``, so channels may be correlated.  It must be
    finite and symmetric to ``f`` relative (``max |C - C^T| <= f max |C|``), with
    ``f = 1e-12``, or ``1e-12 eps32 / eps64`` (about 5.4e-4) for a float32 ``matrix``, so
    that a float32 product ``B D B^T`` nobody symmetrised passes.  The operator then stores
    ``C / 2 + C^T / 2`` (which cannot overflow) as ``matrix``, a read-only float64 copy.

    ``numpy.linalg.eigh`` decomposes it once; a non-finite eigenvalue (the matrix's
    spectrum is beyond float64) and the zero matrix (no direction to keep) raise
    ``ValueError``.  One relative tolerance ``rtol`` does both jobs, with
    ``tol = max |lambda| rtol``: an eigenvalue below ``-tol`` raises ``ValueError`` (``C``
    is not positive semi-definite), and the eigenpairs with ``lambda > tol`` are kept
    (``V``, ``Lambda``).  So ``rtol`` drops the directions with an eigenvalue at or below
    ``rtol max |lambda|`` and forgives negative eigenvalues down to ``-rtol max |lambda|``.
    The default ``rtol=None`` is ``K N eps``, the default of ``numpy.linalg.matrix_rank``
    (``eps`` is float32's for a float32 ``matrix`` and float64's otherwise).  A given
    ``rtol`` must be ``>= 0`` and ``< 1`` (from 1 on, no eigenvalue is below ``-tol``); when
    one refuses an eigenvalue within ``K N eps max |lambda|`` of zero, the message says that
    it is round-off and that the default accepts it.  ``op.rtol`` reads back the relative
    tolerance used, never ``None``, so ``dataclasses.replace(op)`` rebuilds the same
    operator.
```

Replace the `rtol` field and `__post_init__` from its first line to `keep = w > tol`. The rest of `__post_init__`, from `V, w = V[:, keep], w[keep]` on, is unchanged:

```python
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
        eps = float(np.finfo(np.float32 if C.dtype == np.float32 else np.float64).eps)
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
        tol = top * n * eps if rtol is None else rtol * top
        if w[0] < -tol:
            raise ValueError(f"matrix is not positive semi-definite: its smallest eigenvalue "
                             f"{float(w[0])!r} is below -tol = {-tol!r}")
        keep = w > tol
```

with

```python
    #: The relative tolerance (class docstring).  ``None``, the default, becomes ``K N eps``
    #: at construction, so after it ``rtol`` is the float used, never ``None``.
    rtol: float = field(default=None, kw_only=True)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        N, _ = _record(self.domain, "domain")
        names = _channel_names(self.channel_names)
        n = len(names) * N
        if isinstance(self.matrix, np.ma.MaskedArray):
            raise TypeError("matrix: masked arrays are refused (numpy.asarray drops the mask)")
        C = np.asarray(self.matrix)
        require_dtype("matrix", C, REAL_KINDS, "a real numeric array")
        eps64 = float(np.finfo(np.float64).eps)
        eps = float(np.finfo(np.float32).eps) if C.dtype == np.float32 else eps64
        f = 1e-12 * (eps / eps64)                       # 1e-12, or about 5.4e-4 for float32
        if C.shape != (n, n):
            raise ValueError(f"matrix has shape {C.shape}; {len(names)} channel(s) of {N} "
                             f"coordinates need shape ({n}, {n})")
        wilson.validate_finite(C, "matrix")
        C = C.astype(np.float64)
        peak = float(np.max(np.abs(C)))
        asym = float(np.max(np.abs(C - C.T)))
        if asym > f * peak:
            raise ValueError(f"matrix is not symmetric: max |C - C^T| is {asym / peak:.2g} of "
                             f"max |C| (more than {f:.2g})")
        if self.rtol is None:
            rtol = n * eps
        else:
            rtol = require_real("rtol", self.rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError(f"rtol must be finite and >= 0; got {rtol!r}")
            if rtol >= 1.0:
                raise ValueError(f"rtol must be < 1; got {rtol!r}: no eigenvalue is below "
                                 f"-rtol max |lambda| when rtol >= 1, so every matrix would pass")
        S = 0.5 * C + 0.5 * C.T                         # 0.5 * (C + C.T) overflows above ~9e307
        w, V = np.linalg.eigh(S)
        if not np.isfinite(w).all():
            raise ValueError("matrix: numpy.linalg.eigh returned a non-finite eigenvalue (the "
                             "spectrum of the matrix is beyond float64); scale it down")
        top = float(np.max(np.abs(w)))
        if top == 0.0:
            raise ValueError("matrix is zero: an operator needs a direction to keep")
        tol = top * rtol                                # the default: top * (n * eps)
        if w[0] < -tol:
            floor = top * (n * eps)
            if self.rtol is not None and w[0] >= -floor:
                raise ValueError(f"matrix is not positive semi-definite at rtol={rtol!r}: its "
                                 f"smallest eigenvalue {float(w[0])!r} is below -tol = {-tol!r}, "
                                 f"but it is within round-off of zero (at or above -K N eps "
                                 f"max |lambda| = {-floor!r}), and the default rtol=None accepts it")
            raise ValueError(f"matrix is not positive semi-definite: its smallest eigenvalue "
                             f"{float(w[0])!r} is below -tol = {-tol!r}")
        keep = w > tol
```

In `CovarianceOperator.__post_init__`, replace

```python
        if not isinstance(cov, Covariance):
            raise TypeError(f"cov must be a domainia.covariance.Covariance; got {type(cov).__name__}")
        if not isinstance(support, WDMDomain):
            raise TypeError(f"support must be a WDMDomain; got {type(support).__name__}")
```

with

```python
        require_instance("cov", cov, Covariance, "a domainia.covariance.Covariance")
        require_instance("support", support, WDMDomain, "a WDMDomain")
```

then replace

```python
        nu = np.asarray(self.nu)
        if nu.dtype.kind not in "iuf":
            raise TypeError(f"nu must be a real numeric array; got dtype {nu.dtype}")
```

with

```python
        nu = np.asarray(self.nu)
        require_dtype("nu", nu, REAL_KINDS, "a real numeric array")
```

In `operator_matrix`, replace

```python
        raise TypeError(f"op must implement domainia.translate.Operator; got {type(op).__name__}")
```

with

```python
        raise TypeError(f"op must implement domainia.translate.Operator, got {type(op).__name__}")
```

The masked-array `TypeError` (`"matrix: masked arrays are refused (numpy.asarray drops the mask)"`) stays as it is. It names no type, so it has no `got` clause, and its pin (`match="masked"`) and `translate.py`'s two masked refusals use the same form.

- [ ] **Step 5: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_operators_dense.py tests/test_operators_covariance.py 2>&1 | tail -2`
Expected: `71 passed` (44 and 27), with no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "; got" src/domainia/operators.py && grep -n "raise TypeError" src/domainia/operators.py`
Expected: the first grep shows three lines, all `raise ValueError(`: `rtol must be finite and >= 0; got`, `rtol must be < 1; got`, and `every ChannelStack channel needs n = support.size = {s}; got`. The second shows two lines: the masked-array refusal, and `op must implement domainia.translate.Operator, got`.

- [ ] **Step 6: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, with no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: all pass, with no warnings summary.

This task does not touch `wilson/transform.py` or `maps/`, so the float64 bit gate is not run (Conventions).

- [ ] **Step 7: Commit** (sandbox disabled for both git commands)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/operators.py tests/test_operators_dense.py tests/test_operators_covariance.py
git commit -m "operators: DenseOperator keeps one rtol and fails closed

rtol >= 1 raises ValueError, and a given rtol that refuses an eigenvalue
within round-off of zero says that the default accepts it. tol = top * (n * eps)
and S = 0.5 * C + 0.5 * C.T cannot overflow. A non-finite eigenvalue from
eigh and the zero matrix raise ValueError. op.rtol stores the tolerance
used, so dataclasses.replace rebuilds a float32-built operator. The
symmetry factor is 1e-12, or 1e-12 eps32/eps64 for float32 input, and the
refusal prints it. rtol's kind goes through require_real, and the
operators' TypeError messages take the ', got' form.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The message gate and the rule's documents (spec §2.2, §2.11 step 6)

**Goal:** Prove that no `TypeError` message in `src/` keeps the old `"; got"` form, then correct the three documents that still overstate the wrong-kind rule. The fourth, `tests/wilson/test_wrong_kinds.py:1-2`, is now true and stays.

**Assumes the tree after Task 8.** Tasks 2–6 and 8 converted the `TypeError` messages in the files they own.

**Files:**
- Modify: `README.md` (the "## Errors" paragraph)
- Modify: `docs/superpowers/specs/2026-10-31-domainia-general-library-design.md` (§5 "**Errors.**" list)
- Modify: `docs/superpowers/specs/2026-11-04-domainia-units-operators-fixes-design.md` (the Context bullet "The rest of domainia raises `TypeError` for a wrong kind")

**Acceptance Criteria:**
- [ ] The gate script prints `0 TypeError messages with '; got'` (it prints 33 at 779550d).
- [ ] The README's Errors paragraph states the rule as spec §2.1 does, including dtypes, and names its three exceptions: complex values on a time or WDM domain (`ValueError`), complex window samples (`WindowError`), and unchecked `be=` handles.
- [ ] Both specs carry a dated note; their original sentences are kept.
- [ ] `tests/test_readme.py`, the full suite and the NumPy lane pass.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/gate9.py` → `0 TypeError messages with '; got'`

**Steps:**

- [ ] **Step 1: Write and run the gate**

Write `$TMPDIR/gate9.py` (scratch, outside the repo):

```python
import ast
import pathlib

hits = []
for path in sorted(pathlib.Path("src/domainia").rglob("*.py")):
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "TypeError"):
            text = "".join(c.value for c in ast.walk(node.exc)
                           if isinstance(c, ast.Constant) and isinstance(c.value, str))
            if "; got" in text:
                hits.append(f"{path}:{node.lineno}: {text[:70]}")
print(f"{len(hits)} TypeError messages with '; got'", *hits, sep="\n")
```

Run: `cd /Users/aaron/Documents/lisa/domainia && ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/gate9.py`
Expected: `0 TypeError messages with '; got'`. If a line is listed, convert it to the `", got"` form in the task that owns its file (Interface contract, "Kind message form"), grep `tests/` for an anchored `match=` on it and update that too, and record the fix in the errata. The `"; got"` lines that remain in `src/` are all inside `raise ValueError(...)` and keep their text (`grep -rn "; got" src | wc -l` gives about 75).

- [ ] **Step 2: Confirm the three exceptions the README will name**

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P - <<'PYEOF'
import numpy as np
from domainia import translate
from domainia.covariance import DiagonalCovariance
from domainia.domains import TimeDomain
def kind(f):
    try:
        f()
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:60]}"
td = TimeDomain(16, 1.0)
print(kind(lambda: translate.transform({"A": np.ones(16) * 1j}, td, td)))
print(kind(lambda: DiagonalCovariance(lambda nu: nu, 4).factor(np.ones(4), be="numpy")))
PYEOF
```
Expected:
```
ValueError: channel 'A': complex values on a real TimeDomain
AttributeError: 'str' object has no attribute 'name'
```
The third exception, complex window samples, raises `WindowError` (`wilson/windows.py`, spec §3.5's last row).

- [ ] **Step 3: The README's Errors paragraph**

In `README.md`, replace

```markdown
Everything fails closed. `WindowError` (a `ValueError`) means an inadmissible window.
`ValueError` covers shape, finiteness, inactive-cell, endpoint, domain or window
mismatches. `TypeError` means the wrong kind of object.
```

with

```markdown
Everything fails closed. `TypeError` means the wrong kind of argument: a type that can never
be valid in that place (a float or a `bool` where an integer is taken, a string where a flag
is, a `Grid` where a domain is), or an array whose dtype can never be (a complex or bool
array where real numbers are taken). `ValueError` means the right kind with a wrong value:
shape, range, finiteness, integrality, an unknown option string, inactive cells, endpoints,
or a domain or window mismatch. `WindowError` (a `ValueError`) means an inadmissible window,
complex window samples included. Complex values on a time or WDM domain raise `ValueError`,
because they are valid on a `FourierDomain`. Backend handles passed as `be=` are not checked.
```

- [ ] **Step 4: The two specs**

In `docs/superpowers/specs/2026-10-31-domainia-general-library-design.md`, replace

```markdown
* `TypeError` for a wrong kind of object.
```

with

```markdown
* `TypeError` for a wrong kind of object.
  *(Note, 2026-11-08: this held only in parts of the package until spec 2026-11-08 §2, which
  makes it hold everywhere and counts an array's dtype as its kind. Its exceptions: complex
  values on a time or WDM domain raise `ValueError`, complex window samples raise
  `WindowError`, and `be=` backend handles are not checked.)*
```

In `docs/superpowers/specs/2026-11-04-domainia-units-operators-fixes-design.md`, replace

```markdown
  * The rest of domainia raises `TypeError` for a wrong kind (`domains._int`).
```

with

```markdown
  * The rest of domainia raises `TypeError` for a wrong kind (`domains._int`).
    *(Note, 2026-11-08: overstated. At 779550d about 24 entry points raised a bare
    `AttributeError` and about 26 scalar sites converted silently; spec 2026-11-08 §2 makes
    it hold.)*
```

`tests/wilson/test_wrong_kinds.py:1-2` ("refuses a wrong kind with `TypeError` … as the rest of domainia does") is true after Tasks 2–8. It is not changed.

- [ ] **Step 5: Run the README gate and the suites**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_readme.py 2>&1 | tail -1`
Expected: all pass (the paragraph holds no python block).

Run the full suite and the NumPy lane (Conventions). Expected: both pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/aaron/Documents/lisa/domainia
git add README.md docs/superpowers/specs/2026-10-31-domainia-general-library-design.md docs/superpowers/specs/2026-11-04-domainia-units-operators-fixes-design.md
git commit -m "Docs: the wrong-kind rule as it now holds

No TypeError message in src keeps the '; got' form. The README states the
rule with dtypes and names its three exceptions; the 2026-10-31 and
2026-11-04 specs get dated notes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(sandbox disabled for both git commands)

---

### Task 10: Maps compute in double (spec §1.1–§1.3, §1.5, §1.6)

**Goal:** A map that computes returns float64 or complex128 whatever precision it is given, integers included. A map that only selects, slices or zero-fills returns its input's dtype, except that `BinSelect.adjoint` returns the input's complex counterpart. The five methods that take a real array refuse a complex, bool, str or object dtype with `TypeError` before they convert to float64. `FourierToWDM(impl="fft")` runs `forward_bands` in double, so complex64 bins at a large `dt` no longer overflow to NaN. `maps/base.py` states the rule, and the docstrings that promised single precision or an unqualified bitwise identity follow it. float64 results do not change by a single bit.

**Assumes the tree after Task 9.** Task 4 edited the constructors, the `rows` sites and the `_checks` import lines of `maps/`. It did not touch the `apply`/`adjoint` methods, `BinSelect.adjoint`, the `FourierToWDM` docstring or the module docstrings. Tasks 5–9 do not touch `src/domainia/maps/`. Task 2's scope in `wilson/transform.py` (the `k0` ints, `_check_interior_bands`, the `wmn`/`w` `TypeError`s, messages) does not reach the module docstring, so Step 8's old text is as at 779550d. Task 3 made `FourierDomain` store `dt` and `sigma0` as Python `float`s after `require_real` (spec §1.5): this task only checks that, in a test that passes from the start.

**Files:**
- Modify: `src/domainia/maps/base.py` (the module docstring: the rule paragraph after the `built_for` paragraph; `BinSelect.adjoint`)
- Modify: `src/domainia/maps/time.py` (the module docstring; the `_checks` import; `TimeToWDM.apply` and `.adjoint`; `WDMToTime.apply` and `.adjoint`; the `TimeToFourier` docstring, `.apply`, `.apply_complex` and `.adjoint`)
- Modify: `src/domainia/maps/fourier.py` (the `FourierToWDM` docstring; the fft branch of `FourierToWDM.apply`)
- Modify: `src/domainia/wilson/transform.py` (the module docstring's "Where it reaches, and where it stops" list only)
- Create: `tests/test_dom_map_dtypes.py`
- No existing test changes (Step 2).

**Acceptance Criteria:**
- [ ] `maps/base.py`'s module docstring holds spec §1.1's rule sentence word for word (with RST double backticks), then one sentence on the `"real"` push through `Identity` (Step 5) and one on the `TypeError` of the real inputs.
- [ ] `TimeToWDM.apply` (`x`), `TimeToWDM.adjoint` (`y`), `WDMToTime.apply` (`y`), `WDMToTime.adjoint` (`x`) and `TimeToFourier.apply` (`x`) call `require_dtype(name, …, REAL_KINDS, "a real numeric array")` first, then `be.xp.asarray(…, dtype=np.float64)`. A complex64, complex128 or bool input raises `TypeError("{name} must be a real numeric array, got dtype {dtype}")` on NumPy and on JAX, eager and under `jit`.
- [ ] `TimeToFourier.apply_complex` converts complex input to complex128 and real input to float64. `TimeToFourier.adjoint` converts `y` to complex128. `FourierToWDM(impl="fft")` converts `z` to complex128 before `forward_bands`. `BinSelect.adjoint` allocates `np.result_type(np.dtype(y.dtype), np.complex64)`: float32 gives complex64, float64 and integers complex128, complex input keeps its dtype.
- [ ] The dtype table (122 rows: every public map method that takes data, and `push(...).rows`) holds on NumPy and on JAX with x64 on.
- [ ] At `dt = 1e4`, `FourierToWDM(impl="fft").apply` of complex64 bins of amplitude 1e36 returns finite float64 cells, with no warning, equal to the dense path's to 1e-12 of their maximum.
- [ ] `TimeToWDM.adjoint` and `WDMToTime.adjoint` of float32 input return float64, bit for bit the float64 run's, whether `wilson.Grid` holds `dt` as a `float` or an `np.float64` (the `Grid.dt` leak is gone; `wilson.Grid` is unchanged).
- [ ] `tests/test_dom_map_dtypes.py`: `279 passed` (NumPy lane: `141 passed, 138 skipped`). The existing maps tests pass unchanged. The full suite and the NumPy lane pass. The float64 bit gate prints `0 of 1504 arrays differ`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_map_dtypes.py` → `279 passed`; then the bit gate (Step 10) → `0 of 1504 arrays differ`

**Steps:**

- [ ] **Step 1: Grep the sites in `src/domainia/maps`**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "be.asarray(\|dtype=complex" src/domainia/maps/*.py`
Expected: 19 hits.
- `be.asarray(` (15 hits). Six change in Step 6: `TimeToWDM.apply` (`x`), `TimeToWDM.adjoint` (`y`), `WDMToTime.apply` (`y`), and the three `TimeToFourier` methods (`apply`, `apply_complex`, `adjoint`). Nine stay. `on_backend` (`base.py`) moves constants. `ZeroPadTime.apply` (`time.py`) zero-fills, so it keeps the input's dtype. The five in `FourierZeroPad` and `FourierCrop` (`fourier.py`), and `FourierToWDM.apply` and `.adjoint`, are double already through their complex128 constants. `FourierToWDM.apply`'s `z = be.asarray(z)` also feeds the dense path, so it stays, and Step 7 converts `z` on the fft path only.
- `dtype=complex` (4 hits). `BinSelect.adjoint` (`base.py`) changes in Step 5. Three stay: `FourierToWDM.adjoint` (`fourier.py`; its `part` is complex128 through `C`), the spectrum of `TimeToFourier.adjoint` (`time.py`; complex128 is now the rule), and `orthonormal_rows` (`time.py`; host constants).

`WDMToTime.adjoint` has no `be.asarray`: it hands `x` to `TimeToWDM.apply`. It still gets its own check and conversion in Step 6, so each of the five methods checks its own argument.

What these methods do today (probed on NumPy with Task 4 applied, which does not touch them; JAX with x64 on gives the same dtypes):
- `TimeToWDM`, `WDMToTime` and `TimeToFourier.apply` follow float32 input. `FourierToWDM(impl="fft")` returns float32 for complex64 bins. `TimeToFourier.apply_complex` returns complex64 for float32 or complex64 input. `BinSelect.adjoint` always returns complex128.
- Complex input to the five real-taking methods is refused only by accident: NumPy's rfft `TypeError` (`ufunc 'rfft_n_even' not supported for the input types, …`), wilson's `wmn` check, or JAX's `ValueError: only real valued inputs supported for rfft`. A bool array is refused only by the `wmn` check (`TimeToWDM.adjoint`, `WDMToTime.apply`), and not at all by the other three.
- `TimeToWDM.adjoint` multiplies by `g.dt` and `WDMToTime.adjoint` divides by it. `wilson.Grid` keeps the raw `dt` type, so with `dt=np.float64(0.5)` their float32 results come back float64 (after a single-precision transform), and with `dt=0.5` they stay float32.
- With `dt = 1e4`, `FourierToWDM(impl="fft").apply` of complex64 bins of amplitude 1e36 returns NaN, with overflow `RuntimeWarning`s from `phi.astype(rdtype) * z[..., ell]` in `forward_bands`. The dense path returns finite cells (the largest is 3.6e37 for the bins of Step 3's large-`dt` test, which fits float32).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "require_dtype\|REAL_KINDS\|be.xp.asarray" src/domainia/maps/*.py`
Expected: no output.

- [ ] **Step 2: Grep the tests that pin a map's dtype**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "complex64\|float32" tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_fourier_time.py`
Expected: no output.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rlE "TimeToWDM|WDMToTime|TimeToFourier|FourierToWDM|BinSelect|FourierCrop|FourierZeroPad|CropTime|ZeroPadTime|Restrict\(|Identity\(" tests --include='*.py' | xargs grep -c "complex64\|float32" | grep -v ":0$"`
Expected: 2 files.
- `tests/test_wrong_kinds_maps.py:1`: `np.float32(0.25)` as `fmin` in `test_fmin_is_stored_as_a_float`, a scalar, not data. No change.
- `tests/wilson/test_single_precision.py:32`: it calls the wilson kernels directly, which keep single precision (spec §1.3). `FourierToWDM` appears only in its module docstring's prose. No change.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rln "must be real\|only real valued\|rfft_n_even" tests --include='*.py' | grep -v oracle`
Expected: `tests/test_translate.py` (`translate`'s own DC/Nyquist checks, which do not use the maps). No hit goes through a map, so nothing changes.

A bare `pytest.raises` or a dtype assertion through a map would not show in these greps. With Steps 5–8 applied and no test edits, the full suite and the NumPy lane have no failure from these edits: no existing test pins a map's single-precision output, a complex input to a real-taking method, or `BinSelect.adjoint`'s complex128.

- [ ] **Step 3: Write the failing tests `tests/test_dom_map_dtypes.py`**

```python
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
```

The table's inputs follow spec §1.6: float32, float64 and int64 where a method takes reals, complex64 and complex128 where it takes complex. `BinSelect.adjoint` takes all five, because spec §1.2 names the outcome of each. `TimeToFourier.adjoint` also takes a float32 `y` (spec §1.2: "a real `y` converts too"). For `push.rows` the input dtype is the covariance's. A `"real"` push reads `cov.matvec` on the map's float64 unit rows, so it is float64 even through `Identity(time)`. A `"complex"` push (`Identity(bins)`, `BinSelect`) slices `cov.dense()` and keeps its dtype. The JAX legs are separate `jax`-marked tests, so they skip in the NumPy lane. The `TypeError` matches are anchored: a bare `pytest.raises(TypeError)` would pass on NumPy's rfft error today. Under `jit` the match has no `$`, because JAX appends a traceback note to an exception raised while tracing.

- [ ] **Step 4: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_map_dtypes.py 2>&1 | tail -1`
Expected: `58 failed, 221 passed`.
- `test_the_output_dtype_of_every_map_method`: 12 of the 122 fail, and the same 12 of `test_the_output_dtype_of_every_map_method_on_jax`.
  - The float32 case of `TimeToWDM.apply`, `TimeToWDM.adjoint`, `WDMToTime.apply`, `WDMToTime.adjoint`, `Compose.apply` and `Compose.adjoint`: `assert dtype('float32') == dtype('float64')`.
  - `TimeToFourier.apply-float32`, `TimeToFourier.apply_complex-float32` and `TimeToFourier.apply_complex-complex64`: `assert dtype('complex64') == dtype('complex128')`.
  - `FourierToWDM(fft).apply-complex64`: `assert dtype('float32') == dtype('float64')`.
  - `BinSelect.adjoint-float32` and `BinSelect.adjoint-complex64`: `assert dtype('complex128') == dtype('complex64')`.
  - The other 110 pass today. The select, slice and zero-fill maps follow their input. `FourierZeroPad`, `FourierCrop`, the dense, `restrict=True` and `after()` maps, and every `FourierToWDM` adjoint are double through their complex128 constants. int64 input promotes to double. `TimeToFourier.adjoint` allocates complex128. Every push is already double, or keeps the covariance's dtype.
- `test_a_real_taking_method_refuses_a_complex_or_bool_input` (15) and its `_on_jax` twin (15): all fail. The bool cases give `Failed: DID NOT RAISE <class 'TypeError'>` on both backends, except the two that reach the `wmn` check (`TimeToWDM.adjoint`, `WDMToTime.apply`): they raise its `TypeError` `wmn must be a real numeric array, got dtype bool`, which does not match. The complex cases raise the accidental refusals of Step 1, which do not match the anchored pattern: NumPy's rfft `TypeError` (`TimeToWDM.apply`, `WDMToTime.adjoint`, `TimeToFourier.apply`), JAX's `ValueError: only real valued inputs supported for rfft` (the same three), or wilson's `wmn` check (`TimeToWDM.adjoint`, `WDMToTime.apply`). That check is Task 2's `TypeError` `wmn must be a real numeric array, got dtype complex64` (or `complex128`), which does not match either.
- `test_the_conversion_runs_under_jit`: `assert (True and dtype('float32') == <class 'numpy.float64'>)`.
- `test_large_dt_complex64_bins_give_finite_float64_cells_on_the_fft_path`: `RuntimeWarning: overflow encountered in multiply` from `src/domainia/wilson/transform.py` (`forward_bands`), raised as an error by the test's warnings filter.
- `test_the_time_maps_do_not_follow_the_grid_dt_type[float]`: `assert dtype('float32') == <class 'numpy.float64'>`. `[np.float64]`: the dtype is float64 already (the leak), but the values come from a single-precision transform, so `assert_array_equal` fails.
- `test_fourier_domains_store_python_floats` passes: Task 3 stores Python floats (spec §1.5 is checked here, not changed).

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_map_dtypes.py 2>&1 | tail -1`
Expected: `30 failed, 111 passed, 138 skipped` (the 138 `jax`-marked cases skip).

- [ ] **Step 5: `maps/base.py`: the rule and `BinSelect.adjoint` (`src/domainia/maps/base.py`)**

The module docstring: the rule, as a new paragraph after the `built_for` paragraph and before "Map kinds". Its first two sentences are spec §1.1's rule sentence word for word, with RST double backticks. The sentence in parentheses after it records what a `"real"` push through `Identity` returns (Step 3's table pins it): spec §1.1's "a push through `Identity` … returns the dtype of the covariance's `dense()`" holds only for an `Identity` on a `FourierDomain`, whose push is `"complex"`.

```python
``dataclasses.replace(map, dst=...)`` raises ``ValueError`` instead of reusing constants
built for the old domain.

```
→
```python
``dataclasses.replace(map, dst=...)`` raises ``ValueError`` instead of reusing constants
built for the old domain.

Maps that compute (``TimeToWDM``, ``WDMToTime``, ``TimeToFourier``, ``FourierToWDM``,
``FourierZeroPad``, ``FourierCrop``), and a push through one of them, return
float64/complex128 whatever the input's precision; maps that only select, slice or
zero-fill (``Identity``, ``CropTime``, ``ZeroPadTime``, ``Restrict``, ``BinSelect``), and their
adjoints, return the input's dtype (``BinSelect.adjoint``: the input's complex
counterpart), and a push through ``Identity`` or ``BinSelect`` returns the dtype of the
covariance's ``dense()``.  ``Compose`` returns what its parts return.  (A push through an
``Identity`` on a ``TimeDomain`` or a ``WDMDomain`` is a ``"real"`` push: it returns what
``cov.matvec`` returns on float64 unit rows, float64 for a float32 matrix.)  The real inputs
of ``TimeToWDM``, ``WDMToTime`` and ``TimeToFourier.apply`` refuse a complex, bool, str or
object dtype with ``TypeError``.

```

`BinSelect.adjoint`: the input's complex counterpart, computed on the host as a NumPy dtype, so NumPy and JAX allocate the same dtype. `be.set` then writes `y` without a cast: complex64 stays complex64, float32 gives complex64, float64 and integers give complex128.

```python
        out = be.xp.zeros((*y.shape[:-1], self.src.n), dtype=complex)
        return be.set(out, (..., slice(self.offset, self.offset + self.dst.n)), y, copy=False)
```
→
```python
        # the input's complex counterpart, computed on the host so both backends agree
        out = be.xp.zeros((*y.shape[:-1], self.src.n),
                          dtype=np.result_type(np.dtype(y.dtype), np.complex64))
        return be.set(out, (..., slice(self.offset, self.offset + self.dst.n)), y, copy=False)
```

No import changes: `base.py` already imports `numpy as np`.

- [ ] **Step 6: `maps/time.py`: the five real-taking methods, `apply_complex` and `TimeToFourier.adjoint` (`src/domainia/maps/time.py`)**

The module docstring: the bitwise identity holds for float64 input, and the real inputs refuse a wrong dtype.

```python
Every evaluator here makes the library call of the WDM Paper II toy models it was ported
from (they do not ship), so the NumPy path is bitwise:
``TimeToWDM.apply(d) == forward_time(d, window=dst.window)[mask][keep]``.  The Wilson window
is the ``WDMDomain``'s (``dst.window`` of ``TimeToWDM``, ``src.window`` of ``WDMToTime``).
"""
```
→
```python
Every evaluator here makes the library call of the WDM Paper II toy models it was ported
from (they do not ship), so for float64 input (other inputs are converted to float64 first)
the NumPy path is bitwise:
``TimeToWDM.apply(d) == forward_time(d, window=dst.window)[mask][keep]``.  The Wilson window
is the ``WDMDomain``'s (``dst.window`` of ``TimeToWDM``, ``src.window`` of ``WDMToTime``).

``TimeToWDM``, ``WDMToTime`` and ``TimeToFourier.apply`` take real arrays: a complex, bool,
str or object dtype raises ``TypeError`` before the conversion, which would otherwise drop
an imaginary part with only a ``ComplexWarning``.
"""
```

The `_checks` import (Task 4's line) gains `REAL_KINDS` and `require_dtype`:

```python
from domainia.wilson._checks import index_array, require_instance, require_int, require_real
```
→
```python
from domainia.wilson._checks import (REAL_KINDS, index_array, require_dtype, require_instance,
                                     require_int, require_real)
```

`TimeToWDM.apply`, its head:

```python
    def apply(self, x, be=None):
        be = self._be(x, be)
        x = be.asarray(x)
        g = self.grid
        if x.ndim == 1:
```
→
```python
    def apply(self, x, be=None):
        require_dtype("x", x, REAL_KINDS, "a real numeric array")
        be = self._be(x, be)
        x = be.xp.asarray(x, dtype=np.float64)
        g = self.grid
        if x.ndim == 1:
```

`TimeToWDM.adjoint`, its head. With `y` in float64, `g.dt * …` is float64 whatever `g.dt`'s type: the `Grid.dt` leak is gone.

```python
    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        g = self.grid
        x = g.dt * wilson.inverse_time_many(self._scatter(y, be), g.Nf, g.Nt, g.dt, be=be,
```
→
```python
    def adjoint(self, y, be=None):
        require_dtype("y", y, REAL_KINDS, "a real numeric array")
        be = self._be(y, be)
        y = be.xp.asarray(y, dtype=np.float64)
        g = self.grid
        x = g.dt * wilson.inverse_time_many(self._scatter(y, be), g.Nf, g.Nt, g.dt, be=be,
```

`WDMToTime.apply`, its head:

```python
    def apply(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        g = self.grid
        Y = y.reshape(-1, y.shape[-1])
```
→
```python
    def apply(self, y, be=None):
        require_dtype("y", y, REAL_KINDS, "a real numeric array")
        be = self._be(y, be)
        y = be.xp.asarray(y, dtype=np.float64)
        g = self.grid
        Y = y.reshape(-1, y.shape[-1])
```

`WDMToTime.adjoint`, its head. `TimeToWDM.apply` would check and convert `x` too. The method does it itself so that each of the five checks its own argument, and `/ self.grid.dt` then divides a float64 array (the second half of the leak):

```python
    def adjoint(self, x, be=None):
        be = self._be(x, be)
        fwd = TimeToWDM(self.dst, self.src, m_idx=self.m_idx, n_idx=self.n_idx,
```
→
```python
    def adjoint(self, x, be=None):
        require_dtype("x", x, REAL_KINDS, "a real numeric array")
        be = self._be(x, be)
        x = be.xp.asarray(x, dtype=np.float64)
        fwd = TimeToWDM(self.dst, self.src, m_idx=self.m_idx, n_idx=self.n_idx,
```

The `TimeToFourier` docstring, its second paragraph: the bitwise statement holds for float64 input.

```python
    ``apply`` returns the complex bins exactly as the ported toy models compute them
    (``np.fft.rfft(r)[keep]``), because the Whittle density's ``2/N |R|^2`` order is
    bitwise only on them.  Their real orthonormal coordinates are
    ``sqrt(2/N) (Re R_k, -Im R_k)``, bin by bin, with rows :meth:`orthonormal_rows`.
    ``dst`` has ``units="rfft"``.
```
→
```python
    For float64 input (other inputs are converted to float64 first), ``apply`` returns the
    complex bins exactly as the ported toy models compute them (``np.fft.rfft(r)[keep]``),
    because the Whittle density's ``2/N |R|^2`` order is bitwise only on them.  Their real
    orthonormal coordinates are ``sqrt(2/N) (Re R_k, -Im R_k)``, bin by bin, with rows
    :meth:`orthonormal_rows`.  ``dst`` has ``units="rfft"``.
```

`TimeToFourier.apply`, `.apply_complex` and the head of `.adjoint`. `apply` checks the kind before `_last_axis` checks the length. `np.iscomplexobj` reads the dtype of a list, a NumPy or JAX array or a tracer without converting it. `adjoint`'s spectrum stays `dtype=complex`; with `y` in complex128, `be.set` writes it without a cast.

```python
    def apply(self, x, be=None):
        _last_axis(x, self.src.N, "TimeToFourier.apply")
        be = self._be(x, be)
        k0, n = self.dst.k0, self.dst.n
        return be.fft.rfft(be.asarray(x), axis=-1)[..., k0:k0 + n]

    def apply_complex(self, x, be=None):
        """``F x`` for a complex series (``fft``, the kept bins)."""
        _last_axis(x, self.src.N, "TimeToFourier.apply_complex")
        be = self._be(x, be)
        k0, n = self.dst.k0, self.dst.n
        return be.fft.fft(be.asarray(x), axis=-1)[..., k0:k0 + n]

    def adjoint(self, y, be=None):
        """``Re(F^H y)``: the real-inner-product adjoint, ``(N/2) irfft`` of the kept bins."""
        _last_axis(y, self.dst.n, "TimeToFourier.adjoint")
        be = self._be(y, be)
        y = be.asarray(y)
```
→
```python
    def apply(self, x, be=None):
        require_dtype("x", x, REAL_KINDS, "a real numeric array")
        _last_axis(x, self.src.N, "TimeToFourier.apply")
        be = self._be(x, be)
        k0, n = self.dst.k0, self.dst.n
        return be.fft.rfft(be.xp.asarray(x, dtype=np.float64), axis=-1)[..., k0:k0 + n]

    def apply_complex(self, x, be=None):
        """``F x`` for a complex series (``fft``, the kept bins), in complex128: complex input
        is converted to complex128 first, real input to float64."""
        _last_axis(x, self.src.N, "TimeToFourier.apply_complex")
        be = self._be(x, be)
        k0, n = self.dst.k0, self.dst.n
        x = be.xp.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
        return be.fft.fft(x, axis=-1)[..., k0:k0 + n]

    def adjoint(self, y, be=None):
        """``Re(F^H y)``: the real-inner-product adjoint, ``(N/2) irfft`` of the kept bins.
        ``y`` is converted to complex128 first (a real ``y`` too), so the result is float64."""
        _last_axis(y, self.dst.n, "TimeToFourier.adjoint")
        be = self._be(y, be)
        y = be.xp.asarray(y, dtype=np.complex128)
```

- [ ] **Step 7: `maps/fourier.py`: the fft path of `FourierToWDM` (`src/domainia/maps/fourier.py`)**

The `FourierToWDM` docstring: the single-precision fft route is gone.

```python
    ``impl="fft"`` evaluates ``src.rfft_scale * wilson.transform.forward_bands`` (opt-in).
    For complex128 bins it agrees with dense to <= 1e-12 relative; complex64 bins run
    ``forward_bands`` in single precision on the fft path, so it returns float32 where dense
    returns float64.  For a record ``x`` and its bins ``z`` in ``src.units``, the cells equal
    ``TimeToWDM.apply(x)`` on the same cells.
```
→
```python
    ``impl="fft"`` evaluates ``src.rfft_scale * wilson.transform.forward_bands`` (opt-in) on
    the bins converted to complex128 first.  Both impls return float64 cells whatever the
    bins' precision, and they agree to <= 1e-12 relative.  For a record ``x`` and its bins
    ``z`` in ``src.units``, the cells equal ``TimeToWDM.apply(x)`` on the same cells.
```

`FourierToWDM.apply`, the fft branch. The dense branch above it returns first and is unchanged: `z @ self.C.T` is double through the complex128 `C`. A complex128 `z` is not copied, so the float64 bits do not move.

```python
        g = self.dst.cells.grid
        geo = _geometry(self.dst)
        w = wilson.transform.forward_bands(z, self.src.k0, geo["bands"], g.Nf, g.Nt, g.dt, be=be,
```
→
```python
        g = self.dst.cells.grid
        geo = _geometry(self.dst)
        z = be.xp.asarray(z, dtype=np.complex128)        # double, whatever the bins' precision
        w = wilson.transform.forward_bands(z, self.src.k0, geo["bands"], g.Nf, g.Nt, g.dt, be=be,
```

No import changes.

- [ ] **Step 8: `wilson/transform.py`: "Where it reaches, and where it stops" (`src/domainia/wilson/transform.py`)**

The module docstring's list gains one item, after the `matrix`/`stationary` item. The rest of the docstring is unchanged; Task 11 owns its "Precision" text.

```python
* **``matrix`` and ``stationary`` stay double** on purpose. They are $O(N^2)$
  verification and covariance machinery, where precision *is* the deliverable
  and the arrays are too large to want in the first place.
```
→
```python
* **``matrix`` and ``stationary`` stay double** on purpose. They are $O(N^2)$
  verification and covariance machinery, where precision *is* the deliverable
  and the arrays are too large to want in the first place.
* **The maps of ``domainia.maps`` compute in double.** ``TimeToWDM``,
  ``WDMToTime``, ``TimeToFourier`` and ``FourierToWDM(impl="fft")`` convert their
  input to float64 (bins to complex128) before they call these kernels, so a
  ``float32`` record or ``complex64`` bins give float64 results there.  Single
  precision is reached only by calling the kernels here directly.
```

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "require_dtype(\|be.xp.asarray(\|np.result_type(np.dtype(y.dtype)" src/domainia/maps/*.py`
Expected: 14 hits. `time.py` (12): the five `require_dtype(` calls, and the seven `be.xp.asarray(` conversions of the same five methods, `apply_complex` and `TimeToFourier.adjoint`. `base.py` (1): the `BinSelect.adjoint` allocation. `fourier.py` (1): the fft branch of `FourierToWDM.apply`.

- [ ] **Step 9: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_map_dtypes.py 2>&1 | tail -1`
Expected: `279 passed`, no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_map_dtypes.py 2>&1 | tail -1`
Expected: `141 passed, 138 skipped`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_maps_base.py tests/test_dom_maps_dirichlet.py tests/test_dom_maps_fourier_time.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_wdm.py tests/test_dom_built_for.py tests/test_dom_windows.py tests/test_wrong_kinds_maps.py 2>&1 | tail -1`
Expected: `326 passed`, the same count as before Step 5. A failure here is a pin that Step 2 did not find: if it expects the old dtype, change it to the rule's dtype (Step 3's table), and record it in the errata.

- [ ] **Step 10: Full suite, NumPy lane and the float64 bit gate**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes (the oracle tests skip), 0 errors, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, 0 errors, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz`
Expected: `0 of 1504 arrays differ`. Each conversion is a no-op on float64 and complex128 input (`asarray` returns the same array, no copy), and `BinSelect.adjoint` still allocates complex128 for it. If an array differs, compare the dtype of each map method's input with and without Steps 5–7 on the case that differs.

- [ ] **Step 11: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/maps/base.py src/domainia/maps/time.py src/domainia/maps/fourier.py src/domainia/wilson/transform.py tests/test_dom_map_dtypes.py && git commit -m "domainia.maps: the maps compute in double

TimeToWDM, WDMToTime and TimeToFourier.apply refuse a complex, bool, str
or object dtype with TypeError (require_dtype), then convert to float64.
float32 input no longer runs the transform in single precision; complex
input is no longer refused by accident (NumPy's rfft TypeError, wilson's
wmn check, JAX's ValueError) or a bool array taken silently; and the
Grid.dt leak of the two adjoints is gone. TimeToFourier.apply_complex
converts to complex128 (real input to float64), TimeToFourier.adjoint
converts y to complex128, and the fft path of FourierToWDM converts the
bins to complex128 before forward_bands, so complex64 bins at dt=1e4 no
longer overflow to NaN. BinSelect.adjoint returns the input's complex
counterpart. maps/base.py states the rule; the FourierToWDM, maps/time.py,
TimeToFourier and wilson.transform docstrings follow it. float64 results
are unchanged bit for bit. tests/test_dom_map_dtypes.py tables every map
method that takes data, on NumPy and JAX.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: The single-precision kernel guard (spec §1.4, §1.6)

**Goal:** A single-precision run of a `domainia.wilson` transform entry point that produces a non-finite value from finite input raises `ValueError` naming the entry point and suggesting float64 input, instead of returning `nan` with only NumPy's `RuntimeWarning`s. Each of the ten entry points checks the arrays it computes itself, through one new helper, `_check_single_output(out, entry)`. The check runs only on concrete single-precision arrays, so float64 runs gain no pass and `jax.jit` is unchanged. float64 results do not change by a single bit.

**Assumes the tree after Task 10.** In `wilson/transform.py`, Task 2 changed the `_checks` import, `inverse` (its Raises, one doctest, the `wmn` check), `inverse_many` (the `wmn` check), `atom_spectrum`, `_check_interior_bands`, `forward_bands` (`k0`) and `inverse_bands` (`k0`, `n_bins`, `w`). Task 10 added one item to the module docstring's "Where it reaches, and where it stops" list, and left the "Precision" text to this task. Tasks 3–9 do not touch `transform.py`. No earlier task touches `tests/wilson/test_single_precision.py`. None of the text this task anchors to was changed: the first paragraph of the "Precision" section, `_check_finite`, and the last lines of the ten entry points. Task 10 made `FourierToWDM(impl="fft")` convert its bins to complex128, so its large-`dt` test (`tests/test_dom_map_dtypes.py`) runs `forward_bands` in double and the guard does not fire there.

**Files:**
- Modify: `src/domainia/wilson/transform.py` (one paragraph in the module docstring's "Precision" section; the new `_check_single_output` after `_check_finite`; the last lines of `forward`, `forward_many`, `forward_time_many`, `inverse`, `inverse_many`, `inverse_time_many`, `forward_time`, `inverse_time`, `forward_bands` and `inverse_bands`)
- Modify: `tests/wilson/test_single_precision.py` (`import re`; the guard tests appended at the end)
- No other test changes (Step 2).

**Acceptance Criteria:**
- [ ] `wilson.transform._check_single_output(out, entry)` returns at once when `out`'s working dtype (`_working_dtypes`) is double, or when `out` is a JAX tracer. Otherwise, if any entry of `out` is `nan` or `inf`, it raises `ValueError("{entry}: {k} of {size} entries of the single-precision result are not finite ({n} nan, {i} inf), from finite input: the computation overflowed float32, whose largest value is 3.4e38. Pass float64 (or complex128) input to compute in double.")`. It does not change `np.errstate`.
- [ ] The ten entry points call it once each (spec §1.4): `forward` and `forward_many` on `_gather_bands`' result; `forward_time` and `forward_time_many` on their rfft, before they delegate; `inverse` and `inverse_many` on their bins; `inverse_time` and `inverse_time_many` on the irfft; `forward_bands` and `inverse_bands` on their results. For each, a float32/complex64 input that is finite but overflows inside the entry point raises `ValueError` matching `^{entry}: .* single-precision result .*float64`, and the same input in double gives a finite result.
- [ ] `forward_bands` with complex64 bins of amplitude 1e36 at `dt=1e4` raises; the complex128 call returns finite cells below 3.4e38. `forward_time` with float32 `np.full(N, 1e37)` raises `ValueError` matching `^forward_time: .*float64`, not `forward`'s "Fill or excise the gaps". `inverse_time` of float32 cells whose `inverse` is finite but whose irfft overflows raises `ValueError` naming `inverse_time`.
- [ ] On JAX, each entry point run eagerly on concrete float32/complex64 arrays raises the same `ValueError`. Under `jax.jit` nothing raises, and the result holds `inf`/`nan`.
- [ ] `inner_product`, `stationary`, `evolutionary` and `matrix` are unchanged (they inherit the guard). The variance builders are not covered.
- [ ] `tests/wilson/test_single_precision.py`: `84 passed` (NumPy lane: `63 passed, 21 skipped`). `src/domainia/wilson` still has 72 doctest items. The full suite and the NumPy lane pass with no warnings summary. The float64 bit gate prints `0 of 1504 arrays differ`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_single_precision.py` → `84 passed`; then the bit gate (Step 10) → `0 of 1504 arrays differ`

**Steps:**

- [ ] **Step 1: Grep the sites in `src/domainia/wilson/transform.py`**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "_check_finite(\|return _gather_bands\|be.fft.rfft(x\|be.fft.irfft(\|^    return X$\|^    return be.xp.real\|^    return out \*" src/domainia/wilson/transform.py`
Expected: 21 hits (line numbers as after Task 10; Task 2 and Task 10 moved them a few lines from 779550d).

```text
199:def _check_finite(values: NDArray[np.generic], context: str) -> None:
371:    return X
400:    return X
541:    _check_finite(x_rfft, "forward: x_rfft")
542:    return _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
606:    _check_finite(x_rfft, "forward_many: x_rfft")
607:    return _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
653:    _check_finite(x, "forward_time_many: x")
654:    return forward_many(be.fft.rfft(x, axis=-1), Nf, Nt, dt, be=be, window=window)
722:    _check_finite(wmn, "inverse: wmn")
733:    return X
786:    _check_finite(wmn, "inverse_many: wmn")
796:    return X
836:    return be.fft.irfft(inverse_many(wmn, Nf, Nt, dt, be=be, window=window), n=N, axis=-1)
899:    _check_finite(x, "forward_time: x")
900:    return forward(be.fft.rfft(x), Nf, Nt, dt, be=be, window=window)
945:    return be.fft.irfft(x_rfft, n=N)
1159:    _check_finite(z, "forward_bands: z")
1165:    return be.xp.real(phase * Y) * rdtype.type(np.sqrt(2.0 * dt / n_samples(Nf, Nt)))
1214:    _check_finite(w, "inverse_bands: w")
1224:    return out * rdtype.type(np.sqrt(n_samples(Nf, Nt) / (2.0 * dt)))
```

- `_check_finite` (line 199): the new helper goes after it (Step 5).
- Lines 371 and 400 are `_scatter_interior_bands` and `_scatter_edge_bands`, helpers that `inverse` and `inverse_many` call. They stay: the entry points check what the helpers return.
- Each of the ten entry points has one result line here, and Step 6 changes it. `forward`, `forward_many`, `forward_time_many`, `forward_time`, `forward_bands` and `inverse_bands` also show their input `_check_finite`: the guard comes after it, so a `nan` or `inf` the guard finds was made by the transform. `inverse` and `inverse_many` show it too, ten lines above their `return X`.
- `inverse_time` and `inverse_time_many` have no input check of their own. `inverse`/`inverse_many` check `wmn` and then the bins, so the irfft input is finite when the guard runs.

What the entry points do today (probed on Nf=8, Nt=16, NumPy; eager JAX with x64 on gives the same, except that JAX emits no warning):
- `forward_bands` and `forward` at `dt=1e4` with complex64 bins of amplitude 1e36: every cell comes back `nan` or `inf`, with `RuntimeWarning`s (`overflow encountered in multiply`, `overflow encountered in ifft`, `invalid value encountered in ifft`). The window samples reach 400 at `dt=1e4`, so `phi * z` passes float32's 3.4e38. With complex128 bins the cells are finite: the largest is 2.5e37 for Step 3's bins on bands 2 and 3, which fits float32.
- `forward_time` of float32 `np.full(128, 1e37)`: the rfft's bin 0 is 128 · 1e37 = 1.3e39, which overflows (`RuntimeWarning: overflow encountered in cast`). `forward`'s input check then raises `ValueError: forward: x_rfft: 1 of 65 entries are not finite (0 nan, 1 inf); first at index 0. ... Fill or excise the gaps before transforming.` The record has no gap.
- `inverse_time` of the float32 cells of an impulse: `inverse` is finite, since every bin has modulus equal to the impulse's height, but the irfft forms the unnormalised sum `N · height` before it divides by `N`. Probed at heights 1e36, 3e36, 1e37 and 1e38: 1e36 comes back finite, and from 3e36 on the irfft returns `inf` while `inverse` stays finite. The tests use 1e37.
- The other entry points overflow the same way on inputs that are finite in single precision (Step 3's `_overflowing` table). Each returns `inf`/`nan` today, and the double run of the same input is finite.

- [ ] **Step 2: Grep the tests that could pin the old behaviour**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "Fill or excise\|x_rfft: .*not finite\|_check_single_output" tests src --include='*.py' | grep -v "tests/wilson/oracle"`
Expected: one hit, `src/domainia/wilson/_checks.py`: the last line of `validate_finite`'s message (`"silently. Fill or excise the gaps before transforming."`). No test pins `forward`'s message for an overflowed rfft.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rln "float32\|complex64" tests --include='*.py' | grep -v "tests/wilson/oracle" | sort`
Expected: 13 files. None runs a single-precision transform whose result is not finite, so none changes:
- `tests/test_dom_map_dtypes.py` (Task 10): the maps convert to float64/complex128 before they call the kernels, so no single-precision run reaches the guard. The large-`dt` complex64 case runs `forward_bands` in double.
- `tests/test_operators_dense.py`, `tests/test_translate.py`: float32 `DenseOperator` matrices, and `translate`, which is float64 by contract. No single-precision transform.
- `tests/test_wrong_kinds_domains.py`, `tests/test_wrong_kinds_maps.py`, `tests/wilson/test_wrong_kinds.py`, `tests/wilson/test_checks.py`, `tests/wilson/test_windows.py`: `np.float32` scalars as `dt`, `fmin` or helper arguments, and complex64 as a refused kind.
- `tests/wilson/test_backend.py`: dtype casts of the backend helpers.
- `tests/wilson/test_noise_stationary.py`: float32 PSDs into the variance builders, which the guard does not cover.
- `tests/wilson/test_inner_product_fixes.py`: `test_float32_strain_scale_does_not_underflow` runs `inner_product` on float32 cells of amplitude 1e-24. Its `inverse` calls now run the guard, and their results are finite.
- `tests/wilson/test_single_precision.py`, `tests/wilson/test_transform_shim.py`: float32 runs of the entry points on standard normal draws, on NumPy and JAX, eager and under `jit`. Their results are finite, so the guard passes them.


A bare `pytest.raises` or an `isnan` assertion would not show in these greps. In the scratch prototype, the full suite and the NumPy lane with Steps 5–6 applied had no failure outside the new tests.

`tests/wilson/test_transform_shim.py` imports `output_dtype` from `test_single_precision.py`. Step 3 only appends to that file and adds an import, so `OUTPUT_DTYPE`, `output_dtype` and `_cases` keep their names.

- [ ] **Step 3: Write the failing tests (`tests/wilson/test_single_precision.py`)**

The imports:

```python
import dataclasses

import numpy as np
import pytest
```
→
```python
import dataclasses
import re

import numpy as np
import pytest
```

Append to the end of the file, after `test_jax_float32_runs_are_single_precision`:

```python


# ---------------------------------------------------------------------------
# The single-precision guard (spec 2026-11-08 §1.4): a float32 run that overflows raises
# ---------------------------------------------------------------------------


def _overflowing() -> dict:
    """``{entry point: (a, call(a, be))}``: a double-precision input ``a`` that stays finite
    when rounded to single precision, and whose single-precision run overflows float32
    inside the entry point.  The double-precision run is finite."""
    rng = np.random.default_rng(3)
    flat = np.full(N, 1e37)                    # rfft bin 0 is N * 1e37 = 1.3e39
    impulse = np.zeros(N)
    impulse[37] = 1e37                         # every rfft bin has modulus 1e37
    bins = (rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1)) * 1e36
    w_flat = T.forward_time(flat, NF, NT, DT)  # its inverse has bin 0 = 1.3e39 again
    w_impulse = T.forward_time(impulse, NF, NT, DT)
    w_bands = T.forward_bands(bins * 1e3, 0, BANDS, NF, NT, 1e-4)

    def two(a):
        return np.stack([a, a])

    return {
        "forward_time": (flat, lambda a, be: T.forward_time(a, NF, NT, DT, be=be)),
        "forward_time_many": (two(flat), lambda a, be: T.forward_time_many(a, NF, NT, DT, be=be)),
        "forward": (bins, lambda a, be: T.forward(a, NF, NT, 1e4, be=be)),
        "forward_many": (two(bins), lambda a, be: T.forward_many(a, NF, NT, 1e4, be=be)),
        "inverse": (w_flat, lambda a, be: T.inverse(a, NF, NT, DT, be=be)),
        "inverse_many": (two(w_flat), lambda a, be: T.inverse_many(a, NF, NT, DT, be=be)),
        "inverse_time": (w_impulse, lambda a, be: T.inverse_time(a, NF, NT, DT, be=be)),
        "inverse_time_many": (two(w_impulse),
                              lambda a, be: T.inverse_time_many(a, NF, NT, DT, be=be)),
        "forward_bands": (bins, lambda a, be: T.forward_bands(a, 0, BANDS, NF, NT, 1e4, be=be)),
        "inverse_bands": (w_bands, lambda a, be: T.inverse_bands(a, 0, N // 2 + 1, BANDS, NF, NT,
                                                                 1e-4, be=be)),
    }


def _single(a: np.ndarray) -> np.ndarray:
    """``a`` rounded to single precision (float32, or complex64 for complex ``a``)."""
    return a.astype(np.complex64 if np.iscomplexobj(a) else np.float32)


def test_the_guard_reads_single_precision_results_only():
    """A double result is not scanned at all, so float64 runs gain no pass; a single one is,
    and its message names the entry point and float64."""
    T._check_single_output(np.array([1.0, np.inf, np.nan]), "f")      # float64: skipped
    T._check_single_output(np.array([np.inf + 0j]), "f")              # complex128: skipped
    T._check_single_output(np.ones(3, np.float32), "f")               # finite: passes
    msg = ("f: 2 of 3 entries of the single-precision result are not finite (1 nan, 1 inf), "
           "from finite input: the computation overflowed float32, whose largest value is "
           "3.4e38. Pass float64 (or complex128) input to compute in double.")
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        T._check_single_output(np.array([1.0, np.inf, np.nan], np.float32), "f")
    with pytest.raises(ValueError, match=r"^f: 1 of 2 entries .* \(0 nan, 1 inf\)"):
        T._check_single_output(np.array([1.0, np.inf], np.complex64), "f")


def test_forward_bands_of_complex64_bins_at_a_large_dt_raises():
    """At ``dt = 1e4`` the window samples (max 400) scale complex64 bins of amplitude 1e36
    past float32's 3.4e38 inside ``forward_bands``, which returned NaN cells with only
    ``RuntimeWarning``s.  The cells themselves fit float32: complex128 bins give them."""
    z, call = _overflowing()["forward_bands"]
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="^forward_bands: .*float64"):
            call(z.astype(np.complex64), NUMPY)
    w = call(z, NUMPY)
    assert w.dtype == np.float64 and np.all(np.isfinite(w))
    assert np.max(np.abs(w)) < np.finfo(np.float32).max


def test_forward_time_of_a_record_that_overflows_its_rfft_names_forward_time():
    """The rfft of a float32 record of 1e37 overflows (bin 0 is 1.3e39).  ``forward``'s input
    check used to report it as a gap: "forward: x_rfft: ... Fill or excise the gaps"."""
    x = np.full(N, 1e37, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="^forward_time: .*float64") as info:
            T.forward_time(x, NF, NT, DT)
    assert "Fill or excise" not in str(info.value)


def test_inverse_time_that_overflows_only_in_its_irfft_names_inverse_time():
    """The cells of an impulse of height 1e37: ``inverse`` is finite (every bin has modulus
    1e37), but the irfft's unnormalised sum, N * 1e37 = 1.3e39, overflows float32."""
    w, _ = _overflowing()["inverse_time"]
    w32 = w.astype(np.float32)
    assert np.all(np.isfinite(T.inverse(w32, NF, NT, DT)))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="^inverse_time: .*float64"):
            T.inverse_time(w32, NF, NT, DT)


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_a_single_precision_run_that_overflows_raises_naming_its_entry_point(name):
    """Every entry point checks the arrays it computes.  Its input is finite in single
    precision, so the input check does not fire; the double run is finite."""
    a, call = _overflowing()[name]
    a32 = _single(a)
    assert np.all(np.isfinite(a32))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match=f"^{name}: .* single-precision result .*float64"):
            call(a32, NUMPY)
    out = call(a, NUMPY)
    assert out.dtype == output_dtype(name, np.float64) and np.all(np.isfinite(out))


@pytest.mark.jax
@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_the_guard_runs_on_concrete_jax_arrays_and_is_skipped_under_jit(name, jax_be):
    """Eager JAX runs on concrete float32 arrays, so the guard reads them on the host as it
    reads NumPy ones.  Under ``jax.jit`` the arrays are tracers with no values: nothing
    raises, and the overflow comes back as ``inf``/``nan``."""
    import jax

    a, call = _overflowing()[name]
    a32 = jax_be.asarray(_single(a))
    with pytest.raises(ValueError, match=f"^{name}: .*float64"):
        call(a32, jax_be)
    out = jax.jit(lambda v: call(v, jax_be))(a32)
    assert isinstance(out, jax.Array) and out.dtype == output_dtype(name, np.float32)
    assert not np.all(np.isfinite(np.asarray(out)))
```

`_overflowing` uses the module's grid (`NF, NT, DT = 8, 16, 0.5`) and `BANDS = [2, 3]`. Each input overflows float32 for its own reason:
- `forward_time(_many)`: the rfft's bin 0 (1.3e39).
- `forward(_many)` and `forward_bands`: the window samples at `dt=1e4` times the bins.
- `inverse(_many)`: the bins of the flat record (1.3e39 again).
- `inverse_time(_many)`: the irfft only.
- `inverse_bands`: the bins of cells taken at `dt=1e-4` from bins of amplitude 1e39 (the cells fit float32, the bins they give back do not).

The `_many` forms take two copies of the single input.

NumPy emits overflow `RuntimeWarning`s before the guard runs, so the NumPy calls sit in `np.errstate(over="ignore", invalid="ignore")`, as `tests/test_translate.py`'s `test_a_non_finite_result_is_refused` does. JAX emits none.

The message counts are not pinned on JAX: its rfft of the flat record gives `1 nan, 1 inf` where NumPy's gives `0 nan, 1 inf`.

The matches are anchored at `^{name}: `, so `forward` does not match `forward_many`'s message. A message from an inner call (for example `forward:` from inside `forward_time`) fails the match.

- [ ] **Step 4: Run the tests to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_single_precision.py --tb=line 2>&1 | tail -3`
Expected: `24 failed, 60 passed`, no warnings. The 60 existing tests pass. The 24 new ones fail:
- `test_the_guard_reads_single_precision_results_only`: `AttributeError: module 'domainia.wilson.transform' has no attribute '_check_single_output'`.
- `test_forward_bands_of_complex64_bins_at_a_large_dt_raises` and `test_inverse_time_that_overflows_only_in_its_irfft_names_inverse_time`: `Failed: DID NOT RAISE <class 'ValueError'>`.
- `test_forward_time_of_a_record_that_overflows_its_rfft_names_forward_time`: `AssertionError: Regex pattern did not match.` (the message is `forward: x_rfft: 1 of 65 entries are not finite (0 nan, 1 inf); ... Fill or excise the gaps before transforming.`).
- `test_a_single_precision_run_that_overflows_raises_naming_its_entry_point` (10) and `test_the_guard_runs_on_concrete_jax_arrays_and_is_skipped_under_jit` (10): `[forward_time]` and `[forward_time_many]` with `AssertionError: Regex pattern did not match.` (`forward`'s and `forward_many`'s input messages), the other eight with `Failed: DID NOT RAISE <class 'ValueError'>`.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_single_precision.py 2>&1 | tail -1`
Expected: `14 failed, 49 passed, 21 skipped` (the 10 new JAX tests skip, with the 11 existing ones).

- [ ] **Step 5: `_check_single_output` and the "Precision" paragraph (`src/domainia/wilson/transform.py`)**

The module docstring's "Precision" section gains a paragraph after its first one. The rest of the docstring is unchanged, including Task 10's item in "Where it reaches, and where it stops".

```python
and on JAX.  Anything else, integers included, promotes to double.  It is
opt-in by dtype rather than by flag, so the precision of a result is readable
off its own ``.dtype``.
```
→
```python
and on JAX.  Anything else, integers included, promotes to double.  It is
opt-in by dtype rather than by flag, so the precision of a result is readable
off its own ``.dtype``.

A single-precision run that overflows raises ``ValueError`` instead of
returning ``inf`` or ``nan``.  float32 tops out at 3.4e38 and the window
samples grow with ``sqrt(dt)``, so finite input can overflow inside a product
or an FFT (complex64 bins of order 1e36 at ``dt=1e4`` do, in
``forward_bands``), and NumPy only warns on the way.  Each of the ten entry
points (``forward``/``inverse``, their ``_time`` and ``_many`` forms, and
``forward_bands``/``inverse_bands``) checks the single-precision arrays it
computes, and the message names the function whose array overflowed and
suggests float64 input.  The check is host work, like the input check: it is
skipped for double results and under ``jax.jit``, where a tracer has no values
to read and an overflow still comes back as ``inf``/``nan``.
```

The helper, after `_check_finite`:

```python
    if not is_tracer(values):
        validate_finite(values, context)
```
→
```python
    if not is_tracer(values):
        validate_finite(values, context)


def _check_single_output(out: NDArray[np.generic], entry: str) -> None:
    """Raise ``ValueError`` if a single-precision result of ``entry`` is not finite.

    The entry points check their input first (:func:`_check_finite`), so a
    ``nan`` or ``inf`` here was made by the transform itself: a float32
    product, FFT or scaling overflowed.  NumPy only warns on the way and JAX
    says nothing, so the caller would get ``nan`` cells.  Each entry point
    checks the arrays it computes itself, so the message names the function
    whose array overflowed (spec 2026-11-08 §1.4).

    Host work, as :func:`_check_finite` is: skipped for a double-precision
    result (float64 runs gain no pass) and for a JAX tracer.  It does not
    change ``np.errstate``.
    """
    if _working_dtypes(out)[0] != np.float32 or is_tracer(out):
        return
    arr = np.asarray(out)
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    n_bad = int(np.count_nonzero(bad))
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    raise ValueError(
        f"{entry}: {n_bad} of {arr.size} entries of the single-precision result are not "
        f"finite ({n_nan} nan, {n_bad - n_nan} inf), from finite input: the computation "
        "overflowed float32, whose largest value is 3.4e38. Pass float64 (or complex128) "
        "input to compute in double."
    )
```

The dtype test comes first, so a float64 or complex128 result costs one dtype comparison and nothing else. `_working_dtypes` reads only `.dtype`, which a tracer has too. `_check_single_output` does not call `validate_finite`, whose message tells the caller to fill gaps. It has no doctest: an overflowing example would warn, and `src/domainia/wilson` keeps its 72 doctest items.

- [ ] **Step 6: The ten call sites (`src/domainia/wilson/transform.py`)**

Each result is bound to a name, checked, then returned. The arithmetic is unchanged.

`forward`:

```python
    _check_finite(x_rfft, "forward: x_rfft")
    return _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
```
→
```python
    _check_finite(x_rfft, "forward: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
    _check_single_output(out, "forward")
    return out
```

`forward_many`:

```python
    _check_finite(x_rfft, "forward_many: x_rfft")
    return _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
```
→
```python
    _check_finite(x_rfft, "forward_many: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
    _check_single_output(out, "forward_many")
    return out
```

`forward_time_many`:

```python
    _check_finite(x, "forward_time_many: x")
    return forward_many(be.fft.rfft(x, axis=-1), Nf, Nt, dt, be=be, window=window)
```
→
```python
    _check_finite(x, "forward_time_many: x")
    x_rfft = be.fft.rfft(x, axis=-1)
    # Before forward_many's input check, which would report an overflow here as a gap.
    _check_single_output(x_rfft, "forward_time_many")
    return forward_many(x_rfft, Nf, Nt, dt, be=be, window=window)
```

`inverse` (the blank line before `return X` goes):

```python
    X = be.xp.zeros(N // 2 + 1, dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)

    return X
```
→
```python
    X = be.xp.zeros(N // 2 + 1, dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
    _check_single_output(X, "inverse")
    return X
```

`inverse_many`:

```python
    X = be.xp.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
    return X
```
→
```python
    X = be.xp.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
    _check_single_output(X, "inverse_many")
    return X
```

`inverse_time_many`:

```python
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    return be.fft.irfft(inverse_many(wmn, Nf, Nt, dt, be=be, window=window), n=N, axis=-1)
```
→
```python
    N = n_samples(Nf, Nt)
    be = be or get_backend(wmn)
    x = be.fft.irfft(inverse_many(wmn, Nf, Nt, dt, be=be, window=window), n=N, axis=-1)
    _check_single_output(x, "inverse_time_many")
    return x
```

`forward_time`:

```python
    _check_finite(x, "forward_time: x")
    return forward(be.fft.rfft(x), Nf, Nt, dt, be=be, window=window)
```
→
```python
    _check_finite(x, "forward_time: x")
    x_rfft = be.fft.rfft(x)
    # Before forward's input check, which would report an overflow here as a gap.
    _check_single_output(x_rfft, "forward_time")
    return forward(x_rfft, Nf, Nt, dt, be=be, window=window)
```

`inverse_time`:

```python
    x_rfft = inverse(wmn, Nf, Nt, dt, be=be, window=window)
    return be.fft.irfft(x_rfft, n=N)
```
→
```python
    x_rfft = inverse(wmn, Nf, Nt, dt, be=be, window=window)
    x = be.fft.irfft(x_rfft, n=N)
    _check_single_output(x, "inverse_time")
    return x
```

`forward_bands`:

```python
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    return be.xp.real(phase * Y) * rdtype.type(np.sqrt(2.0 * dt / n_samples(Nf, Nt)))
```
→
```python
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    out = be.xp.real(phase * Y) * rdtype.type(np.sqrt(2.0 * dt / n_samples(Nf, Nt)))
    _check_single_output(out, "forward_bands")
    return out
```

`inverse_bands`:

```python
    # rfft units: N / (2 dt) times the adjoint of forward_bands' sqrt(2 dt / N).
    return out * rdtype.type(np.sqrt(n_samples(Nf, Nt) / (2.0 * dt)))
```
→
```python
    # rfft units: N / (2 dt) times the adjoint of forward_bands' sqrt(2 dt / N).
    out = out * rdtype.type(np.sqrt(n_samples(Nf, Nt) / (2.0 * dt)))
    _check_single_output(out, "inverse_bands")
    return out
```

The wrappers check only what they compute themselves:
- `forward_time(_many)` checks its rfft. When the rfft fits float32 but the gather overflows (for example a float32 record at `dt=1e4`), the check inside `forward(_many)` raises, and the message says `forward:` (or `forward_many:`). `forward`'s input check already names `forward` in the same way.
- `inverse_time(_many)` checks the irfft. An overflow in the bins is reported by `inverse(_many)`.

`inner_product`, `stationary`, `evolutionary` and `matrix` call these entry points and inherit the guard with no change. The variance builders are not covered: `whittle_variance`, `band_averaged_variance` and `evolutionary_variance` call none of the ten. `empirical_variance` calls `forward` on `colored_noise`'s spectrum, which is complex128 even for a float32 PSD, so the guard returns at once there.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -c "_check_single_output(" src/domainia/wilson/transform.py`
Expected: `11` (the definition and the ten calls).

- [ ] **Step 7: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_single_precision.py 2>&1 | tail -1`
Expected: `84 passed`, no warnings.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_single_precision.py 2>&1 | tail -1`
Expected: `63 passed, 21 skipped`.

- [ ] **Step 8: The wilson suite, its doctests and the jit tests**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson src/domainia/wilson 2>&1 | tail -1`
Expected: passes, no warnings summary. This includes `test_transform_shim.py`, whose float32 runs on JAX (eager, under `jit` and `vmap`, and in a subprocess that turns `FutureWarning` into an error) now pass through the guard.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q --collect-only src/domainia/wilson | tail -1`
Expected: `72 tests collected` (and a time).

- [ ] **Step 9: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, 0 errors, no warnings summary.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -3`
Expected: passes, 0 errors, no warnings summary.

A failure outside `test_single_precision.py` is a single-precision run with a non-finite result that Step 2 did not find. Check whether its input was finite. If it was, the test pinned the old `nan`: change it to expect the `ValueError`, and record it in the errata.

- [ ] **Step 10: The float64 bit gate**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz`
Expected: `0 of 1504 arrays differ`. On a float64 or complex128 result the guard returns after one dtype comparison, and Step 6 only binds each result to a name before it returns it.

- [ ] **Step 11: Commit** (sandbox disabled for both git commands)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/wilson/transform.py tests/wilson/test_single_precision.py
git commit -m "domainia.wilson: a single-precision run that overflows raises

The ten transform entry points check the single-precision arrays they
compute (_check_single_output): forward and forward_many their cells,
forward_time and forward_time_many their rfft before they delegate,
inverse and inverse_many their bins, inverse_time and inverse_time_many
the irfft, and forward_bands and inverse_bands their results. A float32
or complex64 result that is not finite, from finite input, raises
ValueError naming the entry point and suggesting float64 input. Before,
complex64 bins of 1e36 at dt=1e4 came back from forward_bands as NaN with
only RuntimeWarnings, and forward_time reported an overflowed rfft as a
gap. The check is skipped for double results and for JAX tracers, so
float64 runs gain no pass and jit is unchanged (spec 2026-11-08 §1.4).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `cd /Users/aaron/Documents/lisa/domainia && git status --short`
Expected: no line for any file of this task.

---

### Task 12: Packaging (spec §4)

**Goal:** The sdist lists what it includes (`only-include = ["src", "tests"]`), `.superpowers/` is ignored, and the metadata gains `Typing :: Typed` and a `jax` extra without `jaxlib`. A built sdist, made in a scratch environment, is inspected.

**Assumes the tree after Task 11.** No earlier task touches `pyproject.toml`, `.gitignore` or `tests/test_package.py`.

**Files:**
- Modify: `pyproject.toml` (`[project]` classifiers, `[project.optional-dependencies]` `jax`, `[tool.hatch.build.targets.sdist]`)
- Modify: `.gitignore` (append `.superpowers/`)
- Modify: `tests/test_package.py` (`test_the_sdist_leaves_out_the_working_documents` is replaced; `test_the_project_metadata` is new)

**Acceptance Criteria:**
- [ ] `pyproject.toml`'s sdist table has `only-include = ["src", "tests"]` and no `exclude` key. `[project]` has `classifiers = ["Typing :: Typed"]`, and the `jax` extra is `["jax>=0.9"]`.
- [ ] `.gitignore` ends with `.superpowers/`.
- [ ] `tests/test_package.py`: `test_the_sdist_lists_what_it_includes` and `test_the_project_metadata` fail before the `pyproject.toml` change and pass after it (7 passed in the file).
- [ ] An sdist built with hatchling (scratch env `~/miniconda3/envs/sksparse_test`) from a scratch copy of the committed tree, with `.superpowers/sdd/brief.md`, `stray.txt` and `strayfolder/a.txt` planted at its root, has exactly these top-level entries under `domainia-2.0.0/`: `.gitignore`, `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml`, `src`, `tests`.
- [ ] The full suite and the NumPy lane pass.

**Verify:** `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py` → `7 passed`

**Steps:**

- [ ] **Step 1: Grep the patterns this task changes**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "exclude\|jaxlib\|classifiers" pyproject.toml tests/test_package.py README.md`
Expected: three hits, `pyproject.toml` (`jax = ["jax>=0.9", "jaxlib>=0.9"]`, `exclude = ["docs/superpowers"]`) and `tests/test_package.py` (`...["sdist"]["exclude"]`). If README.md names `jaxlib` in an install line, change it to match in Step 4 and record the change.

- [ ] **Step 2: Write the failing tests**

In `tests/test_package.py`, replace the whole function `test_the_sdist_leaves_out_the_working_documents` with:

```python
def test_the_sdist_lists_what_it_includes():
    """The sdist takes ``src/`` and ``tests/`` only (hatchling adds README.md, LICENSE,
    pyproject.toml, .gitignore and PKG-INFO itself), so the specs and plans under docs/ and
    any untracked folder at the root stay out.  hatchling is not installed in the test env,
    so this reads the build config rather than building."""
    sdist = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert sdist["only-include"] == ["src", "tests"]
    assert "exclude" not in sdist
    assert not [path for path in sdist["only-include"] if path.split("/")[0] == "docs"]


def test_the_project_metadata():
    """The package ships ``py.typed`` and says so; the ``jax`` extra names ``jax`` alone
    (``jax`` 0.9 requires its matching ``jaxlib`` itself)."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert "Typing :: Typed" in project["classifiers"]
    assert (SRC / "domainia" / "py.typed").is_file()
    assert project["optional-dependencies"]["jax"] == ["jax>=0.9"]
```

- [ ] **Step 3: Run them to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -4`
Expected: `2 failed, 5 passed`. `test_the_sdist_lists_what_it_includes` fails with `KeyError: 'only-include'`, and `test_the_project_metadata` with `KeyError: 'classifiers'`.

- [ ] **Step 4: Change `pyproject.toml` and `.gitignore`**

In `pyproject.toml`, replace

```toml
authors = [{ name = "Aaron Johnson" }]
```

with

```toml
authors = [{ name = "Aaron Johnson" }]
classifiers = ["Typing :: Typed"]
```

then replace

```toml
jax = ["jax>=0.9", "jaxlib>=0.9"]
```

with

```toml
jax = ["jax>=0.9"]
```

then replace

```toml
[tool.hatch.build.targets.sdist]
# The design specs and implementation plans are working documents, not part of the release.
exclude = ["docs/superpowers"]
```

with

```toml
[tool.hatch.build.targets.sdist]
# Only the package and its tests: the specs and plans under docs/ and any untracked folder at
# the root stay out. hatchling adds README.md, LICENSE, pyproject.toml, .gitignore and PKG-INFO.
only-include = ["src", "tests"]
```

Append one line to `.gitignore` (after `.claude/`):

```
.superpowers/
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -2`
Expected: `7 passed`

Run the full suite and the NumPy lane (Conventions). Expected: both pass. Run `git status --short`. Expected: ` M .gitignore`, ` M pyproject.toml`, ` M tests/test_package.py` (and the coordinator's `.tasks.json`). `.superpowers/` no longer shows as untracked.

- [ ] **Step 6: Commit**

```bash
cd /Users/aaron/Documents/lisa/domainia
git add pyproject.toml .gitignore tests/test_package.py
git commit -m "Packaging: the sdist lists what it includes

only-include = [src, tests] replaces the docs/superpowers exclude, so an untracked
folder at the root (.superpowers/ was one) cannot reach the sdist; .gitignore gains
.superpowers/. Metadata: Typing :: Typed, and the jax extra drops jaxlib (jax 0.9
pins its own).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(sandbox disabled for both git commands)

- [ ] **Step 7: Build and inspect an sdist in a scratch environment**

hatchling is not in `wdmrel`. Build with `~/miniconda3/envs/sksparse_test` (hatchling 1.27, Python 3.13.5), from a scratch copy of the committed tree with three stray paths planted at its root. Nothing is built in the repo.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia
rm -rf $TMPDIR/sdist-check && mkdir -p $TMPDIR/sdist-check/tree
git archive HEAD | tar -x -C $TMPDIR/sdist-check/tree
cd $TMPDIR/sdist-check/tree && mkdir -p .superpowers/sdd strayfolder
echo x > .superpowers/sdd/brief.md && echo y > stray.txt && echo z > strayfolder/a.txt
~/miniconda3/envs/sksparse_test/bin/python -m hatchling build -t sdist -d $TMPDIR/sdist-check/dist
tar -tzf $TMPDIR/sdist-check/dist/domainia-2.0.0.tar.gz | awk -F/ '{print $2}' | sort | uniq -c
```
Expected: the build prints `…/domainia-2.0.0.tar.gz`, and the listing is exactly

```
   1 .gitignore
   1 LICENSE
   1 PKG-INFO
   1 README.md
   1 pyproject.toml
  NN src
  NN tests
```

(the counts NN are the number of tracked files under each). There is no `.superpowers`, `stray.txt`, `strayfolder` or `docs`. Record the listing in the task report.

---

### Task 13: Test gaps (spec §5)

**Goal:** Close the test gaps of spec §5 with the fewest, shortest additions, each inside an existing test: `match=` on the three bare `pytest.raises(TypeError)` of `tests/test_dom_domains.py`; `to_coordinates`' role word; the padded layouts in the never-dense test; a jitted `WDMToTime.adjoint` and a jitted `after()` adjoint; `MidpointVariance` on JAX. The window scan gains `_cells_and_window`, so its two window-taking callers pass `window=` by keyword, and `MidpointVariance` reads its cells through a new cells-only helper `domains._cells_of`. Nothing changes behaviour.

**Assumes the tree after Task 12.** Earlier tasks edited these files, and the old texts below are their result:
* Task 3: `tests/test_dom_domains.py` (the `CellSet` dtype pin at :149, so the third bare `pytest.raises(TypeError)` is at :188, not the spec's :185), `src/domainia/domains.py`, and the new `tests/test_wrong_kinds_domains.py`.
* Task 5: `src/domainia/covariance/evolutionary.py` (the kind checks; the `_cells_and_window` call is at :134, not the spec's :132).
* Task 6: `tests/test_translate_coordinates.py` (the `R + 0j` case) and `src/domainia/variance.py`.
* Task 7: `tests/test_dom_cov_multiply.py` (the `b` shape tests and the `COVS` table this task reuses).
* Task 1: `SCANNED` in `tests/wilson/test_window_threading.py`.

No earlier task touches `tests/test_dom_restored_coverage.py`, `tests/test_dom_maps_fourier_wdm.py` or `tests/test_dom_windows.py`. Task 9 edits none of this task's files.

**Files:**
- Modify: `src/domainia/domains.py` (new `_cells_of`, after `_cells_and_window`; one phrase of `_cells_and_window`'s docstring)
- Modify: `src/domainia/variance.py` (the `.domains` import; `BandAveragedVariance.__post_init__` passes `window=` by keyword; `MidpointVariance.__post_init__` calls `_cells_of`)
- Modify: `src/domainia/covariance/evolutionary.py` (`EvolutionaryCovariance.__init__` passes `window=` by keyword)
- Modify: `tests/test_dom_domains.py` (`test_time_domain`, `test_fourier_units_carry_sigma0_and_the_rfft_scale`, `test_wdm_domain_and_channels`)
- Modify: `tests/test_translate_coordinates.py` (`test_argument_refusals`)
- Modify: `tests/test_dom_cov_multiply.py` (`test_block_tridiagonal_multiply_never_forms_the_dense_matrix`)
- Modify: `tests/test_dom_restored_coverage.py` (`test_wdm_maps_under_jit_with_the_map_as_argument`)
- Modify: `tests/test_dom_maps_fourier_wdm.py` (`test_jax_parity_with_the_map_as_a_jit_argument`). Deviation: spec §5 files the jitted `after()` adjoint under `tests/test_dom_restored_coverage.py:63-66`. It goes here instead, inside the loop that already builds and jits the two `after()` maps (spec §5's "only `apply` is jitted today, at `tests/test_dom_maps_fourier_wdm.py:253-268`"). In `test_dom_restored_coverage.py` it would need three more imports and a rebuilt crop case.
- Modify: `tests/test_dom_windows.py` (`test_the_window_is_static_under_jit`)
- Modify: `tests/wilson/test_window_threading.py` (`CONSUMERS`)
- Modify: `tests/test_wrong_kinds_domains.py` (the `domainia.domains` import; `test_cells_of_the_wrong_kind_take_the_got_form`)
- Not changed (spec §5's last bullet): `from .conftest import _oracle_paths`, and `tests/wilson/test_transform_shim.py` importing from `test_single_precision.py`.

**Acceptance Criteria:**
- [ ] `grep -n "pytest.raises(TypeError)" tests/test_dom_domains.py` gives no output. The three cases match `N must be an integer, got float`, `positional argument` (`sigma0` is keyword-only) and `unexpected keyword argument 'scale'`.
- [ ] `test_argument_refusals` pins `to_coordinates` with a partial `WDMDomain` (`ValueError`, `^domain: translate handles only WDMDomain.full`) and with a `wilson.Grid` (`TypeError`, `^domain must be a full-record domain`), right after the same `from_coordinates` pair.
- [ ] The never-dense test runs `multiply` on four covariances, with the dense routes patched to raise: `BlockTridiagonalCovariance` and `EvolutionaryCovariance`, each in the ragged and the padded layout (`COVS` entries `blocktri-ragged`, `blocktri-padded`, `evolutionary-blocktri`, `evolutionary-padded`).
- [ ] `jax.jit` of `WDMToTime.adjoint` matches NumPy (`rel <= 1e-12`). `jax.jit` of the adjoint of each of the four maps in `test_jax_parity_with_the_map_as_a_jit_argument` (two `FourierToWDM`, two `after()` maps) matches NumPy to `1e-12` of its peak.
- [ ] `MidpointVariance(psd, CELLS).on(JB)` has no leaves, `tree_unflatten` rebuilds it with cells equal to `CELLS`, and it gives exactly the NumPy values.
- [ ] `"_cells_and_window"` is in `CONSUMERS`. `variance.py` and `covariance/evolutionary.py` call it with `window=` by keyword. `MidpointVariance` calls `domains._cells_of`: a `WDMDomain` gives its `.cells`, a `CellSet` passes, anything else raises `TypeError("cells must be a CellSet or a WDMDomain, got <type>")`, the text of `_cells_and_window` (pinned in `tests/test_wrong_kinds_domains.py`).
- [ ] No test function is added or removed, so the full suite and the NumPy lane give the same totals as before this task.
- [ ] The full suite and the NumPy lane pass, with no new warnings.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py tests/test_wrong_kinds_domains.py` → `755 passed`; the six files of Step 3 → `181 passed`; then the full suite and the NumPy lane → all pass

**Steps:**

- [ ] **Step 1: Grep the gaps**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "pytest.raises(TypeError)" tests/test_dom_domains.py`
Expected: three hits.
- `:27`, around `TimeDomain(8.0, 1.0)`. The message is `N must be an integer, got float`.
- `:62`, around `FourierDomain(64, 0.5, 0, 33, "sigma0", 2.0)`. The message is Python's `FourierDomain.__init__() takes 6 positional arguments but 7 were given`. The pin matches only `positional argument`, so that a new field does not break it.
- `:188`, around `WDMDomain(cells, scale=2.0)`. The message is Python's `WDMDomain.__init__() got an unexpected keyword argument 'scale'`.

The other `pytest.raises(TypeError, ...)` lines of the file (:149, :186) already have a `match=`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -n "to_coordinates(y, GRID\|to_coordinates(y, WDMDomain\|covs = \[" tests/test_translate_coordinates.py tests/test_dom_cov_multiply.py`
Expected: one hit, `tests/test_dom_cov_multiply.py:88` (`covs = [BlockTridiagonalCovariance(_blocks, SIZES), EvolutionaryCovariance(PSD, CELLS)]`, the ragged layout only). No `to_coordinates` call is given a bad domain.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "jit(.*adjoint" tests`
Expected: three hits, and none jits a `WDMToTime` or a `FourierToWDM`:
- `tests/test_dom_restored_coverage.py:60` (`TimeToWDM`);
- `tests/test_dom_maps_base.py:202` (`Restrict`);
- `tests/test_dom_windows.py:476` (the jitted `adjoint` of `test_the_window_is_static_under_jit`, used on `TimeToWDM` only).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "MidpointVariance(.*)\.on(" tests`
Expected: no output.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "_cells_and_window\|_cells_of" src tests`
Expected: ten hits.
- `src/domainia/domains.py:400` (the definition).
- `src/domainia/variance.py:29` (the import), `:68` (`_cells_and_window(self.cells, self.window)`, window positional) and `:100` (`_cells_and_window(self.cells)`, no window).
- `src/domainia/covariance/evolutionary.py:34` (the import) and `:134` (`_cells_and_window(cells, window)`, window positional).
- `tests/test_wrong_kinds_domains.py:16` (the import) and `:71` (the call in `test_cells_of_the_wrong_kind_take_the_got_form`).
- Two test names that match `_cells_of` and are not calls: `tests/test_wrong_kinds_domains.py:66` (`test_cells_of_the_wrong_kind_take_the_got_form`) and `tests/wilson/test_wrong_kinds.py:289` (`test_cells_of_the_wrong_kind_raise_type_error`).
- `tests/wilson/test_window_threading.py` is not a hit: `_cells_and_window` is not in `CONSUMERS` yet.

- [ ] **Step 2: Add the pins (tests only)**

These pin behaviour the code already has, so they pass as soon as they are written. Every addition goes inside an existing test.

`tests/test_dom_domains.py`, in `test_time_domain`, replace

```python
    with pytest.raises(TypeError):
        TimeDomain(8.0, 1.0)
```
→
```python
    with pytest.raises(TypeError, match="N must be an integer, got float"):
        TimeDomain(8.0, 1.0)
```

in `test_fourier_units_carry_sigma0_and_the_rfft_scale`, replace

```python
    with pytest.raises(TypeError):
        FourierDomain(64, 0.5, 0, 33, "sigma0", 2.0)                    # sigma0 is keyword-only
```
→
```python
    with pytest.raises(TypeError, match="positional argument"):
        FourierDomain(64, 0.5, 0, 33, "sigma0", 2.0)                    # sigma0 is keyword-only
```

and in `test_wdm_domain_and_channels`, replace

```python
    with pytest.raises(TypeError):
        WDMDomain(cells, scale=2.0)                 # the field is gone
```
→
```python
    with pytest.raises(TypeError, match="unexpected keyword argument 'scale'"):
        WDMDomain(cells, scale=2.0)                 # the field is gone
```

`tests/test_translate_coordinates.py`, at the end of `test_argument_refusals`, replace

```python
    with pytest.raises(TypeError, match="^domain must be a full-record domain"):
        from_coordinates(R, GRID, ("A", "E"))
```
→
```python
    with pytest.raises(TypeError, match="^domain must be a full-record domain"):
        from_coordinates(R, GRID, ("A", "E"))
    with pytest.raises(ValueError, match="^domain: translate handles only WDMDomain.full"):
        to_coordinates(y, WDMDomain(CellSet.bands(GRID, (2, 3))))
    with pytest.raises(TypeError, match="^domain must be a full-record domain"):
        to_coordinates(y, GRID)
```

`tests/test_dom_cov_multiply.py`, in `test_block_tridiagonal_multiply_never_forms_the_dense_matrix`, replace

```python
    """With the dense routes made to raise, the ragged product still runs."""
    covs = [BlockTridiagonalCovariance(_blocks, SIZES), EvolutionaryCovariance(PSD, CELLS)]
```
→
```python
    """With the dense routes made to raise, the ragged product still runs on both layouts."""
    covs = [COVS[k]() for k in ("blocktri-ragged", "blocktri-padded", "evolutionary-blocktri",
                                "evolutionary-padded")]
```

The first and third `COVS` entries are the two covariances the test built before. The imports of `BlockTridiagonalCovariance` and `EvolutionaryCovariance` stay: `COVS` uses them.

`tests/test_dom_restored_coverage.py`, at the end of `test_wdm_maps_under_jit_with_the_map_as_argument`, replace

```python
    assert rel(inv, Mi.apply(y)) <= 1e-12
    assert rel(inv, d) <= 1e-12
```
→
```python
    assert rel(inv, Mi.apply(y)) <= 1e-12
    assert rel(inv, d) <= 1e-12
    adj = jax.jit(lambda m, v: m.adjoint(v))(Mi.on(JB), JB.asarray(d))
    assert isinstance(adj, jax.Array)
    assert rel(adj, Mi.adjoint(d)) <= 1e-12
```

(`WDMToTime.adjoint` takes a time record, so its input is `d`, not `y`.)

`tests/test_dom_maps_fourier_wdm.py`, at the end of the loop of `test_jax_parity_with_the_map_as_a_jit_argument`, replace

```python
        got = jax.jit(lambda mm, vv: mm.apply(vv))(m.on(be), jnp.asarray(v))
        assert isinstance(got, jax.Array)
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-12 * np.max(np.abs(want)))
```
→
```python
        got = jax.jit(lambda mm, vv: mm.apply(vv))(m.on(be), jnp.asarray(v))
        assert isinstance(got, jax.Array)
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-12 * np.max(np.abs(want)))
        ref = m.adjoint(want)
        back = jax.jit(lambda mm, yy: mm.adjoint(yy))(m.on(be), jnp.asarray(want))
        assert isinstance(back, jax.Array)
        np.testing.assert_allclose(np.asarray(back), ref, rtol=0, atol=1e-12 * np.max(np.abs(ref)))
```

The loop's cases already hold the two `after()` maps (`W.after(crop)`, `Wp.after(zp)`) and two plain `FourierToWDM` maps, so the jitted adjoint covers all four. Spec §5 lists this item under `tests/test_dom_restored_coverage.py`. It goes here instead, where the `after()` maps are already built (see Files).

`tests/test_dom_windows.py`, at the end of `test_the_window_is_static_under_jit`, replace

```python
    psd, _ = _stationary_psd()
    v = BandAveragedVariance(psd, CELLS, window=ALT_WINDOW).on(JB)
    leaves, tree = jax.tree_util.tree_flatten(v)
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves).window is ALT_WINDOW
```
→
```python
    psd, nu = _stationary_psd()
    v = BandAveragedVariance(psd, CELLS, window=ALT_WINDOW).on(JB)
    leaves, tree = jax.tree_util.tree_flatten(v)
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves).window is ALT_WINDOW
    mv = MidpointVariance(psd, CELLS).on(JB)
    leaves, tree = jax.tree_util.tree_flatten(mv)
    assert leaves == [] and jax.tree_util.tree_unflatten(tree, leaves).cells == CELLS
    np.testing.assert_array_equal(mv(nu), MidpointVariance(psd, CELLS)(nu))
```

`MidpointVariance` is already imported there.

- [ ] **Step 3: Run the pins**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_domains.py tests/test_translate_coordinates.py tests/test_dom_cov_multiply.py tests/test_dom_restored_coverage.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_windows.py 2>&1 | tail -1`
Expected: `181 passed`. The same command gives `181 passed` before Step 2 too, because no test function was added. These pins pass before and after any source change: they guard behaviour, they do not drive a change. In the scratch prototype, two pins were checked against a broken copy. A `to_coordinates` that named its domain `source` failed `test_argument_refusals`. A padded `_multiply` routed through `_matrix` failed the never-dense test.

- [ ] **Step 4: Put `_cells_and_window` in the window scan, and pin `_cells_of`**

`tests/wilson/test_window_threading.py`, at the end of `CONSUMERS`, replace

```python
    "WDMDomain", "EvolutionaryCovariance", "BandAveragedVariance",
})
```
→
```python
    "WDMDomain", "EvolutionaryCovariance", "BandAveragedVariance",
    # The helper those constructors resolve the window with; MidpointVariance uses _cells_of.
    "_cells_and_window",
})
```

`tests/test_wrong_kinds_domains.py`, the import:

```python
from domainia.domains import (CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain,
                              _cells_and_window)
```
→
```python
from domainia.domains import (CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain,
                              _cells_and_window, _cells_of)
```

and at the end of `test_cells_of_the_wrong_kind_take_the_got_form`, replace

```python
    with pytest.raises(TypeError, match=_exact(f"cells must be a CellSet or a WDMDomain, got {kind}")):
        _cells_and_window(bad)
```
→
```python
    for fn in (_cells_and_window, _cells_of):
        with pytest.raises(TypeError, match=_exact(f"cells must be a CellSet or a WDMDomain, got {kind}")):
            fn(bad)
```

- [ ] **Step 5: Run them to see them fail**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q --continue-on-collection-errors tests/wilson/test_window_threading.py tests/test_wrong_kinds_domains.py 2>&1 | grep -E "^E   |^FAILED|^ERROR|passed"`
Expected: `2 failed, 563 passed, 2 errors`.
- `tests/test_wrong_kinds_domains.py` does not collect: `ImportError: cannot import name '_cells_of' from 'domainia.domains'`. It is reported twice, once by the test module and once by `--doctest-modules`.
- `test_no_kernel_call_drops_the_window[variance.py]`: `Left contains 2 more items, first extra item: (68, '_cells_and_window', 'drops the window')`. The second item is line 100, the call with no window.
- `test_no_kernel_call_drops_the_window[covariance/evolutionary.py]`: `Left contains one more item: (134, '_cells_and_window', 'drops the window')`.

A positional window counts as dropped. That is the scan's rule for every consumer (the comment above the constructors in `CONSUMERS`).

- [ ] **Step 6: `_cells_of`, and `window=` by keyword**

`src/domainia/domains.py`, in the docstring of `_cells_and_window`, replace

```python
    (``EvolutionaryCovariance``, ``BandAveragedVariance``; ``MidpointVariance`` reads the
    cells only).
```
→
```python
    (``EvolutionaryCovariance``, ``BandAveragedVariance``; ``MidpointVariance`` reads the
    cells only, with :func:`_cells_of`).
```

and after the end of `_cells_and_window`, replace

```python
    require_instance("cells", cells, CellSet, "a CellSet or a WDMDomain")
    return cells, require_window(default if window is None else window)
```
→
```python
    require_instance("cells", cells, CellSet, "a CellSet or a WDMDomain")
    return cells, require_window(default if window is None else window)


def _cells_of(cells: object) -> CellSet:
    """The cell set of a consumer that reads the cells only (``MidpointVariance``): a
    :class:`WDMDomain`'s cells, or a :class:`CellSet` as it is.  Any other ``cells`` raises
    the ``TypeError`` of :func:`_cells_and_window`."""
    if isinstance(cells, WDMDomain):
        return cells.cells
    return require_instance("cells", cells, CellSet, "a CellSet or a WDMDomain")
```

`require_instance` (Task 1) returns the value it checked. `domains.py` already imports it.

`src/domainia/variance.py`, the import:

```python
from .domains import CellSet, _cells_and_window
```
→
```python
from .domains import CellSet, _cells_and_window, _cells_of
```

in `BandAveragedVariance.__post_init__`:

```python
        cells, window = _cells_and_window(self.cells, self.window)
```
→
```python
        cells, window = _cells_and_window(self.cells, window=self.window)
```

and in `MidpointVariance.__post_init__`:

```python
        cells, _ = _cells_and_window(self.cells)
        object.__setattr__(self, "cells", cells)
```
→
```python
        object.__setattr__(self, "cells", _cells_of(self.cells))
```

`src/domainia/covariance/evolutionary.py`, in `EvolutionaryCovariance.__init__`:

```python
        cells, window = _cells_and_window(cells, window)
```
→
```python
        cells, window = _cells_and_window(cells, window=window)
```

None of this changes behaviour. `MidpointVariance` gets the same cells as before: a `WDMDomain`'s `.cells`, or the `CellSet` itself. The old call also checked `require_window(MEYER)`, and Meyer always passes. Any other `cells` gets the same `TypeError` text.

- [ ] **Step 7: Run them to see them pass**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py tests/test_wrong_kinds_domains.py 2>&1 | tail -1`
Expected: `755 passed`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_domains.py tests/test_translate_coordinates.py tests/test_dom_cov_multiply.py tests/test_dom_restored_coverage.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_windows.py 2>&1 | tail -1`
Expected: `181 passed`.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "_cells_and_window(\|_cells_of(" src`
Expected: five hits. The two definitions are in `domains.py`. The calls are `variance.py` `_cells_and_window(self.cells, window=self.window)` and `_cells_of(self.cells)`, and `covariance/evolutionary.py` `_cells_and_window(cells, window=window)`.

- [ ] **Step 8: Full suite and NumPy lane**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`
Expected: all pass, 0 errors, no warnings. The totals are those before this task, because no test function was added or removed. In the scratch prototype (Tasks 1-8 and 10-12 applied, without Task 9): `3151 passed, 3 skipped`, before and after.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`
Expected: all pass, 0 errors, no warnings. The totals are again those before this task (prototype: `2887 passed, 267 skipped`). The three jitted tests of Step 2 are `jax`-marked and skip here.

The float64 bit gate is not run here (Conventions: Tasks 2, 4, 10, 11 and 15). This task touches neither `wilson/transform.py` nor `maps/`.

- [ ] **Step 9: Commit** (sandbox disabled)

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/domains.py src/domainia/variance.py src/domainia/covariance/evolutionary.py tests/test_dom_domains.py tests/test_translate_coordinates.py tests/test_dom_cov_multiply.py tests/test_dom_restored_coverage.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_windows.py tests/wilson/test_window_threading.py tests/test_wrong_kinds_domains.py
git commit -m "Tests: close the spec 2026-11-08 section 5 gaps

match= on the three bare TypeError pins of test_dom_domains; to_coordinates'
role word for a partial WDMDomain and a Grid; the padded layouts in the
never-dense multiply test; jitted adjoints of WDMToTime and of the
after() maps; MidpointVariance on JAX (no leaves, equal cells on rebuild,
the NumPy values). The window scan lists _cells_and_window, so its
callers pass window= by keyword, and MidpointVariance reads its cells
through the new cells-only helper domains._cells_of. No behaviour change.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Text: the 1.x tags, units notes and trims (spec §6)

**Goal:** The 25 leading 1.x plan tags are gone from the tests, and the four sentences that lean on a tag say what they mean. `dense_lambda_generic` and `wilson.evolutionary.eq45_separable_kernels` say what units their result is in, and a test pins `dense_lambda_generic(W.C, ...)` to `W.push_rows`. `linalg.py` and `maps/fourier.py` lose their second mention of the galactic-binary toy model. The spec citations that Tasks 1-13 added to code are gone. No code changes.

**Assumes the tree after Task 13.**
* Task 13 edits `tests/test_dom_domains.py` (the `match=` additions at the old `:27`, `:62`, `:185`). This task changes only that file's first line, `"""D-A2: domains and cell sets."""`, and Task 13 does not touch it. No other file of this task is in Task 13's scope.
* Earlier tasks edited some of these files. Task 6 edited `testing.py` (imports, module docstring, `dense_W`, `eq45_dense`, `eq54_covariance`) and `linalg.py`. Tasks 4 and 10 edited `maps/fourier.py`. Task 2 edited `wilson/evolutionary.py`, `tests/wilson/test_grid_bands.py` and `tests/wilson/test_noise_stationary.py`. Tasks 3, 4 and 6 edited `tests/test_dom_domains.py`, `tests/test_dom_maps_dirichlet.py` and `tests/test_dom_spectra.py`. None of them changed the lines this task replaces. Every edit below is anchored to the current text.
* The spec's line numbers are at 779550d. The trim sites have moved: `linalg.py:420` is now about `:432`, `maps/fourier.py:308` about `:320`, and `dense_lambda_generic`'s docstring about `testing.py:143-144`.

**Files:**
- Modify (first line of the module docstring; tag only, or one of the four rewordings): `tests/test_dom_cov_api.py`, `tests/test_dom_cov_base.py`, `tests/test_dom_cov_blocktri.py`, `tests/test_dom_data.py`, `tests/test_dom_db_imports.py`, `tests/test_dom_domains.py`, `tests/test_dom_ev_ported.py`, `tests/test_dom_linalg_blocktri.py`, `tests/test_dom_linalg_factors.py`, `tests/test_dom_linalg_jax.py`, `tests/test_dom_linalg_pcg.py`, `tests/test_dom_maps_base.py`, `tests/test_dom_maps_dirichlet.py`, `tests/test_dom_maps_fourier_time.py`, `tests/test_dom_maps_wdm.py`, `tests/test_dom_spectra.py`, `tests/test_dom_testing.py`, `tests/wilson/test_backend.py`, `tests/wilson/test_evolutionary_kernels.py` (also the `G5:` docstring of `test_wrap_rule_on_the_sinusoid_grid`), `tests/wilson/test_grid_bands.py`, `tests/wilson/test_inner_product_fixes.py`, `tests/wilson/test_lanes.py`, `tests/wilson/test_noise_stationary.py` (also the docstring of `test_colored_noise_rejects_bad_N_and_dt`), `tests/wilson/test_transform_shim.py`
- Modify: `tests/test_dom_testing.py` (imports; new test `test_dense_lambda_generic_of_a_maps_C_is_its_push`)
- Modify: `src/domainia/testing.py` (module docstring bullet; `dense_lambda_generic` docstring)
- Modify: `src/domainia/wilson/evolutionary.py` (`eq45_separable_kernels` docstring)
- Modify: `src/domainia/linalg.py` (the `Complex Hermitian block-tridiagonal` section banner)
- Modify: `src/domainia/maps/fourier.py` (`FourierToWDM.window` docstring)
- Modify (drop the spec citations): `src/domainia/wilson/transform.py` (`_check_single_output` docstring), `tests/test_dom_cov_multiply.py`, `tests/test_dom_map_dtypes.py`, `tests/test_wrong_kinds_covariance.py`, `tests/test_wrong_kinds_domains.py`, `tests/test_wrong_kinds_maps.py`, `tests/test_wrong_kinds_misc.py`, `tests/wilson/test_checks.py`, `tests/wilson/test_single_precision.py`, `tests/wilson/test_wrong_kinds.py`

**Acceptance Criteria:**
- [ ] `grep -rnE '"""(D-[A-C]|[CG])[0-9]+: ' src tests --include='*.py'` prints nothing (25 lines in 24 files before).
- [ ] `grep -rnE "section D-C|in C8|since C4|the section's" src tests --include='*.py'` prints nothing (4 lines before).
- [ ] `tests/wilson/vendored/` is untouched (its `S2` comment stays): `git diff --stat HEAD -- tests/wilson/vendored` prints nothing before the commit.
- [ ] `dense_lambda_generic`'s docstring says that Λ is in the units of `U`, that `Λ(U) = Λ(C)/κ²` for `C = κU`, so `N/(2 dt)` times the Wilson-unit Λ only for rfft-unit bins, and that `FourierToWDM.C` gives Wilson units for a directly built map, not for a `restrict=True` or `after()` map. The `testing` module docstring no longer says "its WDM form". `eq45_separable_kernels` has the same one-line units note.
- [ ] `test_dense_lambda_generic_of_a_maps_C_is_its_push` passes: `tests/test_dom_testing.py` → `11 passed`.
- [ ] `grep -rn "12-month exact\|480 / 876 / 1800" src` prints only `linalg.py`'s table row. The ten "ported order" / "ported line for line" / "ported toy model's" warnings in `src` stay.
- [ ] `grep -rnE "§|spec 2026|2026-1[01]-[0-9]{2}" src tests --include='*.py' | grep -v -e tests/wilson/vendored -e tests/wilson/oracle` prints nothing (25 lines in 10 files before).
- [ ] No added line is longer than 100 characters. The full suite and the NumPy lane pass.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_testing.py` → `11 passed`; the two tag greps and the citation grep above → no output; then the full suite and the NumPy lane → pass

**Steps:**

- [ ] **Step 1: Grep the sites**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE '"""(D-[A-C]|[CG])[0-9]+: ' src tests --include='*.py' | cut -c1-60`
Expected: exactly 25 lines in 24 files, all in `tests/` (line 1 unless noted):
- `tests/`: `test_dom_cov_api.py` (`D-C11`), `test_dom_cov_base.py` (`D-C1`), `test_dom_cov_blocktri.py` (`D-C2`), `test_dom_data.py` (`D-A2`), `test_dom_db_imports.py` (`D-B7`), `test_dom_domains.py` (`D-A2`), `test_dom_ev_ported.py` (`D-A5`), `test_dom_linalg_blocktri.py` (`D-A4`), `test_dom_linalg_factors.py` (`D-A5`), `test_dom_linalg_jax.py` (`D-A7`), `test_dom_linalg_pcg.py` (`D-A6`), `test_dom_maps_base.py` (`D-B1`), `test_dom_maps_dirichlet.py` (`D-B4`), `test_dom_maps_fourier_time.py` (`D-B3`), `test_dom_maps_wdm.py` (`D-B2`), `test_dom_spectra.py` (`D-B6`), `test_dom_testing.py` (`D-A3`);
- `tests/wilson/`: `test_backend.py` (`C2`), `test_evolutionary_kernels.py` (`C8` at line 1 and `G5` at about line 119), `test_grid_bands.py` (`C6`), `test_inner_product_fixes.py` (`C5`), `test_lanes.py` (`C9`), `test_noise_stationary.py` (`C4`), `test_transform_shim.py` (`C3`).

Run the same grep with `| wc -l` and with `| cut -d: -f1 | sort -u | wc -l`. Expected: `25` and `24`. `tests/wilson/vendored/` has no hit; it is out of scope, and its `S2` comment (`tests/wilson/vendored/test_evolutionary.py:12`) stays. If a hit is missing or extra, stop and report it: the script in Step 4 lists the 25 by exact text.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "section D-C|in C8|since C4|the section's" src tests --include='*.py'`
Expected: 4 lines, in any order, the four sentences that lean on a tag:
```
tests/test_dom_cov_api.py:1:"""D-C11: the covariance package's public surface (section D-C gate).
tests/test_dom_ev_ported.py:1:"""D-A5: the wdm_dev EV linear-algebra tests retired from ``wdm`` in C8 (part-1 plan note 7),
tests/test_dom_db_imports.py:1:"""D-B7: the section's public names, and no JAX import from maps, spectra or variance."""
tests/wilson/test_noise_stationary.py:106:    """``colored_noise`` is public since C4, so it validates like every entry point.
```
The other single-letter-plus-digit names in `src` and `tests` (`S8`, `S2`, `C4` as local array names in `wilson/stationary.py`, `tests/test_translate_operator.py` and `tests/wilson/test_window_threading.py`) are variables, not tags. Leave them.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "12-month exact\|480 / 876 / 1800" src tests README.md`
Expected: 3 lines. `src/domainia/linalg.py:12` (the module table row `galactic-binary toy model (12-month exact)`) stays. `src/domainia/linalg.py` about `:432` (the section banner) and `src/domainia/maps/fourier.py` about `:320` (`FourierToWDM.window`) are the two repeats trimmed in Step 8.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rniE "ported order|ported line for line|ported toy model's" src | wc -l`
Expected: `10` (six in `linalg.py`, two in `maps/fourier.py`, two in `wilson/evolutionary.py`). These warnings guard bit-identical results, and they stay: the count is the same after Step 8.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "dense_lambda_generic\|eq45_separable_kernels" src tests README.md`
Expected: `testing.py` (module docstring, `__all__`, the `def`), `wilson/evolutionary.py` (module docstring, `__all__`, the `def`), `tests/test_dom_testing.py` (one call), `tests/wilson/test_evolutionary_kernels.py` (two calls) and `tests/test_wrong_kinds_misc.py` (the `dense_lambda_generic margin` case). No test reads a docstring.

- [ ] **Step 2: Write the units test**

In `tests/test_dom_testing.py`, replace

```python
from domainia.domains import CellSet
```

with

```python
from domainia.domains import CellSet, FourierDomain, WDMDomain
from domainia.maps import FourierToWDM
```

then replace the last two lines of `test_eq54_is_the_covariance_of_a_modulated_stationary_process`

```python
    lam = T.dense_lambda_generic(U, np.zeros(G), Nr * S[src], h, margin=margin)
    assert np.array_equal(lam, 0.5 * np.real(U @ got @ U.conj().T))
```

with

```python
    lam = T.dense_lambda_generic(U, np.zeros(G), Nr * S[src], h, margin=margin)
    assert np.array_equal(lam, 0.5 * np.real(U @ got @ U.conj().T))


def test_dense_lambda_generic_of_a_maps_C_is_its_push(rng):
    """``U = W.C`` of a ``FourierToWDM`` ``W`` built directly over the band window gives
    ``Lambda`` in Wilson units: the push of the Eq. (54) covariance through ``W``."""
    dst = WDMDomain(CellSet.bands(GRID, (2, 3, 4)))
    W = FourierToWDM(FourierToWDM(FourierDomain(N, DT, 0, N // 2 + 1, "rfft"), dst).window, dst)
    G, margin = W.src.n, 3
    h = np.zeros(2 * (G + margin) + 1, complex)
    h[G + margin - 1:G + margin + 2] = 0.2 - 0.1j, 1.0, 0.2 + 0.1j      # lags -1, 0, 1
    sI, sG = rng.uniform(0.5, 1.0, G), rng.uniform(1.0, 2.0, G + 2 * margin)
    lam = T.dense_lambda_generic(W.C, sI, sG, h, margin=margin)
    M = T.eq54_covariance(sI, sG, h, margin=margin)
    assert rel(lam, W.push_rows(M, np.arange(W.shape[0]))) < 1e-14
```

`W.src` is the band window (`G = 32` bins, `k0 = 8`), so `W.C` is `kappa U` on exactly the `G` bins that `eq54_covariance` works on, and `push_rows` slices nothing. `rel` is the file's own helper. In the scratch prototype the two sides were equal bit for bit. The test keeps a 1e-14 tolerance so that it does not depend on the last bit of a BLAS product. The same call with `W.C / kappa` (`kappa = sqrt(2 dt / N)`, rfft units) gave exactly `N / (2 dt) = 1024` times the push, which is the factor the docstring in Step 6 states.

- [ ] **Step 3: Run the test**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_testing.py 2>&1 | tail -2`
Expected: `11 passed`. The new test passes on today's code. It pins the fact that Step 6 writes down; this task changes docstrings only, so there is no red run.

- [ ] **Step 4: The tag script** (keep it in `$TMPDIR`, do not commit it)

Save as `$TMPDIR/strip_tags.py`:

```python
"""One-off: strip the 25 leading 1.x plan tags and reword the four sentences that lean on one.

Usage: python -I strip_tags.py REPO
Each old text must occur exactly once in its file; otherwise the script writes nothing.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])

EDITS = {
    "tests/test_dom_cov_api.py": [
        ('"""D-C11: the covariance package\'s public surface (section D-C gate).\n',
         '"""The covariance package\'s public surface.\n'),
    ],
    "tests/test_dom_cov_base.py": [
        ('"""D-C1: the covariance protocol,', '"""The covariance protocol,'),
    ],
    "tests/test_dom_cov_blocktri.py": [
        ('"""D-C2: BlockTridiagonalCovariance', '"""BlockTridiagonalCovariance'),
    ],
    "tests/test_dom_data.py": [
        ('"""D-A2: data vectors', '"""Data vectors'),
    ],
    "tests/test_dom_db_imports.py": [
        ('"""D-B7: the section\'s public names, and no JAX import from maps, spectra or variance."""',
         '"""The public names of maps, spectra and variance, and no JAX import from them."""'),
    ],
    "tests/test_dom_domains.py": [
        ('"""D-A2: domains and cell sets."""', '"""Domains and cell sets."""'),
    ],
    "tests/test_dom_ev_ported.py": [
        ('"""D-A5: the wdm_dev EV linear-algebra tests retired from ``wdm`` in C8 (part-1 plan note 7),\n'
         're-homed on ``domainia``',
         '"""The wdm_dev EV linear-algebra tests that ``wdm`` retired when it kept only the stateless\n'
         'kernels, re-homed on ``domainia``'),
    ],
    "tests/test_dom_linalg_blocktri.py": [
        ('"""D-A4: diagonal, dense-Cholesky', '"""Diagonal, dense-Cholesky'),
    ],
    "tests/test_dom_linalg_factors.py": [
        ('"""D-A5: the complex Hermitian', '"""The complex Hermitian'),
    ],
    "tests/test_dom_linalg_jax.py": [
        ('"""D-A7: the linear algebra on JAX', '"""The linear algebra on JAX'),
    ],
    "tests/test_dom_linalg_pcg.py": [
        ('"""D-A6: preconditioned CG', '"""Preconditioned CG'),
    ],
    "tests/test_dom_maps_base.py": [
        ('"""D-B1: the LinearMap base', '"""The LinearMap base'),
    ],
    "tests/test_dom_maps_dirichlet.py": [
        ('"""D-B4: pad_maps / crop_maps', '"""pad_maps / crop_maps'),
    ],
    "tests/test_dom_maps_fourier_time.py": [
        ('"""D-B3: TimeToFourier, CropTime', '"""TimeToFourier, CropTime'),
    ],
    "tests/test_dom_maps_wdm.py": [
        ('"""D-B2: TimeToWDM and WDMToTime', '"""TimeToWDM and WDMToTime'),
    ],
    "tests/test_dom_spectra.py": [
        ('"""D-B6: TabulatedSpectrum and GenericPSD."""', '"""TabulatedSpectrum and GenericPSD."""'),
    ],
    "tests/test_dom_testing.py": [
        ('"""D-A3: the dense oracles', '"""The dense oracles'),
    ],
    "tests/wilson/test_backend.py": [
        ('"""C2: the backend shim."""', '"""The backend shim."""'),
    ],
    "tests/wilson/test_evolutionary_kernels.py": [
        ('"""C8: the stateless WDM Paper II kernels', '"""The stateless WDM Paper II kernels'),
        ('"""G5: (6, 5) at (64, 256, 2**-10).', '"""The cut is (6, 5) at (64, 256, 2**-10).'),
    ],
    "tests/wilson/test_grid_bands.py": [
        ('"""C6: packing helpers', '"""Packing helpers'),
    ],
    "tests/wilson/test_inner_product_fixes.py": [
        ('"""C5: the b5973b7 fixes', '"""The b5973b7 fixes'),
    ],
    "tests/wilson/test_lanes.py": [
        ('"""C9: the NumPy-only CI lane.', '"""The NumPy-only CI lane.'),
    ],
    "tests/wilson/test_noise_stationary.py": [
        ('"""C4: domainia.wilson.noise and', '"""domainia.wilson.noise and'),
        ('``colored_noise`` is public since C4, so it validates',
         '``colored_noise`` is public, so it validates'),
    ],
    "tests/wilson/test_transform_shim.py": [
        ('"""C3: the transform over the backend shim.', '"""The transform over the backend shim.'),
    ],
}

assert sum(map(len, EDITS.values())) == 26 and len(EDITS) == 24
out = {}
for rel, pairs in EDITS.items():
    assert not rel.startswith("tests/wilson/vendored/")
    text = (ROOT / rel).read_text()
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            sys.exit(f"STOP, nothing written: {rel}: {old[:60]!r} occurs {n} times")
        text = text.replace(old, new)
    out[rel] = text
for rel, text in out.items():             # every check passed: write all 24 files
    (ROOT / rel).write_text(text)
    print("rewrote", rel)
```

What the 26 edits do:
- 21 lines lose only their tag. A plain word after the tag gets a capital letter ("the backend shim" → "The backend shim"). A name keeps its case (`pad_maps / crop_maps`, `domainia.wilson.noise`, `TimeToFourier`, `BlockTridiagonalCovariance`).
- `G5:` becomes "The cut is", so that the sentence still reads: `"""The cut is (6, 5) at (64, 256, 2**-10).  The last cut columns ...`.
- The four sentences that lean on a tag:
  - `test_dom_cov_api.py`: "(section D-C gate)" is dropped.
  - `test_dom_db_imports.py`: "the section's public names" names the three modules its `EXPORTS` dict covers.
  - `test_dom_ev_ported.py`: "in C8 (part-1 plan note 7)" becomes what happened, from the header of `tests/wilson/vendored/test_evolutionary.py`: wdm-paper 1.0 kept only the stateless kernels in `wdm`.
  - `test_noise_stationary.py`: "since C4" is dropped.

- [ ] **Step 5: Run it once, then the gates**

Run: `cd /Users/aaron/Documents/lisa/domainia && ~/miniconda3/envs/wdmrel/bin/python -I $TMPDIR/strip_tags.py .`
Expected: 24 `rewrote tests/...` lines and no `STOP` line. A `STOP` line means an earlier task changed one of these lines: the script wrote nothing. Fix that one entry against the file's current text, run the script again, and record the change in the Execution errata.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE '"""(D-[A-C]|[CG])[0-9]+: ' src tests --include='*.py'`
Expected: no output.
Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "section D-C|in C8|since C4|the section's" src tests --include='*.py'`
Expected: no output.
Run: `cd /Users/aaron/Documents/lisa/domainia && git diff --stat -- tests/wilson/vendored tests/wilson/oracle`
Expected: no output.

- [ ] **Step 6: `dense_lambda_generic` and the `testing` module docstring**

In `src/domainia/testing.py`, in the module docstring, replace

```python
  Eq. (54) covariance of a modulated stationary process on Fourier bins and its WDM
  form (ported from the galactic-binary toy model, which does not ship).
```

with

```python
  Eq. (54) covariance of a modulated stationary process on Fourier bins, and its
  congruence by atoms ``U``, in Wilson units for ``U = FourierToWDM.C`` (ported from the
  galactic-binary toy model, which does not ship).
```

The README's units rule says that a WDM value is always a Wilson coefficient. `dense_lambda_generic` with unit-norm atoms is not one, so the bullet no longer calls it a "WDM form".

Then replace the end of `dense_lambda_generic`'s docstring

```python
    :func:`eq54_covariance` and ``U`` the unit-norm atoms on the ``G`` grid bins."""
```

with

```python
    :func:`eq54_covariance` and ``U`` the unit-norm atoms on the ``G`` grid bins.

    ``Lambda`` is in the units of ``U``: ``Lambda(U) = Lambda(C) / kappa**2`` for
    ``C = kappa U`` (:class:`~domainia.maps.FourierToWDM`), so ``N / (2 dt)`` times the
    Wilson-unit ``Lambda`` for ``"rfft"``-unit bins only.  ``U = FourierToWDM.C`` gives Wilson
    units for a directly built map, not for a ``restrict=True`` or ``after()`` map (its ``C``
    is not ``kappa U`` on the window's ``G`` bins)."""
```

The function body is unchanged. `kappa = src.rfft_scale * sqrt(2 dt / N)` (`maps/fourier.py`, `_kappa`), so `1 / kappa**2 = N / (2 dt)` only when `rfft_scale = 1`. "Directly built" means `FourierToWDM(src, dst)` with `restrict=False`, for any `src` that covers the window: its `C` is `kappa U` on the window's `G` bins, and `push_rows` reads those bins of a wider `src`. A `restrict=True` map's `C` has only the sub-window's columns (truncated atoms), and an `after()` map's `C` lies on the bins of the outer record.

- [ ] **Step 7: The same note on `eq45_separable_kernels`**

In `src/domainia/wilson/evolutionary.py`, in the docstring of `eq45_separable_kernels`, replace

```python
    spectrum on the window bins.  Ported verbatim (host NumPy,
```

with

```python
    spectrum on the window bins.  ``K^X`` is in the units of ``U`` (Wilson units for the
    ``C = kappa U`` of ``domainia.maps.FourierToWDM``).  Ported verbatim (host NumPy,
```

The next line, `    built once per noise component).`, stays. The docstring has no `>>>` line, so the `wilson` doctest collection does not change.

- [ ] **Step 8: Trim the two port repeats**

In `src/domainia/linalg.py`, replace the section banner line

```python
# Complex Hermitian block-tridiagonal (the galactic-binary 12-month exact model)
```

with

```python
# Complex Hermitian block-tridiagonal
```

The module docstring's table already says that `cblock_factor` is ported from the galactic-binary toy model (12-month exact).

In `src/domainia/maps/fourier.py`, in the docstring of the `FourierToWDM.window` property, replace

```python
        """The band windows' bins ``[g0, g0 + G)`` of the cells' grid record
        (480 / 876 / 1800 for the galactic-binary toy model).  The record is ``dst``'s grid,
        not ``src``'s: the two agree for a built map (the constructor checks it), while a map
        returned by :meth:`after` reads the bins of the outer record.  It is
```

with

```python
        """The band windows' bins ``[g0, g0 + G)`` of the cells' grid record.  The record is
        ``dst``'s grid, not ``src``'s: the two agree for a built map (the constructor checks
        it), while a map returned by :meth:`after` reads the bins of the outer record.  It is
```

The rest of the docstring stays. The module docstring already says once that the cell matrices are ported from the galactic-binary toy model. The "ported order" warnings, and the two "ported toy model's ... call site" docstrings in `maps/fourier.py`, stay.

- [ ] **Step 9: Drop the spec citations in code**

The specs live under `docs/`, which the sdist does not ship after Task 12, so a spec citation in `src` or `tests` points nowhere. Tasks 1-13 added 25. None of their files is touched by Steps 1-8.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "§|spec 2026|2026-1[01]-[0-9]{2}" src tests --include='*.py' | grep -v -e tests/wilson/vendored -e tests/wilson/oracle`
Expected: 25 lines in 10 files:
- `src/domainia/wilson/transform.py:229`
- `tests/test_dom_cov_multiply.py:3`, `:140`, `:151`
- `tests/test_dom_map_dtypes.py:1`, `:250`
- `tests/test_wrong_kinds_covariance.py:2`, `:47`, `:102`, `:135`, `:195`, `:219`
- `tests/test_wrong_kinds_domains.py:2`, `:131`
- `tests/test_wrong_kinds_maps.py:2`, `:273`
- `tests/test_wrong_kinds_misc.py:113`, `:158`, `:171`, `:178`
- `tests/wilson/test_checks.py:1`, `:255`
- `tests/wilson/test_single_precision.py:176`
- `tests/wilson/test_wrong_kinds.py:123`, `:334`

The bare `§` catches the three that do not say "spec " (`Spec §1.5:` twice, `Before §2.8`). If a hit is missing or extra, stop and report it.

Save as `$TMPDIR/strip_specs.py` (keep it in `$TMPDIR`, do not commit it):

```python
"""One-off: drop the 25 spec citations that Tasks 1-13 added to src and tests.

Usage: python -I strip_specs.py REPO
Each old text must occur exactly once in its file; otherwise the script writes nothing.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])

EDITS = {
    "src/domainia/wilson/transform.py": [
        ("whose array overflowed (spec 2026-11-08 §1.4).", "whose array overflowed."),
    ],
    "tests/test_dom_map_dtypes.py": [
        ('"""The maps compute in double (spec 2026-11-08 §1): a map',
         '"""The maps compute in double: a map'),
        ('"""Spec §1.5: ``FourierDomain`` stores', '"""``FourierDomain`` stores'),
    ],
    "tests/test_dom_cov_multiply.py": [
        ("check the leading shape of ``b`` (spec 2026-11-08 §2.8).",
         "check the leading shape of ``b``."),
        ("# --- the leading shape of b (spec 2026-11-08 §2.8) ------------------------------------------",
         "# --- the leading shape of b -----------------------------------------------------------------"),
        ("the factor's)``.  Before\n#: §2.8 a ``DiagonalCovariance``",
         "the factor's)``.  Before,\n#: a ``DiagonalCovariance``"),
    ],
    "tests/test_wrong_kinds_covariance.py": [
        ("``ValueError`` (spec 2026-11-08 §2.3, §2.5, §2.6, §2.9).", "``ValueError``."),
        ("# --- integers (spec §2.3, §2.9) -------------------------------------------------------------",
         "# --- integers -------------------------------------------------------------------------------"),
        ("# --- flags (spec §2.5) ----------------------------------------------------------------------",
         "# --- flags ----------------------------------------------------------------------------------"),
        ("# --- string options (spec §2.5) -------------------------------------------------------------",
         "# --- string options -------------------------------------------------------------------------"),
        ("# --- covariances and callables (spec §2.6) --------------------------------------------------",
         "# --- covariances and callables --------------------------------------------------------------"),
        ("# --- t_max (spec §2.4, §2.9) ----------------------------------------------------------------",
         "# --- t_max ----------------------------------------------------------------------------------"),
    ],
    "tests/test_wrong_kinds_misc.py": [
        ("taken as a one-name axis (spec §2.6).", "taken as a one-name axis."),
        ("raise ``ValueError`` (spec §3.5): ``R`` is never complex.",
         "raise ``ValueError``: ``R`` is never complex."),
        ("the domain makes them wrong (spec §3.5).", "the domain makes them wrong."),
        ("come back as ``inf`` (spec §2.9).", "come back as ``inf``."),
    ],
    "tests/test_wrong_kinds_domains.py": [
        ("a wrong value with ``ValueError``\n(spec 2026-11-08 §1.5, §2.3-§2.6): grids",
         "a wrong value with\n``ValueError``: grids"),
        ('"""Spec §1.5: the domains store', '"""The domains store'),
    ],
    "tests/test_wrong_kinds_maps.py": [
        ("a wrong value with ``ValueError``\n(spec 2026-11-08 §2.3-§2.7): domains",
         "a wrong value with\n``ValueError``: domains"),
        ("arrays it carries (spec §2.7): the checks", "arrays it carries: the checks"),
    ],
    "tests/wilson/test_wrong_kinds.py": [
        ("# The other wilson sites (spec 2026-11-08 §2.3-§2.5, §3.5)\n",
         "# The other wilson sites\n"),
        ("#: The real-array checks of spec §3.5: the call", "#: The real-array checks: the call"),
    ],
    "tests/wilson/test_checks.py": [
        ("``domainia.wilson._checks`` (spec 2026-11-08 §2.2): each helper",
         "``domainia.wilson._checks``: each helper"),
        ("the integer check (spec 2026-11-08, Context) call", "the integer check call"),
    ],
    "tests/wilson/test_single_precision.py": [
        ("# The single-precision guard (spec 2026-11-08 §1.4): a float32",
         "# The single-precision guard: a float32"),
    ],
}

assert sum(map(len, EDITS.values())) == 25 and len(EDITS) == 10
out = {}
for rel, pairs in EDITS.items():
    text = (ROOT / rel).read_text(encoding="utf-8")
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            sys.exit(f"STOP, nothing written: {rel}: {old[:60]!r} occurs {n} times")
        text = text.replace(old, new)
    out[rel] = text
for rel, text in out.items():             # every check passed: write all 10 files
    (ROOT / rel).write_text(text, encoding="utf-8")
    print("rewrote", rel)
```

What the 25 edits do:
- 11 parentheticals go; nothing else in their sentences changes.
- The six `# --- name (spec ...) ---` banners lose the parenthetical and keep their width of 94 characters. The two boxed banners (`test_single_precision.py:176`, `wilson/test_wrong_kinds.py:123`) change only their text line.
- `test_wrong_kinds_domains.py` and `test_wrong_kinds_maps.py`: `ValueError` moves from line 1 to line 2, in place of the parenthetical, as in `test_wrong_kinds_covariance.py`. This also brings `test_wrong_kinds_domains.py:1` from 101 characters to 86.
- `Spec §1.5: ` goes from two docstrings; `of spec §3.5` from "The real-array checks"; `Before §2.8 a` becomes `Before, a`.

Run: `cd /Users/aaron/Documents/lisa/domainia && ~/miniconda3/envs/wdmrel/bin/python -I $TMPDIR/strip_specs.py .`
Expected: 10 `rewrote` lines and no `STOP` line. A `STOP` line means the script wrote nothing: fix that entry against the file's current text, as in Step 5.

Run the grep at the top of this step again. Expected: no output.

- [ ] **Step 10: Check and run**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rn "12-month exact\|480 / 876 / 1800" src tests README.md`
Expected: one line, `src/domainia/linalg.py:12:` (the table row).
Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rniE "ported order|ported line for line|ported toy model's" src | wc -l`
Expected: `10`, as in Step 1.
Run: `cd /Users/aaron/Documents/lisa/domainia && git diff -U0 | grep '^+' | grep -v '^+++' | awk 'length($0) > 101'`
Expected: no output (no added line is over 100 characters).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_testing.py 2>&1 | tail -2`
Expected: `11 passed`.
Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_ev_ported.py tests/test_dom_db_imports.py tests/test_dom_cov_api.py tests/wilson/test_noise_stationary.py tests/wilson/test_evolutionary_kernels.py src/domainia/wilson/evolutionary.py 2>&1 | tail -2`
Expected: `92 passed, 2 skipped` (the two skips are the oracle-gated tests, which need `DOMAINIA_ORACLE_ROOT`, as before).

Run the full suite and the NumPy lane (Conventions). Expected: both pass, with no new warnings and one more test than before this task. In the scratch prototype (the tree after Task 12 plus this task, without Task 13) they gave `3152 passed, 3 skipped` and `2888 passed, 267 skipped`, one more each than the tree before.

Run: `cd /Users/aaron/Documents/lisa/domainia && git status --short`
Expected: the 38 files of Step 11, all ` M` (and the plan's own files).

- [ ] **Step 11: Commit**

```bash
cd /Users/aaron/Documents/lisa/domainia
git add src/domainia/testing.py src/domainia/wilson/evolutionary.py src/domainia/linalg.py \
  src/domainia/maps/fourier.py \
  tests/test_dom_cov_api.py tests/test_dom_cov_base.py tests/test_dom_cov_blocktri.py \
  tests/test_dom_data.py tests/test_dom_db_imports.py tests/test_dom_domains.py \
  tests/test_dom_ev_ported.py tests/test_dom_linalg_blocktri.py tests/test_dom_linalg_factors.py \
  tests/test_dom_linalg_jax.py tests/test_dom_linalg_pcg.py tests/test_dom_maps_base.py \
  tests/test_dom_maps_dirichlet.py tests/test_dom_maps_fourier_time.py tests/test_dom_maps_wdm.py \
  tests/test_dom_spectra.py tests/test_dom_testing.py \
  tests/wilson/test_backend.py tests/wilson/test_evolutionary_kernels.py \
  tests/wilson/test_grid_bands.py tests/wilson/test_inner_product_fixes.py tests/wilson/test_lanes.py \
  tests/wilson/test_noise_stationary.py tests/wilson/test_transform_shim.py \
  src/domainia/wilson/transform.py tests/test_dom_cov_multiply.py tests/test_dom_map_dtypes.py \
  tests/test_wrong_kinds_covariance.py tests/test_wrong_kinds_domains.py \
  tests/test_wrong_kinds_maps.py tests/test_wrong_kinds_misc.py tests/wilson/test_checks.py \
  tests/wilson/test_single_precision.py tests/wilson/test_wrong_kinds.py
git commit -m "Text: the 1.x plan tags go; units notes; two port repeats trimmed

The 25 leading 1.x plan tags (24 test modules, one test docstring) are
stripped, and the four sentences that leaned on a tag say what they mean.
dense_lambda_generic and eq45_separable_kernels say that their result is in
the units of U (Wilson units for FourierToWDM.C), and a new test pins
dense_lambda_generic(W.C, ...) to W.push_rows. linalg.py and maps/fourier.py
drop a second mention of the galactic-binary toy model. The commit also
drops the spec citations Tasks 1-13 added to code.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git status --short
```

(sandbox disabled for the git commands). Expected: `git status --short` shows only the plan's own files.

---

### Task 15: The record and the final gates (spec §7, Gates)

**Goal:** Record the four corrections in the Nov 7 plan's errata and add the §1.7 note to spec 2026-11-04. Run the Task 8 `solve` probe of spec §7 and record its counts. Then run every gate of the spec on the final tree, and record the final state in this plan's "Execution errata".

**Assumes the tree after Task 14.** Every code change is in.

**Files:**
- Modify: `docs/superpowers/plans/2026-11-07-domainia-units-operators-fixes.md` (append "### Corrections (2026-11-08)" at the end of its "Execution errata")
- Modify: `docs/superpowers/specs/2026-11-04-domainia-units-operators-fixes-design.md` (§1.7, the bullet "The test docstring records the reason")
- Modify: `docs/superpowers/plans/2026-11-08-domainia-open-decisions.md` (this plan: append "## Execution errata", or extend it if the coordinator has started it)
- Create (scratch, outside the repo): `$TMPDIR/probe8/test_solve_exact.py`, `$TMPDIR/sdist-final/`

**Acceptance Criteria:**
- [ ] The Nov 7 plan ends with a "### Corrections (2026-11-08)" list of four items (spec §7).
- [ ] Spec 2026-11-04 §1.7 carries the dated note on the dropped citation.
- [ ] The `solve` probe ran 30 times with default threads and 30 times with `VECLIB_MAXIMUM_THREADS=1`. Both failure counts are in this plan's errata. The committed test keeps its 1e-12 tolerance.
- [ ] Gates: the full suite and the NumPy lane pass with no warnings summary; `meyer_cases.py --check` prints `baseline reproduces: 1504 arrays`; the bit gate prints `0 of 1504 arrays differ`; the final sdist listing is the seven entries of Task 12; `git status --short` is clean apart from this plan's `.tasks.json`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz` → `0 of 1504 arrays differ`

**Steps:**

- [ ] **Step 1: The Nov 7 plan's corrections**

In `docs/superpowers/plans/2026-11-07-domainia-units-operators-fixes.md`, the file ends with the line below (its "Minor notes, left open" list). Replace it:

```markdown
  - The toy-model ports are mentioned more than once in a few modules.
```

with

```markdown
  - The toy-model ports are mentioned more than once in a few modules.

### Corrections (2026-11-08)

Found by the survey for spec 2026-11-08 (`docs/superpowers/specs/2026-11-08-domainia-open-decisions-design.md`), which settles the open points above:

- "22 files" (Known open points) is 24 files and 25 lines: 24 module docstrings and one function docstring (`G5:` at `tests/wilson/test_evolutionary_kernels.py:119`).
- "The integer-kind check is written out five times in `domainia.wilson`" (Review notes left open) is six times in `wilson`, plus `domains._int`.
- "No `jax.jit` test of … an `after()` map" held only for the adjoint: at 779550d `apply` was jitted at `tests/test_dom_maps_fourier_wdm.py:253-268`. Plan 2026-11-08 Task 13 jits the adjoint in the same loop.
- The `DenseOperator` `rtol` question was raised after the final review, in neither list above. Spec 2026-11-08 §3 settles it.
```

- [ ] **Step 2: The spec 2026-11-04 §1.7 note**

In `docs/superpowers/specs/2026-11-04-domainia-units-operators-fixes-design.md`, replace

```markdown
* The test docstring records the reason: units, this spec §1.
```

with

```markdown
* The test docstring records the reason: units, this spec §1.
  *(Note, 2026-11-08: the Nov 7 plan's text sweep, Task 15, dropped the citation of this spec
  on purpose. Its zero-hit gate on spec citations in code overrides the request; the
  docstring gives the reason without naming the spec.)*
```

- [ ] **Step 3: The Task 8 `solve` probe (spec §7)**

The Nov 7 plan's Task 8 compares `solve` to 1e-12 in `tests/test_operators_covariance.py::test_a_reversed_channel_dict_meets_the_same_blocks`, because the exact form failed in some pytest runs. Make a scratch copy that compares `solve` exactly, outside the repo:

```bash
cd /Users/aaron/Documents/lisa/domainia
rm -rf $TMPDIR/probe8 && mkdir -p $TMPDIR/probe8 && cp tests/test_operators_covariance.py $TMPDIR/probe8/test_solve_exact.py
PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P - <<'PYEOF'
import os
from pathlib import Path
p = Path(os.environ["TMPDIR"]) / "probe8/test_solve_exact.py"
s = p.read_text()
old = '''            if method == "solve":   # triangular solves are not bit-reproducible on every BLAS
                _close(b[k], a[k], 1e-12)
            else:
                np.testing.assert_array_equal(a[k], b[k])
'''
assert s.count(old) == 1
p.write_text(s.replace(old, "            np.testing.assert_array_equal(a[k], b[k])\n"))
print("ok")
PYEOF
```
Expected: `ok`. If the assert fails, an earlier task changed that test's loop. Make the same edit by hand, so that `solve` uses `np.testing.assert_array_equal` like `apply` and `project`, and record this.

Run 30 times with default threads, then 30 times with one Accelerate thread:
```bash
cd /Users/aaron/Documents/lisa/domainia
f=0; for i in $(seq 30); do PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -c pyproject.toml --rootdir . $TMPDIR/probe8/test_solve_exact.py -k reversed_channel >/dev/null 2>&1 || f=$((f+1)); done; echo "default threads: $f of 30 runs failed"
f=0; for i in $(seq 30); do VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q -c pyproject.toml --rootdir . $TMPDIR/probe8/test_solve_exact.py -k reversed_channel >/dev/null 2>&1 || f=$((f+1)); done; echo "VECLIB_MAXIMUM_THREADS=1: $f of 30 runs failed"
```
Expected: two lines, `default threads: A of 30 runs failed` and `VECLIB_MAXIMUM_THREADS=1: B of 30 runs failed`. Any A and B are a result. While planning (at 3c1dc13, before Task 1), A was 3 and B was 6, so one thread does not make the last bit reproducible. If a run fails, check that the failure is on `solve`: run the probe command once without `>/dev/null 2>&1` and read the `FAILED` lines and the mismatch. The committed test is not changed.

- [ ] **Step 4: The final gates**

Run each from `/Users/aaron/Documents/lisa/domainia`:

1. The full suite and the NumPy lane (Conventions). Expected: both pass, with no warnings summary. Note the two summary lines.
2. `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`. Expected: `baseline reproduces: 1504 arrays`.
3. `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/domainia_bits.py check $TMPDIR/domainia-bits-779550d.npz`. Expected: `0 of 1504 arrays differ`.
4. `~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/gate9.py`. Expected: `0 TypeError messages with '; got'`.
5. The sdist, from the final commit, with stray paths planted (as Task 12 Step 7):
```bash
cd /Users/aaron/Documents/lisa/domainia
rm -rf $TMPDIR/sdist-final && mkdir -p $TMPDIR/sdist-final/tree
git archive HEAD | tar -x -C $TMPDIR/sdist-final/tree
cd $TMPDIR/sdist-final/tree && mkdir -p .superpowers/sdd strayfolder
echo x > .superpowers/sdd/brief.md && echo y > stray.txt && echo z > strayfolder/a.txt
~/miniconda3/envs/sksparse_test/bin/python -m hatchling build -t sdist -d $TMPDIR/sdist-final/dist
tar -tzf $TMPDIR/sdist-final/dist/domainia-2.0.0.tar.gz | awk -F/ '{print $2}' | sort -u
```
Expected: exactly `.gitignore`, `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml`, `src`, `tests`.

6. `git status --short`. Expected: nothing, or only ` M docs/superpowers/plans/2026-11-08-domainia-open-decisions.md.tasks.json` (the coordinator's file).

- [ ] **Step 5: This plan's errata**

At the end of `docs/superpowers/plans/2026-11-08-domainia-open-decisions.md`, append this section. If the coordinator has already started a `## Execution errata` section, add the two subsections to it instead. Fill in the numbers you measured in Steps 3 and 4:

```markdown
## Execution errata

### Final state

- Full suite: <summary line of Step 4.1>. NumPy lane: <summary line>.
- `meyer_cases.py --check`: baseline reproduces, 1504 arrays. Float64 bit gate against the baseline recorded before Task 1: `0 of 1504 arrays differ`.
- No `TypeError` message in `src` keeps the `"; got"` form.
- The sdist built from the final commit (hatchling, scratch env) holds `.gitignore`, `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml`, `src/` and `tests/` only, with stray paths planted at the root.

### The Nov 7 Task 8 `solve` probe (spec §7)

- A scratch copy of `test_a_reversed_channel_dict_meets_the_same_blocks` that compares `solve` exactly, run 30 times under pytest: <A> of 30 runs failed with default threads, <B> of 30 with `VECLIB_MAXIMUM_THREADS=1`.
- The committed test keeps its 1e-12 tolerance.
```

Write the measured values in place of each `<…>`. No `<` may remain in the section.

- [ ] **Step 6: Commit**

```bash
cd /Users/aaron/Documents/lisa/domainia
git add docs/superpowers/plans/2026-11-07-domainia-units-operators-fixes.md docs/superpowers/specs/2026-11-04-domainia-units-operators-fixes-design.md docs/superpowers/plans/2026-11-08-domainia-open-decisions.md
git commit -m "Record: the Nov 7 corrections, the solve probe and the final gates

The Nov 7 plan's errata get the four corrections of spec 2026-11-08 §7,
and spec 2026-11-04 §1.7 a note on the dropped citation. This plan's
errata hold the final state and the Task 8 solve probe's counts.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(sandbox disabled for both git commands)

---

## Known open points (FOR USER, found while drafting; not in this plan's scope)

### Size

The user said, on seeing the drafted totals: "The overall code should be fairly small for this project." A later round is expected to cut code. These are the plan's numbers, measured on a scratch tree that holds every drafted patch (Tasks 1–14).

| | before (779550d) | after Task 14 | change |
|---|---|---|---|
| `src` lines (`*.py`) | 11762 | 12302 | +540 (+5%) |
| `tests` lines (`*.py`, without `vendored`/`oracle`) | 9155 | 11623 | +2468 (+27%) |
| tests run, full suite | 1768 | 3155 (3152 passed, 3 skipped) | +1387 |
| full suite, wall time | 68–72 s | 77–90 s | about +12 s (runs vary by about 10 s) |

Where the new tests come from (collected cases):

* the wrong-kind files, one case per slot × kind: `tests/wilson/test_wrong_kinds.py` +216, `tests/test_wrong_kinds_covariance.py` 206, `tests/test_wrong_kinds_domains.py` 190, `tests/test_wrong_kinds_maps.py` 181, `tests/test_wrong_kinds_misc.py` 111 (904 in all);
* `tests/test_dom_map_dtypes.py`, every map × input dtype: 279;
* `tests/wilson/test_checks.py`, the helpers: 103;
* everything else, doctests included: about 100.

Cut candidates for that round, in order of size: one test per module that loops over its slots and kinds, instead of one parametrised case per pair; a sample of the map × dtype matrix (one map per module, one real and one complex single-precision dtype) instead of all of it; `test_checks.py` cases the wrong-kind files already cover. In `src`, `_checks.py` is 277 lines, about half of them docstrings and doctests.

### Arguments still unchecked

These take a wrong kind without a `TypeError`, or raise a different error. They are outside §2's list, so the plan leaves them.

* `be=` everywhere: `be='numpy'` raises a bare `AttributeError`. One shared check in `get_backend` would cover it.
* `wilson`, not listed in §2.3: `carrier_phase` `Nt`, `edge_columns` `Nt`, `band_indices` `m`, `delta_T` `Nf`, `evolutionary_band_variance` `bands`, `wrap_fractions`/`wrap_cut_columns` `bands`, `wrap_cut_columns` `tol`, `eq45_coeffs_interp` `k`, `eq45_coeffs_bessel`/`bessel_i` `P`, `time_domain_factor`/`time_domain_covariance`/`draw_noise` `N` and `dt`, `stationary`/`is_diagonal` `rtol`.
* Value arrays in Task 6's modules (`spectra`, `data`, `translate`, `variance`, `linalg`) are converted by `np.asarray` without a dtype check; only the slots the spec lists are checked.
* `DenseOperator.rtol` is annotated `float` with `default=None` and a `# type: ignore[assignment]` (Task 8), so that the annotation says the stored value is never `None`. Under mypy an explicit `rtol=None` is an arg-type error. Nothing in the repo runs mypy.
* Found while prototyping, not new: `jax.jit(lambda nu, b: cov.multiply(nu, b, be=jax_be))` on a covariance whose JAX constants are not cached yet leaks a tracer (`UnexpectedTracerError` on the second call): `Covariance.constants()` caches what it `device_put`s inside the trace. `multiply`'s docstring says "eagerly", so nothing promises otherwise.

### Text left for later

* Task 14's sweep removes the spec citations from code (the gate is a bare `§`, `spec 2026` or a 2026 date in `src`/`tests` `*.py`). Three references to documents stay, because the gate does not match them: `src/domainia/translate.py:1` ("domainia 2.0 spec, section 3", already at 779550d), `tests/test_dom_restored_coverage.py:1` ("the Task 2 pruning", already at 779550d), and the "Task 1" in a `#:` comment that Task 3 adds to `tests/test_wrong_kinds_domains.py` (`INT_SLOTS`).

### Where the spec and the code disagree

The plan follows the code in each case, and the task that met it says why.

* §3.5, "the `wmn`-complex tests the grep finds": there are none. Task 2 adds them.
* §2.10: no test pins `Meyer d must be an int` or `must be an int;` (Task 1's grep).
* §2.3 lists `pcg`'s `maxit` among the `int()` sites, but `pcg` uses `maxit` (and `tol`) as given, with no conversion. Task 6 checks both.
* §2.8, "three silent wrong answers ... on the multiply path and on the likelihood path": a `(1,)` `b` through `DiagonalCovariance`'s `quad` (solve form) already raised NumPy's matmul `ValueError`. The other paths were silent, as the spec says.
* §2.8: a ragged `BlockTriCovFactor` whose factorisation failed keeps raising `"the factorisation failed ..."`. The shape check runs only when the factorisation succeeded (Task 7).
* §1.1, "a push through `Identity` or `BinSelect` returns the dtype of the covariance's `dense()`": true for `BinSelect` and for an `Identity` on a `FourierDomain`. An `Identity` on a `TimeDomain` or `WDMDomain` is a `"real"` push and returns float64 for a float32 matrix. Task 10's docstring says so.
* Context, float32 overflow: the bins of Task 10's large-`dt` test give true cells up to 3.6e37, where the spec's probe measured 1.7e33. Both fit float32, which is all the test needs.
* §3.6: `rtol=0` on `_psd(n, n - 7)` is refused on most seeds, not all: of 20000 seeds, 19983 were refused with the round-off message and 17 were accepted. Task 8 fixes seed 17 and asserts the premise first.
* §5 files the jitted `after()` adjoint under `tests/test_dom_restored_coverage.py:63-66`. Task 13 puts it in `tests/test_dom_maps_fourier_wdm.py`'s jit-parity loop instead, which already builds those maps.

## Execution errata

### Final state

- Full suite: 3152 passed, 3 skipped in 132.27s. NumPy lane: 2888 passed, 267 skipped in 40.02s.
- `meyer_cases.py --check`: baseline reproduces, 1504 arrays. Float64 bit gate against the baseline recorded before Task 1: `0 of 1504 arrays differ`.
- No `TypeError` message in `src` keeps the `"; got"` form.
- The sdist built from the final commit (hatchling, scratch env) holds `.gitignore`, `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml`, `src/` and `tests/` only, with stray paths planted at the root.

### The Nov 7 Task 8 `solve` probe (spec §7)

- A scratch copy of `test_a_reversed_channel_dict_meets_the_same_blocks` that compares `solve` exactly, run 30 times under pytest: 4 of 30 runs failed with default threads, 10 of 30 with `VECLIB_MAXIMUM_THREADS=1`.
- The committed test keeps its 1e-12 tolerance.

### Coordinator notes (subagent-driven execution, 2026-11-08)

- One implementer and one reviewer (spec compliance, then quality, in one report) per task; a final whole-branch review on top. Task 0 has no code and was run by the coordinator. Every task's committed `src`/`tests` change matched its prototype: byte for byte for Tasks 1–10 and 12–13, with only hunk offsets differing for Task 11 (from Task 10's docstring), and for Task 14 the tree equals the merged prototype. Every per-task review: approved, no Critical or Important finding.
- Commits: T1 c6072d1, T2 6022076, T3 98cadc5, T4 7dfe85a, T5 7f9b9a4, T6 1795907 + 14ad16d, T7 4f912de, T8 5b6c0e9, T9 dad8058, T10 a63bef4, T11 3d58b5e, T12 f2a2bed, T13 46e3f8b, T14 d168739, T15 8bf4653, final-review fixes af54874.
- Deviations: from Task 2 on, each commit's `Co-Authored-By` line is the implementer's own model (Sonnet 5.5; Haiku 5.5 for Task 12), not the plan's line. Task 8's red run came after the source edit (done by restoring HEAD's `operators.py`: 20 failed, 51 passed, as expected). Task 12 Step 1's grep has four hits, not three: `tests/test_package.py:69` (the existing `('jax', 'jaxlib')` filter) also matches; it is untouched.
- The final review found a regression the suite missed. Task 7's `if bool(self.ok):` in `BlockTriCovFactor._apply` raised `TracerBoolConversionError` for a ragged factor passed into `jax.jit`; it now uses `linalg._failed`, with a pin in `test_the_check_runs_on_jax_and_under_jit`. It also fixed `wilson.inner_product.gaussian_from_func`, which truncated a complex PSD (now `sanitize_psd` first, as `matrix.py` does), and limited the README's complex-values sentence to `domainia.translate`. After af54874: full suite 3153 passed, 3 skipped; NumPy lane 2889 passed, 267 skipped; bit gate `0 of 1504 arrays differ`; gate9 `0`.
- Left open (minor, for the cut): some kind `TypeError`s lack the `", got"` form (`variance.py:45`, `linalg.py:269`, the masked-array messages; `ChannelAxis(3)`, `BandLayout.from_sizes(3)`, `BlockTridiagonalCovariance(blocks, 3)` and `ChannelStack` give Python's "not iterable"); the same check is worded differently across modules; silent complex casts outside the spec's lists (`testing.wdm_congruence` `Sigma`, `PointwiseOperator`, `Covariance.factor`'s state key); `DenseOperator.rtol`'s `float` annotation with a `None` default; `index_array` lets integral floats >= 2**63 wrap; a few checks run twice (`quad_form`'s `form`, `BlockTriCovFactor._b_shape(C)`, `WDMToTime.adjoint`).
- Size cut, ranked by the final review: (1) the wrong-kind matrices, about 1745 lines and 1007 cases, to one wrong value per slot (about 700 lines, 200 cases); (2) one table-driven wrong-kind file instead of six (150–200 lines more); (3) `src` docstrings, about 200 lines (`_checks.py` is 180 docstring lines of 277); (4) `tests/test_dom_map_dtypes.py`, 279 cases to 60–80; (5) optionally `require_str`/`require_callable` folded into `require_instance`.
