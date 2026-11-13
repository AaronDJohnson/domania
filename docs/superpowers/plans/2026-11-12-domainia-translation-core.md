# domainia 3.0 Implementation Plan: the translation core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved spec `docs/superpowers/specs/2026-11-12-domainia-translation-core-design.md` on branch `translation-core`. domainia becomes a small fp64 NumPy library with two jobs:
* translating data arrays between time, frequency and WDM (`domainia.transform`);
* evaluating a noise model as a per-point K×K covariance on a frequency or WDM grid (`domainia.frequency_covariance`, `domainia.wdm_covariance`).

Everything else goes: the likelihoods, factorisations, covariance classes, operators, coordinates, the dict API, the maps, the JAX backend and float32.

**Architecture:**
* Task 1 makes the Meyer replay a 1e-12 tolerance gate that survives every later deletion. Tasks 2–7 then cut the package down under that gate.
* Tasks 3, 8 and 9 add the three public functions, test first, in two new private modules (`_transform.py`, `_covariance.py`) on the kept Wilson kernels.
* Tasks 10–12 remove the dead code and the docstring text, and Task 13 runs the gates on Python 3.12–3.14, at the dependency floors and at current versions.

**Tech Stack:** Python ≥3.12, NumPy ≥2.1, SciPy ≥1.14.1, pytest ≥8. Test environments:
* `~/miniconda3/envs/wdmrel`: Python 3.12.4, numpy 2.3.5, scipy 1.16.3, jax 0.9.0.1. It is the main lane until Task 5 removes JAX, and it stays a lane afterwards.
* `$TMPDIR/dom3-envs/{floor312,cur313,cur314}`: uv venvs without JAX that Task 0 creates. floor312 has numpy 2.1.0 and scipy 1.14.1; cur313 and cur314 have current numpy and scipy.

**User decisions (already made):**
* 2026-11-12, the spec's quoted decisions: fp64 only, with no jit and no float32; delete JAX; Enchilada holds no likelihood code; drop the channel layer and the dict API; "domainia doesn't need to compute the log likelihood right now … The only thing that domainia needs to do is translate data and covariance between domains"; "The Wheel absolutely asks domainia"; "keep the evolutionary noise parts"; "The wheel must hold the total noise model"; "yes, per-point KxK is fine for now. write the spec".
* 2026-11-12, the written spec: "create a branch and let's go". This approves the spec as written, with its numbered decisions. Decision 3 therefore takes option (a), its recommended option. Decision 11's cuts are "made only if the user says so", and the user named none, so none is made: `stationary.py`, the Gram form, `RampWindow` and `SampledWindow`, and `matrix.py`'s test-only helpers all stay.
* Standing rules:
  * commit on `translation-core` only, and never push;
  * Enchilada and `~/Documents/lisa/wilson` are read-only;
  * keep code and tests small, and flag growth early;
  * no tag, rename or publication.

**Deviation from the spec's Order, recorded here:**
* The spec removes JAX (its step 3) before it deletes the data route (its step 4). This plan deletes `translate.py`, `maps/`, `data.py` and `domains.py` first (Tasks 3–4), so their ~1,700 lines are never converted to NumPy only to be deleted.
* `spectra.py` and `variance.py` import `domains` and `maps`, so they go in Task 4 instead of the spec's step 5.
* Folding the kernels (eight entry points into four) moves to Task 7, after JAX and float32 are gone, so that it touches each kernel once.
* The end state is the spec's.

**Where the spec is silent or loose, this plan decides** (the user can overrule any of these):
* `transform` refuses a `numpy.ma.MaskedArray` with `TypeError`, as 2.0's `translate` did, because `numpy.asarray` drops the mask and the masked values would leak into the result (Task 3).
* The `testing` oracles take a boolean `(Nf + 1, Nt)` cell mask, the layout of `wilson.active_mask` and of `wdm_covariance`'s output, with the grid first: `dense_W(Nf, Nt, dt, mask, *, window)`. The spec says the mask is the one "`eq45_plan` already takes"; `eq45_plan`'s `keep` is instead a mask over the `N` packed active cells, and the private helper `testing._keep` converts (Task 4). A mask that selects an inactive cell is refused.
* The replay fixture for `transform` is `tests/data/translate_reference.npz`, recorded from 2.0's `translate.transform` before Task 4 deletes it (Task 1).
* §3.1's rule `|S_ab − conj(S_ba)| ≤ tol` is applied for a ≠ b. The diagonal has its own rule, `|Im S_aa| ≤ tol`. Read literally at a = b, the first rule would bound `|Im S_aa|` by tol/2 and make the second redundant (Task 8).
* The folded kernels keep their first parameter's name: `x_rfft`, `wmn` or `x`, not the spec's `X` or `w` (Task 7).
* `wdm_covariance` calls the model once on band 0 and two columns before the chunks. That call tells a stationary model (size-1 time axis) from an evolutionary one, and gives K. Without it, a stationary model was evaluated once per column chunk: 26,637 calls and 14.4 s at 2048×2048, against 50 calls and 0.69 s with it. The chunk size is the module constant `CHUNK = 2**16` points, so the signature stays the spec's. A K that changes between calls is refused (Task 9).
* A float32 scalar `dt` is accepted and computed in float64, as float32 arrays are (Task 10).
* `evolutionary.time_domain_factor` checks `dt` with `validate_dt`, like every other entry point. Through it, so do `time_domain_covariance`, `draw_noise` and `exact_covariance`. Before, `dt=True` ran as 1.0 and `dt=-1.0` returned NaN entries; now both are refused (Task 10).
* Of the 10 vendored Eq. 39 tests, one Monte-Carlo test (8 lines) stays, with `band_averaged_variance` as its reference. It is the only test that `colored_noise` followed by `forward` converges to the analytic variance (Task 10).
* A name that leaves an `__all__` also leaves that package's namespace, and stays importable from its submodule. `WindowError` is not in §5's list, so it is found only at `domainia.wilson.windows.WindowError`. It subclasses `ValueError`, so a caller can catch that instead (Task 11).
* `domainia.__all__` keeps `__version__` beside §5's four names (Task 11).
* `test_dom_testing`'s `edge_columns(NT, 3, 1)` becomes the explicit columns `np.r_[0:3, NT - 1]`, which are the same columns. §3.3 says the test "is rewritten with `wrap_cut_columns`", but `wrap_cut_columns(8, 16, 1/16)` is `(6, 5)` and would exclude 11 of the 16 columns (Task 11).
* The docstring trim keeps every doctest verbatim. It keeps each notes `.tex` path once, in a module docstring. It also keeps two test docstrings that name deleted code (`meyer_cases.domainia_cases`, `test_meyer_baseline`'s dropped-families test), because they explain recorded baseline key names (Task 12).

**Scope check:** one plan. The spec covers one package, with one Gates section. Enchilada's changes (spec §8) are a hand-off to an Enchilada session and are not tasks here.

---

## Conventions (every task)

* **Repository.** `REPO=/Users/aaron/Documents/lisa/domainia`, the main checkout, on branch `translation-core`. There is no worktree. Work only in `$REPO`, and never `git checkout` another branch.
* **`$PY`** is shorthand for the prefix `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P`, run from `$REPO`.
  * The Bash tool runs zsh, which does not word-split a variable. So `PY="…"; $PY -m pytest` fails, and so does `${=PY}`, because the `NAME=value` words it yields are not taken as assignments. Write the prefix out in full in each command, as every task's commands do.
  * Quote globs too (`--include='*.py'`): in zsh an unmatched glob is an error.
  * A probe script outside pytest needs `PYTHONPATH=src`, because the package is not installed in `wdmrel`.
  * From Task 5 on, `JAX_PLATFORMS=cpu` changes nothing, because nothing imports jax. Task 13 leaves it out.
* **`$FLOOR`** is shorthand for `PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P`.
* **Lanes.** Each must pass at the end of every task, with no collection errors and no new warnings:
  * the full suite, `$PY -m pytest -q`;
  * the NumPy lane, `WDM_TESTS_HIDE_JAX=1` + `$PY -m pytest -q`, until Task 5 removes JAX;
  * the floor lane, `$FLOOR -m pytest -q`.

  Task 13 adds the `cur313` and `cur314` lanes (Python 3.13 and 3.14 at current numpy and scipy), which Task 0 creates.
* **Order.** The tasks run one at a time, in order. Most files are edited by several tasks (see "Files edited by several tasks" below). Every Old text is quoted from the tree after the task before.

  Steps state pass counts only for the task's own test files. For the rest they say "passes", because suite totals drift between tasks. The totals the prototype saw are labelled "(prototype)".
  The suite totals in each task's **Verify** line are the prototype's too. A fix-up commit during execution may shift them; that is not a failure if the task's own tests and the replay pass.
* **The replay gate.** `tests/wilson/test_meyer_baseline.py` replays `tests/wilson/meyer_cases.py` against `tests/wilson/data/meyer_baseline.npz`.
  * From Task 1 on, it passes at 1e-12 with 0 mismatches at the end of every task.
  * Never re-record the npz.
  * A task that deletes or renames a name that `meyer_cases.py` calls rewires those keys, under their old key names, in the same commit.
* **Size.** `$PY $TMPDIR/dom3-size.py` prints src lines and code lines, and the test lines with the oracle copies shown apart. Every task ends by recording it.
  * Budget (spec): src ≤ about 5,400 lines (2,150 code) before the docstring trim (Task 12), and tests ≤ about 5,700 non-oracle lines.
  * A task that finds the budget overrun says so in its report. The coordinator records it under "Known open points".
* **Edits are anchored to code, not line numbers.** Every step names the function, class or exact old text it changes. The spec cites lines at 9f53225, and earlier tasks shift them.
* **Each task starts from the tree after the task before it**, as the Interface contract below defines it.
* **Git.** The Claude Code sandbox blocks writes inside `.git`, so every `git add`, `git mv`, `git rm` and `git commit` runs with the sandbox disabled (`dangerouslyDisableSandbox: true`).
  * End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  * Stage named paths only. Nothing is pushed. Never amend, and never `git stash`.
* **Scratch** scripts go in `$TMPDIR`, never in the repo.
* **Errata.** Deviations agreed during execution go in an "Execution errata" section at the end of this plan.

## File structure (the end state)

`src/domainia/` (15 files, 4,059 lines and 1,849 code lines in the prototype):

| File | Responsibility | Tasks that edit it |
|---|---|---|
| `__init__.py` | the public API: `transform`, `frequency_covariance`, `wdm_covariance`, `wilson`, `__version__` | 2, 3, 4, 5, 8, 9, 11 |
| `_transform.py` (new) | `transform`: data arrays between time, frequency and WDM, with its input checks | 3, 7, 11, 12 |
| `_covariance.py` (new) | the §3.1 model check and projection (`_hermitian`), `frequency_covariance`, `wdm_covariance` (its evaluator and chunking) | 8, 9, 10, 12 |
| `testing.py` | dense oracles for tests (`dense_W`, `wdm_congruence`, `eq45_dense` on a cell mask), plus `colored_noise` and `empirical_variance` | 2, 4, 11, 12 |
| `wilson/transform.py` | the four kernels `forward`, `inverse`, `forward_time`, `inverse_time`, any leading axes, fp64 NumPy | 2, 4, 5, 6, 7, 11, 12 |
| `wilson/grid.py` | grids, the active mask, PSD sanitising, the grid and `dt` checks | 2, 5, 6, 10, 11, 12 |
| `wilson/_checks.py` | argument checks | 4, 5, 11, 12 |
| `wilson/filter.py` | the Meyer filter `phi_centered` | 10, 11, 12 |
| `wilson/windows.py` | `Window`, `Meyer`, `MEYER`, `RampWindow`, `SampledWindow`, `WindowError`, `sample_window` | 5, 6 |
| `wilson/noise.py` | Eq. 36: `band_averaged_variance`, `evolutionary_band_variance` | 2, 6, 10, 11, 12 |
| `wilson/evolutionary.py` | the Eq. 45 builders, the exact time-domain references, the wrap rules | 2, 4, 5, 7, 10, 12 |
| `wilson/stationary.py` | stationary builders | 5, 10, 11, 12 |
| `wilson/matrix.py` | dense references (`build_W`, `active_W`, `build_C_time`, `build_C_wdm`) | 6, 10, 12 |
| `wilson/__init__.py` | `domainia.wilson`'s exports (spec §5) | 2, 5, 7, 10, 11, 12 |

Deleted: `translate.py`, `maps/`, `data.py`, `domains.py`, `spectra.py`, `variance.py`, `linalg.py`, `operators.py`, `covariance/`, `wilson/inner_product.py`, `wilson/backend.py`, `wilson/special.py`.

Tests:
* new: `tests/test_transform.py` (Task 3), `tests/test_covariance.py` (Tasks 8–9) and the fixture `tests/data/translate_reference.npz` (Task 1);
* the replay: `tests/wilson/meyer_cases.py` and `tests/wilson/test_meyer_baseline.py`, against the untouched `tests/wilson/data/meyer_baseline.npz`;
* the oracle copies under `tests/wilson/oracle/` stay whole.

## Files edited by several tasks

The tasks run strictly in order (each `blockedBy` the one before), because these files are edited by four or more tasks. Each task quotes its Old text from the tree the task before it leaves.

| File | Tasks |
|---|---|
| `tests/wilson/test_window_threading.py` | 1, 2, 3, 4, 5, 7, 8, 9, 11 |
| `src/domainia/__init__.py` | 2, 3, 4, 5, 8, 9, 11 |
| `src/domainia/wilson/transform.py` | 2, 4, 5, 6, 7, 11, 12 |
| `tests/test_package.py` | 2, 4, 5, 8, 9, 11 |
| `tests/wilson/test_wrong_kinds.py` | 2, 5, 6, 7, 10, 11 |
| `src/domainia/wilson/grid.py` | 2, 5, 6, 10, 11, 12 |
| `src/domainia/wilson/__init__.py` | 2, 5, 7, 10, 11, 12 |
| `src/domainia/wilson/evolutionary.py` | 2, 4, 5, 7, 10, 12 |
| `README.md` | 2, 3, 4, 5, 11 |
| `src/domainia/wilson/noise.py` | 2, 6, 10, 11, 12 |
| `tests/wilson/meyer_cases.py` | 1, 4, 7, 11 |
| `tests/wilson/test_grid_bands.py` | 2, 5, 7, 11 |
| `tests/wilson/test_noise_stationary.py` | 5, 6, 10, 11 |
| `src/domainia/wilson/_checks.py` | 4, 5, 11, 12 |
| `src/domainia/_covariance.py` | 8, 9, 10, 12 |
| `src/domainia/_transform.py` | 3, 7, 11, 12 |
| `src/domainia/testing.py` | 2, 4, 11, 12 |
| `src/domainia/wilson/stationary.py` | 5, 10, 11, 12 |

## Interface contract (names later tasks rely on)

* **Task 0**:
  * `$TMPDIR/dom3-size.py` prints src lines and code lines, and the test lines with the oracle copies shown apart; `-v` adds the per-file counts.
  * the venvs `$TMPDIR/dom3-envs/{floor312,cur313,cur314}`.
* **Task 1**:
  * `tests/wilson/meyer_cases.py`: `DROPPED` (7 key prefixes), `RTOL = 1e-12`, `mismatches(got, want)`, `domainia_cases(wl, Nf, Nt, dt)`, `compute(wl, wilson_kw, domainia_kw, *, only=None)`, `psd_ft(T, depth=0.5)`, `THREADED`.
  * The fixture `tests/data/translate_reference.npz` holds `routes` (30, 2), `grids` (30, 6) = (N, dt, source Nf, source Nt, target Nf, target Nt), and `x{i}`, `y{i}`.
* **Task 3**, `src/domainia/_transform.py`: `transform(x, source, target, *, N, dt, source_wdm=None, target_wdm=None, window=MEYER)`, re-exported as `domainia.transform`; `RESIDUE = 1e-10`.
* **Task 4**, `src/domainia/testing.py`:
  * `dense_W(Nf, Nt, dt, mask, *, window=MEYER)`, `wdm_congruence(Sigma, Nf, Nt, dt, mask, *, window=MEYER)` and `eq45_dense(Nf, Nt, dt, mask, Sn, *, P=24, t_max=None, window=MEYER)`, where `mask` is a bool `(Nf + 1, Nt)` array;
  * the private `_keep(Nf, Nt, mask)`, which converts the mask to `eq45_plan`'s packed `keep`.
* **Task 7**, `domainia.wilson.transform`:
  * `forward(x_rfft, Nf, Nt, dt, *, window=MEYER)`, `inverse(wmn, Nf, Nt, dt, *, window=MEYER)`, `forward_time(x, Nf, Nt, dt, *, window=MEYER)` and `inverse_time(wmn, Nf, Nt, dt, *, window=MEYER)`, each with any leading axes.
  * Until Task 5 they also take `be=`, and the `_many` forms exist until Task 7.
  * The spec writes the first parameter as `X` or `w`. The plan keeps today's names, because every caller passes it by position and the names appear in pinned messages.
* **Task 8**:
  * `src/domainia/_covariance.py`: `TOL = 1e-12`; `_hermitian(out, points, name)`, the check and projection, which returns `H = ½(S + Sᴴ)` in the output's own shape; `frequency_covariance(S, *, N, dt)`.
  * `tests/test_covariance.py`: `ROUTES`, `sigma(S, N, dt)`, `error(got, want)`, and the models `psd`, `delayed`, `second`, `rounded` and `plus`.
* **Task 9**, same module: `CHUNK = 2**16`, a module constant (the test patches `domainia._covariance.CHUNK`); `wdm_covariance(S, *, Nf, Nt, dt, window=MEYER)`.
* **Task 10**:
  * `wilson.noise` keeps `band_averaged_variance`, `evolutionary_band_variance`, `colored_noise` and `empirical_variance`; the four Eq. 39 names are gone.
  * `evolutionary.time_domain_factor` calls `validate_dt`.
* **Task 11**:
  * `domainia.__all__` is `__version__`, `frequency_covariance`, `transform`, `wdm_covariance` and `wilson`.
  * `domainia.wilson.__all__` is spec §5's 19 names.
  * The names that left live in submodules: `wilson.grid.*` (`active_mask`, `band_frequencies`, `delta_F`, `delta_T`, `n_samples`, `time_pixel_centers`, `validate_dt`, `validate_finite`, `validate_grid`), `wilson.windows.{WindowError, require_window, sample_window}` and `wilson.filter.phi_centered`.
  * `colored_noise` and `empirical_variance` are `domainia.testing.colored_noise` and `domainia.testing.empirical_variance`.
  * `tests/test_package.py` has `PUBLIC` and `WILSON_PUBLIC`.
  * The version is `3.0.0`.
* **Task 12**: `$TMPDIR/dom3-ast.py REV` compares every src file at `REV` with the working tree, with docstrings stripped and doctests compared.
* **Task 13**: `$TMPDIR/dom3-drift.py` (the replay's worst drift and its mismatches) and `$TMPDIR/dom3-imports.py` (what `import domainia` loads, sorted by origin).

## Spec coverage

| Spec section | Tasks |
|---|---|
| Decisions (user), Decisions in this spec | the header ("User decisions"); decision 3 (a) and decision 11 (no cuts) |
| §1 The boundary | 2 (the likelihood layers go), 3, 8, 9 (the three functions); the Wheel's side is §8 |
| §2 Data translation, its kernels and the 1e-10 residue | 3 (`transform`), 7 (the fold to four kernels), 1 (the recorded `translate.transform` outputs) |
| §3.1 The model: the check and the projection | 8 (`_hermitian`); 9 applies it per chunk |
| §3.2 `frequency_covariance` | 8 |
| §3.3 `wdm_covariance` | 9 (and Task 11 for the `test_dom_testing` columns) |
| §3.4 Linearity | 8 and 9 (tests) |
| §3.5 What a block does with it | 11 (README); no code |
| §4 What goes | 2 (likelihoods, factorisations, covariance classes, operators, coordinates); 4 (data route, `spectra.py`, `variance.py`); 5 (JAX, `special.py`); 6 (float32); 7 (band-limited and `_many` entry points); 10 (Eq. 39); 11 (dead code inside the kept files) |
| §5 What stays, and how it changes | 5 (`evolutionary.py`: `pytree_dataclass`, `scipy.special.ive`); 6 and 10 (fp64, including `dt`); 7 (the kernels); 11 (the public API) |
| §6.1 Replay gate first | 1; every task ends with the replay at 1e-12 |
| §6.2 New tests | 3, 8, 9 |
| §6.3 What happens to today's tests | 2, 4, 5, 6, 7, 10, 11 |
| §7 Packaging and text | 5 (the `[jax]` extra, the `jax` and `slow` markers); 11 (version, description, README, module docstrings); 12 (the docstring trim) |
| §8 Hand-off to Enchilada | not a task (Enchilada is read-only here); see Known open points |
| Size budget and gates | every task records the size; 13 runs the gates |
| Order | the header's "Deviation from the spec's Order" |
| Risks | 9 (timing of `wdm_covariance`); Known open points |
| Non-goals | none is built |

## Size after each task (prototype)

| After | src lines / code | tests (+ oracle 4,135) |
|---|---|---|
| start (Task 0) | 12,252 / 5,669 | 13,249 |
| Task 1 | 12,252 / 5,669 | 13,164 |
| Task 2 | 8,592 / 3,850 | 8,677 |
| Task 3 | 8,731 / 3,941 | 8,806 |
| Task 4 | 6,560 / 2,605 | 5,769 |
| Task 5 | 5,796 / 2,174 | 4,760 |
| Task 6 | 5,624 / 2,125 | 4,528 |
| Task 7 | 5,219 / 1,956 | 4,337 |
| Task 8 | 5,310 / 2,005 | 4,486 |
| Task 9 | 5,367 / 2,037 | 4,596 |
| Task 10 | 5,188 / 1,983 | 4,490 |
| Task 11 | 4,825 / 1,849 | 4,346 |
| Task 12 | 4,059 / 1,849 | 4,346 |

The budget holds. Before the trim (Task 11), src is 4,825 lines and 1,849 code lines, against at most about 5,400 and 2,150. The tests are 4,346 lines, against at most about 5,700. src is within budget from Task 7 on, and the tests from Task 5 on. Overall, src shrinks by 67% in lines (8,193) and 67% in code lines (3,820), and the tests by 67% (8,903 lines).

## Known open points

None of these blocks a task. Each is either something the plan leaves as it is or a fact the user may want to act on.

* **Floors on Python 3.12 only.** The spec's gate is "green at the dependency floors … and at current versions, on Python 3.12–3.14". The floors (numpy 2.1.0, scipy 1.14.1) run on 3.12; current versions run on 3.13 and 3.14 (numpy 2.5.4 and scipy 1.18.1 in the prototype). A floor lane on 3.13 was not tried. Whether wheels exist for it was not checked.
* **The replay is bit-exact.** On this machine, every kept replay array equals the baseline exactly in all four lanes, so the 1e-12 tolerance is never used. The spec's 9.24e-14 drift was measured on NumPy 1.26.4, which is below the floor.
* **Three skips in every lane:** `DOMAINIA_ORACLE_ROOT is not set` (`tests/wilson/conftest.py`). The galactic-binary window test needs the paper repo's oracle files.
* **Evolutionary `wdm_covariance` is slow at full size**, as spec Risks expects (Task 9, K = 3). Measured: 10–13 s at 512², 78 s at 1024², and about 10–13 min extrapolated at 2048². A stationary model takes 0.69 s at 2048². The separable fast path is the spec's first follow-up.
* **`Grid` keeps a float32 `dt` as given.** `Grid(6, 10, np.float32(0.1)).dt` is a float32. Everything computed from a `Grid` goes through the functions converted in Task 10, so the results are exact.
* **`WindowError` and `__version__`.** See "this plan decides" in the header. Adding `WindowError` to `domainia.wilson.__all__`, or removing `__version__` from `domainia.__all__`, would each be a one-line change and a spec change.
* **Spec text that the plan reads differently**, listed in the header's "this plan decides": §3.1 at a = b, the `testing` mask layout, and §3.3's `wrap_cut_columns` sentence. The spec may want these three edits after execution.
* **Enchilada has moved on.** Enchilada is now at ce66774; the spec cites d1e0419. Only documents changed since: its spec was revised and a plan added, on 2026-11-12. The revised spec does these things:
  * takes §8's drops: the operator view, `TranslatedCovariance`, the seam;
  * removes `from_psd`, which §8 kept;
  * keeps one native covariance on the owner's grid, and refuses a block on another grid (§8 reversed that row);
  * leaves the components layer to a spec after domainia 3.0.

  None of this changes a domainia task. Its noise convention, C_k = (T_obs / 2) S(f_k) under X = dt · rfft(x), is `frequency_covariance`'s diagonal.
* **Stale `__pycache__` directories in the real repo** (`src/domainia/maps/__pycache__`, `src/domainia/covariance/__pycache__`, `tests/__pycache__`, `tests/wilson/__pycache__`). They are untracked. Tasks 2 and 4 delete the two package directories whole. Task 13 Step 9's `find` lists any directory left behind, which would otherwise import as a namespace package.
* **One historical comment in the test tree:** `tests/wilson/vendored/test_evolutionary.py`'s header comment ("… move to domainia in S2"). Task 12 trims src only.

---

## Tasks

### Task 0: Starting state (no commit)

**Goal:** Confirm the code is 9f53225's, the four Python environments and the size script exist, and record the suite totals and size that every later task is measured against.

Assumes the repo at `/Users/aaron/Documents/lisa/domainia` on branch `translation-core`, whose only commits since 9f53225 are the spec (3d4dbbc) and this plan.

**Files:**
- Create: none in the repo. Outside it, only if missing: `$TMPDIR/dom3-size.py` and the venvs `$TMPDIR/dom3-envs/floor312`, `cur313` and `cur314`.
- Modify / Delete: none.
- Test: none new.

**Acceptance Criteria:**
- [ ] `git diff --quiet 9f53225 HEAD -- src tests` exits 0.
- [ ] The three venvs print the versions listed in Step 2.
- [ ] `$TMPDIR/dom3-size.py` prints `src 12252 lines / 5669 code` and `tests 13249 lines (+ oracle 4135)`.
- [ ] PY: `3153 passed, 3 skipped`. NumPy lane: `2889 passed, 267 skipped`. FLOOR: `2889 passed, 267 skipped`. No warnings and no collection errors in any lane.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && git diff --quiet 9f53225 HEAD -- src tests && echo same` -> `same`

The shell is zsh, which does not word-split variables. So every command below spells out its interpreter prefix in full:
- PY = `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P`
- NumPy lane = `WDM_TESTS_HIDE_JAX=1` followed by the PY prefix
- FLOOR = `PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P`

`$TMPDIR` is `/tmp/claude-501`. The package is not installed: pytest's `pythonpath = ["src"]` finds it, and a script run outside pytest needs `PYTHONPATH=src`.

**Steps:**

- [ ] **Step 1: Confirm the branch, a clean tree and 9f53225's code**

  Run: `cd /Users/aaron/Documents/lisa/domainia && git branch --show-current && git status --short && git diff --quiet 9f53225 HEAD -- src tests && echo same && git diff --stat 9f53225 HEAD`

  Expected: `translation-core`, no `git status` lines, then `same`. The stat lists only `docs/superpowers/specs/2026-11-12-domainia-translation-core-design.md` and this plan's files under `docs/superpowers/plans/`.

- [ ] **Step 2: The environments**

  Run: `for e in floor312 cur313 cur314; do $TMPDIR/dom3-envs/$e/bin/python -c "import sys, numpy, scipy, pytest; print(sys.version.split()[0], numpy.__version__, scipy.__version__, pytest.__version__)"; $TMPDIR/dom3-envs/$e/bin/python -c "import jax" 2>&1 | tail -1; done`

  Expected (prototype):
```
3.12.11 2.1.0 1.14.1 9.2.0
ModuleNotFoundError: No module named 'jax'
3.13.14 2.5.4 1.18.1 9.2.0
ModuleNotFoundError: No module named 'jax'
3.14.7 2.5.4 1.18.1 9.2.0
ModuleNotFoundError: No module named 'jax'
```
  - floor312 must show numpy 2.1.0 and scipy 1.14.1 exactly.
  - cur313 and cur314 hold the current numpy and scipy, unpinned, so a fresh install may show later versions than the 2.5.4 and 1.18.1 above.
  - None has jax.

  If all three exist like this, keep them. If one is missing or wrong, delete it with `rm -rf $TMPDIR/dom3-envs/<name>` and create it again with the matching command below.
  - The commands need network access to `pypi.org` and `files.pythonhosted.org`.
  - In auto mode, put both hosts in the Bash call's `allowed_domains`.
  - If `uv` must download the interpreter itself, it also needs `github.com` and `objects.githubusercontent.com`.

  Run: `uv venv -q --python 3.12 $TMPDIR/dom3-envs/floor312 && uv pip install -q --python $TMPDIR/dom3-envs/floor312/bin/python numpy==2.1.0 scipy==1.14.1 "pytest>=8"`

  Run: `uv venv -q --python 3.13 $TMPDIR/dom3-envs/cur313 && uv pip install -q --python $TMPDIR/dom3-envs/cur313/bin/python numpy scipy "pytest>=8"`

  Run: `uv venv -q --python 3.14 $TMPDIR/dom3-envs/cur314 && uv pip install -q --python $TMPDIR/dom3-envs/cur314/bin/python numpy scipy "pytest>=8"`

  Expected: no output. Then repeat the version check at the start of this step.

- [ ] **Step 3: The size script**

  If `$TMPDIR/dom3-size.py` exists with exactly this text, keep it. Otherwise write it with this text:

```python
"""Line counts for the 2026-11-12 plan.  Run from the repo root: ``python size.py``.

Prints src total/code lines and tests total lines (oracle copies excluded, and shown apart).
Code lines exclude blank lines, comment-only lines and docstrings."""
import ast
import io
import sys
import tokenize
from pathlib import Path


def code_lines(path: Path) -> int:
    text = path.read_text()
    doc = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc.update(range(body[0].lineno, body[0].end_lineno + 1))
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(lines - doc)


def total(paths):
    return sum(len(p.read_text().splitlines()) for p in paths)


src = sorted(Path("src").rglob("*.py"))
tests = sorted(Path("tests").rglob("*.py"))
oracle = [p for p in tests if "oracle" in p.parts]
rest = [p for p in tests if "oracle" not in p.parts]
print(f"src {total(src)} lines / {sum(code_lines(p) for p in src)} code")
print(f"tests {total(rest)} lines (+ oracle {total(oracle)})")
if "-v" in sys.argv:
    for p in src:
        print(f"  {total([p]):6d} {code_lines(p):6d}  {p}")
```

  Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

  Expected:
```
src 12252 lines / 5669 code
tests 13249 lines (+ oracle 4135)
```

- [ ] **Step 4: Record the starting suite totals in the three lanes**

  The three runs are independent, so they can run in parallel, at about 40–80 s each.

  Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

  Expected: `3153 passed, 3 skipped in …`. The 3 skips are the oracle-root tests (`tests/wilson/conftest.py:81`, `DOMAINIA_ORACLE_ROOT is not set`).

  Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

  Expected: `2889 passed, 267 skipped in …`

  Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

  Expected: `2889 passed, 267 skipped in …`

  - None of the three summary lines contains `warning` or `error`.
  - These totals and Step 3's size are the reference for every later task. Each later task states its own totals as "(prototype)" figures.

  No commit: this task changes nothing in the repo.

Size after Task 0 (prototype): src 12252 lines / 5669 code; tests 13249 (+ oracle 4135).

---

### Task 1: Replay gate at a tolerance (spec §6.1; Order step 1)

**Goal:** The Meyer replay compares every array to 1e-12 with one comparator. It sets aside the 96 keys of code that goes, recomputes the 238 keys of deleted classes with kept kernels, and collects without importing any module that goes. `translate.transform`'s outputs are recorded as a fixture for Task 3.

Assumes the tree after Task 0 (9f53225's code on branch `translation-core`).

**Files:**
- Create: `tests/data/translate_reference.npz`.
  - The fixture for Task 3's agreement test.
  - It is written by `$TMPDIR/record_translate.py` (Step 5), which lives outside the repo and is not committed.
- Modify `tests/wilson/meyer_cases.py`, which is rewritten whole (Step 2).
  - Changed: the module docstring, `THREADED`, `mismatches`, `psd_ft` (new optional `depth`), `wilson_cases`, `domainia_cases` (new signature `(wl, Nf, Nt, dt)`), `compute` (new signature `(wl, wilson_kw, domainia_kw, *, only=None)`) and `main`.
  - New: `DROPPED`, `RTOL` and `_family`.
  - Removed: `_wilson_units`, `RESCALED`, `RESCALED_RTOL`, `RETIRED`, `_KEY`, `rescale_factor`, and the imports `re` and `SimpleNamespace`.
- Modify `tests/wilson/test_meyer_baseline.py`, which is rewritten whole (Step 1).
  - `test_the_rescaled_families_are_exactly_the_spec_table` and `test_the_rescaled_comparison_rejects_nan_and_a_wrong_factor` go.
  - They are replaced by `test_the_dropped_families_are_96_of_the_1504_keys` and `test_the_comparator`.
  - The `_bit_for_bit` tests are renamed.
  - The module-level imports of `domainia.covariance`, `domains`, `maps`, `spectra`, `testing` and `variance` go, and so does `DM`.
- Modify `tests/wilson/test_window_threading.py`: the module imports, and the `_DM` / `_DCASES` block (Step 3).
- Test: `tests/wilson/test_meyer_baseline.py` and `tests/wilson/test_window_threading.py`.

**What the replay computes after this task.** The 1,504 recorded keys split as follows:
- **Dropped:** 96 keys, set aside by `meyer_cases.DROPPED`:
  - maps: `TimeToWDM.*`, `WDMToTime.*` and `FourierToWDM.*`, 54 keys;
  - `inner_product.*`: 24 keys, 12 of which were already RETIRED;
  - `transform.band_atoms`, `forward_bands` and `inverse_bands`: 18 keys.
- **Rewired now**, under the old key names, to kept kernels:
  - `EvolutionaryCovariance.interp` (230 keys): `eq45_plan(Nf, Nt, dt)` with the default `P = 24`, then `eq45_coeffs_interp` and `eq45_blocks_cheb`;
  - `EvolutionaryCovariance.direct` (2 keys): `eq45_direct(eq45_plan(Nf, Nt, dt, P=2))`. `P=2` is what the class's `_DirectMatrix` passed;
  - `BandAveragedVariance` (6 keys): `noise.evolutionary_band_variance(...)[m, n]` with `m, n = np.nonzero(active_mask(Nf, Nt))`. That is `packed_cells`' order, and `packed_cells` itself goes in Task 11.
  - The model is `GenericPSD.bind([0.4])`, which becomes `psd_ft(T, depth=0.4)`.
  - Each class covered every active cell, so `keep` is the default (all cells).
- **Still today's code; later tasks rewire them under the same key names:**
  - `transform.forward_many`, `inverse_many`, `forward_time_many` and `inverse_time_many` (24 keys): Task 7;
  - `testing.dense_W`, `wdm_congruence` and `eq45_dense` (6 keys): Task 4. Until then they take a `CellSet`, which `domainia_cases.cells()` imports only when a case runs.
  - `noise.empirical_variance` (6 keys): Task 11, which moves it to `testing`.
- **Inputs.** `w = forward(rfft(x))`, and `Wm` stacks `forward` over the rows of `rfft(X)`. `forward` is 1-D only until Task 7.
  - Measured: both equal the old `forward_time(x)` and `forward_time_many(X)` bit for bit on all 6 grid × dt.
  - `wb` and `bands` go.
- **The rng stream of `domainia_cases`.** Before `Sigma`, the old code drew the inputs of the maps cases, which are now dropped:
  - `x` (N), `X` (2N), `y` (N active cells), `xt` (N) and `wf` (2·Nt, bands 1 and 2).
  - One `rng.standard_normal(5 * N + 2 * Nt)` advances the stream identically, so `Sigma`, and with it `testing.wdm_congruence`, is unchanged (measured: replays exactly).

**The comparator** (`meyer_cases.mismatches`):
- `want`'s `DROPPED` keys are set aside. The remaining keys must equal `got`'s exactly, or `mismatches` raises `AssertionError("baseline keys differ: …")`. Coverage is therefore never lost silently.
- dtype and shape must match exactly, and integer arrays must be equal.
- Float and complex arrays pass when `max|got − want| ≤ 1e-12·s`.
  - `s` is `max|want|` over the array.
  - For an all-zero recorded array, `s` is the largest `|want|` over every array of the same function (`layer::case`) in `want`. There are 128 such arrays, in `stationary.build_lambda_blocks_full_active` and `EvolutionaryCovariance.interp`, and both families have a nonzero scale.
  - A function whose arrays were all zero would need exact zeros. No such function exists.
- A NaN fails.

**Acceptance Criteria:**
- [ ] `pytest -q tests/wilson/test_meyer_baseline.py` → `5 passed` in PY, the NumPy lane and FLOOR.
- [ ] `python tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`.
- [ ] `pytest -q tests/wilson/test_window_threading.py` → `533 passed`. That is 565 before, minus 32 parametrised items of deleted code:
  - in `test_every_kernel_samples/validates_the_window_it_is_given`: `inner_product.*` ×4, `transform.band_atoms`, `forward_bands` and `inverse_bands`, ×2 tests;
  - in `test_every_domainia_consumer_samples/validates_the_window_it_is_given`: the 9 maps cases, ×2 tests.
- [ ] Step 4's first grep prints nothing. Its second prints exactly three indented lines, all inside function bodies.
- [ ] `tests/data/translate_reference.npz` holds 30 cases in 108764 bytes. Step 6's check prints `cases 30 worst rel vs independent rfft-hub chain 0.0`.
- [ ] Full suite (prototype): PY `3121 passed, 3 skipped`; NumPy lane and FLOOR `2857 passed, 267 skipped`. No warnings and no collection errors.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`

**Steps:**

- [ ] **Step 1: Write the new replay test (it fails first)**

Replace the whole of `tests/wilson/test_meyer_baseline.py` with:

```python
"""The default window reproduces the Meyer baseline recorded at d991ac4, to 1e-12 of each
array's scale (domainia 3.0 spec, section 6.1).

``meyer_cases.mismatches`` is the one comparator, shared with ``meyer_cases.py --check``.  The
``meyer_cases.DROPPED`` families, of code that 3.0 deletes, are set aside; every other recorded
key is recomputed under its old name.
"""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson

from .meyer_cases import BASELINE, DROPPED, RTOL, THREADED, compute, mismatches


@pytest.fixture(scope="module")
def baseline() -> dict[str, np.ndarray]:
    with np.load(BASELINE) as z:
        return {k: z[k] for k in z.files}


def test_the_dropped_families_are_96_of_the_1504_keys(baseline):
    """maps 54, inner_product 24 and the band-limited entry points 18."""
    assert len(baseline) == 1504
    assert sum(k.startswith(DROPPED) for k in baseline) == 96


def test_the_comparator():
    """dtype and shape exact, integers exact, floats to ``RTOL`` of the array's largest value,
    an all-zero array to ``RTOL`` of its function's; a NaN fails."""
    a, z, i = ("wilson::noise.band_averaged_variance::8x16::dt=1.0",
               "wilson::noise.band_averaged_variance::8x16::dt=0.25",
               "wilson::evolutionary.wrap_cut_columns::8x16::dt=1.0")
    w = np.random.default_rng(0).standard_normal((3, 4))
    want = {a: w, z: np.zeros(3), i: np.arange(3)}
    peak = float(np.max(np.abs(w)))
    nan = w.copy()
    nan[1, 2] = np.nan
    assert mismatches({a: w + 0.5 * RTOL * peak, z: np.full(3, 0.5 * RTOL * peak), i: np.arange(3)},
                      want) == []
    for k, got in ((a, w + 2 * RTOL * peak), (a, nan), (a, w.astype(np.float32)), (a, w[:2]),
                   (z, np.full(3, 2 * RTOL * peak)), (i, np.arange(3) + 1), (i, np.arange(3.0))):
        assert mismatches({**want, k: got}, want) == [k]
    with pytest.raises(AssertionError, match="baseline keys differ"):
        mismatches({a: w}, want)
    assert mismatches({}, {k: w for k in ("domainia::TimeToWDM.apply::8x16::dt=1.0",
                                          "wilson::inner_product.gaussian::8x16::dt=1.0")}) == []


def test_default_window_reproduces_the_meyer_baseline(baseline):
    assert not mismatches(compute(wilson, {}, {}), baseline)


def test_explicit_meyer_reproduces_the_baseline(baseline):
    got = compute(wilson, {"window": wilson.MEYER}, {}, only=THREADED)
    want = {k: v for k, v in baseline.items()
            if k.startswith("wilson::") and k.split("::")[1].startswith(THREADED)}
    assert want and not mismatches(got, want)


def test_explicit_meyer_on_both_layers_reproduces_the_whole_baseline(baseline):
    got = compute(wilson, {"window": wilson.MEYER}, {"window": wilson.MEYER})
    assert not mismatches(got, baseline)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_meyer_baseline.py 2>&1 | tail -5`

Expected: a collection error, `ImportError: cannot import name 'DROPPED' from 'wilson.meyer_cases'`. It appears twice, because `--doctest-modules` collects the file a second time, followed by `2 errors`.

- [ ] **Step 2: Rewrite `tests/wilson/meyer_cases.py`**

Replace the whole file with:

```python
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
#: Baseline key prefixes of code that domainia 3.0 deletes: maps (54 keys), inner_product (24)
#: and the band-limited entry points (18).  They are not replayed.
DROPPED = ("domainia::TimeToWDM.", "domainia::WDMToTime.", "domainia::FourierToWDM.",
           "wilson::inner_product.", "wilson::transform.band_atoms::",
           "wilson::transform.forward_bands::", "wilson::transform.inverse_bands::")
#: The comparison: ``max |got - want| <= RTOL * s`` (see :func:`mismatches`).
RTOL = 1e-12

Case = tuple[str, Callable[[dict], Any]]


def _family(key: str) -> str:
    """``"layer::case"``: the function that recorded ``key``."""
    return "::".join(key.split("::")[:2])


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
    """The kernel window consumers of ``wl`` (``domainia.wilson``)."""
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
    Wm = np.stack([wl.forward(r, Nf, Nt, dt) for r in Xr])
    mask = wl.active_mask(Nf, Nt)
    Tr, M, St, No, E = wl.transform, wl.matrix, wl.stationary, wl.noise, wl.evolutionary
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
        # nan (inactive cells) -> -1, as recorded; the fractions lie in [0, 1].
        ("evolutionary.wrap_fractions",
         lambda kw: np.nan_to_num(E.wrap_fractions(Nf, Nt, dt, **kw), nan=-1.0)),
        ("evolutionary.wrap_cut_columns", lambda kw: np.asarray(E.wrap_cut_columns(Nf, Nt, dt, **kw))),
        ("evolutionary.eq45_plan", lambda kw: E.eq45_plan(Nf, Nt, dt, **kw).base),
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
    m, n = np.nonzero(wl.active_mask(Nf, Nt))

    def interp(kw: dict) -> Any:
        plan = E.eq45_plan(Nf, Nt, dt, **kw)
        return E.eq45_blocks_cheb(plan, E.eq45_coeffs_interp(Sn, plan.f, plan.t_max, plan.P,
                                                             k=plan.k_used))

    def cells() -> Any:
        """Every active cell as a ``CellSet``, which ``testing`` takes for now (imported on
        call, so collecting this module never needs ``domainia.domains``)."""
        from domainia.domains import CellSet
        return CellSet.all_active(wl.Grid(Nf, Nt, dt))

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
            ("testing.dense_W", lambda kw: testing.dense_W(cells(), **kw)),
            ("testing.wdm_congruence", lambda kw: testing.wdm_congruence(Sigma, cells(), **kw)),
            ("testing.eq45_dense", lambda kw: testing.eq45_dense(cells(), Sn, **kw)),
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
    with np.load(BASELINE) as z:
        stored = {k: z[k] for k in z.files}
    try:
        bad = mismatches(out, stored)
    except AssertionError as exc:
        raise SystemExit(str(exc)) from None
    if bad:
        raise SystemExit(f"{len(bad)} arrays differ from the baseline: {bad[:10]}")
    print(f"baseline reproduces: {len(out)} arrays ({len(stored) - len(out)} DROPPED keys set aside)")


if __name__ == "__main__":
    main()
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_meyer_baseline.py 2>&1 | tail -1`

Expected: `5 passed`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`.

Run the same `--check` with the FLOOR prefix: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: the same line.
- In the prototype, every kept array, the rewired families included, reproduced with a worst relative difference of 0.0 in the wdmrel, floor312, cur313 and cur314 envs.
- The 6.1e-14 quoted in the spec was measured on NumPy 1.26.4.

- [ ] **Step 3: Point `tests/wilson/test_window_threading.py` at the new `domainia_cases`**

`test_window_threading.py` builds the cases at collection, so it must follow the new signature. That leaves its imports of the modules that go unused, so they go too.

Edit 1, the module imports. Old:

```python
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import domainia
import domainia.covariance
import domainia.domains
import domainia.maps
import domainia.spectra
import domainia.testing
import domainia.variance
from domainia import wilson
```

New:

```python
from pathlib import Path

import numpy as np
import pytest

import domainia
from domainia import wilson
```

Edit 2, above `test_every_domainia_consumer_samples_the_window_it_is_given`. Old:

```python
_DM = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                      variance=domainia.variance, covariance=domainia.covariance, testing=domainia.testing)
#: The domainia-layer cases: ``kw`` goes to ``WDMDomain`` and to the cell-based constructors.
_DCASES = domainia_cases(_DM, wilson, 8, 16, 1.0)
```

New:

```python
#: The domainia-layer cases: ``kw`` goes to every window-taking call.
_DCASES = domainia_cases(wilson, 8, 16, 1.0)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py 2>&1 | tail -1`

Expected: `533 passed`. It was 565.
- The 32 items that leave are cases of code the spec deletes, so this is not a regression:
  - the kernel cases `inner_product.gaussian`, `gaussian_from_func`, `loglike_gaussian` and `loglike_gaussian_many`, and `transform.band_atoms`, `forward_bands` and `inverse_bands`;
  - the 9 maps cases of the domainia layer;
  - each counts twice, once in the "samples" test and once in the "validates" test.
- `BandAveragedVariance`, `EvolutionaryCovariance.interp`/`.direct` and the three `testing.*` cases keep their ids. They now thread `window=` into `eq45_plan`, `evolutionary_band_variance` and `testing`.
- The rest of this file (`SCANNED`, `CONSUMERS`, `St, No, E, I = …, wilson.inner_product`) is left to the tasks that delete those modules.

- [ ] **Step 4: Check the decoupling**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -nE "^(import|from) domainia\.(covariance|domains|maps|spectra|variance|translate|testing)|wl\.inner_product|forward_bands\(|rescale_factor|RESCALED|RETIRED|bit_for_bit|SimpleNamespace" tests/wilson/meyer_cases.py tests/wilson/test_meyer_baseline.py`

Expected: no output.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep "import" tests/wilson/meyer_cases.py | grep domainia`

Expected: three indented lines, all inside function bodies (`domainia_cases`, its `cells()`, and `main`):
```
    from domainia import testing
        from domainia.domains import CellSet
    from domainia import wilson
```
- `domainia.domains` is imported only when a `testing.*` case runs.
- `domainia.testing` stays in 3.0. Its own module-level `from .domains import …` goes in Task 4.

- [ ] **Step 5: Record `translate.transform`'s outputs for Task 3**

Write this script to `$TMPDIR/record_translate.py`. It sits outside the repo and is not committed:

```python
"""Record domainia.translate.transform on seeded canonical inputs, for the domainia 3.0
``transform`` agreement test.  Run once from the repo root (Task 1); not committed.

tests/data/translate_reference.npz holds, for case i:
  routes[i] = (source, target), each "time", "frequency" or "wdm";
  grids[i]  = (N, dt, source Nf, source Nt, target Nf, target Nt), 0 on a non-WDM side;
  x{i}      = the input, (K, ...) with K channels on the leading axis, or one channel with
              no leading axis (the 32x8 record);
  y{i}      = translate.transform's output, in the same layout.
"""
import itertools
from pathlib import Path

import numpy as np

from domainia import wilson
from domainia.domains import FourierDomain, TimeDomain, WDMDomain
from domainia.translate import transform

OUT = Path("tests/data/translate_reference.npz")
#: (K, N, dt, WDM grid): all six pairs on a WDM grid; time <-> frequency only without one.
#: K None: one channel, no leading axis.
RECORDS = [(2, 128, 0.25, (8, 16)), (3, 60, 0.5, (6, 10)), (None, 256, 5.0, (32, 8)),
           (2, 24, 0.1, (4, 6)), (2, 45, 0.75, None), (3, 15, 0.5, None)]
#: WDM -> WDM between two grids of one record: (K, N, dt, source grid, target grid).
REGRIDS = [(3, 60, 0.5, (6, 10), (10, 6)), (2, 256, 5.0, (32, 8), (16, 16))]
rng = np.random.default_rng(20261112)


def domain(kind, N, dt, g):
    if kind == "time":
        return TimeDomain(N, dt)
    if kind == "frequency":
        return FourierDomain.full(N, dt)
    return WDMDomain.full(wilson.Grid(*g, dt))


def draw(kind, K, N, g):
    """A seeded canonical input: real time; complex dt-rfft bins with real DC and even-N
    Nyquist; real WDM with inactive cells exactly 0."""
    if kind == "time":
        return rng.standard_normal((K, N))
    if kind == "frequency":
        X = rng.standard_normal((K, N // 2 + 1)) + 1j * rng.standard_normal((K, N // 2 + 1))
        X[:, 0] = X[:, 0].real
        if N % 2 == 0:
            X[:, -1] = X[:, -1].real
        return X
    w = rng.standard_normal((K, g[0] + 1, g[1]))
    w[:, ~wilson.active_mask(*g)] = 0.0
    return w


cases = []
for K, N, dt, g in RECORDS:
    kinds = ("time", "frequency", "wdm") if g else ("time", "frequency")
    for s, t in itertools.permutations(kinds, 2):
        cases.append((s, t, K, N, dt, g if s == "wdm" else None, g if t == "wdm" else None))
cases += [("wdm", "wdm", K, N, dt, a, b) for K, N, dt, a, b in REGRIDS]
out = {"routes": np.array([c[:2] for c in cases]),
       "grids": np.array([(N, dt, *(a or (0, 0)), *(b or (0, 0))) for *_, N, dt, a, b in cases])}
for i, (s, t, K, N, dt, a, b) in enumerate(cases):
    x = draw(s, K or 1, N, a)
    names = [f"ch{j}" for j in range(K or 1)]
    y = transform(dict(zip(names, x)), domain(s, N, dt, a), domain(t, N, dt, b))
    y = np.stack([y[n] for n in names])
    out[f"x{i}"], out[f"y{i}"] = (x, y) if K else (x[0], y[0])
OUT.parent.mkdir(exist_ok=True)
np.savez(OUT, **out)
print(f"{OUT}: {len(cases)} cases, {OUT.stat().st_size} bytes")
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/record_translate.py`

Expected: `tests/data/translate_reference.npz: 30 cases, 108764 bytes`.
- In the prototype, a second run gave a byte-identical file.
- `translate.transform` would refuse a non-canonical input, so the script also confirms that its inputs are exactly canonical.

The cases, by index i:

| i | Record | Routes |
|---|---|---|
| 0–5 | 8×16, N = 128, dt = 0.25, K = 2 | all six pairs |
| 6–11 | 6×10, N = 60, dt = 0.5, K = 3 | all six pairs |
| 12–17 | 32×8, N = 256, dt = 5.0, one channel, x has no leading axis | all six pairs |
| 18–23 | 4×6, N = 24, dt = 0.1, K = 2 | all six pairs |
| 24–25 | odd N = 45, dt = 0.75, K = 2 | time ↔ frequency |
| 26–27 | odd N = 15, dt = 0.5, K = 3 | time ↔ frequency |
| 28 | WDM 6×10 → 10×6, N = 60, dt = 0.5, K = 3 | WDM → WDM |
| 29 | WDM 32×8 → 16×16, N = 256, dt = 5.0, K = 2 | WDM → WDM |

Within each six-pair record the order is: time→frequency, time→wdm, frequency→time, frequency→wdm, wdm→time, wdm→frequency.

- [ ] **Step 6: Check the fixture against an independent route**

Write this script to `$TMPDIR/check_translate_reference.py`. It is not committed:

```python
"""Check tests/data/translate_reference.npz against an independent rfft-hub chain (kernels per
row, then the canonical fix-ups).  Run once from the repo root (Task 1); not committed."""
import numpy as np

from domainia import wilson as wl

z = np.load("tests/data/translate_reference.npz")


def to_rfft(x, kind, N, dt, g):
    if kind == "time":
        return np.fft.rfft(x, axis=-1)
    if kind == "frequency":
        return x / dt
    rows = [wl.inverse(v, *g, dt) for v in x.reshape(-1, g[0] + 1, g[1])]
    return np.stack(rows).reshape(x.shape[:-2] + (N // 2 + 1,))


def from_rfft(R, kind, N, dt, g):
    if kind == "time":
        return np.fft.irfft(R, n=N, axis=-1)
    if kind == "frequency":
        X = dt * R
        X[..., 0] = X[..., 0].real
        if N % 2 == 0:
            X[..., -1] = X[..., -1].real
        return X
    w = np.stack([wl.forward(r, *g, dt) for r in R.reshape(-1, R.shape[-1])])
    w = w.reshape(R.shape[:-1] + (g[0] + 1, g[1]))
    w[..., ~wl.active_mask(*g)] = 0.0
    return w


worst = 0.0
for i, ((s, t), (N, dt, a, b, c, d)) in enumerate(zip(z["routes"], z["grids"])):
    N, sg, tg, y = int(N), (int(a), int(b)), (int(c), int(d)), z[f"y{i}"]
    want = from_rfft(to_rfft(z[f"x{i}"], s, N, dt, sg), t, N, dt, tg)
    assert want.shape == y.shape and want.dtype == y.dtype, i
    worst = max(worst, float(np.max(np.abs(want - y)) / np.max(np.abs(y))))
    if t == "frequency":
        assert not np.any(y[..., 0].imag) and (N % 2 or not np.any(y[..., -1].imag)), i
    if t == "wdm":
        assert not np.any(y[..., ~wl.active_mask(*tg)]), i
print("cases", len(z["routes"]), "worst rel vs independent rfft-hub chain", worst)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/check_translate_reference.py`

Expected: `cases 30 worst rel vs independent rfft-hub chain 0.0`. The FLOOR prefix prints the same line.

This shows three things:
- every recorded output is `forward`/`inverse`/`rfft`/`irfft` per row, through the rfft hub;
- every output is exactly canonical;
- WDM → WDM equals the chain through frequency.

- [ ] **Step 7: Gates**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `3121 passed, 3 skipped`, which is Task 0's 3153 minus Step 3's 32.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2857 passed, 267 skipped`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2857 passed, 267 skipped`.

None of the three lines contains `warning` or `error`. The Meyer replay is part of each run (`test_meyer_baseline.py`, 5 passed): 1408 arrays at 1e-12, 0 mismatches.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 12252 lines / 5669 code
tests 13164 lines (+ oracle 4135)
```

- [ ] **Step 8: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add tests/wilson/meyer_cases.py tests/wilson/test_meyer_baseline.py tests/wilson/test_window_threading.py tests/data/translate_reference.npz && git commit -m "Replay: one 1e-12 comparator; 96 keys of deleted code dropped; translate reference recorded" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

Expected: 4 files changed. The npz is new (108764 bytes); `meyer_cases.py` has 83 insertions and 145 deletions, `test_meyer_baseline.py` 41 and 55, and `test_window_threading.py` 2 and 11.

Size after Task 1 (prototype): src 12252 lines / 5669 code; tests 13164 (+ oracle 4135).

---

### Task 2: Delete the likelihoods, factorisations, covariance classes, operators and coordinates, with their tests (spec Order step 2; §1, §4; decision 8)

**Goal:** Delete four layers, with every test whose subject is that code:
- the likelihoods (`wilson/inner_product.py`);
- the factorisations (`linalg.py`);
- the covariance classes (`covariance/`);
- the operators (`operators.py`);
- and, inside `translate.py`, the real coordinates, the coordinate Jacobian and the `Operator` layer.

Every kept module and kept test still passes, and the replay still reproduces 1408 arrays. Nothing left in `src`, `tests` or the README names the deleted code.

Assumes the tree after Task 1, whose commit is "Replay: one 1e-12 comparator; 96 keys of deleted code dropped; translate reference recorded".

**Files:**
- Delete, source: 10 files, 3,119 lines.
  - `src/domainia/wilson/inner_product.py` (1008 lines);
  - `src/domainia/linalg.py` (620);
  - `src/domainia/operators.py` (329);
  - `src/domainia/covariance/`: `__init__`, `base`, `blocktri`, `dense`, `diagonal`, `evolutionary` and `stack` (1162).
- Delete, tests: 19 files, 3,723 lines and 591 items.

| File | Items |
|---|---|
| `tests/test_dom_cov_api.py` | 13 |
| `tests/test_dom_cov_base.py` | 12 |
| `tests/test_dom_cov_blocktri.py` | 11 |
| `tests/test_dom_cov_multiply.py` | 56 |
| `tests/test_dom_cov_stack.py` | 2 |
| `tests/test_dom_ev_ported.py` | 2 |
| `tests/test_dom_evolutionary_cov.py` | 9 |
| `tests/test_dom_linalg_blocktri.py` | 15 |
| `tests/test_dom_linalg_factors.py` | 2 |
| `tests/test_dom_linalg_jax.py` | 6 |
| `tests/test_dom_linalg_pcg.py` | 7 |
| `tests/test_dom_restored_coverage.py` | 8 |
| `tests/test_operators_covariance.py` | 27 |
| `tests/test_operators_dense.py` | 44 |
| `tests/test_translate_coordinates.py` | 50 |
| `tests/test_translate_operator.py` | 93 |
| `tests/test_wrong_kinds_covariance.py` | 206 |
| `tests/wilson/test_inner_product_fixes.py` | 12 |
| `tests/wilson/vendored/test_inner_product.py` | 16 |

- The 12 doctests of `inner_product.py` go with it.
- Modify, source:
  - `src/domainia/__init__.py`: rewritten whole (Step 3).
  - `src/domainia/wilson/__init__.py`: the inner-product docstring sections, and the `inner_product` import and export.
  - `src/domainia/translate.py` keeps `transform` and its helpers. These go: `_packed`, `log_coordinate_scale`, `_channel_names`, `_real_coordinates`, `_canonical_layout`, `to_coordinates`, `from_coordinates`, `Operator`, `TranslatedOperator` and `translate_operator`.
  - `src/domainia/testing.py`: rewritten whole (Step 6). `blocktri_dense`, `hhat`, `eq54_covariance`, `dense_lambda_generic` and `PointwiseOperator` go (spec §4).
  - Docstrings, and one error message, that name deleted code: `maps/base.py`, `domains.py`, `variance.py`, `wilson/grid.py`, `wilson/noise.py`, `wilson/transform.py` and `wilson/evolutionary.py`.
- Modify, tests:
  - `tests/test_dom_variance_generic.py`, `tests/test_dom_maps_base.py`, `tests/test_dom_windows.py`, `tests/test_dom_testing.py`, `tests/test_translate.py`;
  - `tests/test_wrong_kinds_misc.py`, `tests/test_package.py`;
  - `tests/wilson/test_window_threading.py`, `tests/wilson/test_wrong_kinds.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_evolutionary_kernels.py`;
  - `tests/wilson/vendored/test_evolutionary.py`, `tests/wilson/vendored/conftest.py`.
- Modify: `README.md`.
- Test: no new test function.
  - Two kept tests move into `test_grid_bands.py`.
  - `test_bessel_blocks_match_the_direct_sum_and_gram` gains the interp builder's check.

**Which tests go.** A test goes when its subject is deleted code. A test that also checks kept code stays, trimmed to the kept part.
- The 19 files above test only deleted code, with one exception. Two tests of `test_dom_restored_coverage.py`, `test_time_fourier_crop_and_pad_maps_under_jit_with_the_map_as_argument` and `test_wdm_maps_under_jit_with_the_map_as_argument`, run the maps under `jax.jit`. Maps go in Task 4 and JAX in Task 5, so that file goes whole now.
- `test_inner_product_fixes.py` holds 2 tests of kept code that nothing else pins:
  - `delta_F`'s "Nf must be even";
  - the odd-`N` refusals of `real_to_rfft` and `rfft_to_real`.
  They move verbatim to `test_grid_bands.py` (Step 17).
- `test_dom_evolutionary_cov.py` was the only check of the interp Eq. 45 builder against the direct sum on a padded plan. Two lines in `test_evolutionary_kernels.py` keep that check (Step 17). Measured: 3.7e-15 at Nt=16 and 4.1e-15 at Nt=20, the same in PY and FLOOR.
- `test_dom_testing.py`'s `blocktri_dense` test goes with `blocktri_dense`. Its kernels stay covered by `test_evolutionary_kernels.py::test_bessel_blocks_match_the_direct_sum_and_gram` and `test_window_threading.py::test_eq45_plan_carries_the_window_to_every_eq45_builder`.
- In `vendored/test_evolutionary.py`, a test goes if it calls no domainia function, or if its subject is an oracle factorisation, likelihood or covariance class: `BlockTri`, `Solver`, `Schur`, `PaddedMarginal`, `marginal_lnlike`, `nuisance_basis`, `FourierBand`/`wdm_discarded`, `make_variant` or `TriCovariance`.
  - 16 of its 24 tests go.
  - The 8 that stay check these against the oracle: `packed_grid`, `atoms`, `apply_mask_energy_fraction`, `band_average` (2 tests), `stationary_limit`, `exact_covariance_transformed` and `wrap_fractions`.
- The oracle copies stay whole.
- The replay is untouched. Since Task 1, `meyer_cases.py` recomputes the deleted classes' keys from kept kernels.
- Task 4 deletes `maps/base.py`, `domains.py` and `variance.py`, but their docstrings are fixed now as well, so Step 19's grep gate holds after this task.

**Acceptance Criteria:**
- [ ] `src/domainia` has no `covariance/`, `linalg.py`, `operators.py` or `wilson/inner_product.py`. `domainia.__all__` has 21 names, and `domainia.translate.__all__` is `["Domain", "transform"]`.
- [ ] Step 19's grep prints exactly one line, `tests/wilson/meyer_cases.py:169:…`.
- [ ] `python tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`.
- [ ] Full suite (prototype): PY `2325 passed, 3 skipped`; NumPy lane and FLOOR `2085 passed, 243 skipped`. No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 8592 lines / 3850 code` and `tests 8677 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1` → `2325 passed, 3 skipped in …`

**Steps:**

Edits are anchored to code text, never to line numbers. Each Old text occurs exactly once in its file, and New replaces it. "Delete" means the shown lines go, newline included.

**Deleting a named function** (Steps 8–16) follows one rule:
- Delete from its first decorator line, or its `def` line if it has none, through its last line, with the two blank lines after it.
- If it is the last thing in the file, delete the two blank lines before it instead, so the file still ends in one newline.
- Deleting the listed functions in any order gives the same file.

The suite does not collect cleanly until Step 16 is done: until then some kept test files still import deleted names. `tests/test_readme.py` fails until Step 18, because the README's Operators block imports `PointwiseOperator`.

Steps 1, 2 and 21 are git writes in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: Delete the tests of the deleted layers**

Run: `cd /Users/aaron/Documents/lisa/domainia && git rm -q tests/test_dom_cov_api.py tests/test_dom_cov_base.py tests/test_dom_cov_blocktri.py tests/test_dom_cov_multiply.py tests/test_dom_cov_stack.py tests/test_dom_ev_ported.py tests/test_dom_evolutionary_cov.py tests/test_dom_linalg_blocktri.py tests/test_dom_linalg_factors.py tests/test_dom_linalg_jax.py tests/test_dom_linalg_pcg.py tests/test_dom_restored_coverage.py tests/test_operators_covariance.py tests/test_operators_dense.py tests/test_translate_coordinates.py tests/test_translate_operator.py tests/test_wrong_kinds_covariance.py tests/wilson/test_inner_product_fixes.py tests/wilson/vendored/test_inner_product.py && git status --short | wc -l`

Expected: `19`. `wc` may pad it with spaces.

- [ ] **Step 2: Delete the four source modules**

Run: `cd /Users/aaron/Documents/lisa/domainia && git rm -q -r src/domainia/covariance src/domainia/linalg.py src/domainia/operators.py src/domainia/wilson/inner_product.py && rm -rf src/domainia/covariance && git status --short src | wc -l`

Expected: `10`.
- The `rm -rf` removes the untracked `__pycache__` that `git rm` leaves behind in `covariance/`.
- No kept module imports these four. That includes `maps`, `spectra`, `variance`, `data` and `domains`, which Task 4 deletes.

- [ ] **Step 3: Rewrite `src/domainia/__init__.py`**

Replace the whole file with the text below.
- The docstring loses the `operators` and `linalg` bullets, and `translate_operator` and the coordinates.
- `__all__` loses `CovarianceOperator`, `DenseOperator`, `Operator`, `TranslatedOperator`, `from_coordinates`, `linalg`, `log_coordinate_scale`, `operator_matrix`, `operators`, `to_coordinates` and `translate_operator`.
- `__version__` stays `"2.0.0"` until Task 11.
- The JAX paragraph stays until Task 5.

```python
"""domainia: data, linear maps and covariance across the time, Fourier and time-frequency
(Wilson / WDM) domains.

Importing ``domainia`` (any of its modules) never imports JAX.  JAX is imported only on use:
when the JAX backend is built (:func:`domainia.wilson.backend.jax_backend` or
``domainia.wilson.backend.get_backend(backend="jax")``) to pass as the ``be=`` of a kernel,
map or covariance that takes one, when float64 is turned on
(:func:`domainia.wilson.backend.enable_x64`), or when JAX device constants are built.  The
JAX backend needs float64: call ``enable_x64()`` (or set ``JAX_ENABLE_X64=true``) first, or
it raises ``RuntimeError``.  A kernel that takes ``be=``, given JAX arrays and no ``be=``,
infers the JAX backend, with the same requirement.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`MEYER`); every kernel that samples the window takes
  ``window=`` (any admissible :class:`Window`: ``Meyer(d)``, ``RampWindow``,
  ``SampledWindow``).
* :mod:`domainia.translate`: ``transform`` (``{channel: array}`` dicts) between the
  full-record ``TimeDomain``, ``FourierDomain.full`` and ``WDMDomain.full``.
* :mod:`domainia.domains`: ``TimeDomain``, ``FourierDomain``, ``CellSet``, ``WDMDomain``,
  ``ChannelAxis`` (frozen, hashable: static under ``jax.jit``).
* :mod:`domainia.data`: ``TimeSeries``, ``FourierBins``, ``WDMCells`` (pytrees).
* Not imported here: :mod:`domainia.maps` (linear maps between partial domains),
  :mod:`domainia.variance`, :mod:`domainia.spectra`, and :mod:`domainia.testing` (dense
  oracles and test windows).
"""

from __future__ import annotations

from . import data, domains, translate, wilson
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from .translate import transform
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
    "RampWindow",
    "SampledWindow",
    "TimeDomain",
    "TimeSeries",
    "WDMCells",
    "WDMDomain",
    "Window",
    "WindowError",
    "__version__",
    "data",
    "domains",
    "transform",
    "translate",
    "wilson",
]
```

- [ ] **Step 4: `src/domainia/wilson/__init__.py`**

Edit 1, the module docstring. Old:

```python
Inner products and log-likelihoods (in :mod:`domainia.wilson.inner_product`):

* :func:`inner_product.whittle`
* :func:`inner_product.band_averaged`
* :func:`inner_product.evolutionary_whittle`
* :func:`inner_product.gaussian`, :func:`inner_product.gaussian_from_func`
* :func:`inner_product.loglike_whittle`
* :func:`inner_product.loglike_band_averaged`
* :func:`inner_product.loglike_evolutionary_whittle`
* :func:`inner_product.loglike_gaussian`

Batched over a stack of coefficient grids (one call, ``(B,)`` out):

* :func:`inner_product.whittle_many`,
  :func:`inner_product.loglike_whittle_many` -- these cover the
  ``band_averaged`` and ``evolutionary_whittle`` flavours too, which differ
  only in the variance grid

Explicit matrix construction (in :mod:`domainia.wilson.matrix`):
```

New:

```python
Explicit matrix construction (in :mod:`domainia.wilson.matrix`):
```

Edit 2, the submodule import list. Old:

```python
    filter,
    inner_product,
    matrix,
```

New:

```python
    filter,
    matrix,
```

Edit 3, `__all__`. Old:

```python
    "forward_time_many",
    "inner_product",
    "inverse",
```

New:

```python
    "forward_time_many",
    "inverse",
```

- [ ] **Step 5: `src/domainia/translate.py` keeps `transform` only**

Task 4 deletes this file, so only what names deleted code goes now. `_check_pair` keeps its `roles` parameter, which every remaining call now leaves at its default.

Edit 1, the module docstring's tail. Old:

```python
or packed cell sets stay with :mod:`domainia.maps`.

In the real coordinates (time samples; WDM cells; Fourier ``Re X_0``, ``sqrt(2) Re/Im X_k``,
and ``Re X_{N/2}`` for even ``N`` only) every domain is the time record scaled by
``exp(log_coordinate_scale(domain))``: 1 for time, ``sqrt(dt)`` for WDM (any admissible
window), ``dt sqrt(N)`` for Fourier.
:func:`to_coordinates` turns a dict into these coordinates, one row per channel, and
:func:`from_coordinates` turns the rows back into a dict in the canonical layout.

A covariance :class:`Operator` lives on one full-record domain.  :func:`translate_operator`
gives a :class:`TranslatedOperator` that sees it from another domain lazily: every call
moves the dicts with :func:`transform` and applies the coordinate Jacobian
``Delta = log_coordinate_scale(target) - log_coordinate_scale(native)``.  It moves dicts
only: the native operator applies the real-coordinate convention (sqrt(2) Re/Im interiors,
real endpoints, masks) itself, for example through :func:`to_coordinates` and
:func:`from_coordinates`.
"""
```

New:

```python
or packed cell sets stay with :mod:`domainia.maps`.
"""
```

Edit 2, the imports, `__all__` and the `Domain` comment. Old:

```python
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from domainia import wilson
from domainia.domains import ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from domainia.wilson._checks import REAL_KINDS, require_dtype

__all__ = ["Domain", "Operator", "TranslatedOperator", "from_coordinates", "log_coordinate_scale",
           "to_coordinates", "transform", "translate_operator"]

#: The domains :func:`transform` and :func:`log_coordinate_scale` take (full-record ones only).
```

New:

```python
from collections.abc import Mapping
from typing import Any

import numpy as np

from domainia import wilson
from domainia.domains import FourierDomain, TimeDomain, WDMDomain

__all__ = ["Domain", "transform"]

#: The domains :func:`transform` takes (full-record ones only).
```

Edit 3, `_packed`. Its two callers, `_real_coordinates` and `_canonical_layout`, go in Edit 4. Delete, with the two blank lines after it:

```python
def _packed(domain: WDMDomain) -> tuple[np.ndarray, np.ndarray]:
    """``(m, n)`` of every active cell in ``wilson.grid.packed_cells`` order."""
    g = domain.cells.grid
    return wilson.grid.packed_cells(g.Nf, g.Nt)
```

Edit 4: delete everything from the line `def log_coordinate_scale(domain: Domain) -> float:` to the end of the file, and the two blank lines before it.
- That removes `log_coordinate_scale`, `_channel_names`, `_real_coordinates`, `_canonical_layout`, `to_coordinates`, `from_coordinates`, `Operator`, `TranslatedOperator` and `translate_operator`: 291 lines.
- The file now ends with `transform`'s last line, `    return _finish(names, Y, target)`, and one newline. It is 214 lines.

- [ ] **Step 6: Rewrite `src/domainia/testing.py`**

Replace the whole file with the text below.
- `blocktri_dense`, `hhat`, `eq54_covariance`, `dense_lambda_generic` and `PointwiseOperator` go, with their docstring bullets, their imports and `CArray`.
- Only tests of deleted code, and the README's Operators section (Step 18), used them.

```python
"""Dense oracles for tests and small grids (NumPy only, never on a
likelihood path).  Every oracle here is ``O(N^2)`` memory or worse: keep ``N`` <~ 2048.

* :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a cell set
  (``wilson.matrix.active_W``) and ``W_K Sigma W_K^T``.
* :func:`eq45_dense`: Eq. (45) on a cell set by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).

:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.

The oracles follow the package's rule: a wrong kind (``cells`` that is not a ``CellSet``)
raises ``TypeError``, a wrong value ``ValueError``.

Test windows (``domainia.wilson.windows``):

* :func:`smoothstep`: a non-Meyer ramp for :class:`~domainia.wilson.RampWindow`.
* :func:`wide_profile`, :data:`ALT_WINDOW`: a non-Meyer admissible profile and the
  :class:`~domainia.wilson.SampledWindow` built from it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from domainia import wilson
from domainia.wilson._checks import require_callable, require_instance
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
from domainia.wilson.windows import MEYER, SampledWindow, Window

from .domains import CellSet

__all__ = ["ALT_WINDOW", "dense_W", "eq45_dense", "smoothstep", "wdm_congruence", "wide_profile"]

Array = NDArray[np.float64]


def dense_W(cells: CellSet, *, window: Window = MEYER) -> Array:
    """``(cells.size, N)``: row ``p`` is cell ``p``'s atom sampled in time, so ``W x`` are
    the cells of ``x`` (``wilson.matrix.active_W`` rows, ``W_a W_a^T = dt I``) in the
    Wilson basis of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    return wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=window)[cells.packed_positions()]


def wdm_congruence(Sigma: Array, cells: CellSet, *, window: Window = MEYER) -> Array:
    """``W_K Sigma W_K^T`` for a time-domain covariance ``Sigma (N, N)`` (``W_K`` is
    :func:`dense_W` with ``window``)."""
    W = dense_W(cells, window=window)
    return W @ np.asarray(Sigma, dtype=float) @ W.T


def eq45_dense(cells: CellSet, Sn: Callable[[Any, Any], Any], *, P: int = 24,
               t_max: float | None = None, window: Window = MEYER) -> Array:
    """Eq. (45) on ``cells`` by the direct sum, ``Sn(f, t)`` elementwise (``A_n`` inside),
    with the atoms of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    require_callable("Sn", Sn)
    plan = eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max, window=window)
    return eq45_direct(plan, Sn)


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

- [ ] **Step 7: Docstrings and the message that name deleted code**

Edit 1, `src/domainia/maps/base.py`, the module docstring. Old:

```python
WDM -> time push ``W^T Lambda W / dt^2`` is not available).  Fix the
covariance at a noise state as a :class:`domainia.operators.CovarianceOperator` and see it
from time through :func:`domainia.translate.translate_operator` instead.
"""
```

New:

```python
WDM -> time push ``W^T Lambda W / dt^2`` is not available).
"""
```

Edit 2, `src/domainia/maps/base.py`, the `PushedCovariance` docstring. Old:

```python
    A covariance that carries a Wilson ``window`` (``EvolutionaryCovariance``, a
    ``DiagonalCovariance`` over a ``BandAveragedVariance``, a ``PushedCovariance`` onto WDM
    cells) lives on WDM cells in that window's basis: on a map whose ``src`` is not a
```

New:

```python
    A covariance that carries a Wilson ``window`` (a ``PushedCovariance`` onto WDM cells)
    lives on WDM cells in that window's basis: on a map whose ``src`` is not a
```

Edit 3, `src/domainia/maps/base.py`, the `_matvec` docstring. Old:

```python
        """``cov.matvec``, or a TypeError: a covariance on WDM cells (``DiagonalCovariance``,
        ``EvolutionaryCovariance``, ...) has no fixed-noise-state ``matvec``, so the
        WDM -> time push ``W^T Lambda W / dt^2`` is not available; the error names the
        route: :class:`domainia.operators.CovarianceOperator`, then
        :func:`domainia.translate.translate_operator`."""
```

New:

```python
        """``cov.matvec``, or a TypeError: without it the WDM -> time push
        ``W^T Lambda W / dt^2`` is not available."""
```

Edit 4, `src/domainia/maps/base.py`, the `_matvec` TypeError message. Old:

```python
                            "(X -> X Sigma), so the WDM -> time push is not available; fix the "
                            "covariance at a noise state as a domainia.operators.CovarianceOperator "
                            "(a domainia.translate.Operator) and see it from time with "
                            "domainia.translate.translate_operator")
```

New:

```python
                            "(X -> X Sigma), so the WDM -> time push is not available")
```

Edit 5, `src/domainia/domains.py`, the `_cells_and_window` docstring. Old:

```python
    (``EvolutionaryCovariance``, ``BandAveragedVariance``; ``MidpointVariance`` reads the
    cells only, with :func:`_cells_of`).
```

New:

```python
    (``BandAveragedVariance``; ``MidpointVariance`` reads the cells only, with
    :func:`_cells_of`).
```

Edit 6, `src/domainia/variance.py`, the module docstring. Old:

```python
metadata), but evaluating them calls the PSD on the host, so neither is jittable
(``jittable = False``): a ``DiagonalCovariance`` built on one factorises eagerly on every
backend.
"""
```

New:

```python
metadata), but evaluating them calls the PSD on the host, so neither is jittable
(``jittable = False``).
"""
```

Edit 7, `src/domainia/variance.py`, `BandAveragedVariance.on`. Old:

```python
    def on(self, be: Backend) -> "BandAveragedVariance":
        """The variance for ``be``: it has no array leaves (the PSD runs on the host).  On
        NumPy this returns ``self``; on JAX it calls ``require_x64`` and returns a re-validated
        copy (``dataclasses.replace`` with no changes).  ``as_evaluator`` recognises a variance
        by this method."""
```

New:

```python
    def on(self, be: Backend) -> "BandAveragedVariance":
        """The variance for ``be``: it has no array leaves (the PSD runs on the host).  On
        NumPy this returns ``self``; on JAX it calls ``require_x64`` and returns a re-validated
        copy (``dataclasses.replace`` with no changes)."""
```

Edit 8, `src/domainia/variance.py`, `MidpointVariance.on`. Old:

```python
    def on(self, be: Backend) -> "MidpointVariance":
        """The variance for ``be``: it has no array leaves (the PSD runs on the host).  On
        NumPy this returns ``self``; on JAX it calls ``require_x64`` and returns a re-validated
        copy (``dataclasses.replace`` with no changes).  ``as_evaluator`` recognises a variance
        by this method."""
```

New:

```python
    def on(self, be: Backend) -> "MidpointVariance":
        """The variance for ``be``: it has no array leaves (the PSD runs on the host).  On
        NumPy this returns ``self``; on JAX it calls ``require_x64`` and returns a re-validated
        copy (``dataclasses.replace`` with no changes)."""
```

Edit 9, `src/domainia/wilson/grid.py`. Old:

```python
    from a weighted average (a different answer), and
    ``inner_product.loglike_gaussian`` propagated it to ``+inf``. The same PSD
    gave three answers depending on which function you called.
```

New:

```python
    from a weighted average (a different answer), and the Gaussian log-likelihood
    propagated it to ``+inf``. The same PSD gave three answers depending on which
    function you called.
```

Edit 10, `src/domainia/wilson/noise.py`, the module docstring. Old:

```python
(``Nf, Nt, dt``), then the PSD. It matches :mod:`domainia.wilson.matrix` and
:mod:`domainia.wilson.inner_product`, so the PSD is always the trailing argument.
```

New:

```python
(``Nf, Nt, dt``), then the PSD. It matches :mod:`domainia.wilson.matrix`, so the PSD is
always the trailing argument.
```

Edit 11, `src/domainia/wilson/transform.py`. Old:

```python
every non-trivial consumer (``matrix``, ``stationary``, ``inner_product``)
either assumed or hard-coded the normalized one anyway.
```

New:

```python
every non-trivial consumer (``matrix``, ``stationary``) either assumed or
hard-coded the normalized one anyway.
```

Edit 12, `src/domainia/wilson/evolutionary.py`, the module docstring. Old:

```python
Everything here is a pure function of ``(grid, spectrum, arrays)``.  Anything with
state or a factorisation lives outside :mod:`domainia.wilson`: the covariances of
:mod:`domainia.covariance` (``EvolutionaryCovariance`` builds the Eq. 45 blocks from these
kernels), the factors of :mod:`domainia.linalg` (``BlockTriFactor``) and the maps of
:mod:`domainia.maps`.
```

New:

```python
Everything here is a pure function of ``(grid, spectrum, arrays)``.
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -c "import domainia, domainia.maps, domainia.spectra, domainia.testing, domainia.variance; print(len(domainia.__all__), domainia.translate.__all__, domainia.testing.__all__)"`

Expected: `21 ['Domain', 'transform'] ['ALT_WINDOW', 'dense_W', 'eq45_dense', 'smoothstep', 'wdm_congruence', 'wide_profile']`

- [ ] **Step 8: `tests/test_dom_variance_generic.py` and `tests/test_dom_maps_base.py`**

Edit 1, `tests/test_dom_variance_generic.py`, the module docstring. Old:

```python
"""BandAveragedVariance and MidpointVariance of a GenericPSD (replaces the golden
test_dom_variance), and DiagonalCovariance over them."""
```

New:

```python
"""BandAveragedVariance and MidpointVariance of a GenericPSD (replaces the golden
test_dom_variance)."""
```

Edit 2, `tests/test_dom_variance_generic.py`. Old:

```python
from domainia import wilson
from domainia.covariance import DiagonalCovariance
```

New:

```python
from domainia import wilson
```

In `tests/test_dom_variance_generic.py`, delete these functions, which are the last two in the file: `test_diagonal_covariance_of_a_variance_carries_its_cells_and_factors` and `test_callable_and_array_variances_carry_no_cells`.

In `tests/test_dom_maps_base.py`, delete `test_push_without_matvec_is_a_type_error_naming_translate_operator`.
- It builds its input with `DiagonalCovariance`, and it checks that the error names the operator route, which Step 7 removed from the message. Both go.
- The `_matvec` TypeError itself is then untested until Task 4 deletes the maps.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_variance_generic.py tests/test_dom_maps_base.py 2>&1 | tail -1`

Expected: `20 passed`. That is 11 + 9; before this task it was 13 + 10.

- [ ] **Step 9: `tests/test_dom_windows.py`**

These tests stay: the maps tests, `BandAveragedVariance`, `MidpointVariance` and the oracles. Only the covariance parts go.

Edit 1, the module docstring. Old:

```python
"""The window through the domainia layer: WDMDomain, maps, variances, covariance, oracles."""
```

New:

```python
"""The window through the domainia layer: WDMDomain, maps, variances, oracles."""
```

Edit 2, the covariance import. Old:

```python
from domainia import testing, wilson
from domainia.covariance import (ChannelStack, Covariance, DenseCovariance, DiagonalCovariance,
                                 EvolutionaryCovariance)
```

New:

```python
from domainia import testing, wilson
```

Edit 3, the maps import (`PushedCovariance` is no longer used). Old:

```python
from domainia.maps import (Compose, FourierCrop, FourierToWDM, Identity, PushedCovariance, Restrict,
                           TimeToWDM, WDMToTime)
```

New:

```python
from domainia.maps import (Compose, FourierCrop, FourierToWDM, Identity, Restrict, TimeToWDM,
                           WDMToTime)
```

Edit 4, at the end of `test_equal_but_distinct_windows_are_one_basis`. Old:

```python
    np.testing.assert_array_equal(C.apply(X), TimeToWDM(TIME, alt).apply(X)[pos])
    psd, _ = _stationary_psd()
    ev = EvolutionaryCovariance(psd, CELLS, window=meyer4)
    assert Identity(WDMDomain(CELLS)).push(ev).cov is ev
```

New:

```python
    np.testing.assert_array_equal(C.apply(X), TimeToWDM(TIME, alt).apply(X)[pos])
```

Edit 5, the section comment. Old:

```python
# Variances, covariance, oracles
```

New:

```python
# Variances and oracles
```

Edit 6, in `test_the_cell_set_consumers_take_a_wdm_domain_and_its_window`. Old:

```python
    dom = WDMDomain(cut, window=ALT_WINDOW)
    ev = EvolutionaryCovariance(psd, dom)
    assert ev.window is ALT_WINDOW and ev.cells == cut
    np.testing.assert_array_equal(ev.dense(nu), EvolutionaryCovariance(psd, cut, window=ALT_WINDOW).dense(nu))
    bv = BandAveragedVariance(psd, dom)
```

New:

```python
    dom = WDMDomain(cut, window=ALT_WINDOW)
    bv = BandAveragedVariance(psd, dom)
```

Edit 7, at the end of the same test. The equal-not-identical check moves to `BandAveragedVariance`. Old:

```python
    np.testing.assert_array_equal(mv(nu), MidpointVariance(psd, cut)(nu))
    dg = DiagonalCovariance(bv, cut.size)
    assert dg.window is ALT_WINDOW and dg.cells == cut
    # the domain's own window passed again (equal, not identical) is accepted
    assert EvolutionaryCovariance(psd, dom, window=_roundtrip(ALT_WINDOW)).window is ALT_WINDOW
    assert BandAveragedVariance(psd, dom, window=ALT_WINDOW).window is ALT_WINDOW
```

New:

```python
    np.testing.assert_array_equal(mv(nu), MidpointVariance(psd, cut)(nu))
    # the domain's own window passed again (equal, not identical) is accepted
    assert BandAveragedVariance(psd, dom, window=_roundtrip(ALT_WINDOW)).window is ALT_WINDOW
```

Edit 8, in `test_a_wdm_domain_with_another_explicit_window_is_refused`. The loop over the two builders becomes the `BandAveragedVariance` calls. Old:

```python
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
```

New:

```python
    alt = WDMDomain(CELLS, window=ALT_WINDOW)
    with pytest.raises(ValueError, match=r"^window meyer\(d=4\) is not the WDMDomain's window "
                                         r"wide-smoothstep: pass window=None"):
        BandAveragedVariance(psd, alt, window=MEYER)
    with pytest.raises(ValueError, match=r"window wide-smoothstep is not the WDMDomain's window meyer"):
        BandAveragedVariance(psd, WDMDomain(CELLS), window=ALT_WINDOW)
    with pytest.raises(TypeError, match="Window"):
        BandAveragedVariance(psd, alt, window="meyer")
    assert BandAveragedVariance(psd, CELLS, window=None).window is MEYER     # None on a CellSet
```

Delete these seven functions:
- `test_a_map_refuses_a_covariance_built_for_another_window`
- `test_a_covariance_on_wdm_cells_is_refused_by_a_map_from_another_domain`
- `test_covariance_windows_are_read_only_and_declared_on_the_base`
- `test_a_diagonal_covariance_reports_the_window_of_a_plain_callable_variance`
- `test_a_channel_stack_reports_its_channels_common_window`
- `test_evolutionary_covariance_stationary_limit_with_another_window`
- `test_evolutionary_covariance_interp_builder_uses_its_window`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_windows.py 2>&1 | tail -1`

Expected: `20 passed`. It was 27.

- [ ] **Step 10: `tests/test_dom_testing.py`**

Edit 1, the imports. Old:

```python
from domainia.wilson.evolutionary import (eq45_blocks_cheb, eq45_coeffs_bessel, eq45_plan, exact_covariance,
                              time_domain_covariance)
from domainia import testing as T
from domainia.domains import CellSet, FourierDomain, WDMDomain
from domainia.maps import FourierToWDM
```

New:

```python
from domainia.wilson.evolutionary import exact_covariance, time_domain_covariance
from domainia import testing as T
from domainia.domains import CellSet
```

Edit 2: delete everything from the line `@pytest.mark.parametrize("cut", [None, (3, 1)])` to the end of the file, and the two blank lines before it. That is 79 lines, the two blank lines included, and five tests of deleted helpers:
- `test_blocktri_dense_of_the_fast_blocks_is_the_direct_sum`
- `test_hhat_is_the_dft_of_the_square_root_envelope`
- `test_eq54_is_the_covariance_of_a_modulated_stationary_process`
- `test_dense_lambda_generic_of_a_maps_C_is_its_push`
- `test_eq54_with_a_constant_envelope_is_diagonal`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_testing.py 2>&1 | tail -1`

Expected: `5 passed`. It was 11.

- [ ] **Step 11: `tests/test_translate.py`**

Edit 1. Old:

```python
import math
import re
```

New:

```python
import re
```

Edit 2. Old:

```python
from domainia.translate import log_coordinate_scale, transform
```

New:

```python
from domainia.translate import transform
```

Edit 3: the Parseval test now states its scales directly instead of through `log_coordinate_scale`. Old:

```python
@pytest.mark.parametrize("name", RECORDS)
def test_parseval_scales_match_log_coordinate_scale(name):
    t, data, d = RECORDS[name]
    on = transform(data, t, d)
    for k, x in data.items():
        want = math.exp(2.0 * log_coordinate_scale(d)) * float(np.sum(x ** 2))
        assert _energy(d, on[k]) == pytest.approx(want, rel=1e-12)
```

New:

```python
@pytest.mark.parametrize("name", RECORDS)
def test_parseval_scales(name):
    """The energy scales by 1 on time, ``dt`` on WDM (any window) and ``dt^2 N`` on Fourier."""
    t, data, d = RECORDS[name]
    on = transform(data, t, d)
    scale = {TimeDomain: 1.0, WDMDomain: t.dt, FourierDomain: t.dt ** 2 * t.N}[type(d)]
    for k, x in data.items():
        assert _energy(d, on[k]) == pytest.approx(scale * float(np.sum(x ** 2)), rel=1e-12)
```

Delete the last two functions in the file: `test_log_coordinate_scale_values` and `test_log_coordinate_scale_refuses_what_transform_refuses`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_translate.py 2>&1 | tail -1`

Expected: `71 passed`. It was 74.

- [ ] **Step 12: `tests/test_wrong_kinds_misc.py`**

Edit 1, the module docstring. Old:

```python
"""``domainia.spectra``, ``domainia.data``, ``domainia.translate``, ``domainia.variance``,
``domainia.linalg`` and ``domainia.testing`` refuse a wrong kind with ``TypeError`` and a wrong
value with ``ValueError``, as ``domainia.wilson`` does."""
```

New:

```python
"""``domainia.spectra``, ``domainia.data``, ``domainia.translate``, ``domainia.variance`` and
``domainia.testing`` refuse a wrong kind with ``TypeError`` and a wrong value with
``ValueError``, as ``domainia.wilson`` does."""
```

Edit 2. Old:

```python
from domainia import linalg as la
from domainia import testing as T
```

New:

```python
from domainia import testing as T
```

Edit 3. Old:

```python
from domainia.translate import (TranslatedOperator, from_coordinates, log_coordinate_scale,
                                to_coordinates, transform, translate_operator)
```

New:

```python
from domainia.translate import transform
```

Edit 4: delete the line

```python
FOURIER = FourierDomain.full(128, 0.5)
```

Edit 5, the end of the parametrize list of `test_translate_kind_messages_say_got`. Old:

```python
    (lambda: transform(Y, GRID, WDM), f"source {FULL_RECORD}"),
    (lambda: log_coordinate_scale(GRID), f"domain {FULL_RECORD}"),
    (lambda: to_coordinates(Y, TIME, "A"), "channel_names must be a tuple of channel names, got str"),
    (lambda: translate_operator(object(), WDM),
     "op must implement domainia.translate.Operator, got object"),
    (lambda: TranslatedOperator(object(), WDM),
     "native must be a native domainia.translate.Operator (translate_operator re-targets a "
     "TranslatedOperator), got object"),
], ids=["mapping", "name", "numeric", "source", "domain", "channel_names", "op", "native"])
```

New:

```python
    (lambda: transform(Y, GRID, WDM), f"source {FULL_RECORD}"),
], ids=["mapping", "name", "numeric", "source"])
```

Edit 6:
- Replace everything from the line `@pytest.mark.parametrize("part", ["real", "imaginary"])` up to, but not including, the line `@pytest.mark.parametrize(("cells", "kind"), [(GRID, "Grid"), (WDM, "WDMDomain"), (None, "NoneType")])`.
- The replacement is the header below followed by two blank lines. The header is 92 characters wide.

```python
# --- domainia.testing ---------------------------------------------------------------------
```

That removes 132 lines:
- `test_to_coordinates_refuses_an_overflowing_sqrt2_scaling`;
- the `domainia.linalg and domainia.testing` header;
- `M4`, `B2`, `G`, `MARGIN`, `S_I`, `S_G`, `H`, `_pcg`, `INTEGER_SLOTS` and `STRING_SLOTS`;
- `test_integer_slots_refuse_other_kinds`, `test_integer_slots_take_numpy_integers`, `test_pcg_tolerances_must_be_real_numbers`, `test_pcg_tolerances_take_numpy_integers_and_floats`, `test_string_options_refuse_other_kinds` and `test_the_layouts_and_factors_are_checked`.

Delete these functions:
- `test_from_coordinates_refuses_coordinates_that_are_not_real`
- `test_from_coordinates_takes_numpy_integers_and_floats`
- `test_eq54_spectra_must_be_real_numbers`
- `test_eq54_spectra_take_numpy_integers_and_floats`
- `test_the_pointwise_test_operator_messages_say_got`

The last three are the last in the file. Afterwards the file ends with `test_eq45_dense_needs_a_callable_spectrum`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_wrong_kinds_misc.py 2>&1 | tail -1`

Expected: `45 passed`. It was 111.

- [ ] **Step 13: `tests/test_package.py`**

`domainia.transform` is still `translate.transform`, and Task 3 changes that. So no identity assert on it is added here.

Edit 1, `PUBLIC`. Old:

```python
PUBLIC = {
    "CellSet", "ChannelAxis", "CovarianceOperator", "DenseOperator", "FourierBins", "FourierDomain",
    "Grid", "MEYER", "Meyer", "Operator", "RampWindow", "SampledWindow", "TimeDomain", "TimeSeries",
    "TranslatedOperator", "WDMCells", "WDMDomain", "Window", "WindowError", "__version__", "data",
    "domains", "from_coordinates", "linalg", "log_coordinate_scale", "operator_matrix", "operators",
    "to_coordinates", "transform", "translate", "translate_operator", "wilson",
}
```

New:

```python
PUBLIC = {
    "CellSet", "ChannelAxis", "FourierBins", "FourierDomain", "Grid", "MEYER", "Meyer", "RampWindow",
    "SampledWindow", "TimeDomain", "TimeSeries", "WDMCells", "WDMDomain", "Window", "WindowError",
    "__version__", "data", "domains", "transform", "translate", "wilson",
}
```

Edit 2, in `test_the_public_names`. Old:

```python
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid
    assert {"from_coordinates", "to_coordinates"} <= set(translate.__all__)
    assert domainia.operators.__all__ == ["CovarianceOperator", "DenseOperator", "operator_matrix"]
    assert domainia.to_coordinates is translate.to_coordinates
    assert domainia.CovarianceOperator is domainia.operators.CovarianceOperator
```

New:

```python
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid
```

Edit 3, in the subprocess test's module set. Old:

```python
    assert {"domainia.translate", "domainia.wilson", "domainia.wilson.backend", "domainia.linalg",
            "domainia.covariance.base", "domainia.maps.time", "domainia.testing"} <= set(info["modules"])
```

New:

```python
    assert {"domainia.translate", "domainia.wilson", "domainia.wilson.backend", "domainia.maps.time",
            "domainia.testing"} <= set(info["modules"])
```

Edit 4, the import-promise test's name and docstring. Old:

```python
def test_importing_domainia_leaves_the_covariance_layer_unimported():
    """The package docstring's promise, with domainia.operators imported: a fresh
    ``import domainia`` imports none of domainia.covariance, domainia.maps,
    domainia.variance, domainia.spectra and domainia.testing (CovarianceOperator imports
    domainia.covariance when it is built)."""
```

New:

```python
def test_importing_domainia_leaves_the_unlisted_modules_unimported():
    """The package docstring's promise: a fresh ``import domainia`` imports none of
    domainia.maps, domainia.variance, domainia.spectra and domainia.testing."""
```

Edit 5, its asserts. Old:

```python
    assert {"domainia.operators", "domainia.translate"} <= loaded
    assert not {m.split(".")[1] for m in loaded} & {"covariance", "maps", "spectra", "testing", "variance"}
```

New:

```python
    assert "domainia.translate" in loaded
    assert not {m.split(".")[1] for m in loaded} & {"maps", "spectra", "testing", "variance"}
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -1`

Expected: `7 passed`, as before.

- [ ] **Step 14: `tests/wilson/test_window_threading.py`**

`test_the_scan_covers_every_module_but_filter_and_windows` requires `SCANNED` to be exactly the package's modules (minus `UNSCANNED`). `CONSUMERS` names the functions the AST scan checks.

Edit 1, `SCANNED` (31 → 21 entries). Old:

```python
SCANNED = ("wilson/transform.py", "wilson/matrix.py", "wilson/stationary.py", "wilson/noise.py",
           "wilson/evolutionary.py", "wilson/inner_product.py", "wilson/grid.py", "wilson/backend.py",
           "wilson/special.py", "wilson/__init__.py", "wilson/_checks.py",
           "domains.py", "maps/base.py", "maps/time.py", "maps/fourier.py", "variance.py",
           "covariance/evolutionary.py", "covariance/base.py", "covariance/diagonal.py",
           "covariance/dense.py", "covariance/blocktri.py", "covariance/stack.py", "testing.py",
           "linalg.py", "data.py", "spectra.py", "__init__.py", "maps/__init__.py",
           "covariance/__init__.py", "translate.py", "operators.py")
```

New:

```python
SCANNED = ("wilson/transform.py", "wilson/matrix.py", "wilson/stationary.py", "wilson/noise.py",
           "wilson/evolutionary.py", "wilson/grid.py", "wilson/backend.py",
           "wilson/special.py", "wilson/__init__.py", "wilson/_checks.py",
           "domains.py", "maps/base.py", "maps/time.py", "maps/fourier.py", "variance.py",
           "testing.py", "data.py", "spectra.py", "__init__.py", "maps/__init__.py",
           "translate.py")
```

Edit 2, `CONSUMERS`. Old:

```python
    "eq45_plan",
    "gaussian", "gaussian_from_func", "loglike_gaussian", "loglike_gaussian_many",
    "dense_W", "wdm_congruence", "eq45_dense", "_dense_atoms",
```

New:

```python
    "eq45_plan",
    "dense_W", "wdm_congruence", "eq45_dense", "_dense_atoms",
```

Edit 3, `CONSUMERS`. Old:

```python
    "WDMDomain", "EvolutionaryCovariance", "BandAveragedVariance",
```

New:

```python
    "WDMDomain", "BandAveragedVariance",
```

Edits 4 and 5 are in the parametrize list of `test_the_window_scan_flags_exactly_the_calls_that_lose_the_window`.

Edit 4: delete the line

```python
    ("EvolutionaryCovariance(psd, cells, builder='direct', window=self.window)", False),
```

Edit 5: delete the line

```python
    ("EvolutionaryCovariance(psd, cells, builder='direct')", True),
```

Edit 6. Old:

```python
St, No, E, I = wilson.stationary, wilson.noise, wilson.evolutionary, wilson.inner_product
```

New:

```python
St, No, E = wilson.stationary, wilson.noise, wilson.evolutionary
```

Delete `test_gaussian_forms_are_the_dense_wdm_forms`, the last function in the file.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py 2>&1 | tail -1`

Expected: `501 passed`. It was 533. The 32 that leave:
- 10 `SCANNED` files;
- the 2 `EvolutionaryCovariance` scan rows;
- 20 items of `test_gaussian_forms_are_the_dense_wdm_forms`.

- [ ] **Step 15: `tests/wilson/test_wrong_kinds.py`**

The `get_backend`, Cholesky and tracer tests stay; they go in Task 5.

Edit 1. Old:

```python
_T, _IP = wilson.transform, wilson.inner_product
```

New:

```python
_T = wilson.transform
```

Edit 2. Old:

```python
_W = wilson.forward_time(np.random.default_rng(1).standard_normal(128), 8, 16, 1.0)
_VAR = wilson.whittle_variance(8, 16, 1.0, np.ones_like)
_SN = np.ones(65)
```

New:

```python
_W = wilson.forward_time(np.random.default_rng(1).standard_normal(128), 8, 16, 1.0)
```

Edit 3: delete everything from this line:

```python
#: The loglike functions that take ``include_logdet``.
```

up to, but not including, this line:

```python
@pytest.mark.parametrize(("value", "kind"), [(1, "int"), (b"numpy", "bytes"), (True, "bool"),
```

That is 56 lines:
- `_LOGLIKE` and `_CELLS`;
- `test_include_logdet_of_the_wrong_kind_raises_type_error`, `test_include_logdet_takes_a_numpy_bool`, `test_cells_of_the_wrong_kind_raise_type_error` and `test_cells_keep_their_choices`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py 2>&1 | tail -1`

Expected: `192 passed`. It was 254.

- [ ] **Step 16: `tests/wilson/vendored/test_evolutionary.py` and `tests/wilson/vendored/conftest.py`**

In `tests/wilson/vendored/test_evolutionary.py`, delete these two helpers and 16 tests:
- `_spd`, `_blocktri`;
- `test_endpoint_weights_are_the_trapezoid_rule`
- `test_block_tridiagonal_solver`, `test_solver_flavours_agree_with_numpy`, `test_schur_edge_exact_matches_the_dense_matrix`
- `test_padded_marginal_is_the_data_block_of_the_extended_covariance`, `test_marginal_lnlike_is_the_flat_prior_limit`, `test_nuisance_basis_keeps_full_rank_and_reduces_degenerate`
- `test_fourier_band_discards_an_orthonormal_low_frequency_set`, `test_fourier_band_zero_cut_is_the_identity`, `test_variants`
- `test_time_domain_factor_is_the_square_root_of_the_paper_covariance`, `test_draw_noise_is_reproducible`
- `test_tri_covariance_is_exact_for_stationary_noise`, `test_tri_covariance_chebyshev_blocks_match_the_direct_sum`, `test_tri_covariance_is_close_to_exact_for_slow_variation`
- `test_wrap_cut_columns_grow_as_the_tolerance_tightens`, the last in the file.

Then, in the same file. Old:

```python
# General S(f, t): band average, exact covariance, Eq. (45), wrap rule
```

New:

```python
# General S(f, t): band average, exact covariance, wrap rule
```

In `tests/wilson/vendored/conftest.py`, delete the `gaussian_pulse` fixture, the last function in the file. Its only user was the deleted `vendored/test_inner_product.py`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/vendored/test_evolutionary.py 2>&1 | tail -1`

Expected: `8 passed`. It was 24.

- [ ] **Step 17: Keep two tests of kept code, and the interp check**

Edit 1, `tests/wilson/test_grid_bands.py`, after `test_unrepresentable_delta_F_raises_instead_of_returning_zeros`. These two tests are moved verbatim from the deleted `test_inner_product_fixes.py`. Old:

```python
            pytest.fail(f"{name} returned for dt={dt!r}")


def test_atoms_are_the_rows_of_the_transform():
```

New:

```python
            pytest.fail(f"{name} returned for dt={dt!r}")


def test_delta_F_rejects_unrepresentable_widths():
    from domainia.wilson.grid import delta_F

    assert delta_F(8, 1.0) == 0.0625
    with pytest.raises(ValueError, match="not representable"):
        delta_F(8, 5e-324)
    with pytest.raises(ValueError, match="Nf must be even"):
        delta_F(7, 1.0)


def test_rfft_packing_rejects_odd_N():
    from domainia.wilson.matrix import real_to_rfft, rfft_to_real

    x = np.random.default_rng(0).standard_normal(32)
    assert np.allclose(real_to_rfft(rfft_to_real(np.fft.rfft(x), 32), 32), np.fft.rfft(x))
    with pytest.raises(ValueError, match="N must be even"):
        rfft_to_real(np.fft.rfft(x[:5]), 5)
    with pytest.raises(ValueError, match="N must be even"):
        real_to_rfft(np.zeros(5), 5)


def test_atoms_are_the_rows_of_the_transform():
```

Edit 2, `tests/wilson/test_evolutionary_kernels.py`, the docstring of `test_bessel_blocks_match_the_direct_sum_and_gram`. Old:

```python
    """bessel (exact Chebyshev) vs direct sum vs gram, log-linear S, both <= 1e-12."""
```

New:

```python
    """bessel (exact Chebyshev) and interp vs direct sum vs gram, log-linear S, all <= 1e-12."""
```

Edit 3, the end of the same test. Old:

```python
    D = E.eq45_direct(plan, _loglinear(ac, ad, T_obs))
    assert np.max(np.abs(C - D)) <= 1e-12 * np.abs(D).max()
```

New:

```python
    D = E.eq45_direct(plan, _loglinear(ac, ad, T_obs))
    assert np.max(np.abs(C - D)) <= 1e-12 * np.abs(D).max()
    coeffs = E.eq45_coeffs_interp(_loglinear(ac, ad, T_obs), plan.f, plan.t_max, plan.P, k=plan.k_used)
    assert np.max(np.abs(_assemble(plan, *E.eq45_blocks_cheb(plan, coeffs)) - D)) <= 1e-12 * np.abs(D).max()
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_grid_bands.py 2>&1 | tail -1`

Expected: `54 passed, 1 skipped`. It was `52 passed, 1 skipped`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_evolutionary_kernels.py -k bessel_blocks 2>&1 | tail -1`

Expected: `2 passed, 1 skipped, 30 deselected`. The two passes are `test_bessel_blocks_match_the_direct_sum_and_gram` at Nt=16 and Nt=20. The skip is an oracle test whose name also matches, skipped because `DOMAINIA_ORACLE_ROOT` is not set.

- [ ] **Step 18: `README.md`**

Only the README blocks that use deleted names change. The README's other stale prose is Task 11's: "inner products" in the first bullet, "maps and covariances" in the layout paragraph, and the JAX text.

Edit 1, the `translate` and `domains` bullets. Old:

```markdown
* `domainia.translate` is the translation layer. `transform` moves `{channel: array}`
  dicts between the full-record domains, and `translate_operator` puts a covariance
  operator native to one domain into another lazily: no dense matrix, and the
  coordinate Jacobian is in `log_determinant`.
* `domainia.domains`, `domainia.data`, `domainia.maps`, `domainia.covariance`,
  `domainia.variance` and `domainia.spectra` cover partial domains (WDM cell sets, Fourier
  bands): maps between them, block-tridiagonal / dense / diagonal / evolutionary covariances
  and cell variances.
```

New:

```markdown
* `domainia.translate` is the translation layer. `transform` moves `{channel: array}`
  dicts between the full-record domains.
* `domainia.domains`, `domainia.data`, `domainia.maps`, `domainia.variance` and
  `domainia.spectra` cover partial domains (WDM cell sets, Fourier bands): maps between them
  and cell variances.
```

Edit 2, the coordinates sentence after the `unpack` line. Old:

```markdown
`domainia.wilson.unpack(y, Nf, Nt)`.
Coordinates scale as `exp(log_coordinate_scale(domain))` relative to the time
samples: 1 for time, `sqrt(dt)` for WDM (any window), `dt * sqrt(N)` for Fourier (in the
real coordinates `Re X_0`, `sqrt(2) Re/Im X_k`, `Re X_{N/2}` for even `N`).
```

New:

```markdown
`domainia.wilson.unpack(y, Nf, Nt)`.
```

Edit 3: delete everything from the line `## Operators` up to, but not including, the line `## Errors`.
- That is the Operators section, 50 lines, including the python block that used `translate_operator` and `PointwiseOperator`.
- `tests/test_readme.py` runs the README's python blocks.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_readme.py 2>&1 | tail -1`

Expected: `2 passed`, as before.

- [ ] **Step 19: Nothing names the deleted code**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE --exclude-dir=oracle "domainia\.(linalg|covariance|operators)\b|wilson\.inner_product|\binner_product\.(py|[a-z_]+\()|from \.{0,2}(linalg|covariance|operators|inner_product)\b|import (linalg|covariance|operators|inner_product)\b|\b(Operator|Covariance|to_coordinates|from_coordinates|log_coordinate_scale|translate_operator|TranslatedOperator|CovarianceOperator|DenseOperator|operator_matrix|PointwiseOperator|blocktri_dense|eq54_covariance|dense_lambda_generic|hhat|DiagonalCovariance|DenseCovariance|BlockTridiagonalCovariance|ChannelStack|CholFactor|DiagFactor|as_evaluator|BandLayout|BlockTriFactor|CBlockFactor|pcg|device_constants|loglike_[a-z_]+|whittle_many|gaussian_from_func)\b|\bEvolutionaryCovariance\b[^.]" src tests README.md`

Expected: exactly one line:
```
tests/wilson/meyer_cases.py:169:    that recorded them are computed by the kernels they wrapped: ``EvolutionaryCovariance``
```
- That is `domainia_cases`' docstring. It names the class that recorded the `EvolutionaryCovariance.*` keys, which keep their old names on purpose.
- `EvolutionaryCovariance.` followed by a dot, as in the key names, is not matched.
- The oracle copies are excluded; they stay whole.

- [ ] **Step 20: Gates**

The three runs are independent and can run in parallel.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2325 passed, 3 skipped`. That is Task 1's 3121 minus 796:
- 591 items in the 19 deleted files;
- the 12 `inner_product.py` doctests;
- a net 193 in the edited files:
  - Step 8: −3;
  - Steps 9–16: −7, −6, −3, −66, 0, −32, −62 and −16;
  - Step 17: +2.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2085 passed, 243 skipped`. It was `2857 passed, 267 skipped`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2085 passed, 243 skipped`.

None of the three lines contains `warning` or `error`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. The FLOOR prefix prints the same line.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 8592 lines / 3850 code
tests 8677 lines (+ oracle 4135)
```

- [ ] **Step 21: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true). The 29 deletions are already staged by Steps 1 and 2. This adds the 25 modified paths by name.

Run: `cd /Users/aaron/Documents/lisa/domainia && git add README.md src/domainia/__init__.py src/domainia/domains.py src/domainia/maps/base.py src/domainia/testing.py src/domainia/translate.py src/domainia/variance.py src/domainia/wilson/__init__.py src/domainia/wilson/evolutionary.py src/domainia/wilson/grid.py src/domainia/wilson/noise.py src/domainia/wilson/transform.py tests/test_dom_maps_base.py tests/test_dom_testing.py tests/test_dom_variance_generic.py tests/test_dom_windows.py tests/test_package.py tests/test_translate.py tests/test_wrong_kinds_misc.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_grid_bands.py tests/wilson/test_window_threading.py tests/wilson/test_wrong_kinds.py tests/wilson/vendored/conftest.py tests/wilson/vendored/test_evolutionary.py && git status --short | grep -vc "^[MD] "`

Expected: `0`. Every remaining status line is a staged change, so nothing is unstaged or untracked.

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Delete the likelihoods, factorisations, covariance classes, operators and coordinates" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected:
- `54 files changed, 106 insertions(+), 8309 deletions(-)`: 29 files deleted and 25 modified.
- Of those, `src` and `README.md` are 22 files with 39 insertions and 3755 deletions, and `tests` is 32 files with 67 insertions and 4554 deletions.

Size after Task 2 (prototype): src 8592 lines / 3850 code; tests 8677 (+ oracle 4135).

---

### Task 3: Add `domainia.transform` (spec §2; §6.2 "transform")

**Goal:** `domainia.transform(x, source, target, *, N, dt, source_wdm=None, target_wdm=None, window=MEYER)` moves data arrays with any leading axes between time, frequency and WDM through the rfft, on today's kernels, written test-first.

Assumes the tree after Task 2 (commit "Delete the likelihoods, factorisations, covariance classes, operators and coordinates").

**Files:**
- Create:
  - `src/domainia/_transform.py`: `transform`, the helpers `_grid` and `_checked`, and the constants `KINDS` and `RESIDUE` (Step 2).
  - `tests/test_transform.py`: five tests (Step 1).
- Modify:
  - `src/domainia/__init__.py`: the module docstring's Modules list, and `transform` is imported from `._transform` instead of `.translate` (Step 3). `__all__` does not change; it already lists `"transform"` and `"translate"`.
  - `tests/wilson/test_window_threading.py`: `SCANNED` gains `"_transform.py"` (Step 5).
  - `README.md`: the two python blocks import `transform` from `domainia.translate` (Step 6).
- Delete: none. `src/domainia/translate.py` and `tests/test_translate.py` stay until Task 4.
- Test: `tests/test_transform.py`. Also affected: `tests/wilson/test_window_threading.py`, `tests/test_readme.py` and `tests/test_package.py`.

**How `transform` works** (spec §2):
- **Checks, in order:**
  - `N`: `require_int` (`TypeError` for a non-integer), then `N >= 1`.
  - `dt`: `wilson.validate_dt`.
  - `source`, then `target` (`_grid`):
    - a non-string raises `TypeError`, and a string other than `"time"`, `"frequency"` or `"wdm"` raises `ValueError`;
    - `source_wdm` or `target_wdm` is required on a `"wdm"` side and refused on the others (`ValueError`);
    - the pair goes through `wilson.validate_grid`, then `Nf * Nt == N` is checked (`ValueError`).
  - `x` (`_checked`):
    - its dtype kind must be one of `iufc` (`require_dtype`, `TypeError` for bool, str or object);
    - complex `x` on time or WDM raises `ValueError`;
    - the trailing shape must be `(N,)`, `(N // 2 + 1,)` or `(Nf + 1, Nt)` (`ValueError`);
    - `x` must be finite (`wilson.validate_finite`, `ValueError`);
    - `x` is then copied to float64, or to complex128 on frequency. A real `x` on frequency is taken as complex.
  - Then the canonical-form residue (below).
- **Route.**
  - Every pair goes through the raw rfft `R`:
    - time: `numpy.fft.rfft` and `irfft(n=N)`;
    - frequency: `X / dt` and `dt * R`;
    - WDM: `wilson.inverse_many` and `wilson.forward_many`.
  - Today's `_many` kernels take exactly one leading axis, so `transform` flattens the leading axes into one around each kernel call and restores them afterwards. Task 7 replaces the two calls with the folded `inverse` and `forward` and drops the reshapes.
  - Both kernel calls pass `window=window`.
  - When `(source, source_wdm) == (target, target_wdm)`, the route is skipped and the result is the validated copy.
- **Canonical output.** Every result, including that copy, is made exactly canonical:
  - inactive WDM cells are set to 0;
  - the imaginary parts at DC, and at Nyquist for even `N`, are set to 0.
- **The residue rule** (`RESIDUE = 1e-10`).
  - The peak is `max |x|` over the domain axes, for each leading index.
  - The residue is `|Im x|` at DC and at even-`N` Nyquist, and `|x|` at an inactive WDM cell.
  - Each residue is divided by its slice's peak, with 0 where the peak is 0. A ratio above `RESIDUE` is refused with `ValueError`, naming the first such place:
    - `x[2, 1]: the imaginary part of rfft bin 0 (DC) is 1e-06 of the slice's peak; at most 1e-10 is accepted`;
    - `x: inactive WDM cell (m=0, n=1) is 0.38 of the slice's peak; at most 1e-10 is accepted` (no leading axes).
  - So a slice that is zero apart from a residue has a ratio of 1 and is refused.
- **Exception kinds the spec leaves open**, which follow the package rule (`TypeError` for a wrong kind, `ValueError` for a wrong value):
  - `validate_grid` raises `TypeError` for a non-integer `Nf` or `Nt` (`(8.0, 16)`), and `ValueError` for one that is not positive and even.
  - A `source_wdm` of the wrong length raises Python's unpacking `ValueError`, and a non-iterable one raises `TypeError`.

**Acceptance Criteria:**
- [ ] Step 1's run gives `5 failed`, each with `TypeError: transform() got an unexpected keyword argument 'N'`.
- [ ] `pytest -q tests/test_transform.py` → `5 passed` in PY, the NumPy lane and FLOOR.
- [ ] Step 4's probe prints `cases 30 fixture worst rel 0.0` in PY and FLOOR.
- [ ] `pytest -q tests/wilson/test_window_threading.py` → `502 passed` (it was 501).
- [ ] `pytest -q tests/test_readme.py tests/test_package.py` → `9 passed`.
- [ ] `grep -nE "be=|float32|jax" src/domainia/_transform.py` prints nothing.
- [ ] `meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`.
- [ ] Full suite (prototype): PY `2331 passed, 3 skipped`; NumPy lane and FLOOR `2091 passed, 243 skipped`. No warnings and no collection errors.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_transform.py 2>&1 | tail -1` → `5 passed in …`

**Steps:**

- [ ] **Step 1: Write the test (it fails first)**

Create `tests/test_transform.py` with this text:

```python
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
    (measured: 0.0) and exactly canonical."""
    with np.load(REFERENCE) as z:
        for i, ((s, t), (n, dt, *g)) in enumerate(zip(z["routes"], z["grids"])):
            n, sg, tg = int(n), (int(g[0]), int(g[1])), (int(g[2]), int(g[3]))
            got = transform(z[f"x{i}"], s, t, N=n, dt=dt, source_wdm=sg if s == "wdm" else None,
                            target_wdm=tg if t == "wdm" else None)
            _close(got, z[f"y{i}"], 1e-12)
            if t == "frequency":
                assert not np.any(got[..., 0].imag) and (n % 2 or not np.any(got[..., -1].imag)), i
            if t == "wdm":
                assert not np.any(got[..., ~wilson.active_mask(*tg)]), i


def test_round_trips_leading_axes_and_parseval():
    """Every pair maps X's three layouts onto each other and back, with a non-default window;
    each row is the kernel's; the energy scales by dt on WDM and dt^2 N on frequency."""
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
    real = np.arange(65.0)
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
```

What each test covers, against the bullets of spec §6.2 "transform":
- `test_agrees_with_the_recorded_translate_transform`:
  - bullet 1, agreement with the outputs recorded in Task 1, on all 30 cases of `tests/data/translate_reference.npz`, to 1e-12 of each output's peak;
  - bullet 4, canonical outputs, exactly, on every case. The 6x10 grids are among them, where the raw kernels leave residue.
- `test_round_trips_leading_axes_and_parseval`:
  - bullets 2 and 3: round trips of every pair, on a `(3, 2, …)` stack;
  - it uses the non-default window `ALT_WINDOW`, and checks one row against `wilson.forward_time` with that window, so the window is threaded on both legs;
  - it also checks the Parseval scales (1 on time, `dt` on WDM, `dt²·N` on frequency). They are carried over from `test_translate.py`'s `test_parseval_scales`, at even N only.
- `test_a_residue_up_to_1e_10_of_the_peak_is_accepted_and_the_copy_is_canonical`:
  - bullet 5: a residue of 1e-12 of the peak at DC, at Nyquist and in an inactive cell of each edge band is accepted, and one of about 1e-6 is refused with a message naming the leading index and the bin or cell;
  - bullet 7: `source == target` returns a fresh canonical copy;
  - the kernels ignore an accepted residue exactly (the time results are equal);
  - a real input on frequency is taken as complex.
- `test_wdm_to_wdm_across_two_grids_is_the_chain_through_frequency`: bullet 6, from 8x16 to 16x8.
- `test_every_refusal`: bullet 8, complex time data (`ValueError`) and bool, str and object input (`TypeError`), plus the other argument and input rules of §2.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_transform.py 2>&1 | tail -1`

Expected: `5 failed in …`. `domainia.transform` is still `translate.transform`, so every test stops at its first call with `TypeError: transform() got an unexpected keyword argument 'N'`. Check with `… -m pytest -q tests/test_transform.py 2>&1 | grep -c "unexpected keyword argument 'N'"`, which prints `5`.

- [ ] **Step 2: Create `src/domainia/_transform.py`**

Create it with this text:

```python
"""Data arrays between time, frequency and WDM (domainia 3.0 spec, section 2).

The layouts, on the trailing axes of an array with any leading axes (typically ``(K, ...)``):

* ``"time"``: real ``(N,)``;
* ``"frequency"``: complex ``(N // 2 + 1,)``, ``dt * numpy.fft.rfft(x)``;
* ``"wdm"``: real ``(Nf + 1, Nt)``, the Wilson coefficients of ``window``, edge bands
  included.

Every route goes through the rfft ``numpy.fft.rfft(x)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import wilson
from .wilson import MEYER, Window
from .wilson._checks import require_dtype, require_int

__all__ = ["transform"]

KINDS = ("time", "frequency", "wdm")
#: The largest canonical-form residue an input may carry, as a fraction of its slice's peak.
RESIDUE = 1e-10


def _grid(role: str, kind: Any, wdm: Any, N: int) -> tuple[int, int] | None:
    """The validated ``(Nf, Nt)`` of a ``"wdm"`` side, ``None`` on the others."""
    if not isinstance(kind, str):
        raise TypeError(f"{role} must be a string, got {type(kind).__name__}")
    if kind not in KINDS:
        raise ValueError(f"{role} must be one of {KINDS}, got {kind!r}")
    if kind != "wdm":
        if wdm is not None:
            raise ValueError(f"{role}_wdm is only for a 'wdm' {role}; {role} is {kind!r}")
        return None
    if wdm is None:
        raise ValueError(f"a 'wdm' {role} needs {role}_wdm=(Nf, Nt)")
    Nf, Nt = wdm
    wilson.validate_grid(Nf, Nt)
    if Nf * Nt != N:
        raise ValueError(f"{role}_wdm=({Nf}, {Nt}) holds {Nf * Nt} samples, not N={N}")
    return int(Nf), int(Nt)


def _checked(x: Any, kind: str, N: int, grid: tuple[int, int] | None) -> np.ndarray:
    """``x`` as a float64 (complex128 on frequency) copy, after the input checks."""
    if isinstance(x, np.ma.MaskedArray):
        raise TypeError("x is a masked array; numpy.asarray would drop the mask, so fill it first")
    a = np.asarray(x)
    require_dtype("x", a, "iufc", "a numeric array")
    if a.dtype.kind == "c" and kind != "frequency":
        raise ValueError(f"x is complex, but {kind} data are real")
    shape = {"time": (N,), "frequency": (N // 2 + 1,)}.get(kind) or (grid[0] + 1, grid[1])
    nd = len(shape)
    if a.shape[a.ndim - nd:] != shape:
        raise ValueError(f"x has shape {a.shape}, but {kind} data end in shape {shape}")
    wilson.validate_finite(a, "x")
    a = a.astype(np.complex128 if kind == "frequency" else np.float64)
    if kind == "time":
        return a
    if kind == "frequency":
        ends = np.zeros(shape, bool)
        ends[-1] = N % 2 == 0
        ends[0] = True
        off = np.where(ends, np.abs(a.imag), 0.0)
    else:
        off = np.where(wilson.active_mask(*grid), 0.0, np.abs(a))
    peak = np.max(np.abs(a), axis=tuple(range(-nd, 0)), keepdims=True)
    rel = np.divide(off, peak, out=np.zeros_like(off), where=peak > 0)
    bad = np.argwhere(rel > RESIDUE)
    if bad.size:
        at = [int(i) for i in bad[0]]
        lead, k = at[:-nd], at[-nd:]
        where = (f"the imaginary part of rfft bin {k[0]} ({'Nyquist' if k[0] else 'DC'})"
                 if kind == "frequency" else f"inactive WDM cell (m={k[0]}, n={k[1]})")
        raise ValueError(f"x{lead if lead else ''}: {where} is {rel[tuple(at)]:.2g} of the "
                         f"slice's peak; at most {RESIDUE:g} is accepted")
    return a


def transform(x: Any, source: str, target: str, *, N: int, dt: float,
              source_wdm: tuple[int, int] | None = None, target_wdm: tuple[int, int] | None = None,
              window: Window = MEYER) -> np.ndarray:
    """``x`` on ``source`` re-expressed on ``target`` (module docstring for the layouts).

    ``source`` and ``target`` are each ``"time"``, ``"frequency"`` or ``"wdm"``.  ``N`` is the
    record's sample count, always required.  ``source_wdm`` and ``target_wdm`` are the
    ``(Nf, Nt)`` of a ``"wdm"`` side, required there and refused elsewhere, with Nf and Nt
    positive and even and ``Nf * Nt == N``.  The two WDM grids may differ: WDM -> WDM then
    regrids through the rfft.  When the source equals the target, grid included, the result
    is a validated canonical copy.

    The result is float64 (complex128 on frequency) and exactly canonical: inactive WDM cells
    are 0, and so are the imaginary parts at DC and even-``N`` Nyquist.  An input may carry a
    residue there up to :data:`RESIDUE` of its slice's peak (``max |x|`` over the domain axes,
    for each leading index); a larger one is refused, naming the cell or bin.

    Raises ``TypeError`` for a masked or non-numeric ``x`` (bool, str, object), a non-string
    ``source``/``target``, or a non-integer ``N``, ``Nf`` or ``Nt``; ``ValueError`` for an
    unknown domain, a missing or misplaced or bad WDM grid, a bad ``N`` or ``dt``, complex
    ``x`` on time or WDM, a wrong trailing shape, a non-finite value, or a residue above
    :data:`RESIDUE`.  A real ``x`` on frequency is taken as complex.
    """
    N = require_int("N", N)
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    wilson.validate_dt(dt)
    dt = float(dt)
    sg, tg = _grid("source", source, source_wdm, N), _grid("target", target, target_wdm, N)
    y = _checked(x, source, N, sg)
    if (source, sg) != (target, tg):
        lead = y.shape[:y.ndim - (2 if source == "wdm" else 1)]
        if source == "time":
            R = np.fft.rfft(y, axis=-1)
        elif source == "frequency":
            R = y / dt
        else:
            R = wilson.inverse_many(y.reshape(-1, sg[0] + 1, sg[1]), *sg, dt,
                                    window=window).reshape(lead + (N // 2 + 1,))
        if target == "time":
            y = np.fft.irfft(R, n=N, axis=-1)
        elif target == "frequency":
            y = dt * R
        else:
            y = wilson.forward_many(R.reshape(-1, N // 2 + 1), *tg, dt,
                                    window=window).reshape(lead + (tg[0] + 1, tg[1]))
    if target == "frequency":
        y[..., 0] = y[..., 0].real
        if N % 2 == 0:
            y[..., -1] = y[..., -1].real
    elif target == "wdm":
        y[..., ~wilson.active_mask(*tg)] = 0.0
    return y
```

- [ ] **Step 3: Re-export `transform` from `src/domainia/__init__.py`**

Edit 1, the module docstring. Old:

```python
* :mod:`domainia.translate`: ``transform`` (``{channel: array}`` dicts) between the
  full-record ``TimeDomain``, ``FourierDomain.full`` and ``WDMDomain.full``.
```

New:

```python
* :func:`transform`: data arrays, with any leading axes, between time, frequency and WDM
  (:mod:`domainia._transform`).
* :mod:`domainia.translate`: the 2.0 ``transform`` of ``{channel: array}`` dicts between the
  full-record ``TimeDomain``, ``FourierDomain.full`` and ``WDMDomain.full``.
```

Edit 2, the imports. Old:

```python
from . import data, domains, translate, wilson
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from .translate import transform
from .wilson import MEYER, Grid, Meyer, RampWindow, SampledWindow, Window, WindowError
```

New:

```python
from . import data, domains, translate, wilson
from ._transform import transform
from .data import FourierBins, TimeSeries, WDMCells
from .domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from .wilson import MEYER, Grid, Meyer, RampWindow, SampledWindow, Window, WindowError
```

`translate` stays imported as a module. `tests/test_package.py` requires it among the modules a fresh `import domainia` loads, and in `__all__`, until Task 4.

- [ ] **Step 4: The new tests pass, and the measured agreement**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_transform.py 2>&1 | tail -1`

Expected: `5 passed in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q tests/test_transform.py 2>&1 | tail -1`

Expected: `5 passed in …`

This probe measures the agreement with the recorded outputs. The test's bound is 1e-12.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P - <<'EOF'
import numpy as np
from domainia import transform
z = np.load("tests/data/translate_reference.npz")
worst = 0.0
for i, ((s, t), (n, dt, *g)) in enumerate(zip(z["routes"], z["grids"])):
    sg, tg = (int(g[0]), int(g[1])), (int(g[2]), int(g[3]))
    got = transform(z[f"x{i}"], s, t, N=int(n), dt=dt, source_wdm=sg if s == "wdm" else None,
                    target_wdm=tg if t == "wdm" else None)
    worst = max(worst, float(np.max(np.abs(got - z[f"y{i}"])) / np.max(np.abs(z[f"y{i}"]))))
print("cases", len(z["routes"]), "fixture worst rel", worst)
EOF
```

Expected: `cases 30 fixture worst rel 0.0`. The FLOOR prefix prints the same line. `transform` runs the same kernels on the same stacks as `translate.transform`, so the outputs are bit-identical.

Until Steps 5 and 6, the suite has two failures: `test_the_scan_covers_every_module_but_filter_and_windows` and `test_readme_python_blocks_run`.

- [ ] **Step 5: Scan the new module for the window (`tests/wilson/test_window_threading.py`)**

`test_the_scan_covers_every_module_but_filter_and_windows` requires every module of the package except `UNSCANNED` to be in `SCANNED`. Before this step it fails, because `_transform.py` is missing.

Old:

```python
           "testing.py", "data.py", "spectra.py", "__init__.py", "maps/__init__.py",
           "translate.py")
```

New:

```python
           "testing.py", "data.py", "spectra.py", "__init__.py", "maps/__init__.py",
           "translate.py", "_transform.py")
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py 2>&1 | tail -1`

Expected: `502 passed in …`. It was 501. The new item is `test_no_kernel_call_drops_the_window[_transform.py]`. It passes because both kernel calls in `_transform.py` pass `window=window`, and `MEYER` appears only as the signature default.

- [ ] **Step 6: The README's python blocks keep the 2.0 dict API (`README.md`)**

After Step 3, `from domainia import transform` is the new function, so the README's two blocks, which call it with dicts and domain objects, fail:

`AssertionError: README.md python block 1 of 2 (line 26) raised TypeError: transform() missing 2 required keyword-only arguments: 'N' and 'dt'`

Until Task 4 deletes `translate.py`, both blocks import the dict form from `domainia.translate`. Task 4 rewrites them, and Task 11 rewrites the README.

Edit 1 (Quick start). Old:

```python
from domainia import FourierDomain, Grid, Meyer, TimeDomain, WDMDomain, transform
```

New:

```python
from domainia import FourierDomain, Grid, Meyer, TimeDomain, WDMDomain
from domainia.translate import transform
```

Edit 2 (Canonical inputs). Old:

```python
from domainia import FourierDomain, Grid, TimeDomain, WDMDomain, transform, wilson
```

New:

```python
from domainia import FourierDomain, Grid, TimeDomain, WDMDomain, wilson
from domainia.translate import transform
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_readme.py tests/test_package.py 2>&1 | tail -1`

Expected: `9 passed in …`

- [ ] **Step 7: Gates**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -nE "be=|float32|jax" src/domainia/_transform.py`

Expected: no output.

The three suite runs are independent and can run in parallel.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2331 passed, 3 skipped`. That is Task 2's 2325 plus the 5 new tests and the new `SCANNED` item.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2091 passed, 243 skipped`. It was `2085 passed, 243 skipped`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `2091 passed, 243 skipped`.

None of the three lines contains `warning` or `error`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. `tests/wilson/test_meyer_baseline.py` passes inside the full runs (`5 passed` on its own).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 8731 lines / 3941 code
tests 8806 lines (+ oracle 4135)
```
That is +139 src lines (+91 code), of which `_transform.py` is 137 lines (91 code); and +129 test lines, of which `tests/test_transform.py` is 129.

- [ ] **Step 8: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/_transform.py src/domainia/__init__.py tests/test_transform.py tests/wilson/test_window_threading.py README.md && git status --short | grep -vc "^[AM] "`

Expected: `0`. Every remaining status line is a staged change, so nothing is unstaged or untracked.

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Add domainia.transform: data arrays between time, frequency and WDM" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `5 files changed, 272 insertions(+), 5 deletions(-)`

Size after Task 3 (prototype): src 8731 lines / 3941 code; tests 8806 (+ oracle 4135).

---

### Task 4: Delete `translate.py`, `maps/`, `data.py`, `domains.py`, `spectra.py` and `variance.py`; the `testing` oracles take a cell mask (spec Order step 4; §4; §6.3; decisions 4 and 8)

**Goal:** Delete the 2.0 data route and its grids, which are `translate.py`, `maps/`, `data.py` and `domains.py`, and also `spectra.py` and `variance.py`, with every test whose subject is that code. The dense oracles `testing.dense_W`, `wdm_congruence` and `eq45_dense` take a boolean `(Nf + 1, Nt)` cell mask in place of a `CellSet`. The replay still reproduces 1408 arrays.

Assumes the tree after Task 3, whose commit is "Add domainia.transform: data arrays between time, frequency and WDM".

**Deviation from the spec's Order, already decided by the plan.** The spec deletes `spectra.py` and `variance.py` in its step 5. They move here because both import `domains`, and `domains.py` goes now.

**Files:**
- Delete, source: 9 files, 2,162 lines.
  - `src/domainia/translate.py` (214 lines);
  - `src/domainia/maps/`: `__init__` (11), `base` (472), `fourier` (397) and `time` (314);
  - `src/domainia/data.py` (93), `src/domainia/domains.py` (457), `src/domainia/spectra.py` (94) and `src/domainia/variance.py` (110).
- Delete, tests: 18 files, 3,043 lines and 936 items. Items are counted in PY.

| File | Lines | Items |
|---|---|---|
| `tests/test_translate.py` | 260 | 71 |
| `tests/test_dom_maps_base.py` | 187 | 9 |
| `tests/test_dom_maps_dirichlet.py` | 148 | 10 |
| `tests/test_dom_maps_fourier_time.py` | 80 | 4 |
| `tests/test_dom_maps_fourier_wdm.py` | 272 | 26 |
| `tests/test_dom_maps_wdm.py` | 101 | 13 |
| `tests/test_dom_map_dtypes.py` | 255 | 279 |
| `tests/test_dom_built_for.py` | 168 | 55 |
| `tests/test_wrong_kinds_maps.py` | 280 | 181 |
| `tests/test_dom_data.py` | 70 | 4 |
| `tests/test_dom_db_imports.py` | 37 | 2 |
| `tests/dom_db_inputs.py` (a helper, no tests) | 42 | 0 |
| `tests/test_dom_domains.py` | 193 | 14 |
| `tests/test_wrong_kinds_domains.py` | 294 | 190 |
| `tests/test_dom_spectra.py` | 40 | 2 |
| `tests/test_dom_variance_generic.py` | 77 | 11 |
| `tests/test_dom_windows.py` | 372 | 20 |
| `tests/test_wrong_kinds_misc.py` | 167 | 45 |

- Modify, source:
  - `src/domainia/testing.py`: the new helper `_keep`; `dense_W`, `wdm_congruence` and `eq45_dense` take `(…, Nf, Nt, dt, mask, …)`; the module docstring; the imports (Step 2).
  - `src/domainia/__init__.py`: rewritten whole (Step 5).
  - Docstrings that name deleted code: `wilson/_checks.py` (`require_instance`), `wilson/evolutionary.py` (`eq45_separable_kernels`) and `wilson/transform.py` (one bullet of the module docstring) (Step 11).
- Modify, tests:
  - `tests/test_dom_testing.py`: rewritten whole (Step 1). It gains the window test of the deleted `test_dom_windows.py` and the kind tests of the deleted `test_wrong_kinds_misc.py`.
  - `tests/wilson/meyer_cases.py`: `domainia_cases` (Step 3).
  - `tests/test_transform.py`: `test_agrees_with_the_recorded_translate_transform`, `test_round_trips_leading_axes_and_parseval` and `test_a_residue_up_to_1e_10_of_the_peak_is_accepted_and_the_copy_is_canonical` (Step 7).
  - `tests/test_package.py`: `PUBLIC`, `test_the_public_names`, `test_importing_every_module_resolves_to_src_and_does_not_import_jax` and `test_importing_domainia_leaves_the_unlisted_modules_unimported` (Step 8).
  - `tests/wilson/test_window_threading.py`: `SCANNED`, `CONSUMERS`, `test_the_scan_covers_every_module_but_filter_and_windows`'s docstring, and the parametrize list of `test_the_window_scan_flags_exactly_the_calls_that_lose_the_window` (Step 9).
  - `tests/wilson/test_checks.py`: `test_the_seven_integer_checks_raise_the_helpers_message`, which is renamed `test_the_six_integer_checks_raise_the_helpers_message`, and `test_the_range_checks_after_require_int_are_unchanged` (Step 10).
- Modify: `README.md`, the two python blocks and the paragraph before the second (Step 12); `pyproject.toml`, one comment (Step 12).
- Test: `tests/test_dom_testing.py` (7 tests). Also affected: `tests/test_transform.py`, `tests/wilson/test_meyer_baseline.py`, `tests/wilson/test_window_threading.py`, `tests/test_readme.py` and `tests/test_package.py`.

**The mask signatures** (spec §4: `testing.py`'s `CellSet` "becomes a bool mask"):
- `dense_W(Nf, Nt, dt, mask, *, window=MEYER)`, `wdm_congruence(Sigma, Nf, Nt, dt, mask, *, window=MEYER)` and `eq45_dense(Nf, Nt, dt, mask, Sn, *, P=24, t_max=None, window=MEYER)`. The grid arguments come first, as in `eq45_plan(Nf, Nt, dt, keep, …)`. A mask alone does not carry `dt`; the `CellSet`'s grid did.
- `mask` is a boolean array of shape `(Nf + 1, Nt)`, the WDM layout of `wilson.active_mask`. `eq45_plan`'s own `keep` is a different layout: a mask over the `N` packed active cells. The helper `_keep` converts one to the other with `mask[wilson.active_mask(Nf, Nt)]`.
- Rows and columns follow `numpy.nonzero(mask)`. That is the packed band order of `wilson.grid.packed_cells` (checked on 8x16, 16x32, 32x8 and 6x10), so with the full active mask every output equals the old `CellSet.all_active` output, bit for bit.
- Refusals, checked in this order:
  - a mask whose dtype is not bool raises `TypeError`: `mask must be a boolean cell mask, got dtype int64`;
  - a wrong shape raises `ValueError`: `mask must have shape (Nf + 1, Nt) = (9, 16), got (144,)`;
  - a mask that selects an inactive cell raises `ValueError`: `mask selects the inactive cell (m=0, n=1)`. The package fails closed, so the cell is not silently dropped.
  - `eq45_dense` then checks `Sn` (`TypeError`: `Sn must be callable, got float`), as before.
- `require_instance` is no longer imported by `testing.py`; `wilson/windows.py` still uses it.

**Which tests go.** A test goes when its subject is deleted code. A test that also checks kept code stays, moved or trimmed to the kept part.
- `test_dom_maps_*`, `test_dom_map_dtypes`, `test_dom_built_for`, `test_wrong_kinds_maps`, `test_dom_data`, `test_dom_db_imports`, `test_dom_spectra`, `dom_db_inputs.py`: their subjects are `maps/`, `data.py` and `spectra.py`. `test_dom_restored_coverage.py` already went in Task 2.
- `test_dom_domains.py` and `test_wrong_kinds_domains.py` go whole (spec §6.3). So do the `domains._int` row and asserts in `tests/wilson/test_checks.py`.
- `test_dom_variance_generic.py` goes whole. No test in it pins kept code that is not pinned elsewhere:
  - the stationary-limit test (`evolutionary_band_variance` of a stationary PSD is the exact covariance's diagonal) is `tests/wilson/test_window_threading.py::test_band_variances_are_the_diagonal_of_the_dense_covariance`, on 5 windows (`ALT_WINDOW` among them) and 4 grids;
  - the white-PSD test (`S/2`) follows from the same test, because `build_C_wdm_active` is an independent route;
  - `evolutionary_band_variance` on the active cells is the replay key `BandAveragedVariance`;
  - the other 8 items test `BandAveragedVariance`, `MidpointVariance` (Eq. 39) and their argument rules.
- `test_dom_windows.py` goes whole. Its `WDMDomain`, maps, variance-class and `jax.jit` tests go with that code. Its two tests of kept code, `test_testing_oracles_take_the_window` and `test_eq45_dense_oracle_takes_the_window`, become `test_the_oracles_compute_in_the_window_they_are_given` in `test_dom_testing.py`. `test_band_averaged_variance_uses_its_window`'s kernel check is the threading test named above.
- `test_wrong_kinds_misc.py` goes whole. Its `spectra`, `variance`, `data` and `translate` tests go. Its two `testing` tests, `test_the_dense_oracles_take_a_cell_set` and `test_eq45_dense_needs_a_callable_spectrum`, become `test_the_oracles_refuse_a_mask_of_the_wrong_kind_or_value` in `test_dom_testing.py`.
- `test_translate.py` goes whole. Task 3's `tests/test_transform.py` covers its definitions, round trips, canonical outputs, even-N Parseval and refusals. What `test_transform.py` lacked:
  - **carried into `test_transform.py`** (Step 7), in 6 added lines and no new test:
    - at odd N the last bin is not a Nyquist bin and keeps its imaginary part (from `test_an_odd_record_has_no_nyquist_bin`);
    - Parseval at odd N (`test_parseval_scales`'s odd records);
    - an integer input is taken as float (from `test_outputs_are_fresh_exact_and_in_input_order`), on the existing real-on-frequency line.
  - **carried into the README** (Step 12): raw `wilson.forward_time` output and a full-FFT slice on the 6x12 grid, which carry round-off in the inactive cells and at Nyquist, are accepted. That is the 1e-10 rule's counterpart of `test_the_named_remedies_make_raw_kernel_and_fft_outputs_valid_inputs`, and `tests/test_readme.py` runs it.
  - **not carried**:
    - `test_a_non_finite_result_is_refused`: `transform` checks its input for finiteness, not its output, and spec §2 does not ask for an output check;
    - the float32 and complex64 promotion: Task 6 removes float32, and the integer input exercises the same conversion;
    - the remedy and exact-zero messages, which the 1e-10 rule replaced in Task 3;
    - the dict and domain rules (`{channel: array}`, `TimeDomain`, `FourierDomain.full`, `WDMDomain.full`, `t0`).
- In `test_window_threading.py`:
  - `SCANNED` loses its 9 deleted modules, so `test_no_kernel_call_drops_the_window` has 9 fewer items;
  - `CONSUMERS` loses `WDMDomain`, `BandAveragedVariance`, `_cells_and_window` (in `domains.py`) and `_dense_atoms` (in `maps/fourier.py`), with their comment;
  - the 7 scanner rows that use `WDMDomain` and `BandAveragedVariance` go. The scanner's other rows, including the generic "domainia layer's patterns" rows (`self.window = window`, `cls(grid, window=window)`, `src.window != dst.window`), name no deleted code and stay.
  - `_forward`, `_inverse`, `_forward_wdm` and `_inverse_wdm` stay in `CONSUMERS`: they are import aliases inside `wilson/stationary.py`.
  - The domainia-consumer cases (`test_every_domainia_consumer_*`) keep their 6 items each. `meyer_cases.domainia_cases` now passes the mask.

**Acceptance Criteria:**
- [ ] Step 1's run gives `7 failed`, each with a `TypeError` about positional arguments of `dense_W`, `wdm_congruence` or `eq45_dense`.
- [ ] `pytest -q tests/test_dom_testing.py` → `7 passed` in PY and FLOOR.
- [ ] `src/domainia` holds `__init__.py`, `_transform.py`, `py.typed`, `testing.py` and `wilson/` only. `domainia.__all__` has 10 names: `Grid`, `MEYER`, `Meyer`, `RampWindow`, `SampledWindow`, `Window`, `WindowError`, `__version__`, `transform` and `wilson`.
- [ ] Step 13's grep prints exactly the two `tests/wilson/meyer_cases.py` lines that hold the recorded key name `BandAveragedVariance`.
- [ ] `python tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`, and `tests/wilson/test_meyer_baseline.py` passes in all three lanes.
- [ ] `pytest -q tests/wilson/test_window_threading.py` → `486 passed` (it was 502).
- [ ] Full suite (prototype): PY `1380 passed, 3 skipped`; NumPy lane and FLOOR `1293 passed, 90 skipped`. No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 6560 lines / 2605 code` and `tests 5769 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1` → `1380 passed, 3 skipped in …`

**Steps:**

Edits are anchored to code text, never to line numbers. Each Old text occurs exactly once in its file, and New replaces it. "Delete these lines" means the shown lines go, newline included.

The suite does not collect cleanly until Step 10 is done: until then some kept test files import deleted modules. Between Steps 2 and 6, the deleted `test_dom_windows.py` and `test_wrong_kinds_misc.py` still call the oracles with a `CellSet` and fail; they go in Step 6.

Steps 4, 6 and 15 are git writes in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: Rewrite `tests/test_dom_testing.py` (the test comes first)**

Replace the whole file `tests/test_dom_testing.py` with this text:

```python
"""The dense oracles of ``domainia.testing`` against independent constructions."""

from __future__ import annotations

import re

import numpy as np
import pytest

from domainia import wilson
from domainia.wilson.evolutionary import exact_covariance, time_domain_covariance
from domainia import testing as T

# The WDM Paper II toy models' test configuration: 8 x 16 cells, dt = 1/16, f_min = 0.5,
# f_ref = 1.
NF, NT, DT = 8, 16, 1.0 / 16
N = NF * NT
F_MIN, F_REF = 0.5, 1.0
#: Every active cell.
MASK = wilson.active_mask(NF, NT)


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


def drift(ac, ad):
    """``S(f, t) = (max(f, f_min)/f_ref)^(ac + ad (t/T - 1/2))``, elementwise (A_n = 1)."""
    T_obs = N * DT
    return lambda f, t: np.exp((ac + ad * (np.asarray(t) / T_obs - 0.5)) * L_of(f))


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def test_dense_W_rows_are_the_atoms(rng):
    W = T.dense_W(NF, NT, DT, MASK)
    x = rng.standard_normal(N)
    assert np.max(np.abs(W @ x - wilson.forward_time(x, NF, NT, DT)[MASK])) < 1e-12
    assert np.max(np.abs(W @ W.T - DT * np.eye(N))) < 1e-14
    sub = MASK.copy()
    sub[:, :3] = sub[:, -1:] = False
    assert np.array_equal(T.dense_W(NF, NT, DT, sub), W[sub[MASK]])


def test_congruence_is_the_exact_covariance():
    Sn = drift(-2.0, 1.2)
    got = T.wdm_congruence(time_domain_covariance(N, DT, Sn), NF, NT, DT, MASK)
    assert rel(got, exact_covariance(NF, NT, DT, Sn)) < 1e-12


@pytest.mark.parametrize("ac", [-2.0, -0.7])
def test_eq45_is_exact_for_stationary_noise(ac):
    """Ported from the toy models' test of the same name."""
    C = T.eq45_dense(NF, NT, DT, MASK, drift(ac, 0.0))
    assert rel(C, exact_covariance(NF, NT, DT, drift(ac, 0.0))) < 1e-10


def test_eq45_differs_from_exact_at_second_order_with_drift():
    """Ported from the toy models' test of the same name (the test's small edges: 3
    leading and 1 trailing column)."""
    C = T.eq45_dense(NF, NT, DT, MASK, drift(-2.0, 1.2))
    exact = exact_covariance(NF, NT, DT, drift(-2.0, 1.2))
    interior = ~np.isin(np.nonzero(MASK)[1], wilson.grid.edge_columns(NT, 3, 1))
    d = np.diag(exact)[interior]
    assert np.max(np.abs(np.diag(C)[interior] / d - 1)) < 0.05


def test_the_oracles_compute_in_the_window_they_are_given():
    """With ALT_WINDOW: ``dense_W`` is ``active_W``, the congruence is ``W S W^T``, and Eq. 45
    of a stationary spectrum is the exact covariance; each differs from Meyer's."""
    W = T.dense_W(NF, NT, DT, MASK, window=T.ALT_WINDOW)
    np.testing.assert_array_equal(W, wilson.matrix.active_W(NF, NT, DT, window=T.ALT_WINDOW))
    S = np.diag(1.0 + np.arange(N) / N)                  # not the identity: W I W^T = dt I for any window
    alt = T.wdm_congruence(S, NF, NT, DT, MASK, window=T.ALT_WINDOW)
    np.testing.assert_allclose(alt, W @ S @ W.T, rtol=0, atol=1e-12)
    assert not np.allclose(alt, T.wdm_congruence(S, NF, NT, DT, MASK))
    got = T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0), window=T.ALT_WINDOW)
    exact = wilson.matrix.build_C_wdm_active(NF, NT, DT, lambda f: drift(-1.2, 0.0)(f, 0.0),
                                             window=T.ALT_WINDOW)
    assert rel(got, exact) < 1e-10
    assert not np.allclose(got, T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0)))


def test_the_oracles_refuse_a_mask_of_the_wrong_kind_or_value():
    """A mask that is not boolean is a wrong kind; a wrong shape or an inactive cell is a wrong
    value.  ``eq45_dense`` also needs a callable spectrum."""
    for mask, err, message in (
        (MASK.astype(int), TypeError, "mask must be a boolean cell mask, got dtype int64"),
        (None, TypeError, "mask must be a boolean cell mask, got dtype object"),
        (MASK.reshape(-1), ValueError, "mask must have shape (Nf + 1, Nt) = (9, 16), got (144,)"),
        (np.ones((NF + 1, NT), bool), ValueError, "mask selects the inactive cell (m=0, n=1)"),
    ):
        match = f"^{re.escape(message)}$"
        with pytest.raises(err, match=match):
            T.dense_W(NF, NT, DT, mask)
        with pytest.raises(err, match=match):
            T.wdm_congruence(np.eye(N), NF, NT, DT, mask)
        with pytest.raises(err, match=match):
            T.eq45_dense(NF, NT, DT, mask, drift(0.0, 0.0))
    with pytest.raises(TypeError, match=r"^Sn must be callable, got float$"):
        T.eq45_dense(NF, NT, DT, MASK, 1.0)
```

What changed:
- The four existing tests call the oracles with `MASK`, every active cell, in place of `CellSet.all_active(GRID)`, and `GRID` goes.
- `test_dense_W_rows_are_the_atoms` compares `W @ x` with `forward_time(x)[MASK]` instead of `wilson.grid.pack(...)`, which Task 11 deletes. The sub-mask drops 3 leading columns and 1 trailing column, as `CellSet.drop_columns(3, 1)` did, and selects rows with `sub[MASK]`.
- `test_eq45_differs_from_exact_at_second_order_with_drift` keeps its `wilson.grid.edge_columns(NT, 3, 1)` pin. It reads the cells' time index from `np.nonzero(MASK)[1]`, in the packed order of `np.diag(C)`. Task 11 deletes `edge_columns` and rewrites this pin with `evolutionary.wrap_cut_columns` (spec §3.3).
- `test_the_oracles_compute_in_the_window_they_are_given` is the deleted `test_dom_windows.py`'s two oracle tests, on the same 8x16 grid at `dt = 1/16`. The stationary spectrum is this file's `drift(-1.2, 0.0)`, in place of `GenericPSD`'s `(1 + f^2)^-1.2`.
- `test_the_oracles_refuse_a_mask_of_the_wrong_kind_or_value` is the deleted `test_wrong_kinds_misc.py`'s two `testing` tests, for masks.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_testing.py 2>&1 | tail -1`

Expected: `7 failed in …`. Each fails with a `TypeError` from the old signatures; `… 2>&1 | grep -E "^E +TypeError" | sort | uniq -c` shows:
```
   1 E               TypeError: dense_W() takes 1 positional argument but 4 were given
   1 E       TypeError: dense_W() takes 1 positional argument but 4 positional arguments (and 1 keyword-only argument) were given
   1 E       TypeError: dense_W() takes 1 positional argument but 4 were given
   3 E       TypeError: eq45_dense() takes 2 positional arguments but 5 were given
   1 E       TypeError: wdm_congruence() takes 2 positional arguments but 5 were given
```

- [ ] **Step 2: The oracles take a mask (`src/domainia/testing.py`)**

Edit 1, `src/domainia/testing.py`. Old:

```python
* :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a cell set
  (``wilson.matrix.active_W``) and ``W_K Sigma W_K^T``.
* :func:`eq45_dense`: Eq. (45) on a cell set by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).

:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.

The oracles follow the package's rule: a wrong kind (``cells`` that is not a ``CellSet``)
raises ``TypeError``, a wrong value ``ValueError``.
```

New:

```python
* :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a set of cells
  (``wilson.matrix.active_W``) and ``W_K Sigma W_K^T``.
* :func:`eq45_dense`: Eq. (45) on a set of cells by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).

The cells are a boolean ``mask`` of shape ``(Nf + 1, Nt)``, the WDM layout of
``wilson.active_mask``, that selects active cells only.  Rows and columns follow
``numpy.nonzero(mask)``: band by band, time within a band.

:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.

The oracles follow the package's rule: a wrong kind (a ``mask`` that is not boolean) raises
``TypeError``, a wrong value (a wrong shape, an inactive cell) ``ValueError``.
```

Edit 2, `src/domainia/testing.py`. Old:

```python
from domainia.wilson._checks import require_callable, require_instance
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
from domainia.wilson.windows import MEYER, SampledWindow, Window

from .domains import CellSet

```

New:

```python
from domainia.wilson._checks import require_callable, require_dtype
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
from domainia.wilson.windows import MEYER, SampledWindow, Window

```

Edit 3, `src/domainia/testing.py`. Old:

```python
def dense_W(cells: CellSet, *, window: Window = MEYER) -> Array:
    """``(cells.size, N)``: row ``p`` is cell ``p``'s atom sampled in time, so ``W x`` are
    the cells of ``x`` (``wilson.matrix.active_W`` rows, ``W_a W_a^T = dt I``) in the
    Wilson basis of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    return wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=window)[cells.packed_positions()]


def wdm_congruence(Sigma: Array, cells: CellSet, *, window: Window = MEYER) -> Array:
    """``W_K Sigma W_K^T`` for a time-domain covariance ``Sigma (N, N)`` (``W_K`` is
    :func:`dense_W` with ``window``)."""
    W = dense_W(cells, window=window)
    return W @ np.asarray(Sigma, dtype=float) @ W.T


def eq45_dense(cells: CellSet, Sn: Callable[[Any, Any], Any], *, P: int = 24,
               t_max: float | None = None, window: Window = MEYER) -> Array:
    """Eq. (45) on ``cells`` by the direct sum, ``Sn(f, t)`` elementwise (``A_n`` inside),
    with the atoms of ``window``."""
    g = require_instance("cells", cells, CellSet, "a CellSet").grid
    require_callable("Sn", Sn)
    plan = eq45_plan(g.Nf, g.Nt, g.dt, cells.keep_mask(), P=P, t_max=t_max, window=window)
    return eq45_direct(plan, Sn)
```

New:

```python
def _keep(Nf: int, Nt: int, mask: Any) -> NDArray[np.bool_]:
    """The cells of ``mask`` as a mask over the ``N`` packed active cells (``eq45_plan``'s
    ``keep``), after the checks of the module docstring."""
    require_dtype("mask", mask, "b", "a boolean cell mask")
    mask = np.asarray(mask)
    active = wilson.active_mask(Nf, Nt)
    if mask.shape != active.shape:
        raise ValueError(f"mask must have shape (Nf + 1, Nt) = {active.shape}, got {mask.shape}")
    bad = np.argwhere(mask & ~active)
    if bad.size:
        raise ValueError(f"mask selects the inactive cell (m={bad[0, 0]}, n={bad[0, 1]})")
    return mask[active]


def dense_W(Nf: int, Nt: int, dt: float, mask: Any, *, window: Window = MEYER) -> Array:
    """``(cells, N)``: row ``p`` is the atom of the ``p``-th cell of ``mask`` sampled in
    time, so ``W x`` are those cells of ``x`` (``wilson.matrix.active_W`` rows,
    ``W_a W_a^T = dt I``) in the Wilson basis of ``window``."""
    keep = _keep(Nf, Nt, mask)
    return wilson.matrix.active_W(Nf, Nt, dt, window=window)[keep]


def wdm_congruence(Sigma: Array, Nf: int, Nt: int, dt: float, mask: Any, *,
                   window: Window = MEYER) -> Array:
    """``W_K Sigma W_K^T`` for a time-domain covariance ``Sigma (N, N)`` (``W_K`` is
    :func:`dense_W` with ``window``)."""
    W = dense_W(Nf, Nt, dt, mask, window=window)
    return W @ np.asarray(Sigma, dtype=float) @ W.T


def eq45_dense(Nf: int, Nt: int, dt: float, mask: Any, Sn: Callable[[Any, Any], Any], *,
               P: int = 24, t_max: float | None = None, window: Window = MEYER) -> Array:
    """Eq. (45) on the cells of ``mask`` by the direct sum, ``Sn(f, t)`` elementwise (``A_n``
    inside), with the atoms of ``window``."""
    keep = _keep(Nf, Nt, mask)
    require_callable("Sn", Sn)
    plan = eq45_plan(Nf, Nt, dt, keep, P=P, t_max=t_max, window=window)
    return eq45_direct(plan, Sn)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_dom_testing.py 2>&1 | tail -1`

Expected: `7 passed in …`. The FLOOR prefix (`PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P`) gives the same.

- [ ] **Step 3: Rewire the replay's `testing.*` keys (`tests/wilson/meyer_cases.py`, `domainia_cases`)**

The keys `testing.dense_W`, `testing.wdm_congruence` and `testing.eq45_dense` keep their names. They are computed through the mask signatures, with every active cell. The `cells()` closure goes, so the replay no longer imports `domainia.domains`.

Edit 1, `tests/wilson/meyer_cases.py`. Old:

```python
    m, n = np.nonzero(wl.active_mask(Nf, Nt))
```

New:

```python
    mask = wl.active_mask(Nf, Nt)
    m, n = np.nonzero(mask)
```

Edit 2, `tests/wilson/meyer_cases.py`. Old:

```python
                                                             k=plan.k_used))

    def cells() -> Any:
        """Every active cell as a ``CellSet``, which ``testing`` takes for now (imported on
        call, so collecting this module never needs ``domainia.domains``)."""
        from domainia.domains import CellSet
        return CellSet.all_active(wl.Grid(Nf, Nt, dt))
```

New:

```python
                                                             k=plan.k_used))
```

Edit 3, `tests/wilson/meyer_cases.py`. Old:

```python
            ("testing.dense_W", lambda kw: testing.dense_W(cells(), **kw)),
            ("testing.wdm_congruence", lambda kw: testing.wdm_congruence(Sigma, cells(), **kw)),
            ("testing.eq45_dense", lambda kw: testing.eq45_dense(cells(), Sn, **kw)),
```

New:

```python
            ("testing.dense_W", lambda kw: testing.dense_W(Nf, Nt, dt, mask, **kw)),
            ("testing.wdm_congruence", lambda kw: testing.wdm_congruence(Sigma, Nf, Nt, dt, mask, **kw)),
            ("testing.eq45_dense", lambda kw: testing.eq45_dense(Nf, Nt, dt, mask, Sn, **kw)),
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`.

- [ ] **Step 4: Delete the source modules**

Run: `cd /Users/aaron/Documents/lisa/domainia && git rm -q -r src/domainia/translate.py src/domainia/maps src/domainia/data.py src/domainia/domains.py src/domainia/spectra.py src/domainia/variance.py && rm -rf src/domainia/maps && git status --short src | wc -l`

Expected: `10`: the 9 deleted files and the modified `testing.py`. `wc` may pad the number with spaces.
- The `rm -rf` removes any untracked `__pycache__` that `git rm` leaves in `maps/`.
- No kept module imports the deleted ones once `testing.py` is done (Step 2) and `__init__.py` is rewritten (Step 5).
- Steps 4 and 5 go together: between them `import domainia` fails on `from . import data, domains, translate, wilson`, so run nothing in between.

- [ ] **Step 5: Rewrite `src/domainia/__init__.py`**

Replace the whole file `src/domainia/__init__.py` with this text:

```python
"""domainia: data and covariance across the time, Fourier and time-frequency (Wilson / WDM)
domains.

Importing ``domainia`` (any of its modules) never imports JAX.  JAX is imported only on use:
when the JAX backend is built (:func:`domainia.wilson.backend.jax_backend` or
``domainia.wilson.backend.get_backend(backend="jax")``) to pass as the ``be=`` of a kernel,
map or covariance that takes one, when float64 is turned on
(:func:`domainia.wilson.backend.enable_x64`), or when JAX device constants are built.  The
JAX backend needs float64: call ``enable_x64()`` (or set ``JAX_ENABLE_X64=true``) first, or
it raises ``RuntimeError``.  A kernel that takes ``be=``, given JAX arrays and no ``be=``,
infers the JAX backend, with the same requirement.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`MEYER`); every kernel that samples the window takes
  ``window=`` (any admissible :class:`Window`: ``Meyer(d)``, ``RampWindow``,
  ``SampledWindow``).
* :func:`transform`: data arrays, with any leading axes, between time, frequency and WDM
  (:mod:`domainia._transform`).
* Not imported here: :mod:`domainia.testing` (dense oracles and test windows).
"""

from __future__ import annotations

from . import wilson
from ._transform import transform
from .wilson import MEYER, Grid, Meyer, RampWindow, SampledWindow, Window, WindowError

__version__ = "2.0.0"

__all__ = [
    "MEYER",
    "Grid",
    "Meyer",
    "RampWindow",
    "SampledWindow",
    "Window",
    "WindowError",
    "__version__",
    "transform",
    "wilson",
]
```

- The docstring's first line drops "linear maps". The `translate`, `domains` and `data` bullets go, and the "Not imported here" bullet names only `domainia.testing`.
- `__all__` loses `CellSet`, `ChannelAxis`, `FourierBins`, `FourierDomain`, `TimeDomain`, `TimeSeries`, `WDMCells`, `WDMDomain`, `data`, `domains` and `translate`.
- The window names, `Grid` and `MEYER` stay for now. Spec §5's final top-level API is `transform`, `frequency_covariance`, `wdm_covariance` and `wilson`.
- `testing` is not exported, as before. `__version__` stays `"2.0.0"` until Task 11. The JAX paragraph stays until Task 5.

- [ ] **Step 6: Delete the tests of the deleted modules**

Run: `cd /Users/aaron/Documents/lisa/domainia && git rm -q tests/test_translate.py tests/test_dom_maps_base.py tests/test_dom_maps_dirichlet.py tests/test_dom_maps_fourier_time.py tests/test_dom_maps_fourier_wdm.py tests/test_dom_maps_wdm.py tests/test_dom_map_dtypes.py tests/test_dom_built_for.py tests/test_wrong_kinds_maps.py tests/test_dom_data.py tests/test_dom_db_imports.py tests/dom_db_inputs.py tests/test_dom_domains.py tests/test_wrong_kinds_domains.py tests/test_dom_spectra.py tests/test_dom_variance_generic.py tests/test_dom_windows.py tests/test_wrong_kinds_misc.py && git status --short tests | wc -l`

Expected: `20`: the 18 deleted files, and the two files modified in Steps 1 and 3 (`tests/test_dom_testing.py` and `tests/wilson/meyer_cases.py`).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rln "dom_db_inputs\|DenseCov" tests`

Expected: no output. No kept test imports the deleted helper.

- [ ] **Step 7: Carry `test_translate.py`'s missing cases into `tests/test_transform.py`**

These edits test behaviour that `transform` already has, so they pass at once.

Edit 1, `tests/test_transform.py`. Old:

```python
    """The six pairs on four WDM grids, time <-> frequency at odd N, one channel with no
    leading axis, and WDM -> WDM across two grids: each output within 1e-12 of its peak
    (measured: 0.0) and exactly canonical."""
```

New:

```python
    """The six pairs on four WDM grids, time <-> frequency at odd N, one channel with no
    leading axis, and WDM -> WDM across two grids: each output within 1e-12 of its peak
    (measured: 0.0) and exactly canonical.  At odd N the last bin is not a Nyquist bin, and
    it keeps its imaginary part."""
```

Edit 2, `tests/test_transform.py`. Old:

```python
                assert not np.any(got[..., 0].imag) and (n % 2 or not np.any(got[..., -1].imag)), i
```

New:

```python
                assert not np.any(got[..., 0].imag), i
                assert np.all(got[..., -1].imag) if n % 2 else not np.any(got[..., -1].imag), i
```

Edit 3, `tests/test_transform.py`. Old:

```python
    each row is the kernel's; the energy scales by dt on WDM and dt^2 N on frequency."""
```

New:

```python
    each row is the kernel's; the energy scales by dt on WDM and dt^2 N on frequency, also at
    odd N."""
```

Edit 4, `tests/test_transform.py`. Old:

```python
    np.testing.assert_allclose(np.sum(two * np.abs(on["frequency"]) ** 2, axis=-1),
                               DT ** 2 * N * energy, rtol=1e-12)
```

New:

```python
    np.testing.assert_allclose(np.sum(two * np.abs(on["frequency"]) ** 2, axis=-1),
                               DT ** 2 * N * energy, rtol=1e-12)
    odd = transform(X[..., :15], "time", "frequency", N=15, dt=DT)             # no Nyquist bin
    np.testing.assert_allclose(np.sum(two[:8] * np.abs(odd) ** 2, axis=-1),
                               DT ** 2 * 15 * np.sum(X[..., :15] ** 2, axis=-1), rtol=1e-12)
```

Edit 5, `tests/test_transform.py`. Old:

```python
    real = np.arange(65.0)
```

New:

```python
    real = np.arange(65)                                                         # integers too
```

- `two[:8]` is the rfft weight `[1, 2, …, 2]` of an odd record of 15 samples: DC once, the 7 other bins twice, no Nyquist bin.
- The odd-N cases of the fixture are 24-27 (N = 45 and N = 15). The new assert holds on cases 24 and 26, the time → frequency ones.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_transform.py 2>&1 | tail -1`

Expected: `5 passed in …`

- [ ] **Step 8: `tests/test_package.py`**

Edit 1, `tests/test_package.py`. Old:

```python
from domainia import translate, wilson
```

New:

```python
from domainia import wilson
```

Edit 2, `tests/test_package.py`. Old:

```python
PUBLIC = {
    "CellSet", "ChannelAxis", "FourierBins", "FourierDomain", "Grid", "MEYER", "Meyer", "RampWindow",
    "SampledWindow", "TimeDomain", "TimeSeries", "WDMCells", "WDMDomain", "Window", "WindowError",
    "__version__", "data", "domains", "transform", "translate", "wilson",
}
```

New:

```python
PUBLIC = {"Grid", "MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError",
          "__version__", "transform", "wilson"}
```

Edit 3, `tests/test_package.py`. Delete these lines:

```python
    assert all(hasattr(translate, name) for name in translate.__all__)
```

Edit 4, `tests/test_package.py`. Old:

```python
    assert {"domainia.translate", "domainia.wilson", "domainia.wilson.backend", "domainia.maps.time",
            "domainia.testing"} <= set(info["modules"])
```

New:

```python
    assert {"domainia._transform", "domainia.wilson", "domainia.wilson.backend",
            "domainia.testing"} <= set(info["modules"])
```

Edit 5, `tests/test_package.py`. Old:

```python
    """The package docstring's promise: a fresh ``import domainia`` imports none of
    domainia.maps, domainia.variance, domainia.spectra and domainia.testing."""
```

New:

```python
    """The package docstring's promise: a fresh ``import domainia`` does not import
    domainia.testing."""
```

Edit 6, `tests/test_package.py`. Old:

```python
    assert "domainia.translate" in loaded
    assert not {m.split(".")[1] for m in loaded} & {"maps", "spectra", "testing", "variance"}
```

New:

```python
    assert "domainia._transform" in loaded and "domainia.testing" not in loaded
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -1`

Expected: `7 passed in …`

- [ ] **Step 9: `tests/wilson/test_window_threading.py`**

Edit 1, `tests/wilson/test_window_threading.py`. Old:

```python
           "domains.py", "maps/base.py", "maps/time.py", "maps/fourier.py", "variance.py",
           "testing.py", "data.py", "spectra.py", "__init__.py", "maps/__init__.py",
           "translate.py", "_transform.py")
```

New:

```python
           "testing.py", "__init__.py", "_transform.py")
```

Edit 2, `tests/wilson/test_window_threading.py`. Old:

```python
    "dense_W", "wdm_congruence", "eq45_dense", "_dense_atoms",
    # The window-taking constructors must receive ``window=`` by keyword: building one without
    # it drops the window.  Known limits of this name-based rule: a positional window
    # (``WDMDomain(cells, window)``) is flagged, a conservative false positive; a
    # ``cls(...)`` inside a classmethod and a call to such a classmethod (``WDMDomain.full(...)``)
    # are not seen, a false negative that the value tests in tests/test_dom_windows.py and
    # tests/test_translate.py cover.
    "WDMDomain", "BandAveragedVariance",
    # The helper those constructors resolve the window with; MidpointVariance uses _cells_of.
    "_cells_and_window",
})
```

New:

```python
    "dense_W", "wdm_congruence", "eq45_dense",
})
```

Edit 3, `tests/wilson/test_window_threading.py`. Old:

```python
    """A new module anywhere in the package (``domainia.translate`` too, once it exists) is
    scanned unless it is added to UNSCANNED on purpose."""
```

New:

```python
    """A new module anywhere in the package is scanned unless it is added to UNSCANNED on
    purpose."""
```

Edit 4, `tests/wilson/test_window_threading.py`. Delete these lines:

```python
    # The window-taking constructors of the domainia layer.
    ("WDMDomain(cells, window=window)", False),
    ("D.WDMDomain(cells, window=self.dst.window)", False),
    ("BandAveragedVariance(psd, cells, window=window)", False),
    ("WDMDomain(cells)", True),
    ("domains.WDMDomain(cells, ALT_WINDOW)", True),           # positional: flagged
    ("BandAveragedVariance(psd, cells)", True),
    ("WDMDomain(cells, window=wilson.MEYER)", True),
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py 2>&1 | tail -1`

Expected: `486 passed in …`. It was 502: 9 fewer `test_no_kernel_call_drops_the_window` items and 7 fewer scanner rows.

- [ ] **Step 10: `tests/wilson/test_checks.py`**

`domains._int` was the seventh copy of the integer check; six remain.

Edit 1, `tests/wilson/test_checks.py`. Old:

```python
from domainia import domains, wilson
```

New:

```python
from domainia import wilson
```

Edit 2, `tests/wilson/test_checks.py`. Old:

```python
    (lambda: wilson.Meyer(2.5), "Meyer d must be an integer, got float"),
    (lambda: domains.TimeDomain(8.0, 1.0), "N must be an integer, got float"),
], ids=["_require_int", "validate_grid", "n_samples", "nu_d", "colored_noise", "Meyer", "domains._int"])
def test_the_seven_integer_checks_raise_the_helpers_message(call, message):
    """The seven copies of the integer check call ``require_int``."""
```

New:

```python
    (lambda: wilson.Meyer(2.5), "Meyer d must be an integer, got float"),
], ids=["_require_int", "validate_grid", "n_samples", "nu_d", "colored_noise", "Meyer"])
def test_the_six_integer_checks_raise_the_helpers_message(call, message):
    """The six copies of the integer check call ``require_int``."""
```

Edit 3, `tests/wilson/test_checks.py`. Delete these lines:

```python
    assert type(domains._int("k0", np.int64(2), minimum=0)) is int
    with pytest.raises(ValueError, match=_exact("k0 must be >= 0; got -1")):
        domains._int("k0", -1, minimum=0)
    with pytest.raises(TypeError, match=_exact("k0 must be an integer, got bool")):
        domains._int("k0", np.True_, minimum=0)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_checks.py 2>&1 | tail -1`

Expected: `102 passed in …` (it was 103).

- [ ] **Step 11: Kept docstrings that name deleted code**

Edit 1, `src/domainia/wilson/_checks.py`. Old:

```python
    (``"a TimeDomain"``, ``"a CellSet or a WDMDomain"``).
```

New:

```python
    (``"a Window"``, ``"a string"``).
```

Edit 2, `src/domainia/wilson/evolutionary.py`. Old:

```python
    spectrum on the window bins.  ``K^X`` is in the units of ``U`` (Wilson units for the
    ``C = kappa U`` of ``domainia.maps.FourierToWDM``).  Ported verbatim (host NumPy,
    built once per noise component).
```

New:

```python
    spectrum on the window bins.  ``K^X`` is in the units of ``U``.  Ported verbatim (host
    NumPy, built once per noise component).
```

Edit 3, `src/domainia/wilson/transform.py`. Delete these lines:

```python
* **The maps of ``domainia.maps`` compute in double.** ``TimeToWDM``,
  ``WDMToTime``, ``TimeToFourier`` and ``FourierToWDM(impl="fft")`` convert their
  input to float64 (bins to complex128) before they call these kernels, so a
  ``float32`` record or ``complex64`` bins give float64 results there.  Single
  precision is reached only by calling the kernels here directly.
```

- The `transform.py` bullet is part of the float32 section of the module docstring, which Task 6 deletes. It goes now so that Step 13's grep holds.

- [ ] **Step 12: `README.md` and `pyproject.toml`**

The README's two python blocks call the 2.0 `transform` with `TimeDomain`, `FourierDomain` and `WDMDomain`. `tests/test_readme.py` runs them, so they move to the array `transform`.
- Block 1 (Quick start) does the same walk on a `(2, N)` stack. `transform` has one window for both sides, so the change of window goes through frequency.
- Block 2 (Canonical inputs) showed the exact-zero refusal, which `transform` no longer has. It now shows raw kernel output and a full-FFT slice accepted on the 6x12 grid at `dt = 0.3`. There they carry round-off, measured at 4.2e-17 of the peak in the inactive cells and 2.4e-17 at Nyquist (DC is exactly real), in PY and FLOOR.
- The paragraph before block 2 described the exact-zero rule, and it is rewritten for the 1e-10 rule. The rest of the README's prose (the intro bullets, the layout paragraph after block 1, Units and Errors) still names the deleted modules. Task 11 rewrites the README.
- The `pyproject.toml` comment named `dom_db_inputs.py`.

Edit 1, `README.md`. Old:

```python
import numpy as np
from domainia import FourierDomain, Grid, Meyer, TimeDomain, WDMDomain
from domainia.translate import transform

Nf, Nt, dt = 16, 64, 0.5
N = Nf * Nt
time = TimeDomain(N, dt)
freq = FourierDomain.full(N, dt)                        # dt * numpy.fft.rfft(x)
wdm = WDMDomain.full(Grid(Nf, Nt, dt))                  # dense (Nf + 1, Nt) layout, Meyer window
wdm8 = WDMDomain.full(Grid(Nf, Nt, dt), window=Meyer(8))

x = {"A": np.random.default_rng(0).standard_normal(N)}
w = transform(x, time, wdm)                             # {"A": (Nf + 1, Nt) float64}
X = transform(w, wdm, freq)                             # {"A": (N // 2 + 1,) complex128}
w8 = transform(w, wdm, wdm8)                            # another window, through the spectrum
assert np.allclose(transform(w8, wdm8, time)["A"], x["A"])
```

New:

```python
import numpy as np
from domainia import transform, wilson

Nf, Nt, dt = 16, 64, 0.5
N = Nf * Nt
rec = {"N": N, "dt": dt}
meyer8 = wilson.Meyer(8)

x = np.random.default_rng(0).standard_normal((2, N))         # two channels: any leading axes
w = transform(x, "time", "wdm", **rec, target_wdm=(Nf, Nt))       # (2, Nf + 1, Nt) float64, Meyer
X = transform(w, "wdm", "frequency", **rec, source_wdm=(Nf, Nt))  # (2, N // 2 + 1): dt * rfft(x)
w8 = transform(X, "frequency", "wdm", **rec, target_wdm=(Nf, Nt), window=meyer8)  # another window
assert np.allclose(transform(w8, "wdm", "time", **rec, source_wdm=(Nf, Nt), window=meyer8), x)
```

Edit 2, `README.md`. Old:

```markdown
`transform` takes its inputs exactly, with no tolerance: inactive WDM cells must be exactly
0, and the Fourier DC bin (and the Nyquist bin for even `N`) must be exactly real. The
canonical producers are `transform` itself and `dt * numpy.fft.rfft(x)`. Raw
`wilson.forward_time` output, and a slice of a full FFT (`dt * numpy.fft.fft(x)[:N // 2 + 1]`),
carry ~1e-16 round-off there on many grids and are then refused; the error names the
fix:
```

New:

```markdown
`transform`'s outputs are exactly canonical: inactive WDM cells are 0, and the Fourier DC
bin (and the Nyquist bin for even `N`) is real. Its inputs may carry a residue there of up
to 1e-10 of their peak, so raw `wilson.forward_time` output and a slice of a full FFT
(`dt * numpy.fft.fft(x)[:N // 2 + 1]`), which carry ~1e-16 round-off there on many grids,
are taken as they are. A larger residue is refused, and the error names the cell or bin:
```

Edit 3, `README.md`. Old:

```python
import numpy as np
from domainia import FourierDomain, Grid, TimeDomain, WDMDomain, wilson
from domainia.translate import transform

Nf, Nt, dt = 16, 64, 0.5
N = Nf * Nt
time, freq, wdm = TimeDomain(N, dt), FourierDomain.full(N, dt), WDMDomain.full(Grid(Nf, Nt, dt))
x = np.random.default_rng(0).standard_normal(N)

w = wilson.forward_time(x, Nf, Nt, dt)        # raw kernel output: ~1e-16 in the inactive cells
# transform({"A": w}, wdm, time) raises ValueError: ... inactive cells must be exactly 0 ...
w[~wilson.active_mask(Nf, Nt)] = 0.0          # zero the inactive cells
assert np.allclose(transform({"A": w}, wdm, time)["A"], x)

# A full-FFT slice: its DC and Nyquist bins can carry ~1e-16 imaginary round-off (not at
# this power-of-two N = 1024, but typically at N = 240), so make them exactly real.
X = dt * np.fft.fft(x)[: N // 2 + 1]
X[[0, -1]] = X[[0, -1]].real                  # odd N: X[0] only
assert np.allclose(transform({"A": X}, freq, time)["A"], x)
```

New:

```python
import numpy as np
from domainia import transform, wilson

Nf, Nt, dt = 6, 12, 0.3
N = Nf * Nt
x = np.random.default_rng(6).standard_normal(N)

w = wilson.forward_time(x, Nf, Nt, dt)        # raw kernel output: ~1e-17 in the inactive cells
assert np.allclose(transform(w, "wdm", "time", N=N, dt=dt, source_wdm=(Nf, Nt)), x)
X = dt * np.fft.fft(x)[: N // 2 + 1]          # a full-FFT slice: ~1e-17 imaginary at Nyquist
assert np.allclose(transform(X, "frequency", "time", N=N, dt=dt), x)
```

Edit 4, `pyproject.toml`. Old:

```toml
# One config for the whole repo.  tests/ has no __init__.py (its modules import the
# sibling helpers dom_db_inputs.py by name); tests/wilson is a package whose conftest
# puts tests/wilson/oracle on sys.path.
```

New:

```toml
# One config for the whole repo.  tests/ has no __init__.py; tests/wilson is a package
# whose conftest puts tests/wilson/oracle on sys.path.
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_readme.py tests/test_package.py 2>&1 | tail -1`

Expected: `9 passed in …`

- [ ] **Step 13: Grep gate**

The pattern looks for the deleted class names, and for the deleted modules as `domainia.X`, as relative or `from domainia import` imports, as `X.py` paths, and as `maps/`. The modules' names are also English words ("variance grid", "maps", "spectra", "data"), which stay in kept docstrings, so the module names are matched only in those forms.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && grep -rnE "\b(CellSet|TimeDomain|FourierDomain|WDMDomain|ChannelAxis|GenericPSD|TabulatedSpectrum|BandAveragedVariance|MidpointVariance|TimeSeries|FourierBins|WDMCells|DenseCov|dom_db_inputs)\b|domainia\.(translate|maps|data|domains|spectra|variance)\b|^\s*from \.+(translate|maps|data|domains|spectra|variance)\b|^\s*from (domainia|\.+) import .*\b(translate|maps|data|domains|spectra|variance)\b|\b(translate|data|domains|spectra|variance)\.py\b|\bmaps/" src tests pyproject.toml
```

Expected: exactly these two lines, the recorded replay key `BandAveragedVariance`, which keeps its name:
```
tests/wilson/meyer_cases.py:171:    ``eq45_direct``, ``BandAveragedVariance`` by ``noise.evolutionary_band_variance`` on the
tests/wilson/meyer_cases.py:189:        ("BandAveragedVariance", lambda kw: wl.noise.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)[m, n]),
```

Other allowed mentions, which the pattern does not match:
- `tests/data/translate_reference.npz` and `test_agrees_with_the_recorded_translate_transform` in `tests/test_transform.py`: the Task 1 fixture of `translate.transform`'s outputs.
- The `DROPPED` comment "maps (54 keys)" in `meyer_cases.py`, the comment "the recorded maps cases' inputs" there, and "maps 54" in `test_meyer_baseline.py`: the replay's bookkeeping of the dropped families.
- The English words in kept docstrings.

The same grep on the tree after Task 3 lists 37 files.

- [ ] **Step 14: Gates**

The three suite runs are independent and can run in parallel.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1380 passed, 3 skipped`. It was `2331 passed, 3 skipped`: 936 items in the deleted files, 16 in `test_window_threading.py` and 1 in `test_checks.py` go, and `test_dom_testing.py` gains 2.

Run: `cd /Users/aaron/Documents/lisa/domainia && WDM_TESTS_HIDE_JAX=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1293 passed, 90 skipped`. It was `2091 passed, 243 skipped`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1293 passed, 90 skipped`.

None of the three logs contains `warning` or `error` (`grep -ciE "warning|error"` prints `0`).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. `tests/wilson/test_meyer_baseline.py` passes inside all three full runs.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 6560 lines / 2605 code
tests 5769 lines (+ oracle 4135)
```
That is −2171 src lines (−1336 code) and −3037 test lines. For scale, the spec's final budget is about 5,400 src lines (2,150 code) and 5,700 non-oracle test lines. Tasks 5–7 and 10–12 still delete the JAX backend, float32, the `_many` and band entry points, Eq. 39, dead code and docstring text, and Tasks 8–9 add the covariance code and its tests.

- [ ] **Step 15: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/testing.py src/domainia/__init__.py src/domainia/wilson/_checks.py src/domainia/wilson/evolutionary.py src/domainia/wilson/transform.py tests/test_dom_testing.py tests/test_transform.py tests/test_package.py tests/wilson/meyer_cases.py tests/wilson/test_window_threading.py tests/wilson/test_checks.py README.md pyproject.toml && git status --short | grep -vc "^[MD] "`

Expected: `0`. Every remaining status line is a staged change (the `git rm` deletions of Steps 4 and 6 are already staged), so nothing is unstaged or untracked.

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Delete translate, maps, data, domains, spectra and variance; the testing oracles take a cell mask" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `40 files changed, 148 insertions(+), 5369 deletions(-)`

Size after Task 4 (prototype): src 6560 lines / 2605 code; tests 5769 (+ oracle 4135).

---

### Task 5: Delete the JAX backend and `special.py`; every `be.` call becomes NumPy (spec Order step 3; §4; §5)

**Goal:** `src/domainia/wilson/backend.py` and `src/domainia/wilson/special.py` are gone. Every kernel calls NumPy directly, and no signature in src takes `be`. `evolutionary.py` loses `@pytree_dataclass`, and `eq45_coeffs_bessel` calls `scipy.special.ive` itself. The JAX test lanes and their fixtures go, and so do the `[jax]` extra and the `jax` and `slow` markers. Every float64 output is bit for bit what it was: the replay reproduces its 1408 arrays exactly.

Assumes the tree after Task 4, whose commit is "Delete translate, maps, data, domains, spectra and variance; the testing oracles take a cell mask".

**Deviation from the spec's Order, already decided by the plan.** Spec step 3 also deletes float32. Task 6 does that. Here the float32 machinery in `transform.py` keeps its behaviour, with NumPy calls only: `_working_dtypes`, the casts, `_check_single_output`, `_set_rounded` and `_add_rounded`. The time-form imports of `evolutionary.py` stay until Task 7.

**Files:**
- Delete, source: `src/domainia/wilson/backend.py` (582 lines) and `src/domainia/wilson/special.py` (49 lines; 1 doctest item).
- Delete, tests: `tests/wilson/test_backend.py` (372 lines, 25 items) and `tests/wilson/test_lanes.py` (26 lines, 1 item: the `WDM_TESTS_HIDE_JAX=1` lane runner).
- Modify, source:
  - `src/domainia/wilson/transform.py` (Step 2);
  - `src/domainia/wilson/evolutionary.py` (Step 3);
  - `src/domainia/wilson/stationary.py`, `grid.py`, `_checks.py` and `windows.py` (Step 4);
  - `src/domainia/wilson/__init__.py` and `src/domainia/__init__.py` (Step 5).
- Modify, packaging and text: `pyproject.toml` (Step 5) and `README.md` (Step 15).
- Modify, tests:
  - `tests/test_package.py` (Step 1, test first);
  - `tests/conftest.py` and `tests/wilson/conftest.py`, rewritten whole (Step 6);
  - `tests/wilson/test_wrong_kinds.py` (Steps 4 and 13);
  - `tests/wilson/test_single_precision.py`, `test_transform_shim.py`, `test_noise_stationary.py`, `test_grid_bands.py`, `test_checks.py`, `test_evolutionary_kernels.py` and `test_window_threading.py` (Steps 7–14).
- Test: `tests/test_package.py` (7 tests) and `tests/wilson/test_meyer_baseline.py` (the replay, 5 items).

**What each `be` call becomes.** The NumPy backend's methods were one-line wrappers, so each call is replaced by the statement the wrapper ran:

| Backend call | NumPy | Sites |
|---|---|---|
| `be.set(a, idx, v, copy=False)` | `a[idx] = v` | `_build_slab`, `_set_rounded`, `_add_rounded`, `unpack`, `eq45_coeffs_bessel`, `eq45_blocks_cheb` |
| `be.add_at(a, idx, v, copy=False)` | `np.add.at(a, idx, v)`: both sites pass an index array, for which the wrapper called `np.add.at` | `inverse_bands` (unique indices `ell[b]`); `eq45_blocks_cheb` (the fold `k % Nt` can repeat, so `np.add.at` is required there) |
| `be.fft.*`, `be.xp.*`, `be.asarray` | `np.fft.*`, `np.*`, `np.asarray` | every kernel |
| `be.bessel_i(P, a)` | `ive(np.arange(P)[:, None], a[None, :]) * np.exp(np.abs(a))[None, :]` with `a = np.asarray(a)` | `eq45_coeffs_bessel` |
| `get_backend(...)`, `be or get_backend(...)` | removed | every entry point |
| `is_tracer(x)` branches | the concrete branch only | `_check_finite` (its calls become `validate_finite`), `_check_single_output`, `apply_lambda_from_psd` |

`_set_rounded` and `_add_rounded` now update `a` in place and return it, as the NumPy backend's `copy=False` did.

**Which tests go.** A test goes when its subject is deleted code. A test that also checks kept code keeps its NumPy part.
- `test_backend.py` and `test_lanes.py` go whole. So do `special.py`'s doctest, every test that takes `jax_be`, and the four `get_backend`/`cholesky` choice tests in `test_wrong_kinds.py`, whose subject is `Backend`.
- A test parametrised by the `backend` fixture keeps its NumPy leg and drops the parameter: 4 tests in `test_grid_bands.py` and `test_separable_kernels_and_assembly` in `test_evolutionary_kernels.py`.
- `test_special` becomes `test_bessel_coefficients_are_the_chebyshev_series_of_the_log_linear_spectrum` (Step 12). It builds the Chebyshev basis itself, so `special.chebyshev_basis` has no user left. The test checks `sum_p c_p T_p(u) = exp(u L)` at `|a| <= 3.2`. The prototype measures 5.0e-16 of the peak against a 1e-13 gate; dropping the doubling of `c_p` for `p >= 1` gives 0.33.
- `test_apply_lambda_from_psd_inf_psd_under_jit` keeps its NumPy check as `test_apply_lambda_from_psd_inf_psd_is_zero_power` (Step 9): an `inf` PSD sample gives the same result as `0.0`.
- `test_apply_lambda_from_psd_is_the_dense_covariance` loses its `jax` and `jax-jit` lanes; its NumPy lane over `WINDOWS` x `GRIDS` stays (Step 14).
- `test_inverse_bands_accumulates_in_single_precision` goes (Step 7). Its hook, `Backend.add_at`, is deleted, and the spec deletes float32 (Task 6) and `inverse_bands` (Task 7).
- The no-JAX-import check is the existing `test_importing_every_module_resolves_to_src_and_does_not_import_jax` in `tests/test_package.py`. It imports every module in a fresh interpreter and asserts that no `jax` or `jaxlib` module was imported. Step 1 adds that the walk finds neither deleted module.

From this task on there is no NumPy lane: PY and FLOOR run the same items.

**Acceptance Criteria:**
- [ ] Step 1's run gives `2 failed, 5 passed`: `test_the_project_metadata` and `test_importing_every_module_resolves_to_src_and_does_not_import_jax`.
- [ ] `pytest -q tests/test_package.py` → `7 passed` in PY and FLOOR.
- [ ] `src/domainia/wilson/backend.py` and `src/domainia/wilson/special.py` do not exist, and `domainia.wilson.__all__` names neither.
- [ ] Step 16's grep prints exactly the 3 `tests/test_package.py` lines listed there, and its second grep prints nothing.
- [ ] `python tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`, and `tests/wilson/test_meyer_baseline.py` passes in PY and FLOOR.
- [ ] Full suite (prototype): PY `1267 passed, 3 skipped`; FLOOR `1267 passed, 3 skipped`. No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 5796 lines / 2174 code` and `tests 4760 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1` → `1267 passed, 3 skipped in …`

**Steps:**

Edits are anchored to code text, never to line numbers. Each Old text occurs exactly once in its file, and New replaces it. "Delete these lines" means the shown lines go, newline included.

**Deleting a named function** (Steps 7–14) follows one rule:
- Delete from its first decorator line, or its `def` line if it has none, through its last line, with the two blank lines after it.
- If it is the last thing in the file, delete the two blank lines before it instead, so the file still ends in one newline.
- A comment block directly above that first line, with no blank line between, goes with it. Only `_F32_NO_WARNINGS` (Step 8), a module-level string, has one.
- Deleting the listed functions in any order gives the same file, and so does doing a step's Old/New edits before or after them.

The suite does not collect cleanly from Step 5 until Step 12 is done: in between, `test_single_precision.py` (and `test_transform_shim.py`, which imports it) imports `domainia.wilson.backend`, and `test_evolutionary_kernels.py` imports `domainia.wilson.special`. Between Steps 2 and 14, tests that pass `be=` or take the `backend` or `jax_be` fixture fail; they change in Steps 7–14.

Steps 5, 6 and 18 are git writes in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: `tests/test_package.py` (the test comes first)**

Edit 1, `tests/test_package.py`. Old:

```python
def test_the_project_metadata():
    """The package ships ``py.typed`` and says so; the ``jax`` extra names ``jax`` alone
    (``jax`` 0.9 requires its matching ``jaxlib`` itself)."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert "Typing :: Typed" in project["classifiers"]
    assert (SRC / "domainia" / "py.typed").is_file()
    assert project["optional-dependencies"]["jax"] == ["jax>=0.9"]
```

New:

```python
def test_the_project_metadata():
    """The package ships ``py.typed`` and says so; ``test`` is the only extra."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert "Typing :: Typed" in project["classifiers"]
    assert (SRC / "domainia" / "py.typed").is_file()
    assert project["optional-dependencies"] == {"test": ["pytest>=8"]}
```

Edit 2, `tests/test_package.py`. Old:

```python
    """Every module of the package (``pkgutil.walk_packages``), imported in a fresh
    interpreter, leaves JAX unimported."""
```

New:

```python
    """Every module of the package (``pkgutil.walk_packages``), imported in a fresh
    interpreter, leaves JAX unimported.  The JAX backend and its Bessel shim are gone."""
```

Edit 3, `tests/test_package.py`. Old:

```python
    assert {"domainia._transform", "domainia.wilson", "domainia.wilson.backend",
            "domainia.testing"} <= set(info["modules"])
```

New:

```python
    assert {"domainia._transform", "domainia.wilson", "domainia.testing"} <= set(info["modules"])
    assert not {"domainia.wilson.backend", "domainia.wilson.special"} & set(info["modules"])
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -1`

Expected: `2 failed, 5 passed in …`. `test_the_project_metadata` fails because `pyproject.toml` still has the `jax` extra, and the import test because `domainia.wilson` still imports `domainia.wilson.backend` and `domainia.wilson.special`. `… 2>&1 | grep -E "^FAILED" | cut -d" " -f2` shows:
```
tests/test_package.py::test_the_project_metadata
tests/test_package.py::test_importing_every_module_resolves_to_src_and_does_not_import_jax
```

- [ ] **Step 2: `src/domainia/wilson/transform.py`**

The entry points lose `be`, the helpers lose their `be` parameter, and `_check_finite`, whose only job was to skip tracers, goes: its eight calls become `validate_finite`, which `transform.py` already imports. Each entry point's signature edit is anchored by the first line of its docstring.

Edit 1, `src/domainia/wilson/transform.py`. Old:

```python
working dtype first, so every FFT receives a single-precision array, on NumPy
and on JAX.  Anything else, integers included, promotes to double.  It is
```

New:

```python
working dtype first, so every FFT receives a single-precision array.  Anything
else, integers included, promotes to double.  It is
```

Edit 2, `src/domainia/wilson/transform.py`. Old:

```python
suggests float64 input.  The check is host work, like the input check: it is
skipped for double results and under ``jax.jit``, where a tracer has no values
to read and an overflow still comes back as ``inf``/``nan``.
```

New:

```python
suggests float64 input.  The check is skipped for double results.
```

Edit 3, `src/domainia/wilson/transform.py`. Delete these lines:

```python
from .backend import NUMPY, Backend, get_backend, is_tracer
```

Edit 4, `src/domainia/wilson/transform.py`. Old:

```python
def _check_finite(values: NDArray[np.generic], context: str) -> None:
    """:func:`domainia.wilson.validate_finite` on concrete input; skipped for JAX tracers.

    Validation is host work, not arithmetic: a tracer has no values to check,
    and a concrete JAX array is checked on the host like a NumPy one.
    """
    if not is_tracer(values):
        validate_finite(values, context)


def _check_single_output(out: NDArray[np.generic], entry: str) -> None:
```

New:

```python
def _check_single_output(out: NDArray[np.generic], entry: str) -> None:
```

Edit 5, `src/domainia/wilson/transform.py`. Old:

```python
    The entry points check their input first (:func:`_check_finite`), so a
    ``nan`` or ``inf`` here was made by the transform itself: a float32
    product, FFT or scaling overflowed.  NumPy only warns on the way and JAX
    says nothing, so the caller would get ``nan`` cells.  Each entry point
    checks the arrays it computes itself, so the message names the function
    whose array overflowed.

    Host work, as :func:`_check_finite` is: skipped for a double-precision
    result (float64 runs gain no pass) and for a JAX tracer.  It does not
    change ``np.errstate``.
    """
    if _working_dtypes(out)[0] != np.float32 or is_tracer(out):
```

New:

```python
    The entry points check their input first (:func:`domainia.wilson.validate_finite`),
    so a ``nan`` or ``inf`` here was made by the transform itself: a float32 product,
    FFT or scaling overflowed.  NumPy only warns on the way, so the caller would get
    ``nan`` cells.  Each entry point checks the arrays it computes itself, so the
    message names the function whose array overflowed.

    Skipped for a double-precision result (float64 runs gain no pass).  It does not
    change ``np.errstate``.
    """
    if _working_dtypes(out)[0] != np.float32:
```

Edit 6, `src/domainia/wilson/transform.py`. Old:

```python
def _set_rounded(a: Array, idx: object, v: Array, be: Backend) -> Array:
    """``a[idx] = v`` with ``v`` cast to ``a.dtype`` (``copy=False``).

    The transforms form ``v`` in the working dtype (their constants are cast
    first), so the cast is a no-op for them.  It stays as the guard that keeps
    the scatter dtype-safe on JAX: storing float64 values into a float32 array
    warns in jax 0.9 (``FutureWarning``: "scatter inputs have incompatible
    types"), and a later JAX will reject it.
    """
    return be.set(a, idx, be.asarray(v, dtype=a.dtype), copy=False)


def _add_rounded(a: Array, idx: object, v: Array, be: Backend) -> Array:
    """``a[idx] += v`` for a duplicate-free ``idx``, stored in ``a.dtype``.

    Gather, add, cast, set: the functional form of NumPy's in-place ``+=``,
    so float64 bits are the in-place statement's.  The inverse transforms hand
    it a ``v`` already in ``a``'s dtype, so a float32 run adds in single
    precision and the cast is a no-op; the cast keeps the scatter dtype-safe
    on JAX (see :func:`_set_rounded`).
    """
    return be.set(a, idx, be.asarray(a[idx] + v, dtype=a.dtype), copy=False)
```

New:

```python
def _set_rounded(a: Array, idx: object, v: Array) -> Array:
    """``a[idx] = v`` in place, with ``v`` cast to ``a.dtype``; returns ``a``.

    The transforms form ``v`` in the working dtype (their constants are cast
    first), so the cast is a no-op for them.
    """
    a[idx] = np.asarray(v, dtype=a.dtype)
    return a


def _add_rounded(a: Array, idx: object, v: Array) -> Array:
    """``a[idx] += v`` in place for a duplicate-free ``idx``, stored in ``a.dtype``; returns
    ``a``.

    Gather, add, cast, set, so float64 bits are the in-place statement's.  The inverse
    transforms hand it a ``v`` already in ``a``'s dtype, so a float32 run adds in single
    precision and the cast is a no-op.
    """
    a[idx] = np.asarray(a[idx] + v, dtype=a.dtype)
    return a
```

Edit 7, `src/domainia/wilson/transform.py`. Old:

```python
    X_rfft: CArray, m: int, Nf: int, Nt: int, cdtype: np.dtype = _C128,
    be: Backend = NUMPY,
) -> CArray:
```

New:

```python
    X_rfft: CArray, m: int, Nf: int, Nt: int, cdtype: np.dtype = _C128,
) -> CArray:
```

Edit 8, `src/domainia/wilson/transform.py`. Old:

```python
    # Every slot is written below, so zeros (not np.empty) costs nothing and is
    # the only allocation JAX offers.  Integer positions replace the boolean
    # masks: the same assignments, and concrete indices for JAX.
    slab = be.xp.zeros((*X_rfft.shape[:-1], Nt), dtype=cdtype)
    neg = np.flatnonzero(ell < 0)
    over = np.flatnonzero(ell > N // 2)
    direct = np.flatnonzero((ell >= 0) & (ell <= N // 2))
    slab = be.set(slab, (..., direct), X_rfft[..., ell[direct]], copy=False)
    if neg.size:
        slab = be.set(slab, (..., neg), be.xp.conj(X_rfft[..., -ell[neg]]), copy=False)
    if over.size:
        slab = be.set(slab, (..., over), be.xp.conj(X_rfft[..., N - ell[over]]), copy=False)
```

New:

```python
    # Every slot is written below.
    slab = np.zeros((*X_rfft.shape[:-1], Nt), dtype=cdtype)
    neg = np.flatnonzero(ell < 0)
    over = np.flatnonzero(ell > N // 2)
    direct = np.flatnonzero((ell >= 0) & (ell <= N // 2))
    slab[..., direct] = X_rfft[..., ell[direct]]
    if neg.size:
        slab[..., neg] = np.conj(X_rfft[..., -ell[neg]])
    if over.size:
        slab[..., over] = np.conj(X_rfft[..., N - ell[over]])
```

Edit 9, `src/domainia/wilson/transform.py`. Old:

```python
    wmn: Array, Nf: int, Nt: int, rdtype: np.dtype = _F64, be: Backend = NUMPY
) -> Array:
```

New:

```python
    wmn: Array, Nf: int, Nt: int, rdtype: np.dtype = _F64
) -> Array:
```

Edit 10, `src/domainia/wilson/transform.py`. Old:

```python
    work = wmn.astype(rdtype)          # NumPy: always a fresh copy, so copy=False below is safe
    mask = active_mask(Nf, Nt)
    root2 = rdtype.type(np.sqrt(2))   # mask[m] * root2 is then in the working dtype too
    # ``work[..., 0, :] *= mask[0] * root2`` as a functional update.
    work = _set_rounded(work, (..., 0, slice(None)), work[..., 0, :] * (mask[0] * root2), be)
    work = _set_rounded(work, (..., Nf, slice(None)), work[..., Nf, :] * (mask[Nf] * root2), be)
```

New:

```python
    work = wmn.astype(rdtype)          # always a fresh copy, so updating it in place is safe
    mask = active_mask(Nf, Nt)
    root2 = rdtype.type(np.sqrt(2))   # mask[m] * root2 is then in the working dtype too
    # ``work[..., 0, :] *= mask[0] * root2``.
    work = _set_rounded(work, (..., 0, slice(None)), work[..., 0, :] * (mask[0] * root2))
    work = _set_rounded(work, (..., Nf, slice(None)), work[..., Nf, :] * (mask[Nf] * root2))
```

Edit 11, `src/domainia/wilson/transform.py`. Old:

```python
    lam: float,
    be: Backend = NUMPY,
) -> CArray:
```

New:

```python
    lam: float,
) -> CArray:
```

Edit 12, `src/domainia/wilson/transform.py`. Old:

```python
    xp = be.xp
    scaled_int = wmn_work[..., 1:Nf, :] * sign_int
    # g = w' * C_mn * (-1)^{(m+1)n}: real where (m+n) even, imaginary otherwise.
    G = xp.where(even_int, scaled_int, 0.0) + 1j * xp.where(even_int, 0.0, scaled_int)
    W_int = be.fft.fft(G, axis=-1) * (lam * phi_c)
```

New:

```python
    scaled_int = wmn_work[..., 1:Nf, :] * sign_int
    # g = w' * C_mn * (-1)^{(m+1)n}: real where (m+n) even, imaginary otherwise.
    G = np.where(even_int, scaled_int, 0.0) + 1j * np.where(even_int, 0.0, scaled_int)
    W_int = np.fft.fft(G, axis=-1) * (lam * phi_c)
```

Edit 13, `src/domainia/wilson/transform.py`. Old:

```python
    # the order is immaterial on any backend.  Each slice is duplicate-free.
    for i, m_i in enumerate(m_int):
        start = int(m_i - 1) * (Nt // 2)
        X = _add_rounded(X, (..., slice(start, start + Nt)), W_int[..., i, :], be)
```

New:

```python
    # the order is immaterial.  Each slice is duplicate-free.
    for i, m_i in enumerate(m_int):
        start = int(m_i - 1) * (Nt // 2)
        X = _add_rounded(X, (..., slice(start, start + Nt)), W_int[..., i, :])
```

Edit 14, `src/domainia/wilson/transform.py`. Old:

```python
    X: CArray, wmn_work: Array, Nf: int, Nt: int, N: int, phi_c: Array, lam: float,
    be: Backend = NUMPY,
) -> CArray:
```

New:

```python
    X: CArray, wmn_work: Array, Nf: int, Nt: int, N: int, phi_c: Array, lam: float,
) -> CArray:
```

Edit 15, `src/domainia/wilson/transform.py`. Old:

```python
    xp = be.xp
    n_arr = np.arange(Nt)
    for m in (0, Nf):
        even, sign = _parity(m, n_arr)
        scaled = wmn_work[..., m, :] * sign.astype(wmn_work.dtype)
        g = xp.where(even, scaled, 0.0) + 1j * xp.where(even, 0.0, scaled)
        Wm = be.fft.fft(g, axis=-1)
```

New:

```python
    n_arr = np.arange(Nt)
    for m in (0, Nf):
        even, sign = _parity(m, n_arr)
        scaled = wmn_work[..., m, :] * sign.astype(wmn_work.dtype)
        g = np.where(even, scaled, 0.0) + 1j * np.where(even, 0.0, scaled)
        Wm = np.fft.fft(g, axis=-1)
```

Edit 16, `src/domainia/wilson/transform.py`. Old:

```python
        X = _add_rounded(X, (..., ell[valid]), lam * phi_c[valid] * Wm[..., valid], be)
```

New:

```python
        X = _add_rounded(X, (..., ell[valid]), lam * phi_c[valid] * Wm[..., valid])
```

Edit 17, `src/domainia/wilson/transform.py`. Old:

```python
def _gather_bands(x_rfft: CArray, Nf: int, Nt: int, dt: float, be: Backend = NUMPY, *,
                  window: Window = MEYER) -> Array:
```

New:

```python
def _gather_bands(x_rfft: CArray, Nf: int, Nt: int, dt: float, *,
                  window: Window = MEYER) -> Array:
```

Edit 18, `src/domainia/wilson/transform.py`. Old:

```python
    xp = be.xp
    rdtype, cdtype = _working_dtypes(x_rfft)
```

New:

```python
    rdtype, cdtype = _working_dtypes(x_rfft)
```

Edit 19, `src/domainia/wilson/transform.py`. Old:

```python
    out: Array = xp.zeros((*x_rfft.shape[:-1], Nf + 1, Nt), dtype=rdtype)
```

New:

```python
    out: Array = np.zeros((*x_rfft.shape[:-1], Nf + 1, Nt), dtype=rdtype)
```

Edit 20, `src/domainia/wilson/transform.py`. Old:

```python
    Y = be.fft.ifft(x_rfft[..., ell] * phi_c, axis=-1)
```

New:

```python
    Y = np.fft.ifft(x_rfft[..., ell] * phi_c, axis=-1)
```

Edit 21, `src/domainia/wilson/transform.py`. Old:

```python
    re_or_im_int = xp.where(even_int, Y.real, Y.imag)
```

New:

```python
    re_or_im_int = np.where(even_int, Y.real, Y.imag)
```

Edit 22, `src/domainia/wilson/transform.py`. Old:

```python
    out = _set_rounded(out, (..., slice(1, Nf), slice(None)), kappa * sign_int * re_or_im_int, be)

    for m in (0, Nf):
        y_m = be.fft.ifft(_build_slab(x_rfft, m, Nf, Nt, cdtype, be) * phi_c, axis=-1)
        even, sign = _parity(m, n_arr)
        out = _set_rounded(out, (..., m, slice(None)),
                           kappa * sign.astype(rdtype) * xp.where(even, y_m.real, y_m.imag), be)
```

New:

```python
    out = _set_rounded(out, (..., slice(1, Nf), slice(None)), kappa * sign_int * re_or_im_int)

    for m in (0, Nf):
        y_m = np.fft.ifft(_build_slab(x_rfft, m, Nf, Nt, cdtype) * phi_c, axis=-1)
        even, sign = _parity(m, n_arr)
        out = _set_rounded(out, (..., m, slice(None)),
                           kappa * sign.astype(rdtype) * np.where(even, y_m.real, y_m.imag))
```

Edit 23, `src/domainia/wilson/transform.py`. Old:

```python
    out = _set_rounded(out, (..., 0, slice(None)), out[..., 0, :] / root2, be)
    out = _set_rounded(out, (..., Nf, slice(None)), out[..., Nf, :] / root2, be)
```

New:

```python
    out = _set_rounded(out, (..., 0, slice(None)), out[..., 0, :] / root2)
    out = _set_rounded(out, (..., Nf, slice(None)), out[..., Nf, :] / root2)
```

Edit 24, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Forward WDM transform: $\\text{rfft}(x) \\to w_{mn}$.
```

New:

```python
    window: Window = MEYER,
) -> Array:
    """Forward WDM transform: $\\text{rfft}(x) \\to w_{mn}$.
```

Edit 25, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(x_rfft)
    x_rfft = be.asarray(x_rfft)
    if x_rfft.shape != (N // 2 + 1,):
```

New:

```python
    x_rfft = np.asarray(x_rfft)
    if x_rfft.shape != (N // 2 + 1,):
```

Edit 26, `src/domainia/wilson/transform.py`. Old:

```python
    _check_finite(x_rfft, "forward: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
```

New:

```python
    validate_finite(x_rfft, "forward: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, window=window)
```

Edit 27, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Forward-transform a stack of spectra
```

New:

```python
    window: Window = MEYER,
) -> Array:
    """Forward-transform a stack of spectra
```

Edit 28, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(x_rfft)
    x_rfft = be.asarray(x_rfft)
    if x_rfft.ndim != 2
```

New:

```python
    x_rfft = np.asarray(x_rfft)
    if x_rfft.ndim != 2
```

Edit 29, `src/domainia/wilson/transform.py`. Old:

```python
    _check_finite(x_rfft, "forward_many: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, be, window=window)
```

New:

```python
    validate_finite(x_rfft, "forward_many: x_rfft")
    out = _gather_bands(x_rfft, Nf, Nt, dt, window=window)
```

Edit 30, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: a stack of time series
```

New:

```python
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: a stack of time series
```

Edit 31, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(x)
    x = be.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.ndim != 2
```

New:

```python
    x = np.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.ndim != 2
```

Edit 32, `src/domainia/wilson/transform.py`. Old:

```python
    _check_finite(x, "forward_time_many: x")
    x_rfft = be.fft.rfft(x, axis=-1)
```

New:

```python
    validate_finite(x, "forward_time_many: x")
    x_rfft = np.fft.rfft(x, axis=-1)
```

Edit 33, `src/domainia/wilson/transform.py`. Old:

```python
    return forward_many(x_rfft, Nf, Nt, dt, be=be, window=window)
```

New:

```python
    return forward_many(x_rfft, Nf, Nt, dt, window=window)
```

Edit 34, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> CArray:
    """Inverse WDM transform
```

New:

```python
    window: Window = MEYER,
) -> CArray:
    """Inverse WDM transform
```

Edit 35, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(wmn)
    wmn = be.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.shape != (Nf + 1, Nt):
```

New:

```python
    wmn = np.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.shape != (Nf + 1, Nt):
```

Edit 36, `src/domainia/wilson/transform.py`. Old:

```python
    _check_finite(wmn, "inverse: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype, be)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = be.xp.zeros(N // 2 + 1, dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
```

New:

```python
    validate_finite(wmn, "inverse: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = np.zeros(N // 2 + 1, dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
```

Edit 37, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> CArray:
    """Inverse-transform a stack
```

New:

```python
    window: Window = MEYER,
) -> CArray:
    """Inverse-transform a stack
```

Edit 38, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(wmn)
    wmn = be.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.ndim != 3
```

New:

```python
    wmn = np.asarray(wmn)
    # A complex grid would have its imaginary part silently discarded.
    require_dtype("wmn", wmn, REAL_KINDS, "a real numeric array")
    if wmn.ndim != 3
```

Edit 39, `src/domainia/wilson/transform.py`. Old:

```python
    _check_finite(wmn, "inverse_many: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype, be)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = be.xp.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam, be)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam, be)
```

New:

```python
    validate_finite(wmn, "inverse_many: wmn")

    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = np.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
```

Edit 40, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: $(B, N_f+1, N_t) \\to$
```

New:

```python
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: $(B, N_f+1, N_t) \\to$
```

Edit 41, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(wmn)
    x = be.fft.irfft(inverse_many(wmn, Nf, Nt, dt, be=be, window=window), n=N, axis=-1)
```

New:

```python
    x = np.fft.irfft(inverse_many(wmn, Nf, Nt, dt, window=window), n=N, axis=-1)
```

Edit 42, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: time-domain
```

New:

```python
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: time-domain
```

Edit 43, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(x)
    x = be.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.shape != (expected,):
```

New:

```python
    x = np.asarray(x)
    expected = n_samples(Nf, Nt)
    if x.shape != (expected,):
```

Edit 44, `src/domainia/wilson/transform.py`. Old:

```python
    _check_finite(x, "forward_time: x")
    x_rfft = be.fft.rfft(x)
```

New:

```python
    validate_finite(x, "forward_time: x")
    x_rfft = np.fft.rfft(x)
```

Edit 45, `src/domainia/wilson/transform.py`. Old:

```python
    return forward(x_rfft, Nf, Nt, dt, be=be, window=window)
```

New:

```python
    return forward(x_rfft, Nf, Nt, dt, window=window)
```

Edit 46, `src/domainia/wilson/transform.py`. Old:

```python
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: $w_{mn} \\to$ time-domain
```

New:

```python
    window: Window = MEYER,
) -> Array:
    """Convenience wrapper: $w_{mn} \\to$ time-domain
```

Edit 47, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(wmn)
    x_rfft = inverse(wmn, Nf, Nt, dt, be=be, window=window)
    x = be.fft.irfft(x_rfft, n=N)
```

New:

```python
    x_rfft = inverse(wmn, Nf, Nt, dt, window=window)
    x = np.fft.irfft(x_rfft, n=N)
```

Edit 48, `src/domainia/wilson/transform.py`. Old:

```python
                  dt: float, *, be: Backend | None = None, window: Window = MEYER) -> Array:
```

New:

```python
                  dt: float, *, window: Window = MEYER) -> Array:
```

Edit 49, `src/domainia/wilson/transform.py`. Old:

```python
    dt : float
    be : Backend, optional
        Array backend (default: the backend of ``z``).

```

New:

```python
    dt : float

```

Edit 50, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(z)
    bands = _check_interior_bands(bands, Nf)
    z = be.asarray(z)
    ell = _band_windows(bands, k0, z.shape[-1], Nt)
    _check_finite(z, "forward_bands: z")
```

New:

```python
    bands = _check_interior_bands(bands, Nf)
    z = np.asarray(z)
    ell = _band_windows(bands, k0, z.shape[-1], Nt)
    validate_finite(z, "forward_bands: z")
```

Edit 51, `src/domainia/wilson/transform.py`. Old:

```python
    Y = be.fft.ifft(phi.astype(rdtype) * z[..., ell], axis=-1)
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    out = be.xp.real(phase * Y)
```

New:

```python
    Y = np.fft.ifft(phi.astype(rdtype) * z[..., ell], axis=-1)
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    out = np.real(phase * Y)
```

Edit 52, `src/domainia/wilson/transform.py`. Old:

```python
                  Nt: int, dt: float, *, be: Backend | None = None,
                  window: Window = MEYER) -> CArray:
```

New:

```python
                  Nt: int, dt: float, *, window: Window = MEYER) -> CArray:
```

Edit 53, `src/domainia/wilson/transform.py`. Old:

```python
    dt : float
    be : Backend, optional
        Array backend (default: the backend of ``w``).

```

New:

```python
    dt : float

```

Edit 54, `src/domainia/wilson/transform.py`. Old:

```python
    be = be or get_backend(w)
    bands = _check_interior_bands(bands, Nf)
    w = be.asarray(w)
```

New:

```python
    bands = _check_interior_bands(bands, Nf)
    w = np.asarray(w)
```

Edit 55, `src/domainia/wilson/transform.py`. Old:

```python
    _check_finite(w, "inverse_bands: w")
```

New:

```python
    validate_finite(w, "inverse_bands: w")
```

Edit 56, `src/domainia/wilson/transform.py`. Old:

```python
    V = phi.astype(rdtype) * be.fft.fft(np.conj(phase).astype(cdtype) * w, axis=-1) / Nt
    out = be.xp.zeros((*w.shape[:-2], n_bins), dtype=cdtype)
    for b in range(bands.size):
        out = be.add_at(out, (..., ell[b]), V[..., b, :], copy=False)
```

New:

```python
    V = phi.astype(rdtype) * np.fft.fft(np.conj(phase).astype(cdtype) * w, axis=-1) / Nt
    out = np.zeros((*w.shape[:-2], n_bins), dtype=cdtype)
    for b in range(bands.size):
        np.add.at(out, (..., ell[b]), V[..., b, :])
```

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -cE "\bbe[.=,):]|\bxp\b|_check_finite|is_tracer|Backend|get_backend" src/domainia/wilson/transform.py`

Expected: `0`. The replay runs after Step 4: until then `stationary.py` still passes `be=` to the transform.

- [ ] **Step 3: `src/domainia/wilson/evolutionary.py` (spec §5)**

The three `@pytree_dataclass` decorators go with the JAX paragraphs of their docstrings; `Eq45BlockSpec`, `Eq45Plan` and `Eq45GramPlan` stay `@dataclass(frozen=True)`. `eq45_coeffs_bessel` computes `I_p(a)` as `ive(p, a) * exp(|a|)`, the statement the NumPy backend's `bessel_i` ran, with the same argument conventions (`p` down the rows, `a` along the columns). In `eq45_blocks_cheb` the fold stays `np.add.at`, because `k % Nt` can repeat.

Edit 1, `src/domainia/wilson/evolutionary.py`. Old:

```python
  - the Gram form for the log-linear family (the JAX build):
```

New:

```python
  - the Gram form for the log-linear family:
```

Edit 2, `src/domainia/wilson/evolutionary.py`. Old:

```python
which ships), so it is bit-identical to it; the JAX paths agree to round-off.
```

New:

```python
which ships), so it is bit-identical to it.
```

Edit 3, `src/domainia/wilson/evolutionary.py`. Old:

```python
from numpy.typing import NDArray

from ._checks import require_dtype, require_real
from .backend import NUMPY, Backend, get_backend, pytree_dataclass

```

New:

```python
from numpy.typing import NDArray
from scipy.special import ive

from ._checks import require_dtype, require_real

```

Edit 4, `src/domainia/wilson/evolutionary.py`. Delete these lines:

```python
@pytree_dataclass(data=("R", "C", "combo", "d", "s", "B"), meta=("m", "m2"))
```

Edit 5, `src/domainia/wilson/evolutionary.py`. Old:

```python
    and ``B = {combo: (k, eta_k G_ma(k) G*_m2b(k) / T)}`` over the support bins.

    A JAX pytree: the band indices ``m``/``m2`` are static (they pick the block's
    slot and whether it is symmetrised); every array is a leaf.  ``R``/``C`` are
    leaves too, because JAX rejects arrays as static fields; only their shapes,
    which stay static under tracing, matter to a kernel.
    """
```

New:

```python
    and ``B = {combo: (k, eta_k G_ma(k) G*_m2b(k) / T)}`` over the support bins.
    """
```

Edit 6, `src/domainia/wilson/evolutionary.py`. Delete these lines:

```python
@pytree_dataclass(
    data=("f", "eta", "m", "n", "t", "base", "keep", "cells", "pos", "band_cells", "Tp",
          "specs", "k_used"),
    meta=("Nf", "Nt", "dt", "N", "T", "K", "P", "t_max", "fold_has_duplicates"),
)
```

Edit 7, `src/domainia/wilson/evolutionary.py`. Old:

```python
    bits).

    A JAX pytree: the grid scalars (``Nf, Nt, dt, N, T, K, P, t_max``) and
    ``fold_has_duplicates`` are static; every array is a leaf.  (Registered
    with JAX lazily, see :func:`domainia.wilson.backend.pytree_dataclass`: at import if JAX
    was imported first, else by the first :func:`domainia.wilson.backend.enable_x64` /
    :func:`domainia.wilson.backend.jax_backend` call.)

    .. warning::
       Under ``jax.jit`` pass the plan as an **argument**
       (``jax.jit(lambda plan, c: eq45_blocks_cheb(plan, c))(plan, coeffs)``);
       never close over it.  A closed-over plan bakes every array (219 MB at
       64 x 256) into the compiled program: 404 MB of HLO text and ~25 s to
       lower and compile, against 2.5 MB and ~4 s as an argument (measured,
       CPU).  ``jax.device_put(plan)`` once keeps the arrays on the device
       between calls.
    """
```

New:

```python
    bits).
    """
```

Edit 8, `src/domainia/wilson/evolutionary.py`. Old:

```python
def eq45_coeffs_bessel(L: Array, alpha_c: Any, alpha_d: Any, P: int, *, r: float,
                       be: Backend | None = None) -> Array:
```

New:

```python
def eq45_coeffs_bessel(L: Array, alpha_c: Any, alpha_d: Any, P: int, *, r: float) -> Array:
```

Edit 9, `src/domainia/wilson/evolutionary.py`. Old:

```python
    plan.  Ported verbatim: keep the operation order (it fixes the float64 bits).  On JAX
    the Bessel functions are the 40-term series of
    :meth:`domainia.wilson.backend.Backend.bessel_i`.
    """
    be = be or get_backend(L, alpha_c, alpha_d)
    xp = be.xp
    a = 0.5 * alpha_d * r * L
    c = be.bessel_i(P, a)
    c = be.set(c, slice(1, None), c[1:] * 2.0, copy=False)          # c[1:] *= 2.0
    return c * xp.exp((alpha_c + alpha_d * (0.5 * r - 0.5)) * L)[None, :]
```

New:

```python
    plan.  Ported verbatim: keep the operation order (it fixes the float64 bits).
    :math:`I_p(a)` is ``scipy.special.ive(p, a) * exp(|a|)``.
    """
    a = np.asarray(0.5 * alpha_d * r * L)
    c = ive(np.arange(P)[:, None], a[None, :]) * np.exp(np.abs(a))[None, :]      # I_p(a)
    c[1:] *= 2.0
    return c * np.exp((alpha_c + alpha_d * (0.5 * r - 0.5)) * L)[None, :]
```

Edit 10, `src/domainia/wilson/evolutionary.py`. Old:

```python
def eq45_blocks_cheb(plan: Eq45Plan, coeffs: Array | Callable[[IntArray], Array], *,
                     be: Backend | None = None) -> tuple[list[Array], list[Array]]:
```

New:

```python
def eq45_blocks_cheb(plan: Eq45Plan, coeffs: Array | Callable[[IntArray], Array]
                     ) -> tuple[list[Array], list[Array]]:
```

Edit 11, `src/domainia/wilson/evolutionary.py`. Old:

```python
    Returns ragged lists (``diag[m]``: kept cells of band ``m`` squared).

    Under ``jax.jit`` pass ``plan`` as an argument (see :class:`Eq45Plan`) and
    ``coeffs`` as the ``(P, K)`` array (a callable would index host arrays with
    the traced ``k``).  The backend follows ``coeffs`` and the plan's arrays.
    """
    if be is None:
        be = get_backend(plan.Tp) if callable(coeffs) else get_backend(coeffs, plan.Tp)
    xp = be.xp
    P, Nt = plan.P, plan.Nt
    Tp = be.asarray(plan.Tp)
```

New:

```python
    Returns ragged lists (``diag[m]``: kept cells of band ``m`` squared).
    """
    P, Nt = plan.P, plan.Nt
    Tp = np.asarray(plan.Tp)
```

Edit 12, `src/domainia/wilson/evolutionary.py`. Old:

```python
        F = xp.zeros((4, P, Nt), complex)
        for cb, (k, Bk) in spec.B.items():
            Ck = coeffs(k) if callable(coeffs) else coeffs[:, k]
            V = Ck * Bk[None, :]                                        # (P, |k|)
            folded = xp.zeros((P, Nt), complex)
            folded = be.add_at(folded, (slice(None), k % Nt), V, copy=False)
            F = be.set(F, cb, be.fft.fft(folded, axis=1), copy=False)   # F_p(d)
        Fg = F[spec.combo, :, spec.d]                                   # (nR, nC, P)
        Tg = Tp[:, spec.s]                                              # (P, nR, nC)
        blk = xp.real(xp.einsum("ijp,pij->ij", Fg, Tg))
```

New:

```python
        F = np.zeros((4, P, Nt), complex)
        for cb, (k, Bk) in spec.B.items():
            Ck = coeffs(k) if callable(coeffs) else coeffs[:, k]
            V = Ck * Bk[None, :]                                        # (P, |k|)
            folded = np.zeros((P, Nt), complex)
            np.add.at(folded, (slice(None), k % Nt), V)                # k % Nt may repeat
            F[cb] = np.fft.fft(folded, axis=1)                          # F_p(d)
        Fg = F[spec.combo, :, spec.d]                                   # (nR, nC, P)
        Tg = Tp[:, spec.s]                                              # (P, nR, nC)
        blk = np.real(np.einsum("ijp,pij->ij", Fg, Tg))
```

Edit 13, `src/domainia/wilson/evolutionary.py`. Delete these lines:

```python
@pytree_dataclass(data=("sup", "slot", "gG", "gW", "gT", "uI1", "uI2", "uW"),
                  meta=("nb", "B", "Kb"))
```

Edit 14, `src/domainia/wilson/evolutionary.py`. Old:

```python
    overlap bins of adjacent bands.  Built line for line in the ported order.

    A JAX pytree (registered lazily, as :class:`Eq45Plan`): the layout sizes
    ``nb``, ``B``, ``Kb`` are static; every array is a leaf.

    .. warning::
       Under ``jax.jit`` pass the plan as an **argument**
       (``jax.jit(lambda gp, a: eq45_blocks_gram(gp, a, gL))(gplan, alpha_cells)``);
       never close over it.  A closed-over plan bakes ``gG`` (51 MB at 64 x 256)
       into the compiled program: 102 MB of HLO text, against 0.2 MB as an
       argument (measured, CPU).
    """
```

New:

```python
    overlap bins of adjacent bands.  Built line for line in the ported order.
    """
```

Edit 15, `src/domainia/wilson/evolutionary.py`. Old:

```python
def eq45_blocks_gram(gplan: Eq45GramPlan, alpha_cells: Array, gL: Array, *,
                     be: Backend | None = None) -> tuple[Array, Array]:
```

New:

```python
def eq45_blocks_gram(gplan: Eq45GramPlan, alpha_cells: Array, gL: Array) -> tuple[Array, Array]:
```

Edit 16, `src/domainia/wilson/evolutionary.py`. Old:

```python
    order.
    Under ``jax.jit`` pass ``gplan`` as an argument (see :class:`Eq45GramPlan`).
    The backend follows ``alpha_cells``, ``gL`` and the plan's arrays.
    """
    be = be or get_backend(alpha_cells, gL, gplan.gG)
    xp = be.xp
    gG, gW = be.asarray(gplan.gG), be.asarray(gplan.gW)
    uI1, uI2, uW = be.asarray(gplan.uI1), be.asarray(gplan.uI2), be.asarray(gplan.uW)
    al = be.asarray(alpha_cells)
    V = xp.exp(0.5 * al[:, :, None] * be.asarray(gL)[:, None, :]) * gG          # (nb, B, Kb)
    diag = xp.real(xp.einsum("bik,bjk->bij", V * gW[:, None, :], xp.conj(V)))
    Va = xp.take_along_axis(V[:-1], uI1[:, None, :], axis=2) * uW[:, None, :]
    Vb = xp.take_along_axis(V[1:], uI2[:, None, :], axis=2)
    up = xp.real(xp.einsum("bik,bjk->bij", Va, xp.conj(Vb)))
```

New:

```python
    order.
    """
    gG, gW = np.asarray(gplan.gG), np.asarray(gplan.gW)
    uI1, uI2, uW = np.asarray(gplan.uI1), np.asarray(gplan.uI2), np.asarray(gplan.uW)
    al = np.asarray(alpha_cells)
    V = np.exp(0.5 * al[:, :, None] * np.asarray(gL)[:, None, :]) * gG          # (nb, B, Kb)
    diag = np.real(np.einsum("bik,bjk->bij", V * gW[:, None, :], np.conj(V)))
    Va = np.take_along_axis(V[:-1], uI1[:, None, :], axis=2) * uW[:, None, :]
    Vb = np.take_along_axis(V[1:], uI2[:, None, :], axis=2)
    up = np.real(np.einsum("bik,bjk->bij", Va, np.conj(Vb)))
```

Edit 17, `src/domainia/wilson/evolutionary.py`. Old:

```python
def eq45_separable_assemble(KI: Array, KG: Array, g_half: Array, half_index: IntArray, *,
                            be: Backend | None = None) -> Array:
```

New:

```python
def eq45_separable_assemble(KI: Array, KG: Array, g_half: Array, half_index: IntArray) -> Array:
```

Edit 18, `src/domainia/wilson/evolutionary.py`. Old:

```python
    be = be or get_backend(g_half)
    return be.asarray(KI) + be.asarray(g_half)[be.asarray(half_index)] * be.asarray(KG)
```

New:

```python
    return np.asarray(KI) + np.asarray(g_half)[np.asarray(half_index)] * np.asarray(KG)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -cE "\bbe[.=,):]|\bxp\b|get_backend|Backend|pytree_dataclass|bessel_i|jax|JAX" src/domainia/wilson/evolutionary.py`

Expected: `0`

- [ ] **Step 4: `stationary.py`, `grid.py`, `_checks.py` and `windows.py`**

`apply_lambda_from_psd` keeps only its concrete branch: `sanitize_psd` maps `inf` to zero power and refuses negative and `nan` samples. `pack` and `unpack` take NumPy arrays. The window docstring and the hashability message drop "static under jax.jit", and `test_wrong_kinds.py`'s copy of the message follows.

Edit 1, `src/domainia/wilson/stationary.py`. Delete these lines:

```python
from .backend import Backend, get_backend, is_tracer
```

Edit 2, `src/domainia/wilson/stationary.py`. Old:

```python
    *,
    be: Backend | None = None,
    window: Window = MEYER,
) -> Array:
    r"""Apply $\Lambda$ to a WDM vector
```

New:

```python
    *,
    window: Window = MEYER,
) -> Array:
    r"""Apply $\Lambda$ to a WDM vector
```

Edit 3, `src/domainia/wilson/stationary.py`. Old:

```python
        power on every path (:func:`domainia.wilson.grid.sanitize_psd`).  Under
        ``jax.jit`` the PSD is a tracer and negative or ``nan`` samples are not
        checked: they are accepted silently instead of raising.
    Nf, Nt : int
    dt : float
    be : Backend, optional
        Array backend; inferred from ``x_wmn`` and ``Sn_rfft`` when omitted.

```

New:

```python
        power (:func:`domainia.wilson.grid.sanitize_psd`).
    Nf, Nt : int
    dt : float

```

Edit 4, `src/domainia/wilson/stationary.py`. Old:

```python
    be = be or get_backend(x_wmn, Sn_rfft)
    # Concrete input goes through sanitize_psd on the host.  A tracer cannot be
    # checked (negative/nan pass through), but it still gets the same inf -> 0.0
    # policy, so every path gives the same answer for the same PSD.
    if is_tracer(Sn_rfft):
        Sn_rfft = be.asarray(Sn_rfft)
        Sn_rfft = be.xp.where(be.xp.isinf(Sn_rfft), 0.0, Sn_rfft)
    else:
        Sn_rfft = be.asarray(sanitize_psd(Sn_rfft, "apply_lambda_from_psd"))
```

New:

```python
    Sn_rfft = sanitize_psd(Sn_rfft, "apply_lambda_from_psd")
```

Edit 5, `src/domainia/wilson/stationary.py`. Old:

```python
    y = _inverse(x_wmn, Nf, Nt, dt, be=be, window=window)
    z = (Sn_rfft / 2.0) * y
    return _forward(z, Nf, Nt, dt, be=be, window=window)
```

New:

```python
    y = _inverse(x_wmn, Nf, Nt, dt, window=window)
    z = (Sn_rfft / 2.0) * y
    return _forward(z, Nf, Nt, dt, window=window)
```

Edit 6, `src/domainia/wilson/grid.py`. Delete these lines:

```python
from .backend import Backend, get_backend
```

Edit 7, `src/domainia/wilson/grid.py`. Old:

```python
def pack(w: Array, Nf: int, Nt: int, *, be: Backend | None = None) -> Array:
    """``(..., Nf+1, Nt) -> (..., N)``: the active cells in band order."""
    be = be or get_backend(w)
    m, n = packed_cells(Nf, Nt)
    w = be.asarray(w)
```

New:

```python
def pack(w: Array, Nf: int, Nt: int) -> Array:
    """``(..., Nf+1, Nt) -> (..., N)``: the active cells in band order."""
    m, n = packed_cells(Nf, Nt)
    w = np.asarray(w)
```

Edit 8, `src/domainia/wilson/grid.py`. Old:

```python
def unpack(y: Array, Nf: int, Nt: int, *, be: Backend | None = None) -> Array:
    """``(..., N) -> (..., Nf+1, Nt)``, inactive edge pixels exactly 0."""
    be = be or get_backend(y)
    m, n = packed_cells(Nf, Nt)
    y = be.asarray(y)
    if y.shape[-1] != m.size:
        raise ValueError(f"y must end in N = {m.size}; got {y.shape}")
    out = be.xp.zeros((*y.shape[:-1], Nf + 1, Nt), dtype=y.dtype)
    return be.set(out, (..., m, n), y, copy=False)
```

New:

```python
def unpack(y: Array, Nf: int, Nt: int) -> Array:
    """``(..., N) -> (..., Nf+1, Nt)``, inactive edge pixels exactly 0."""
    m, n = packed_cells(Nf, Nt)
    y = np.asarray(y)
    if y.shape[-1] != m.size:
        raise ValueError(f"y must end in N = {m.size}; got {y.shape}")
    out = np.zeros((*y.shape[:-1], Nf + 1, Nt), dtype=y.dtype)
    out[..., m, n] = y
    return out
```

Edit 9, `src/domainia/wilson/_checks.py`. Old:

```python
    Reads ``a.dtype`` without converting ``a`` when it has one, so a JAX array or tracer
    passes through untouched; anything else (a list, a scalar) is read through
```

New:

```python
    Reads ``a.dtype`` without converting ``a`` when it has one; anything else (a list, a
    scalar) is read through
```

Edit 10, `src/domainia/wilson/windows.py`. Old:

```python
    """A Wilson window: frozen and hashable (static under ``jax.jit``)."""
```

New:

```python
    """A Wilson window: frozen and hashable (a cache key)."""
```

Edit 11, `src/domainia/wilson/windows.py`. Old:

```python
        raise TypeError(f"{what} must be hashable (a window is a cache key and static under "
                        f"jax.jit), got an unhashable {type(fn).__name__}: use a module-level "
                        f"function or a frozen dataclass instance") from None
```

New:

```python
        raise TypeError(f"{what} must be hashable (a window is a cache key), got an unhashable "
                        f"{type(fn).__name__}: use a module-level function or a frozen dataclass "
                        f"instance") from None
```

Edit 12, `tests/wilson/test_wrong_kinds.py`. Old:

```python
     "ramp must be hashable (a window is a cache key and static under jax.jit), got an unhashable "
     "_UnhashableRamp: use a module-level function or a frozen dataclass instance"),
```

New:

```python
     "ramp must be hashable (a window is a cache key), got an unhashable _UnhashableRamp: use a "
     "module-level function or a frozen dataclass instance"),
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py -k window_kind_messages 2>&1 | tail -1`

Expected: `5 passed, 187 deselected in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_meyer_baseline.py 2>&1 | tail -1`

Expected: `5 passed in …`. Every kernel the replay calls is now NumPy-only; `backend.py` is still imported by `domainia.wilson`, and goes in Step 5.

- [ ] **Step 5: Delete `backend.py` and `special.py`; the package `__init__`s; `pyproject.toml`**

Run: `cd /Users/aaron/Documents/lisa/domainia && git rm -q src/domainia/wilson/backend.py src/domainia/wilson/special.py && git status --short src | wc -l`

Expected: `8`: the 2 deleted files and the 6 files modified in Steps 2–4. `wc` may pad the number with spaces.

Edit 1, `src/domainia/wilson/__init__.py`. Old:

```python
from . import (
    backend,
    evolutionary,
    filter,
    matrix,
    noise,
    special,
    stationary,
    windows,
)
```

New:

```python
from . import (
    evolutionary,
    filter,
    matrix,
    noise,
    stationary,
    windows,
)
```

Edit 2, `src/domainia/wilson/__init__.py`. Delete these lines:

```python
    "backend",
```

Edit 3, `src/domainia/wilson/__init__.py`. Delete these lines:

```python
    "special",
```

Edit 4, `src/domainia/__init__.py`. Old:

```python
Importing ``domainia`` (any of its modules) never imports JAX.  JAX is imported only on use:
when the JAX backend is built (:func:`domainia.wilson.backend.jax_backend` or
``domainia.wilson.backend.get_backend(backend="jax")``) to pass as the ``be=`` of a kernel,
map or covariance that takes one, when float64 is turned on
(:func:`domainia.wilson.backend.enable_x64`), or when JAX device constants are built.  The
JAX backend needs float64: call ``enable_x64()`` (or set ``JAX_ENABLE_X64=true``) first, or
it raises ``RuntimeError``.  A kernel that takes ``be=``, given JAX arrays and no ``be=``,
infers the JAX backend, with the same requirement.

Modules
-------
```

New:

```python
Modules
-------
```

`pyproject.toml`: the `[jax]` extra and the markers go (spec §7). No test uses `pytest.mark.slow`; the `pytest.mark.jax` tests go in Steps 7, 11 and 13.

Edit 1, `pyproject.toml`. Old:

```toml
[project.optional-dependencies]
jax = ["jax>=0.9"]
test = ["pytest>=8"]
```

New:

```toml
[project.optional-dependencies]
test = ["pytest>=8"]
```

Edit 2, `pyproject.toml`. Old:

```toml
xfail_strict = true
markers = [
    "jax: needs jax; skipped when jax is not importable",
    "slow: heavy constructions (opt-in: -m slow)",
]
```

New:

```toml
xfail_strict = true
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -1`

Expected: `7 passed in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. In the prototype every one of the 1408 arrays is bit for bit its Task 4 value, and so are three `eq45_coeffs_bessel` outputs outside the replay (`(alpha_c, alpha_d, r)` = `(-2, 1.2, 1)`, `(0, 1.2, 272/256)` and `(0.5, -3, 1)`, 50 values of `L`): worst relative drift 0.

- [ ] **Step 6: The test infrastructure: both `conftest.py` files; `test_backend.py` and `test_lanes.py`**

Both `_HideJax` finders, `_jax_or_skip`, `jax_be`, the `backend` fixture and the `WDM_TESTS_HIDE_JAX` handling go. The `rng` fixtures keep their seeds (20261012 and 20261011), and `tests/wilson/conftest.py` keeps the oracle path and `_oracle_paths` as they were.

Replace the whole file `tests/conftest.py` with this text:

```python
"""Test configuration for domainia: the rng fixture."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20261012)
```

Replace the whole file `tests/wilson/conftest.py` with this text:

```python
"""Package-wide test configuration for ``domainia.wilson``.

Puts ``tests/oracle`` on ``sys.path`` so the frozen oracle imports as the
top-level package ``vendored_2f7136c`` (never as ``wdm``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ORACLE_DIR = Path(__file__).resolve().parent / "oracle"
if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))


@pytest.fixture
def rng():
    return np.random.default_rng(20261011)


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

Run: `cd /Users/aaron/Documents/lisa/domainia && git rm -q tests/wilson/test_backend.py tests/wilson/test_lanes.py && git status --short tests | wc -l`

Expected: `6`: the 2 deleted files, both `conftest.py` files, and `tests/test_package.py` and `tests/wilson/test_wrong_kinds.py` from Steps 1 and 4.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q --collect-only 2>&1 | grep -E "^ERROR" | sort -u`

Expected: exactly the three files that Steps 7, 8 and 12 fix (`test_transform_shim.py` imports `test_single_precision.py`):
```
ERROR tests/wilson/test_evolutionary_kernels.py
ERROR tests/wilson/test_single_precision.py
ERROR tests/wilson/test_transform_shim.py
```
Until Steps 7, 11 and 13 are done, collection also warns (`PytestUnknownMarkWarning`) about the `jax` marker that Step 5 unregistered.

- [ ] **Step 7: `tests/wilson/test_single_precision.py`**

The cases are called with no argument. The FFT recorder wraps `numpy.fft`'s four transforms with `monkeypatch`, in place of a `Backend` whose `fft` namespace recorded. The spies of `_set_rounded` and `_add_rounded` take their new three arguments.

Edit 1, `tests/wilson/test_single_precision.py`. Old:

```python
import dataclasses
import re

import numpy as np
import pytest

import domainia.wilson.transform as T
from domainia.wilson.backend import NUMPY

```

New:

```python
import re

import numpy as np
import pytest

import domainia.wilson.transform as T

```

Edit 2, `tests/wilson/test_single_precision.py`. Old:

```python
def _cases(precision, rng, to=np.asarray) -> dict:
    """``{entry point: call(be)}`` for every row of the table, on inputs of ``precision``
    (real inputs in it, complex ones in the matching complex dtype), converted by ``to``."""
```

New:

```python
def _cases(precision, rng) -> dict:
    """``{entry point: call()}`` for every row of the table, on inputs of ``precision``
    (real inputs in it, complex ones in the matching complex dtype)."""
```

Edit 3, `tests/wilson/test_single_precision.py`. Old:

```python
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
```

New:

```python
    x, X, w, W, wb = (a.astype(real) for a in (x, X, w, W, wb))
    z, Z = (a.astype(cplx) for a in (z, Z))
    return {
        "forward_time": lambda: T.forward_time(x, NF, NT, DT),
        "forward_time_many": lambda: T.forward_time_many(X, NF, NT, DT),
        "forward": lambda: T.forward(z, NF, NT, DT),
        "forward_many": lambda: T.forward_many(Z, NF, NT, DT),
        "inverse": lambda: T.inverse(w, NF, NT, DT),
        "inverse_many": lambda: T.inverse_many(W, NF, NT, DT),
        "inverse_time": lambda: T.inverse_time(w, NF, NT, DT),
        "inverse_time_many": lambda: T.inverse_time_many(W, NF, NT, DT),
        "forward_bands": lambda: T.forward_bands(z, 0, BANDS, NF, NT, DT),
        "inverse_bands": lambda: T.inverse_bands(wb, 0, N // 2 + 1, BANDS, NF, NT, DT),
    }


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_every_fft_of_a_float32_run_is_single_precision(name, rng, monkeypatch):
    """The inverse family and both band kernels used to run complex128 FFTs on float32 input
    (float64 sign patterns, ``lam`` and window samples promoted the FFT input).  The kernels
    call ``numpy.fft``, whose four transforms are wrapped to record each input dtype."""
    run = _cases(np.float32, rng)[name]          # its inputs are built before the wrappers
    calls: list[np.dtype] = []
    for op in ("fft", "ifft", "rfft", "irfft"):
        def recorded(a, *args, _fn=getattr(np.fft, op), **kw):
            calls.append(np.dtype(a.dtype))
            return _fn(a, *args, **kw)

        monkeypatch.setattr(np.fft, op, recorded)
    run()
    assert calls and all(dt in SINGLE for dt in calls), calls
```

Edit 4, `tests/wilson/test_single_precision.py`. Old:

```python
        def spy(a, idx, v, be, _helper=getattr(T, helper)):
            seen.append(np.dtype(v.dtype))
            return _helper(a, idx, v, be)

        monkeypatch.setattr(T, helper, spy)
    run(NUMPY)
```

New:

```python
        def spy(a, idx, v, _helper=getattr(T, helper)):
            seen.append(np.dtype(v.dtype))
            return _helper(a, idx, v)

        monkeypatch.setattr(T, helper, spy)
    run()
```

Edit 5, `tests/wilson/test_single_precision.py`. Old:

```python
    assert _cases(precision, rng)[name](NUMPY).dtype == output_dtype(name, precision)
```

New:

```python
    assert _cases(precision, rng)[name]().dtype == output_dtype(name, precision)
```

Edit 6, `tests/wilson/test_single_precision.py`. Old:

```python
    lo = np.asarray(_cases(np.float32, np.random.default_rng(5))[name](NUMPY))
    hi = np.asarray(_cases(np.float64, np.random.default_rng(5))[name](NUMPY))
```

New:

```python
    lo = _cases(np.float32, np.random.default_rng(5))[name]()
    hi = _cases(np.float64, np.random.default_rng(5))[name]()
```

Edit 7, `tests/wilson/test_single_precision.py`. Old:

```python
    """``{entry point: (a, call(a, be))}``: a double-precision input ``a`` that stays finite
```

New:

```python
    """``{entry point: (a, call(a))}``: a double-precision input ``a`` that stays finite
```

Edit 8, `tests/wilson/test_single_precision.py`. Old:

```python
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
```

New:

```python
        "forward_time": (flat, lambda a: T.forward_time(a, NF, NT, DT)),
        "forward_time_many": (two(flat), lambda a: T.forward_time_many(a, NF, NT, DT)),
        "forward": (bins, lambda a: T.forward(a, NF, NT, 1e4)),
        "forward_many": (two(bins), lambda a: T.forward_many(a, NF, NT, 1e4)),
        "inverse": (w_flat, lambda a: T.inverse(a, NF, NT, DT)),
        "inverse_many": (two(w_flat), lambda a: T.inverse_many(a, NF, NT, DT)),
        "inverse_time": (w_impulse, lambda a: T.inverse_time(a, NF, NT, DT)),
        "inverse_time_many": (two(w_impulse), lambda a: T.inverse_time_many(a, NF, NT, DT)),
        "forward_bands": (bins, lambda a: T.forward_bands(a, 0, BANDS, NF, NT, 1e4)),
        "inverse_bands": (w_bands, lambda a: T.inverse_bands(a, 0, N // 2 + 1, BANDS, NF, NT, 1e-4)),
```

Edit 9, `tests/wilson/test_single_precision.py`. Old:

```python
            call(z.astype(np.complex64), NUMPY)
    w = call(z, NUMPY)
```

New:

```python
            call(z.astype(np.complex64))
    w = call(z)
```

Edit 10, `tests/wilson/test_single_precision.py`. Old:

```python
            call(a32, NUMPY)
    out = call(a, NUMPY)
```

New:

```python
            call(a32)
    out = call(a)
```

In `tests/wilson/test_single_precision.py`, delete these by the named-function rule:
- `test_inverse_bands_accumulates_in_single_precision` (17 lines)
- `test_jax_float32_runs_are_single_precision` (12 lines)
- `test_the_guard_runs_on_concrete_jax_arrays_and_is_skipped_under_jit` (15 lines; the last thing in the file)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_single_precision.py 2>&1 | tail -1`

Expected: `62 passed in …`. It was 84 items: 2 legs of `test_inverse_bands_accumulates_in_single_precision` and 10 each of the two JAX tests go.

- [ ] **Step 8: `tests/wilson/test_transform_shim.py`**

Edit 1, `tests/wilson/test_transform_shim.py`. Old:

```python
"""The transform over the backend shim.

NumPy: the frozen oracle vendored_2f7136c to rounding on random grids, dt != 1,
```

New:

```python
"""The transform against the frozen oracle vendored_2f7136c.

It matches to rounding on random grids, dt != 1,
```

Edit 2, `tests/wilson/test_transform_shim.py`. Delete these lines:

```python
JAX: <= 1e-13 * max|w| (float64), and jit/vmap-able.
```

In `tests/wilson/test_transform_shim.py`, delete these by the named-function rule:
- `_on_device` (7 lines)
- `test_jax_matches_numpy` (17 lines)
- `test_jax_jit_and_vmap` (12 lines)
- `test_jax_rejects_nonfinite_concrete_input` (5 lines)
- `_F32_NO_WARNINGS` (37 lines; a module-level string, and its 5-line comment block above goes with it)
- `test_jax_float32_keeps_single_precision_without_future_warnings` (17 lines; the last thing in the file)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_transform_shim.py 2>&1 | tail -1`

Expected: `33 passed in …`. It was 40 items.

- [ ] **Step 9: `tests/wilson/test_noise_stationary.py`**

Edit 1, `tests/wilson/test_noise_stationary.py`. Old:

```python
def test_apply_lambda_from_psd_inf_psd_under_jit(jax_be, rng):
    """An inf PSD sample (the LISA f = 0 case) is zero power on every path.

    The host path maps it to 0.0 through ``sanitize_psd``; the tracer branch
    must apply the same policy or the inf leaks into the coefficients.
    """
    Nf, Nt, dt = 8, 16, 0.5
    S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
    S[0] = np.inf
    x = rng.standard_normal((Nf + 1, Nt))
    ref = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
    assert np.all(np.isfinite(ref))
    xj, Sj = jax_be.asarray(x), jax_be.asarray(S)
    eager = _on_device(stat.apply_lambda_from_psd(xj, Nf, Nt, dt, Sj))
    got = _on_device(_jit_apply(Nf, Nt, dt)(xj, Sj))
    assert np.all(np.isfinite(got)), f"{np.count_nonzero(~np.isfinite(got))} non-finite"
    assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))
    assert np.max(np.abs(got - eager)) <= 1e-13 * np.max(np.abs(eager))
```

New:

```python
def test_apply_lambda_from_psd_inf_psd_is_zero_power(rng):
    """An inf PSD sample (the LISA f = 0 case) is zero power: ``sanitize_psd`` maps it to 0.0."""
    Nf, Nt, dt = 8, 16, 0.5
    S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
    x = rng.standard_normal((Nf + 1, Nt))
    S[0] = 0.0
    want = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
    S[0] = np.inf
    assert np.array_equal(stat.apply_lambda_from_psd(x, Nf, Nt, dt, S), want)
```

In `tests/wilson/test_noise_stationary.py`, delete these by the named-function rule:
- `_on_device` (6 lines)
- `test_apply_lambda_from_psd_on_jax` (7 lines)
- `_jit_apply` (5 lines)
- `test_apply_lambda_from_psd_under_jit` (11 lines)
- `test_apply_lambda_from_psd_explicit_numpy_backend` (11 lines; the last thing in the file)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_noise_stationary.py 2>&1 | tail -1`

Expected: `40 passed in …`. It was 43 items.

- [ ] **Step 10: `tests/wilson/test_grid_bands.py`**

The two `rejects` tests are shown whole, because their call lines repeat.

Edit 1, `tests/wilson/test_grid_bands.py`. Old:

```python
def test_forward_bands_is_the_dense_atom_analysis(grid, backend, rng):
```

New:

```python
def test_forward_bands_is_the_dense_atom_analysis(grid, rng):
```

Edit 2, `tests/wilson/test_grid_bands.py`. Old:

```python
    got = np.asarray(forward_bands(backend.asarray(z), k0, bands, Nf, Nt, dt))
```

New:

```python
    got = forward_bands(z, k0, bands, Nf, Nt, dt)
```

Edit 3, `tests/wilson/test_grid_bands.py`. Old:

```python
def test_inverse_bands_is_the_adjoint(backend, rng):
```

New:

```python
def test_inverse_bands_is_the_adjoint(rng):
```

Edit 4, `tests/wilson/test_grid_bands.py`. Old:

```python
    lhs = np.sum(np.asarray(forward_bands(backend.asarray(z), k0, bands, Nf, Nt, dt)) * w)
    rhs = (2.0 * dt / N) * np.real(np.sum(np.conj(z) * np.asarray(
        inverse_bands(backend.asarray(w), k0, n_bins, bands, Nf, Nt, dt))))
```

New:

```python
    lhs = np.sum(forward_bands(z, k0, bands, Nf, Nt, dt) * w)
    rhs = (2.0 * dt / N) * np.real(np.sum(np.conj(z) * inverse_bands(w, k0, n_bins, bands, Nf,
                                                                      Nt, dt)))
```

Edit 5, `tests/wilson/test_grid_bands.py`. Old:

```python
def test_forward_bands_rejects_non_finite_bins(backend):
    Nf, Nt, dt = 8, 16, 1.0
    bands = np.array([2, 3])
    z = np.ones(40, complex)
    z[10] = np.nan
    with pytest.raises(ValueError, match="forward_bands: z: .* not finite"):
        forward_bands(backend.asarray(z), 0, bands, Nf, Nt, dt)
    z[10] = np.inf
    with pytest.raises(ValueError, match="forward_bands: z: .* not finite"):
        forward_bands(backend.asarray(z), 0, bands, Nf, Nt, dt)


def test_inverse_bands_rejects_complex_and_non_finite_cells(backend):
    Nf, Nt, dt = 8, 16, 1.0
    bands = np.array([2, 3])
    w = np.ones((bands.size, Nt))
    with pytest.raises(TypeError, match="w must be a real numeric array, got dtype complex128"):
        inverse_bands(backend.asarray(w + 1j), 0, 40, bands, Nf, Nt, dt)
    w[1, 4] = np.nan
    with pytest.raises(ValueError, match="inverse_bands: w: .* not finite"):
        inverse_bands(backend.asarray(w), 0, 40, bands, Nf, Nt, dt)
```

New:

```python
def test_forward_bands_rejects_non_finite_bins():
    Nf, Nt, dt = 8, 16, 1.0
    bands = np.array([2, 3])
    z = np.ones(40, complex)
    z[10] = np.nan
    with pytest.raises(ValueError, match="forward_bands: z: .* not finite"):
        forward_bands(z, 0, bands, Nf, Nt, dt)
    z[10] = np.inf
    with pytest.raises(ValueError, match="forward_bands: z: .* not finite"):
        forward_bands(z, 0, bands, Nf, Nt, dt)


def test_inverse_bands_rejects_complex_and_non_finite_cells():
    Nf, Nt, dt = 8, 16, 1.0
    bands = np.array([2, 3])
    w = np.ones((bands.size, Nt))
    with pytest.raises(TypeError, match="w must be a real numeric array, got dtype complex128"):
        inverse_bands(w + 1j, 0, 40, bands, Nf, Nt, dt)
    w[1, 4] = np.nan
    with pytest.raises(ValueError, match="inverse_bands: w: .* not finite"):
        inverse_bands(w, 0, 40, bands, Nf, Nt, dt)
```

In `tests/wilson/test_grid_bands.py`, delete these by the named-function rule:
- `test_pack_unpack_on_jax` (12 lines)
- `test_band_analysis_checks_are_skipped_under_jit` (16 lines)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_grid_bands.py 2>&1 | tail -1`

Expected: `46 passed, 1 skipped in …`. It was 55 items. The skip is the out-of-repo oracle test (`DOMAINIA_ORACLE_ROOT` unset).

- [ ] **Step 11: `tests/wilson/test_checks.py`**

In `tests/wilson/test_checks.py`, delete these by the named-function rule:
- `test_require_dtype_passes_a_jax_array_and_a_jit_tracer_through_untouched` (27 lines)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_checks.py 2>&1 | tail -1`

Expected: `101 passed in …`. It was 102.

- [ ] **Step 12: `tests/wilson/test_evolutionary_kernels.py`**

`HLO_LIMIT` goes with the jit tests. `SIN_NF`, `SIN_NT`, `SIN_DT` and `SIN_T_OBS` stay: `test_gram_matches_bessel_on_the_sinusoid_grid` uses them.

Edit 1, `tests/wilson/test_evolutionary_kernels.py`. Old:

```python
"""The stateless WDM Paper II kernels (domainia.wilson.evolutionary,
domainia.wilson.special, domainia.wilson.noise).

Every kernel is checked against its source: the frozen wdm_dev oracle
(``wdm_dev_ev``), the block builder of the WDM Paper II toy models in a lineage-A
subprocess, dense references, and across backends.  Both sources run the
```

New:

```python
"""The stateless WDM Paper II kernels (domainia.wilson.evolutionary, domainia.wilson.noise).

Every kernel is checked against its source: the frozen wdm_dev oracle
(``wdm_dev_ev``), the block builder of the WDM Paper II toy models in a lineage-A
subprocess, and dense references.  Both sources run the
```

Edit 2, `tests/wilson/test_evolutionary_kernels.py`. Delete these lines:

```python
from domainia.wilson import special
```

Edit 3, `tests/wilson/test_evolutionary_kernels.py`. Old:

```python
# -- Eq. (45) plans are pytrees: pass them to jit as ARGUMENTS ------------------------

SIN_NF, SIN_NT, SIN_DT = 64, 256, 2.0**-10        # the sinusoid grid
SIN_T_OBS = SIN_NF * SIN_NT * SIN_DT
HLO_LIMIT = 5_000_000      # chars; measured 0.2 MB (gram) and 2.5 MB (cheb) with the plan as
                           # an argument, 102 MB and 404 MB with the plan closed over
```

New:

```python
# -- the sinusoid grid ----------------------------------------------------------------

SIN_NF, SIN_NT, SIN_DT = 64, 256, 2.0**-10        # the sinusoid grid
SIN_T_OBS = SIN_NF * SIN_NT * SIN_DT
```

Edit 4, `tests/wilson/test_evolutionary_kernels.py`. Old:

```python
def test_separable_kernels_and_assembly(backend, rng):
```

New:

```python
def test_separable_kernels_and_assembly(rng):
```

Edit 5, `tests/wilson/test_evolutionary_kernels.py`. Old:

```python
    got = E.eq45_separable_assemble(backend.asarray(KI), backend.asarray(KG),
                                    backend.asarray(g_half), half, be=backend)
    assert type(got) is type(backend.asarray(ref))      # the backend's array type
    assert np.max(np.abs(np.asarray(got) - ref)) <= 1e-15 * np.abs(ref).max()


# -- special ----------------------------------------------------------------------

def test_special(backend, rng):
    a = rng.uniform(-3.2, 3.2, 40)
    got = special.bessel_i(24, backend.asarray(a), be=backend)
    assert type(got) is type(backend.asarray(a))      # the backend's array type
    got = np.asarray(got)
    from scipy.special import ive
    ref = ive(np.arange(24)[:, None], a[None, :]) * np.exp(np.abs(a))[None, :]
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-13
    T = special.chebyshev_basis(64, 5)
    u = 2.0 * (np.arange(64) / 64 - 0.5)
    assert np.allclose(T[3], 4 * u**3 - 3 * u, rtol=0, atol=1e-14)
```

New:

```python
    got = E.eq45_separable_assemble(KI, KG, g_half, half)
    assert np.max(np.abs(got - ref)) <= 1e-15 * np.abs(ref).max()


def test_bessel_coefficients_are_the_chebyshev_series_of_the_log_linear_spectrum(rng):
    """``sum_p c_p T_p(u) = exp(alpha(t) L)`` on the record (``r = 1``, ``u = 2 t / T - 1``):
    ``c_p`` is ``I_p(a)`` (``scipy.special.ive(p, a) exp|a|``), doubled for ``p >= 1``.  With
    ``alpha_c = 0`` and ``alpha_d = 2``, ``a = L`` and ``alpha(t) = u``; ``|a| <= 3.2`` is the
    range the sinusoid priors need."""
    L, P = rng.uniform(-3.2, 3.2, 40), 24
    u = 2.0 * (np.arange(64) / 64 - 0.5)
    T = np.cos(np.arange(P)[:, None] * np.arccos(u))                 # T_p(u), (P, 64)
    got = T.T @ E.eq45_coeffs_bessel(L, 0.0, 2.0, P, r=1.0)
    want = np.exp(u[:, None] * L[None, :])
    assert np.max(np.abs(got - want)) <= 1e-13 * np.max(want)
```

In `tests/wilson/test_evolutionary_kernels.py`, delete these by the named-function rule:
- `test_eq45_kernels_on_jax` (21 lines)
- `test_eq45_plans_are_pytrees` (33 lines)
- `test_eq45_gram_jits_with_the_plan_as_an_argument` (13 lines)
- `test_eq45_cheb_jits_with_the_plan_as_an_argument` (18 lines)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_evolutionary_kernels.py 2>&1 | tail -1`

Expected: `25 passed, 2 skipped in …`. It was 33 items. The skips are the two out-of-repo oracle tests.

- [ ] **Step 13: `tests/wilson/test_wrong_kinds.py`**

In `tests/wilson/test_wrong_kinds.py`, delete these by the named-function rule:
- `test_get_backend_backend_of_the_wrong_kind_raises_type_error` (5 lines)
- `test_get_backend_keeps_its_choices` (7 lines)
- `test_cholesky_impl_of_the_wrong_kind_raises_type_error` (5 lines)
- `test_cholesky_keeps_its_choices` (7 lines)
- `test_the_real_array_checks_read_a_jit_tracers_dtype` (19 lines; the last thing in the file)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py 2>&1 | tail -1`

Expected: `178 passed in …`. It was 192 items.

- [ ] **Step 14: `tests/wilson/test_window_threading.py`**

`SCANNED` drops the two deleted modules (11 entries). Two scanner rows used `be=` and one rebinding row used `is_tracer`; the rows keep what they test with plain expressions, and the `be=be` duplicate of `forward(x, Nf, Nt, dt)` goes.

Edit 1, `tests/wilson/test_window_threading.py`. Old:

```python
SCANNED = ("wilson/transform.py", "wilson/matrix.py", "wilson/stationary.py", "wilson/noise.py",
           "wilson/evolutionary.py", "wilson/grid.py", "wilson/backend.py",
           "wilson/special.py", "wilson/__init__.py", "wilson/_checks.py",
           "testing.py", "__init__.py", "_transform.py")
```

New:

```python
SCANNED = ("wilson/transform.py", "wilson/matrix.py", "wilson/stationary.py", "wilson/noise.py",
           "wilson/evolutionary.py", "wilson/grid.py", "wilson/__init__.py", "wilson/_checks.py",
           "testing.py", "__init__.py", "_transform.py")
```

Edit 2, `tests/wilson/test_window_threading.py`. Old:

```python
    ("forward(x, Nf, Nt, dt, be=be, window=self.dst.window)", False),
```

New:

```python
    ("forward(x, Nf, Nt, dt, window=self.dst.window)", False),
```

Edit 3, `tests/wilson/test_window_threading.py`. Delete these lines:

```python
    ("forward(x, Nf, Nt, dt, be=be)", True),
```

Edit 4, `tests/wilson/test_window_threading.py`. Old:

```python
    ("def f(x, *, window=MEYER):\n    window = MEYER if is_tracer(x) else window\n" + _PASS_ON, True),
```

New:

```python
    ("def f(x, *, window=MEYER):\n    window = MEYER if x is None else window\n" + _PASS_ON, True),
```

Edit 5, `tests/wilson/test_window_threading.py`. Old:

```python
    ("def f(x, *, window=MEYER):\n    window = window if be.xp is np else wilson.MEYER\n" + _PASS_ON,
```

New:

```python
    ("def f(x, *, window=MEYER):\n    window = window if x.ndim else wilson.MEYER\n" + _PASS_ON,
```

Edit 6, `tests/wilson/test_window_threading.py`. Old:

```python
#: Static arguments of a jitted kernel: the grid and the window (hashable, so a cache key).
_STATIC = ("Nf", "Nt", "dt", "window")


def _on_device(out):
    """``out`` on the host, after checking it came back as a ``jax.Array`` (no NumPy fallback)."""
    import jax

    assert isinstance(out, jax.Array), type(out)
    return np.asarray(out)


#: Every grid on NumPy; the JAX lanes on two grids only (a dropped window shows on any grid).
#: Not GRIDS[1] (16, 8, 0.25): there every test window but ALT_WINDOW samples alike.
_LANES = ([(*g, "numpy") for g in GRIDS]
          + [(*g, lane) for g in (GRIDS[0], GRIDS[2]) for lane in ("jax", "jax-jit")])


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt, lane", _LANES)
def test_apply_lambda_from_psd_is_the_dense_covariance(window, Nf, Nt, dt, lane, request):
    """``apply_lambda_from_psd(x)`` is ``build_C_wdm_active @ x`` on the active cells (junk in
    the inactive cells of ``x`` is ignored) on NumPy, on JAX with ``be=jax`` and under
    ``jax.jit`` with the window static (the JAX lanes skip without jax)."""
```

New:

```python
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_apply_lambda_from_psd_is_the_dense_covariance(window, Nf, Nt, dt):
    """``apply_lambda_from_psd(x)`` is ``build_C_wdm_active @ x`` on the active cells (junk in
    the inactive cells of ``x`` is ignored)."""
```

Edit 7, `tests/wilson/test_window_threading.py`. Old:

```python
    if lane == "numpy":
        got = St.apply_lambda_from_psd(x, Nf, Nt, dt, S, window=window)
    else:
        be = request.getfixturevalue("jax_be")
        if lane == "jax":
            out = St.apply_lambda_from_psd(be.asarray(x), Nf, Nt, dt, be.asarray(S), be=be,
                                           window=window)
        else:
            import jax

            jitted = jax.jit(St.apply_lambda_from_psd, static_argnames=_STATIC)
            out = jitted(be.asarray(x), Nf=Nf, Nt=Nt, dt=dt, Sn_rfft=be.asarray(S), window=window)
        got = _on_device(out)
    _close(got, want, RTOL_KERNEL, f"apply_lambda_from_psd[{lane}]")
```

New:

```python
    got = St.apply_lambda_from_psd(x, Nf, Nt, dt, S, window=window)
    _close(got, want, RTOL_KERNEL, "apply_lambda_from_psd")
```

In `tests/wilson/test_window_threading.py`, delete these by the named-function rule:
- `test_forward_and_inverse_take_the_window_on_jax` (23 lines)

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py 2>&1 | tail -1`

Expected: `461 passed in …`. It was 486: 20 JAX lanes of `test_apply_lambda_from_psd_is_the_dense_covariance`, 2 of `test_forward_and_inverse_take_the_window_on_jax`, 1 scanner row, and the 2 items of `test_no_kernel_call_drops_the_window` that scanned the deleted modules go.

- [ ] **Step 15: `README.md`**

The minimal edit: the JAX install hint, the JAX paragraph and the `be=` sentence go. Task 11 rewrites the README, and the "`transform` is NumPy only" paragraph goes with that rewrite.

Edit 1, `README.md`. Old:

```markdown
domain raise `ValueError`, because they are valid on a `FourierDomain`. Backend handles passed
as `be=` are not checked.
```

New:

```markdown
domain raise `ValueError`, because they are valid on a `FourierDomain`.
```

Edit 2, `README.md`. Old:

```markdown
pip install -e ".[test]"          # add ",jax" for the JAX backend
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu python -P -m pytest -q
```

New:

```markdown
pip install -e ".[test]"
PYTHONDONTWRITEBYTECODE=1 python -P -m pytest -q
```

Edit 3, `README.md`. Old:

```markdown
subdirectories each one expects).

`import domainia` (any of its modules) never imports JAX. JAX is imported only on use:
when you build the JAX backend (`domainia.wilson.backend.jax_backend()` or
`domainia.wilson.backend.get_backend(backend="jax")`) to pass as the `be=` of a kernel, map
or covariance that takes one, turn on float64 with `domainia.wilson.backend.enable_x64()`,
or build JAX device constants. The JAX backend needs float64: call `enable_x64()` (or set
`JAX_ENABLE_X64=true`) first, or it raises `RuntimeError`. A kernel that takes `be=`, given
JAX arrays and no `be=`, infers the JAX backend, with the same requirement.
```

New:

```markdown
subdirectories each one expects).
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_readme.py tests/test_package.py 2>&1 | tail -1`

Expected: `9 passed in …`

- [ ] **Step 16: Grep gate**

The pattern matches code use: `be.`, `be=`, `be` as an argument or parameter, the backend's names, `jax` as a word, the deleted fixtures and lane variable, and the deleted markers. The frozen oracle under `tests/wilson/oracle/` is not searched.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && grep -rnE "\bbe\.|\bbe=|\bbe[,)]|\bbe:|Backend\b|get_backend|is_tracer|pytree_dataclass|bessel_i|\bjax\b|jax_be|HideJax|HIDE_JAX|NUMPY\b|mark\.(jax|slow)" src tests --include='*.py' --exclude-dir=oracle
```

Expected: exactly these 3 lines, the subprocess check of the no-JAX-import test, which names the `jax` modules it looks for:
```
tests/test_package.py:71:            "jax = sorted(k for k in sys.modules if k.split('.')[0] in ('jax', 'jaxlib'))\n"
tests/test_package.py:72:            "print(json.dumps({'file': domainia.__file__, 'modules': mods, 'jax': jax}))")
tests/test_package.py:80:    assert info["jax"] == []
```

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && grep -rnE "wilson\.(backend|special)\b|from \.(backend|special) import|import special\b|\[jax\]|jax>=" src tests pyproject.toml --include='*.py' --include='*.toml' --exclude-dir=oracle | grep -v "^tests/test_package.py:.*assert not"
```

Expected: no output. The one `tests/test_package.py` line the filter drops is Step 1's assertion that neither module is imported.

Other allowed mentions, which the patterns do not match:
- `scipy.special` in `filter.py` and `evolutionary.py` (imports and one docstring) and in the new Bessel test's docstring.
- "JAX" in `tests/test_package.py`'s module docstring and in the no-JAX-import test's name and docstring.
- `README.md`'s "`transform` is NumPy only" paragraph (Task 11).

The same first grep on the tree after Task 4 prints 526 lines in 21 files.

- [ ] **Step 17: Gates**

The two suite runs are independent and can run in parallel.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1267 passed, 3 skipped`. It was `1380 passed, 3 skipped`: 113 items go, which are 25 in `test_backend.py`, 1 in `test_lanes.py`, `special.py`'s doctest and 86 in the 8 modified test files.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1267 passed, 3 skipped`. It was `1293 passed, 90 skipped`. PY and FLOOR now run the same items.

Neither log contains `warning` or `error` (`grep -ciE "warning|error"` prints `0`).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. `tests/wilson/test_meyer_baseline.py` passes inside both full runs.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 5796 lines / 2174 code
tests 4760 lines (+ oracle 4135)
```
That is −764 src lines (−431 code) and −1009 test lines.

- [ ] **Step 18: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/__init__.py src/domainia/wilson/__init__.py src/domainia/wilson/_checks.py src/domainia/wilson/evolutionary.py src/domainia/wilson/grid.py src/domainia/wilson/stationary.py src/domainia/wilson/transform.py src/domainia/wilson/windows.py tests/conftest.py tests/test_package.py tests/wilson/conftest.py tests/wilson/test_checks.py tests/wilson/test_evolutionary_kernels.py tests/wilson/test_grid_bands.py tests/wilson/test_noise_stationary.py tests/wilson/test_single_precision.py tests/wilson/test_transform_shim.py tests/wilson/test_window_threading.py tests/wilson/test_wrong_kinds.py README.md pyproject.toml && git status --short | grep -vc "^[MD] "`

Expected: `0`. Every remaining status line is a staged change (the `git rm` deletions of Steps 5 and 6 are already staged), so nothing is unstaged or untracked.

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Delete the JAX backend and special.py; the kernels call NumPy" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `25 files changed, 243 insertions(+), 2030 deletions(-)`

Size after Task 5 (prototype): src 5796 lines / 2174 code; tests 4760 (+ oracle 4135).

---

### Task 6: Delete float32: fp64 throughout (spec Order step 3; §4 "the float32 machinery"; §5)

**Goal:** The float32 machinery of `src/domainia/wilson/transform.py` is gone (`_working_dtypes`, `_check_single_output`, `_set_rounded`, `_add_rounded` and the working-dtype casts), every wilson entry point computes in and returns float64 or complex128, and `grid.sanitize_psd` returns float64 for every PSD. Every float64 output is bit for bit what it was.

Assumes the tree after Task 5, whose commit is "Delete the JAX backend and special.py; the kernels call NumPy".

**What changes for a caller.**
- A float32 or complex64 input (float16 too) to any of the ten entry points of `transform.py` (`forward`, `inverse`, their `_time` and `_many` forms, `forward_bands` and `inverse_bands`) is computed in double and returns float64 (complex128 for a spectrum). It returned float32 or complex64 before.
- The single-precision overflow guard goes with it: its `ValueError` could fire only on a float32 result.
- `grid.sanitize_psd` returns float64 for a float32 PSD. Every variance builder goes through it (`whittle_variance`, `whittle_variance_from_array`, `band_averaged_variance`, `evaluate_psd_on_band_grid`, `evolutionary_variance`, `colored_noise`, and `evolutionary_band_variance` through `_psd_at`), so a float32 PSD now gives a float64 grid.

**How each piece goes.** Every replacement is a no-op on a float64 run, which is why the replay is unchanged to the bit.

| Before | After |
|---|---|
| `rdtype, cdtype = _working_dtypes(a)` | removed; arrays are allocated float64 (`np.zeros(shape)`) or complex128 (`dtype=complex`) |
| `.astype(rdtype)` on the window samples and the sign patterns; `rdtype.type(c)` on the constants `kappa`, `root2`, `lam` and the band scales; `.astype(cdtype)` on the band phases | removed: each operand is float64 or complex128 already |
| `a = _set_rounded(a, idx, v)` | `a[idx] = v` |
| `a = _add_rounded(a, idx, v)` | `a[idx] += v`: each `idx` is duplicate-free (a slice, or strictly increasing indices) |
| `_check_single_output(out, "name")` | removed |
| `np.fft.rfft(x)` in `forward_time` and `forward_time_many` | `np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False))` |

The last row is the one new line. NumPy 2's `rfft` keeps single precision: a float32 `x` gives a complex64 spectrum (measured on numpy 2.1.0 and 2.3.5). Without the promotion, `forward_time` would run its rfft in single precision and still return float64, because the next product with the window samples promotes. `np.result_type(x, np.float64)` is float64 for float16, float32, integer and bool `x`. A complex `x` stays complex, and the rfft refuses it with the `TypeError` it raises today. A str or object `x` has already been refused by `validate_finite`, as today. `np.asarray(x, dtype=np.float64)` would be wrong here: it drops a complex `x`'s imaginary part with only a `ComplexWarning`.

Every other entry point is double after its first operation, so it needs no new line:
- `x_rfft[..., ell] * phi_c` is complex128 for a complex64 `x_rfft`, because the window samples are float64.
- `inverse` and `inverse_many` copy `wmn` with `wmn.astype(np.float64)`. This keeps the default `copy=True`: `_prepare_edges_for_inverse` updates the copy in place, and `test_inputs_are_not_mutated` pins that the caller's array is untouched.
- The band kernels multiply by complex128 phases.

**Which tests go, and which float32 mentions stay.**
- `tests/wilson/test_single_precision.py` goes whole (223 lines, 62 items). Its subject is the float32 machinery: the FFT and store dtypes of a float32 run, the dtype table, the float32-to-float64 agreement and the overflow guard.
- `tests/wilson/test_transform_shim.py` loses its float32 items: the `dtype` parametrisation, the float32 entry of `ORACLE_RTOL` and its import of `output_dtype` from `test_single_precision`. It goes from 33 to 17 items. The output dtype is still pinned, now against the oracle's.
- `tests/wilson/test_noise_stationary.py` loses the `_psd32` leg and `RTOL32`: 40 → 36 items.
- New, test first: `tests/wilson/test_wrong_kinds.py::test_float32_input_is_computed_in_double` (1 item). It loops over the ten entry points and `sanitize_psd`. Each must give, on float32 or complex64 input, the same dtype and the same values as on that input promoted to double. Without the `forward_time` line above, `forward_time` and `forward_time_many` fail it on values.
- These float32/complex64 mentions stay. Each is an input kind of a kept check, not a single-precision computation:
  - `test_wrong_kinds.py`: `np.complex64(1.0)` as a refused `dt`; `np.float32(0.1)` as an accepted `dt`; `np.complex64` as a refused real-array dtype.
  - `test_checks.py`: float32 scalars and arrays accepted by `require_real`/`require_dtype`; `complex64` refused; `index_array` of a float32 array; `validate_dt(np.float32(0.5))`.
  - `test_meyer_baseline.py`: `w.astype(np.float32)` as a dtype mismatch the comparator must catch.
  - `test_windows.py::test_meyer_sample_converts_dt_to_float64_first`: a float32 `dt` is sampled as `float(dt)`. This is fp64 behaviour.

**Files:**
- Delete, tests: `tests/wilson/test_single_precision.py` (223 lines, 62 items).
- Modify, source:
  - `src/domainia/wilson/transform.py` (Step 2):
    - the module docstring (its Precision section), the constants `_F64`/`_C128` and the FFT-backend comment;
    - `_build_slab`, `_prepare_edges_for_inverse`, `_scatter_interior_bands`, `_scatter_edge_bands` and `_gather_bands`;
    - the ten entry points: `forward`, `forward_many`, `forward_time_many`, `inverse`, `inverse_many`, `inverse_time_many`, `forward_time`, `inverse_time`, `forward_bands` and `inverse_bands`;
    - delete `_working_dtypes`, `_check_single_output`, `_set_rounded` and `_add_rounded`.
  - `src/domainia/wilson/grid.py` (`sanitize_psd`), `src/domainia/wilson/noise.py` (`band_averaged_variance`), `src/domainia/wilson/matrix.py` (`build_C_time`) and `src/domainia/wilson/windows.py` (the `Meyer` docstring) (Step 3).
- Modify, tests:
  - `tests/wilson/test_wrong_kinds.py` (Step 1, test first);
  - `tests/wilson/test_transform_shim.py`, rewritten whole (Step 5);
  - `tests/wilson/test_noise_stationary.py`: `RTOL64`, `_psd32` and `test_variance_builders_are_the_oracle_to_rounding` (Step 5).
- Test: `tests/wilson/test_wrong_kinds.py::test_float32_input_is_computed_in_double`, and `tests/wilson/test_meyer_baseline.py` (the replay, 5 items).

**Acceptance Criteria:**
- [ ] Step 1's run gives `1 failed, 178 passed`. The failure is `test_float32_input_is_computed_in_double`, with `AssertionError: inverse wmn` (complex64 against complex128).
- [ ] Step 4: `test_float32_input_is_computed_in_double` and `test_inputs_are_not_mutated` pass. The exact replay check prints `0 of 1408 arrays differ from the baseline`.
- [ ] `pytest -q` on `tests/wilson/test_wrong_kinds.py`, `test_transform_shim.py` and `test_noise_stationary.py` gives `179 passed`, `17 passed` and `36 passed`, in PY and in FLOOR.
- [ ] Step 6: the first grep prints nothing, the second prints exactly the 15 lines listed there, and the third prints exactly the 16 lines listed there.
- [ ] `python tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)` in PY and in FLOOR. The exact check prints `0 of 1408 arrays differ from the baseline` in PY and in FLOOR.
- [ ] Full suite (prototype): PY `1186 passed, 3 skipped`; FLOOR `1186 passed, 3 skipped`. No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 5624 lines / 2125 code` and `tests 4528 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1` → `1186 passed, 3 skipped in …`

**Steps:**

Edits are anchored to code text, never to line numbers. Each Old text occurs exactly once in its file, and New replaces it. "Delete these lines" means the shown lines go, newline included.

**Deleting a named function** (Step 2) follows one rule:
- Delete from its `def` line through its last line, with the two blank lines after it.
- None of the four has a decorator or a comment block directly above it, and none is the last thing in the file.
- Deleting the four in any order gives the same file, and so does making Step 2's Old/New edits before or after them.

From Step 2 until Step 5, `tests/wilson/test_single_precision.py` fails: it still expects float32 results and calls the deleted helpers. Step 5 deletes it.

Step 5's `git rm` and Step 8 are git writes in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: The test comes first (`tests/wilson/test_wrong_kinds.py`)**

The test goes after `test_real_arrays_take_integer_and_float_dtypes`. It reuses that test's `_REAL_ARRAYS` table (the inverse family, `inverse_bands` and `sanitize_psd`) and adds the forward family. It also reuses the module's `_Z` (the rfft bins of an 8 x 16 record) and `_BANDS`.

Edit 1, `tests/wilson/test_wrong_kinds.py`. Old:

```python
    ints = np.round(10 * good).astype(np.int64)
    assert np.array_equal(call(ints), call(ints.astype(np.float64)))

```

New:

```python
    ints = np.round(10 * good).astype(np.int64)
    assert np.array_equal(call(ints), call(ints.astype(np.float64)))


def test_float32_input_is_computed_in_double():
    """Every array input is promoted to double before the first FFT: a float32 or complex64
    input gives the double run's result on the same values, dtype included.  forward_time's
    rfft would otherwise run in single precision."""
    x = np.fft.irfft(_Z, 128)
    calls = {slot: (call, good) for slot, (call, good, _) in _REAL_ARRAYS.items()} | {
        "forward": (lambda a: wilson.forward(a, 8, 16, 1.0), _Z),
        "forward_many": (lambda a: wilson.forward_many(a, 8, 16, 1.0), _Z[None]),
        "forward_time": (lambda a: wilson.forward_time(a, 8, 16, 1.0), x),
        "forward_time_many": (lambda a: wilson.forward_time_many(a, 8, 16, 1.0), x[None]),
        "forward_bands": (lambda a: _T.forward_bands(a, 0, _BANDS, 8, 16, 1.0), _Z),
    }
    for slot, (call, good) in calls.items():
        low = good.astype(np.complex64 if good.dtype.kind == "c" else np.float32)
        got, want = call(low), call(low.astype(good.dtype))
        assert got.dtype == want.dtype in (np.float64, np.complex128), slot
        assert np.array_equal(got, want), slot

```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py 2>&1 | grep -E "^E +AssertionError|passed|failed"`

Expected:
```
E           AssertionError: inverse wmn
1 failed, 178 passed in …
```
`inverse` of a float32 `wmn` still returns complex64. The loop stops at the first slot. `forward_time` and `forward_time_many` would fail too, and so would every other slot.

- [ ] **Step 2: `src/domainia/wilson/transform.py`**

The Precision section of the module docstring shrinks to two sentences. The FFT-backend comment no longer claims that the replay pins the outputs bit for bit, because the replay's gate is 1e-12 since Task 1. They are still byte-for-byte stable: Step 4 checks it.

Edit 1, `src/domainia/wilson/transform.py`. Old:

```python
centred samples, from :func:`domainia.wilson.windows.sample_window`.  The formulas,
the edge normalisation and the precision rules are the same for every admissible
window.
```

New:

```python
centred samples, from :func:`domainia.wilson.windows.sample_window`.  The formulas
and the edge normalisation are the same for every admissible window.
```

Edit 2, `src/domainia/wilson/transform.py`. Old:

```python
Precision
---------
Single precision propagates: a ``float32`` series (or ``complex64`` spectrum)
is transformed in single precision and returns ``float32``.  Every constant
that multiplies a working array (the sign patterns, ``kappa``, ``lam``, the
window samples, the edge-band ``sqrt(2)``, the band phases) is cast to the
working dtype first, so every FFT receives a single-precision array.  Anything
else, integers included, promotes to double.  It is
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
suggests float64 input.  The check is skipped for double results.

float64 runs are bit for bit what they were before the casts (a cast to
float64 is a no-op, and no float64 expression is regrouped).  The Meyer
baseline, ``tests/wilson/test_meyer_baseline.py``, pins them bit for bit
except the rescaled families (``forward_bands``, ``inverse_bands``,
``FourierToWDM``), which it compares by tolerance.  float32 runs differ
from the frozen oracle at single-precision rounding, because the oracle formed
some products in double before storing them.

**This is a memory feature, not a speed one**, and the docstrings say so
because the obvious guess is wrong.  Measured end to end it is 0.87-1.19x --
i.e. nothing -- for the same reason ``scipy.fft`` was reverted above: the FFT
is not the bottleneck.  What it buys is half the memory.  Round-trip accuracy
drops to ~2e-7 relative, from ~4e-16.

Where it reaches, and where it stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **The transform** -- ``forward``/``inverse`` and their ``_time`` and
  ``_many`` forms, and the band kernels ``forward_bands``/``inverse_bands`` --
  follows the input dtype.
* **The variance builders** ``whittle_variance``, ``band_averaged_variance``
  and ``evolutionary_variance`` follow the *PSD's* dtype, so a ``float32`` PSD
  paired with a ``float32`` series keeps the whole coefficient/variance
  working set single-precision.
* **``empirical_variance`` stays double** on purpose. It is a Monte-Carlo
  reference; computing it in single precision would add numerical noise to a
  noise estimate, which defeats the point of having it.
* **``matrix`` and ``stationary`` stay double** on purpose. They are $O(N^2)$
  verification and covariance machinery, where precision *is* the deliverable
  and the arrays are too large to want in the first place.

An earlier revision claimed "half the memory" while only the coefficient grid
was single -- the variance grid it is invariably paired with stayed double, so
the real working set was ~33% smaller, not 50%. Both are single now, and the
claim is measured rather than assumed.

Before any of this, ``float32`` input was silently upcast and returned
``float64`` -- twice the memory the caller asked for, with no indication.
"""
```

New:

```python
Precision
---------
Every entry point computes in double and returns float64 (complex128 for a
spectrum).  Input of another accepted dtype (a lower-precision float, an
integer) is promoted first.
"""
```

Edit 3, `src/domainia/wilson/transform.py`. Old:

```python
from .windows import MEYER, Window, require_window, sample_window

_F64 = np.dtype(np.float64)
_C128 = np.dtype(np.complex128)

type Array = NDArray[np.floating]
```

New:

```python
from .windows import MEYER, Window, require_window, sample_window

type Array = NDArray[np.floating]
```

Edit 4, `src/domainia/wilson/transform.py`. Old:

```python
# The cost was real: scipy is not bit-identical to numpy (~3.7e-16 relative,
# same algorithm family, different rounding), and the coefficients are kept
# byte-for-byte stable across every change (the Meyer outputs are pinned bit for
# bit by ``tests/wilson/test_meyer_baseline.py``). Trading a verifiable property
# for measurement noise is a bad deal.
```

New:

```python
# The cost was real: scipy is not bit-identical to numpy (~3.7e-16 relative,
# same algorithm family, different rounding), and the Meyer outputs have stayed
# byte-for-byte stable across every change. Trading a verifiable property for
# measurement noise is a bad deal.
```

Edit 5, `src/domainia/wilson/transform.py`. Old:

```python
def _build_slab(
    X_rfft: CArray, m: int, Nf: int, Nt: int, cdtype: np.dtype = _C128,
) -> CArray:
```

New:

```python
def _build_slab(X_rfft: CArray, m: int, Nf: int, Nt: int) -> CArray:
```

Edit 6, `src/domainia/wilson/transform.py`. Old:

```python
    Nf, Nt : int
    cdtype : dtype, optional
        Complex working dtype.

```

New:

```python
    Nf, Nt : int

```

Edit 7, `src/domainia/wilson/transform.py`. Old:

```python
    slab = np.zeros((*X_rfft.shape[:-1], Nt), dtype=cdtype)
```

New:

```python
    slab = np.zeros((*X_rfft.shape[:-1], Nt), dtype=complex)
```

Edit 8, `src/domainia/wilson/transform.py`. Old:

```python
def _prepare_edges_for_inverse(
    wmn: Array, Nf: int, Nt: int, rdtype: np.dtype = _F64
) -> Array:
```

New:

```python
def _prepare_edges_for_inverse(wmn: Array, Nf: int, Nt: int) -> Array:
```

Edit 9, `src/domainia/wilson/transform.py`. Old:

```python
    work = wmn.astype(rdtype)          # always a fresh copy, so updating it in place is safe
    mask = active_mask(Nf, Nt)
    root2 = rdtype.type(np.sqrt(2))   # mask[m] * root2 is then in the working dtype too
    # ``work[..., 0, :] *= mask[0] * root2``.
    work = _set_rounded(work, (..., 0, slice(None)), work[..., 0, :] * (mask[0] * root2))
    work = _set_rounded(work, (..., Nf, slice(None)), work[..., Nf, :] * (mask[Nf] * root2))
    return work
```

New:

```python
    work = wmn.astype(np.float64)      # always a fresh copy, so updating it in place is safe
    mask = active_mask(Nf, Nt)
    root2 = np.sqrt(2.0)
    work[..., 0, :] *= mask[0] * root2
    work[..., Nf, :] *= mask[Nf] * root2
    return work
```

Edit 10, `src/domainia/wilson/transform.py`. Old:

```python
    # In the working dtype, so a float32 G (and so the FFT) stays complex64;
    # the callers pass ``phi_c`` and ``lam`` in that dtype too.
    sign_int = (1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])).astype(wmn_work.dtype)
```

New:

```python
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
```

Edit 11, `src/domainia/wilson/transform.py`. Old:

```python
        X = _add_rounded(X, (..., slice(start, start + Nt)), W_int[..., i, :])
```

New:

```python
        X[..., start:start + Nt] += W_int[..., i, :]
```

Edit 12, `src/domainia/wilson/transform.py`. Old:

```python
        scaled = wmn_work[..., m, :] * sign.astype(wmn_work.dtype)
```

New:

```python
        scaled = wmn_work[..., m, :] * sign
```

Edit 13, `src/domainia/wilson/transform.py`. Old:

```python
        # ``X[..., ell[valid]] += v``: the indices are strictly increasing, so the
        # fancy-index add is one addition per bin either way.
        X = _add_rounded(X, (..., ell[valid]), lam * phi_c[valid] * Wm[..., valid])
```

New:

```python
        # The indices are strictly increasing, so the fancy-index add is one addition per bin.
        X[..., ell[valid]] += lam * phi_c[valid] * Wm[..., valid]
```

Edit 14, `src/domainia/wilson/transform.py`. Old:

```python
    rdtype, cdtype = _working_dtypes(x_rfft)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)          # host constant
    # Every constant below is in the working dtype, so a float32 run stays float32
    # after the FFT (for float64 the casts are no-ops).
    kappa = rdtype.type(np.sqrt(2.0) / Nf)
    n_arr = np.arange(Nt)
    out: Array = np.zeros((*x_rfft.shape[:-1], Nf + 1, Nt), dtype=rdtype)
```

New:

```python
    phi_c = sample_window(window, Nf, Nt, dt)
    kappa = np.sqrt(2.0) / Nf
    n_arr = np.arange(Nt)
    out: Array = np.zeros((*x_rfft.shape[:-1], Nf + 1, Nt))
```

Edit 15, `src/domainia/wilson/transform.py`. Old:

```python
    sign_int = (1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])).astype(rdtype)
    # ``out[..., 1:Nf, :] = (kappa * sign_int) * re_or_im_int``, in the working dtype.
    out = _set_rounded(out, (..., slice(1, Nf), slice(None)), kappa * sign_int * re_or_im_int)

    for m in (0, Nf):
        y_m = np.fft.ifft(_build_slab(x_rfft, m, Nf, Nt, cdtype) * phi_c, axis=-1)
        even, sign = _parity(m, n_arr)
        out = _set_rounded(out, (..., m, slice(None)),
                           kappa * sign.astype(rdtype) * np.where(even, y_m.real, y_m.imag))

    # Unit-norm edge basis functions. Not optional: see the module docstring.
    # ``out[..., 0, :] /= sqrt(2)``, with sqrt(2) in the working dtype.
    root2 = rdtype.type(np.sqrt(2))
    out = _set_rounded(out, (..., 0, slice(None)), out[..., 0, :] / root2)
    out = _set_rounded(out, (..., Nf, slice(None)), out[..., Nf, :] / root2)
    return out
```

New:

```python
    sign_int = 1.0 - 2.0 * ((1 - m_par)[:, None] * n_par[None, :])
    out[..., 1:Nf, :] = kappa * sign_int * re_or_im_int

    for m in (0, Nf):
        y_m = np.fft.ifft(_build_slab(x_rfft, m, Nf, Nt) * phi_c, axis=-1)
        even, sign = _parity(m, n_arr)
        out[..., m, :] = kappa * sign * np.where(even, y_m.real, y_m.imag)

    # Unit-norm edge basis functions. Not optional: see the module docstring.
    root2 = np.sqrt(2.0)
    out[..., 0, :] /= root2
    out[..., Nf, :] /= root2
    return out
```

Edit 16, `src/domainia/wilson/transform.py`. Old:

```python
    out = _gather_bands(x_rfft, Nf, Nt, dt, window=window)
    _check_single_output(out, "forward")

```

New:

```python
    out = _gather_bands(x_rfft, Nf, Nt, dt, window=window)

```

Edit 17, `src/domainia/wilson/transform.py`. Delete these lines:

```python
    _check_single_output(out, "forward_many")
```

Edit 18, `src/domainia/wilson/transform.py`. Old:

```python
    validate_finite(x, "forward_time_many: x")
    x_rfft = np.fft.rfft(x, axis=-1)
    # Before forward_many's input check, which would report an overflow here as a gap.
    _check_single_output(x_rfft, "forward_time_many")
```

New:

```python
    validate_finite(x, "forward_time_many: x")
    # At least double before the rfft, which keeps a lower-precision float's precision.
    # A complex x stays complex, and the rfft refuses it.
    x_rfft = np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False), axis=-1)
```

Edit 19, `src/domainia/wilson/transform.py`. Old:

```python
    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = np.zeros(N // 2 + 1, dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
    _check_single_output(X, "inverse")
```

New:

```python
    phi_c = sample_window(window, Nf, Nt, dt)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt)

    lam = 1.0 / (np.sqrt(2.0) * dt)
    X = np.zeros(N // 2 + 1, dtype=complex)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
```

Edit 20, `src/domainia/wilson/transform.py`. Old:

```python
    rdtype, cdtype = _working_dtypes(wmn)
    phi_c = sample_window(window, Nf, Nt, dt).astype(rdtype)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt, rdtype)

    lam = rdtype.type(1.0 / (np.sqrt(2.0) * dt))
    X = np.zeros((wmn.shape[0], N // 2 + 1), dtype=cdtype)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
    _check_single_output(X, "inverse_many")
```

New:

```python
    phi_c = sample_window(window, Nf, Nt, dt)
    wmn_work = _prepare_edges_for_inverse(wmn, Nf, Nt)

    lam = 1.0 / (np.sqrt(2.0) * dt)
    X = np.zeros((wmn.shape[0], N // 2 + 1), dtype=complex)
    X = _scatter_interior_bands(X, wmn_work, Nf, Nt, phi_c, lam)
    X = _scatter_edge_bands(X, wmn_work, Nf, Nt, N, phi_c, lam)
```

Edit 21, `src/domainia/wilson/transform.py`. Delete these lines:

```python
    _check_single_output(x, "inverse_time_many")
```

Edit 22, `src/domainia/wilson/transform.py`. Old:

```python
    validate_finite(x, "forward_time: x")
    x_rfft = np.fft.rfft(x)
    # Before forward's input check, which would report an overflow here as a gap.
    _check_single_output(x_rfft, "forward_time")
```

New:

```python
    validate_finite(x, "forward_time: x")
    # At least double before the rfft, which keeps a lower-precision float's precision.
    # A complex x stays complex, and the rfft refuses it.
    x_rfft = np.fft.rfft(x.astype(np.result_type(x, np.float64), copy=False))
```

Edit 23, `src/domainia/wilson/transform.py`. Delete these lines:

```python
    _check_single_output(x, "inverse_time")
```

Edit 24, `src/domainia/wilson/transform.py`. Old:

```python
    rdtype, cdtype = _working_dtypes(z)
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi).astype(cdtype)
    Y = np.fft.ifft(phi.astype(rdtype) * z[..., ell], axis=-1)
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    out = np.real(phase * Y) * rdtype.type(np.sqrt(2.0 * dt / n_samples(Nf, Nt)))
    _check_single_output(out, "forward_bands")
    return out
```

New:

```python
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi)
    Y = np.fft.ifft(phi * z[..., ell], axis=-1)
    # Wilson units: the unit-norm-atom cells times sqrt(2 dt / N) (band_atoms' docstring).
    return np.real(phase * Y) * np.sqrt(2.0 * dt / n_samples(Nf, Nt))
```

Edit 25, `src/domainia/wilson/transform.py`. Old:

```python
    rdtype, cdtype = _working_dtypes(w)
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi)
    V = phi.astype(rdtype) * np.fft.fft(np.conj(phase).astype(cdtype) * w, axis=-1) / Nt
    out = np.zeros((*w.shape[:-2], n_bins), dtype=cdtype)
    for b in range(bands.size):
        np.add.at(out, (..., ell[b]), V[..., b, :])
    # rfft units: N / (2 dt) times the adjoint of forward_bands' sqrt(2 dt / N).
    out = out * rdtype.type(np.sqrt(n_samples(Nf, Nt) / (2.0 * dt)))
    _check_single_output(out, "inverse_bands")
    return out
```

New:

```python
    phi = sample_window(window, Nf, Nt, dt)
    phase = _band_phase(bands, Nt, phi)
    V = phi * np.fft.fft(np.conj(phase) * w, axis=-1) / Nt
    out = np.zeros((*w.shape[:-2], n_bins), dtype=complex)
    for b in range(bands.size):
        np.add.at(out, (..., ell[b]), V[..., b, :])
    # rfft units: N / (2 dt) times the adjoint of forward_bands' sqrt(2 dt / N).
    return out * np.sqrt(n_samples(Nf, Nt) / (2.0 * dt))
```

In `src/domainia/wilson/transform.py`, delete these by the named-function rule:
- `_working_dtypes` (15 lines)
- `_check_single_output` (26 lines)
- `_set_rounded` (8 lines)
- `_add_rounded` (10 lines)

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -cE "rdtype|cdtype|_rounded|_working_dtypes|_check_single_output|_F64|_C128|float32|complex64" src/domainia/wilson/transform.py`

Expected: `0`

- [ ] **Step 3: `grid.py`, `noise.py`, `matrix.py` and `windows.py`**

`sanitize_psd` returns float64 for every PSD. In `band_averaged_variance`, the weights then need no cast to the PSD's dtype. In `build_C_time`, the PSD needs no second cast to float, and the comment that justified it goes. The `Meyer` docstring loses its float32 example. `sample` still converts `dt` with `float(dt)`, which `tests/wilson/test_windows.py::test_meyer_sample_converts_dt_to_float64_first` pins.

Edit 1, `src/domainia/wilson/grid.py`. Old:

```python
    Returns
    -------
    ndarray, float
        Finite, non-negative PSD values with non-finite entries set to ``0.0``.
```

New:

```python
    Returns
    -------
    ndarray, float64
        Finite, non-negative PSD values with non-finite entries set to ``0.0``.
```

Edit 2, `src/domainia/wilson/grid.py`. Old:

```python
    # Preserve single precision rather than forcing float64: a float32 PSD
    # paired with a float32 coefficient grid keeps the *whole* working set
    # single-precision. Forcing double here meant the variance grid stayed
    # float64 no matter what, so a caller in "float32 mode" got a working set
    # that was only ~33% smaller, not 50%. Everything else promotes.
    arr = arr.astype(np.float32 if arr.dtype == np.float32 else float)
```

New:

```python
    arr = arr.astype(np.float64)
```

Edit 3, `src/domainia/wilson/noise.py`. Old:

```python
    # Match the window weights to the PSD's precision, so a float32 PSD yields
    # a float32 variance grid instead of being promoted back to float64 by the
    # matmul. The window itself is always computed in double; only the
    # reduction is cast.
    weights = phi2[in_band].astype(Sn_vals.dtype, copy=False)
```

New:

```python
    weights = phi2[in_band]
```

Edit 4, `src/domainia/wilson/matrix.py`. Old:

```python
    # Force double. `sanitize_psd` preserves single precision so that a
    # float32 PSD yields a float32 *variance grid* (see domainia.wilson.transform's
    # Precision section), but this module is the exact-reference tier: an
    # (N, N) covariance is what everything else is verified against, and it is
    # O(N^2) in memory whatever the dtype, so there is nothing to win and a
    # reference to lose. `stationary` forces double for the same reason.
    Sn = np.asarray(sanitize_psd(Sn_func(freqs), "build_C_time"), dtype=float)
```

New:

```python
    Sn = sanitize_psd(Sn_func(freqs), "build_C_time")
```

Edit 5, `src/domainia/wilson/windows.py`. Old:

```python
    ``Meyer(d).sample`` is :func:`domainia.wilson.filter.phi_centered` ``(Nf, Nt, dt, d)``
    bit for bit for a Python float (or float64) ``dt``; ``Meyer()`` (d = 4) is the WDM
    transform's window.  ``sample`` converts ``dt`` to float64 first, so a float32 ``dt``
    gives ``phi_centered(Nf, Nt, float(dt), d)``, about 3e-8 away from ``phi_centered`` on
    the raw float32 ``dt`` (which then computes partly in float32).
```

New:

```python
    ``Meyer(d).sample`` is :func:`domainia.wilson.filter.phi_centered`
    ``(Nf, Nt, float(dt), d)``: ``sample`` converts any real ``dt`` to a Python float first.
    ``Meyer()`` (d = 4) is the WDM transform's window.
```

- [ ] **Step 4: The new test passes, and the float64 outputs are unchanged to the bit**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py tests/wilson/test_transform_shim.py::test_inputs_are_not_mutated src/domainia/wilson 2>&1 | tail -1`

Expected: `239 passed in …`. That is 179 in `test_wrong_kinds.py`, 1 for `test_inputs_are_not_mutated` and 59 doctests.

The exact check compares every kept replay array with the recorded baseline by dtype and `np.array_equal`. It is stricter than the replay's 1e-12 gate.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')"
```

Expected: `0 of 1408 arrays differ from the baseline`. The prototype prints the same with the FLOOR interpreter (`$TMPDIR/dom3-envs/floor312/bin/python` in place of the PY one).

- [ ] **Step 5: The tests of the float32 machinery**

Run: `cd /Users/aaron/Documents/lisa/domainia && git rm -q tests/wilson/test_single_precision.py`

Replace the whole file `tests/wilson/test_transform_shim.py` with this text:

```python
"""The transform against the frozen oracle vendored_2f7136c.

It matches to rounding on random grids, dt != 1, batched and unbatched.
Bit-identical up to v0.11.0; the Hz window of v0.12.0 (Paper I eq:Meyer-def: phi
flat value 1/sqrt(Delta F), kappa = sqrt(2)/Nf, lambda = 1/(sqrt(2) dt), against the
oracle's omega-normalised 1/sqrt(2 pi Delta F), 2 sqrt(pi)/Nf, sqrt(pi)/dt) is the
same product rounded differently, so the gate is ORACLE_RTOL * max|ref|.
"""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.transform as oracle

import domainia.wilson.transform as T

GRIDS = [(2, 2, 1.0), (4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10),
         (32, 64, 10.0), (64, 256, 2.0**-10), (6, 10, 0.3)]

#: max|got - ref| over max|ref| against the omega-era oracle.
ORACLE_RTOL = 1e-14     # Hz window (v0.12.0): measured 4.2e-16 fwd, 5.4e-16 inv


def _assert_is_the_oracle_to_rounding(got, ref):
    """The oracle's dtype, and its values to rounding."""
    assert got.dtype == ref.dtype
    assert np.max(np.abs(got - ref)) <= ORACLE_RTOL * np.max(np.abs(ref))


@pytest.mark.parametrize("grid", GRIDS)
def test_forward_family_is_the_oracle_to_rounding(grid, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = rng.standard_normal(N)
    X = rng.standard_normal((3, N))
    pairs = [
        (T.forward_time(x, Nf, Nt, dt), oracle.forward_time(x, Nf, Nt, dt)),
        (T.forward(np.fft.rfft(x), Nf, Nt, dt), oracle.forward(np.fft.rfft(x), Nf, Nt, dt)),
        (T.forward_time_many(X, Nf, Nt, dt), oracle.forward_time_many(X, Nf, Nt, dt)),
        (T.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt),
         oracle.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref)


@pytest.mark.parametrize("grid", GRIDS)
def test_inverse_family_is_the_oracle_to_rounding(grid, rng):
    Nf, Nt, dt = grid
    w = rng.standard_normal((Nf + 1, Nt))          # junk in inactive pixels on purpose
    W = rng.standard_normal((3, Nf + 1, Nt))
    pairs = [
        (T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
        (T.inverse_many(W, Nf, Nt, dt), oracle.inverse_many(W, Nf, Nt, dt)),
        (T.inverse_time_many(W, Nf, Nt, dt), oracle.inverse_time_many(W, Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref)


def test_inputs_are_not_mutated(rng):
    Nf, Nt, dt = 8, 16, 0.5
    w = rng.standard_normal((Nf + 1, Nt))
    w0 = w.copy()
    T.inverse(w, Nf, Nt, dt)
    assert np.array_equal(w, w0)
```

In `tests/wilson/test_noise_stationary.py`, the variance-builder test keeps its float64 leg. Its `psd` parameter becomes the module's `_psd`.

Edit 1, `tests/wilson/test_noise_stationary.py`. Old:

```python
#: max|got - ref| over max|ref| against the omega-era oracle, by working precision.
RTOL64, RTOL32 = 1e-14, 2e-6
```

New:

```python
#: max|got - ref| over max|ref| against the omega-era oracle.
RTOL64 = 1e-14
```

Edit 2, `tests/wilson/test_noise_stationary.py`. Old:

```python
def _psd32(f):
    return _psd(f).astype(np.float32)


def _psd_ft(f, t):
```

New:

```python
def _psd_ft(f, t):
```

Edit 3, `tests/wilson/test_noise_stationary.py`. Old:

```python
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("psd", [_psd, _psd32])
def test_variance_builders_are_the_oracle_to_rounding(grid, psd):
```

New:

```python
@pytest.mark.parametrize("grid", GRIDS)
def test_variance_builders_are_the_oracle_to_rounding(grid):
```

Edit 4, `tests/wilson/test_noise_stationary.py`. Old:

```python
    Nf, Nt, dt = grid
    got = noise.whittle_variance(Nf, Nt, dt, psd)
    ref = onoise.whittle_variance(Nf, Nt, dt, psd)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)
    got = noise.band_averaged_variance(Nf, Nt, dt, psd)
    ref = onoise.band_averaged_variance(Nf, Nt, dt, psd)
    assert got.dtype == ref.dtype
    # Hz window (v0.12.0): measured 4.8e-16 (float64 PSD), 3.0e-7 (float32 PSD)
    assert _gap(got, ref) <= (RTOL32 if got.dtype == np.float32 else RTOL64)
```

New:

```python
    Nf, Nt, dt = grid
    got = noise.whittle_variance(Nf, Nt, dt, _psd)
    ref = onoise.whittle_variance(Nf, Nt, dt, _psd)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)
    got = noise.band_averaged_variance(Nf, Nt, dt, _psd)
    ref = onoise.band_averaged_variance(Nf, Nt, dt, _psd)
    assert got.dtype == ref.dtype
    assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 4.8e-16
```

Edit 5, `tests/wilson/test_noise_stationary.py`. Old:

```python
    band = noise.evaluate_psd_on_band_grid(Nf, dt, psd)
    assert np.array_equal(band, onoise.evaluate_psd_on_band_grid(Nf, dt, psd))
```

New:

```python
    band = noise.evaluate_psd_on_band_grid(Nf, dt, _psd)
    assert np.array_equal(band, onoise.evaluate_psd_on_band_grid(Nf, dt, _psd))
```

Run: `cd /Users/aaron/Documents/lisa/domainia && for f in test_wrong_kinds test_transform_shim test_noise_stationary; do PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/$f.py 2>&1 | tail -1; done`

Expected:
```
179 passed in …
17 passed in …
36 passed in …
```

The same loop with the FLOOR interpreter prints the same three counts.

- [ ] **Step 6: Grep gates**

First, the float32 machinery and single-precision text in src:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnEi "float32|complex64|single[- ]precision|_rounded|working.dtype|_check_single" src`

Expected: no output.

Second, the task's literal gate. It also matches the English word "single", in docstrings and comments that are not about precision. Those are left for the docstring trim (Task 12).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "float32|complex64|single" src | LC_ALL=C sort`

Expected: exactly these 15 lines, each the word "single":
```
src/domainia/wilson/__init__.py:77:* :func:`sample_window` (the kernels' single entry point: validated, read-only samples of
src/domainia/wilson/_checks.py:225:    implementation routes each band through an FFT.  A single non-finite
src/domainia/wilson/grid.py:376:       too late. Measured against the energy centroid of a single-pixel
src/domainia/wilson/matrix.py:36:work with a single $N \\times N$ matrix in the real-DFT
src/domainia/wilson/stationary.py:160:    the already-scaled covariance. This function applies the single
src/domainia/wilson/stationary.py:267:    This lives at module level on purpose: it is the single most error-prone
src/domainia/wilson/stationary.py:289:    the single definition of that recipe; it used to appear verbatim in three
src/domainia/wilson/stationary.py:293:    module's single convention (``sigma2_full[:N//2+1] / N``).  Interior slabs
src/domainia/wilson/stationary.py:343:    This is the single definition of the slab-IFFT recipe.  Both
src/domainia/wilson/stationary.py:622:    Each band pair $(m, m')$ with $|m - m'| \le 1$ carries a single
src/domainia/wilson/stationary.py:680:    Stitch the slabs into a single rfft-length vector by assigning each
src/domainia/wilson/stationary.py:759:    # One batched FFT for every band, then a single vectorised
src/domainia/wilson/transform.py:111:#   * Isolated, scipy IS faster: 1.64x single-threaded and 2.78x at
src/domainia/wilson/transform.py:198:    read a single row -- 5.1% of ``inverse`` at $N_f=64$, $N_t=128$.
src/domainia/wilson/windows.py:286:    """The kernels' single entry point: the validated, read-only samples of ``window``.
```

Third, the tests. Every remaining hit is an input kind of a kept check (listed above the Files), or the new test.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "float32|complex64|_psd32|RTOL32|output_dtype|_rounded|_working_dtypes|_check_single_output" tests --include='*.py' --exclude-dir=oracle | LC_ALL=C sort`

Expected: exactly these 16 lines:
```
tests/wilson/test_checks.py:129:@pytest.mark.parametrize("a", [[1, 2], [0.5], 1.0, np.float64(1.0), np.zeros(2), np.zeros(2, np.float32),
tests/wilson/test_checks.py:137:    (np.complex64(1), "complex64"), (np.zeros(2, complex), "complex128"), (np.zeros(2, bool), "bool"),
tests/wilson/test_checks.py:151:        dtype = np.dtype(np.float32)
tests/wilson/test_checks.py:161:    ([1, 2.0], [1, 2]), (np.array([[0.0, 4.0]], np.float32), [[0, 4]]), (3, 3), ([], []),
tests/wilson/test_checks.py:189:    assert validate_dt(np.float32(0.5)) is None and validate_dt(Fraction(1, 2)) is None
tests/wilson/test_checks.py:58:@pytest.mark.parametrize("value", [1, 0.5, np.float32(0.25), np.float64(0.5), np.int64(2), np.uint16(3),
tests/wilson/test_checks.py:67:    (np.complex64(1), "complex64"), (Decimal("0.5"), "Decimal"), (np.array(0.5), "ndarray"),
tests/wilson/test_meyer_baseline.py:44:    for k, got in ((a, w + 2 * RTOL * peak), (a, nan), (a, w.astype(np.float32)), (a, w[:2]),
tests/wilson/test_windows.py:345:    dt = np.float32(0.1)
tests/wilson/test_wrong_kinds.py:256:@pytest.mark.parametrize("dtype", [np.complex128, np.complex64, np.bool_, np.str_, object],
tests/wilson/test_wrong_kinds.py:257:                         ids=["complex128", "complex64", "bool", "str", "object"])
tests/wilson/test_wrong_kinds.py:275:def test_float32_input_is_computed_in_double():
tests/wilson/test_wrong_kinds.py:276:    """Every array input is promoted to double before the first FFT: a float32 or complex64
tests/wilson/test_wrong_kinds.py:288:        low = good.astype(np.complex64 if good.dtype.kind == "c" else np.float32)
tests/wilson/test_wrong_kinds.py:60:    (1.0 + 0.0j, "complex"), (np.complex64(1.0), "complex64"), (np.array(1.0 + 0.0j), "ndarray"),
tests/wilson/test_wrong_kinds.py:72:@pytest.mark.parametrize("dt", [0.5, 1, np.float32(0.1), np.float64(0.25), np.int64(2), Fraction(1, 2)])
```

- [ ] **Step 7: Gates**

The two suite runs are independent and can run in parallel.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1186 passed, 3 skipped`. It was `1267 passed, 3 skipped`. 82 items go: 62 in `test_single_precision.py`, 16 in `test_transform_shim.py` and 4 in `test_noise_stationary.py`. The new test adds 1, so the net change is −81.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1186 passed, 3 skipped`.

Neither log contains `warning` or `error` (`grep -ciE "warning|error"` prints `0`).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. The FLOOR interpreter prints the same. `tests/wilson/test_meyer_baseline.py` passes inside both full runs.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 5624 lines / 2125 code
tests 4528 lines (+ oracle 4135)
```
That is −172 src lines (−49 code) and −232 test lines.

- [ ] **Step 8: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/wilson/transform.py src/domainia/wilson/grid.py src/domainia/wilson/noise.py src/domainia/wilson/matrix.py src/domainia/wilson/windows.py tests/wilson/test_wrong_kinds.py tests/wilson/test_transform_shim.py tests/wilson/test_noise_stationary.py && git status --short | grep -vc "^[MD] "`

Expected: `0`. Every status line is a staged change (Step 5's `git rm` is staged already), so nothing is unstaged or untracked.

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Delete float32: the wilson kernels compute in and return float64" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `9 files changed, 115 insertions(+), 519 deletions(-)`

Size after Task 6 (prototype): src 5624 lines / 2125 code; tests 4528 (+ oracle 4135).

---

### Task 7: Fold the kernels 8 → 4; delete the band-limited entry points (spec §2 "Kernels"; §4; §5; §6.1)

**Goal:** `forward`, `inverse`, `forward_time` and `inverse_time` in `src/domainia/wilson/transform.py` take any leading axes. The four `_many` forms and the band-limited entry points (`forward_bands`, `inverse_bands`, `band_atoms` and their helpers `_band_phase`, `_band_windows`, `_check_interior_bands`) are gone, and every caller uses the folded names. Every replay array is bit for bit what it was.

Assumes the tree after Task 6, whose commit is "Delete float32: the wilson kernels compute in and return float64".

**What changes for a caller.**
- `forward(x_rfft, Nf, Nt, dt, *, window=MEYER)` maps `(..., N//2+1)` to `(..., Nf+1, Nt)`. `inverse(wmn, ...)` maps `(..., Nf+1, Nt)` to `(..., N//2+1)`. `forward_time(x, ...)` maps `(..., N)` to `(..., Nf+1, Nt)`, and `inverse_time(wmn, ...)` maps `(..., Nf+1, Nt)` to `(..., N)`. Before, each took exactly one record, and the `_many` forms took exactly one leading axis.
- Each slice of a stack gives what the call on that slice alone gives, bit for bit. This was measured on numpy 2.1.0, 2.3.5 and 2.5.4, on six grids from 2 x 2 to 64 x 256. The new test checks it to 1e-14.
- A wrong trailing shape still raises `ValueError`, and the message now shows the leading axes: `x_rfft must have shape (..., 65) for Nf=8, Nt=16; got (64,)`, `x must have shape (..., 128) = (..., Nf*Nt) for Nf=8, Nt=16; got (127,)`, `wmn must have shape (..., 9, 16); got (9, 15)`.
- `forward_many`, `inverse_many`, `forward_time_many`, `inverse_time_many`, `forward_bands`, `inverse_bands` and `band_atoms` no longer exist. The four `_many` names leave `domainia.wilson.__all__`; the band names were only ever in `domainia.wilson.transform.__all__`.
- The parameter names stay `x_rfft`, `wmn`, `x` and `wmn`. The spec's §2 writes `forward(X, …)` and `inverse(w, …)`, but every caller passes these arguments by position. The names also appear in pinned messages (`wmn must be a real numeric array`, `forward: x_rfft: …`), so a rename would only change those messages.

**How the fold works.** The internals already index with `...` (`_gather_bands`, `_build_slab`, `_prepare_edges_for_inverse`, `_scatter_interior_bands`, `_scatter_edge_bands`). So each entry point changes in two places at most:
- its shape check looks at the trailing axes only (`x_rfft.shape[-1:]`, `x.shape[-1:]`, `wmn.shape[-2:]`);
- `inverse` allocates its accumulator with the leading axes: `np.zeros((*wmn.shape[:-2], N // 2 + 1), dtype=complex)`.

The `rfft` in `forward_time` and the `irfft` in `inverse_time` already act on the last axis (NumPy's default `axis=-1`), so neither line changes. Task 6's promotion line in `forward_time` stays as it is.

**The callers.**

| Caller | Before | After |
|---|---|---|
| `evolutionary.apply_time_mask` | `np.atleast_2d(inverse_time_many(full, …) * chi)`, then `forward_time_many(X, …)[:, mask]` | `inverse_time(full, …) * chi`, then `forward_time(X, …)[:, mask]`. `full` is `(B, Nf+1, Nt)` already, so `atleast_2d` did nothing. |
| `evolutionary.wrap_fractions` | `e = np.zeros((1, Nf + 1, Nt))`, `inverse_time_many(e, …)[0]` | `e = np.zeros((Nf + 1, Nt))`, `inverse_time(e, …)` |
| `evolutionary.exact_covariance` | `forward_time_many(np.ascontiguousarray(G.T), …)` | `forward_time(np.ascontiguousarray(G.T), …)` |
| `transform.base_atom_spectra` | `inverse_many(e[None], …)[0]` | `inverse(e, …)` |
| `_transform.transform` | `wilson.inverse_many(y.reshape(…), …).reshape(lead + …)`, `wilson.forward_many(R.reshape(…), …).reshape(lead + …)` and the `lead = …` line | `wilson.inverse(y, *sg, dt, window=window)` and `wilson.forward(R, *tg, dt, window=window)`; `lead` goes |
| `matrix.build_W` | `inverse_time(seed, …)` | unchanged: it already calls the folded name |
| `testing.py` | no kernel call | unchanged |

**The replay (spec §6.1).** The 24 keys `wilson::transform.forward_many`, `forward_time_many`, `inverse_many` and `inverse_time_many` (× 3 grids × 2 `dt`) keep their names. Their cases call the folded kernels on the same stacks: `Tr.forward(Xr, …)`, `Tr.forward_time(X, …)`, `Tr.inverse(Wm, …)` and `Tr.inverse_time(Wm, …)`. `Wm` becomes `wl.forward(Xr, Nf, Nt, dt)`, which has the same values as Task 1's `np.stack` of per-row calls. All 1408 kept arrays stay bit for bit the recorded baseline (Step 4).

**Which tests change, and which go.**
- `tests/wilson/test_transform_shim.py` is rewritten whole, test first (Step 1). It goes from 17 to 18 items.
  - Its stacked pairs now call the folded kernels. They are compared with the oracle's single-record calls, stacked. The oracle's `_many` forms are its own loops, bit for bit, and their names would be hits of the Step 6 grep.
  - `_assert_is_the_oracle_to_rounding` now checks the shape too.
  - New: `test_leading_axes_are_the_loop_over_the_slices`. For each of the four kernels, it compares a `(3, 2, …)` stack with the double loop over its slices, to `ORACLE_RTOL` with dtype and shape. It also checks that an input one entry short on its last axis is refused with `ValueError`.
- `tests/wilson/test_window_threading.py` goes from 461 to 401 items.
  - `CONSUMERS` loses the seven deleted names.
  - The scanner row `("inverse_many(e, …, window=window if a == 0 else MEYER)", True)` becomes `inverse(…)`. A name outside `CONSUMERS` is not a consumer call, so the row would no longer be flagged and its test would fail.
  - `test_time_round_trip_and_energy` sends its two-row stack through `forward_time`.
  - Deleted, band code: `test_forward_bands_matches_its_atoms_and_forward` (20 items) and `test_inverse_bands_is_the_adjoint_of_forward_bands` (20 items).
  - Deleted, the batched path: `test_batched_inverse_is_the_per_row_inverse_and_round_trips` (20 items). Its subject was the separate code of `inverse_many` and `inverse_time_many`. After the fold, a stack runs the same code as one record. The stack-equals-loop check is the new shim test, and `test_time_round_trip_and_energy` still sends a stack through `forward_time` with every window.
- `tests/wilson/test_wrong_kinds.py` goes from 179 to 117 items. Every item removed called a deleted name.
  - `_BANDS` and `_W_BANDS` go; `_T`, `_Z` and `_W` stay.
  - `_INT_SLOTS` loses `forward_bands k0`, `inverse_bands k0` and `inverse_bands n_bins` (21 items across its two tests).
  - `test_integer_slots_out_of_range_stay_value_errors` loses its two band rows.
  - `_BAND_SLOTS` and its three tests go (21 items).
  - `_REAL_ARRAYS` loses `inverse_many wmn`, `inverse_time_many wmn` and `inverse_bands w` (18 items across its two tests). The folded `inverse` and `inverse_time` keep their rows.
  - `test_float32_input_is_computed_in_double` loses `forward_many`, `forward_time_many` and `forward_bands`. It still loops over 5 cases: `inverse`, `inverse_time`, `sanitize_psd`, `forward` and `forward_time`.
  - The docstring of `test_integer_slots_of_the_wrong_kind_raise_type_error` loses its `k0="0"` example.
- `tests/wilson/test_grid_bands.py` goes from 47 to 29 items.
  - Its seven band definitions go (18 items).
  - `test_window_and_atoms_are_bitwise_the_gb_copies` keeps its window check as `test_window_is_bitwise_the_gb_copy`, and `_gb_reference` no longer computes the galactic-binary atoms. The test skips unless `$DOMAINIA_ORACLE_ROOT/toy_models/galactic_binary` exists, and it skips in every lane here, so only collection checks this edit. The subprocess still copies `config.py`, `wdm_atoms.py` and `geometry.py`, because whether `wdm_atoms.py` imports `geometry.py` cannot be checked here.
- `tests/wilson/vendored/test_evolutionary.py` keeps its 8 items. `test_packed_grid_matches_the_dense_transform` called the oracle's own `PackedGrid.forward_many` and `inverse_many`.
  - `g.forward_many(x)` becomes `np.stack([g.forward(r) for r in x])`, still compared with `x @ Wa.T`, where `Wa` is domainia's `active_W`.
  - The round trip `g.inverse_many(g.forward_many(x))` goes. It calls only the oracle (spec §6.3).
- The doctests of `transform.py` go from 8 to 4: the four `_many` doctests go, and `forward`'s doctest gains a leading-axes example.
- `tests/wilson/test_checks.py` does not change. Its `index_array` tests stay: `index_array` loses its last src caller here (`_check_interior_bands`), and spec §4 deletes it in Task 11.

**Files:**
- Modify, source:
  - `src/domainia/wilson/transform.py` (Step 2): the imports, `__all__`, the docstrings of `_build_slab`, `_prepare_edges_for_inverse`, `_scatter_interior_bands` and `_gather_bands`, the entry points `forward`, `inverse`, `forward_time` and `inverse_time`, the section comment and `base_atom_spectra`; delete `forward_many`, `forward_time_many`, `inverse_many`, `inverse_time_many`, `_check_interior_bands`, `band_atoms`, `_band_phase`, `_band_windows`, `forward_bands` and `inverse_bands`.
  - `src/domainia/wilson/__init__.py` (the docstring, the import from `.transform` and `__all__`), `src/domainia/wilson/evolutionary.py` (the import from `.transform`, `apply_time_mask`, `wrap_fractions` and `exact_covariance`) and `src/domainia/_transform.py` (`transform`) (Step 3).
- Modify, tests:
  - `tests/wilson/test_transform_shim.py`, rewritten whole (Step 1, test first);
  - `tests/wilson/meyer_cases.py`: `wilson_cases` (Step 4);
  - `tests/wilson/test_window_threading.py`, `tests/wilson/test_wrong_kinds.py`, `tests/wilson/test_grid_bands.py` and `tests/wilson/vendored/test_evolutionary.py` (Step 5).
- No change: `src/domainia/wilson/matrix.py` (`build_W` already calls `inverse_time`), `src/domainia/testing.py` (no kernel call), `README.md` (no deleted name), `tests/wilson/test_checks.py`.
- Test: `tests/wilson/test_transform_shim.py::test_leading_axes_are_the_loop_over_the_slices`, and `tests/wilson/test_meyer_baseline.py` (the replay, 5 items).

**Acceptance Criteria:**
- [ ] Step 1's run gives `17 failed, 1 passed`, and the new test fails with `ValueError: x_rfft must have shape (65,) for Nf=8, Nt=16; got (3, 2, 65)`.
- [ ] Step 2's count prints `0`. Step 3's grep prints nothing, and its test run gives `78 passed`.
- [ ] Step 4: the exact check prints `0 of 1408 arrays differ from the baseline` in PY and in FLOOR, and `tests/wilson/test_meyer_baseline.py` gives `5 passed`.
- [ ] Step 5: `401 passed`, `117 passed`, `28 passed, 1 skipped`, `8 passed` and `18 passed`, in PY and in FLOOR.
- [ ] Step 6: the grep gate prints exactly the 6 lines listed there, all in `tests/wilson/meyer_cases.py`.
- [ ] `python tests/wilson/meyer_cases.py --check` → `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)` in PY and in FLOOR.
- [ ] Full suite (prototype): PY `1043 passed, 3 skipped`; FLOOR `1043 passed, 3 skipped`. No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 5219 lines / 1956 code` and `tests 4337 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1` → `1043 passed, 3 skipped in …`

**Steps:**

Edits are anchored to code text, never to line numbers. Each Old text occurs exactly once in its file, and New replaces it. "Delete these lines" means the shown lines go, newline included.

**Deleting a named definition** (Steps 2 and 5) follows one rule:
- Delete from its first line through its last line, with the two blank lines after it.
- Its first line is its first decorator line when it has decorators, and otherwise its `def` line, or the line of its assignment for `_BAND_SLOTS` and `BASELINE_GRIDS`.
- A comment block directly above it, with no blank line between, goes with it. Here that is the one `#:` line above `_BAND_SLOTS` and the one above `BASELINE_GRIDS`.
- `inverse_bands` is the last definition in `transform.py`. Delete it with the two blank lines before it instead, so the file still ends in one newline.
- Deleting the definitions in any order gives the same file, and so does making the step's Old/New edits before or after them.

The tree is broken between steps; each step's own checks say what runs.
- After Step 2, `domainia.wilson` does not import, because its `__init__.py` still imports the `_many` names. Step 3 fixes that.
- After Step 3, the replay fails, because `meyer_cases.py` calls the `_many` names. Step 4 fixes that.
- Until Step 5, `test_wrong_kinds.py` and `test_grid_bands.py` do not collect, and `test_window_threading.py` fails.

Step 8 is a git write in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: The test comes first (`tests/wilson/test_transform_shim.py`)**

Replace the whole file `tests/wilson/test_transform_shim.py` with this text:

```python
"""The transform against the frozen oracle vendored_2f7136c.

It matches to rounding on random grids, dt != 1, one record and a stack of three.
Bit-identical up to v0.11.0; the Hz window of v0.12.0 (Paper I eq:Meyer-def: phi
flat value 1/sqrt(Delta F), kappa = sqrt(2)/Nf, lambda = 1/(sqrt(2) dt), against the
oracle's omega-normalised 1/sqrt(2 pi Delta F), 2 sqrt(pi)/Nf, sqrt(pi)/dt) is the
same product rounded differently, so the gate is ORACLE_RTOL * max|ref|.
"""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.transform as oracle

import domainia.wilson.transform as T

GRIDS = [(2, 2, 1.0), (4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10),
         (32, 64, 10.0), (64, 256, 2.0**-10), (6, 10, 0.3)]

#: max|got - ref| over max|ref| against the omega-era oracle.
ORACLE_RTOL = 1e-14     # Hz window (v0.12.0): measured 4.2e-16 fwd, 5.4e-16 inv


def _assert_is_the_oracle_to_rounding(got, ref):
    """The oracle's dtype and shape, and its values to rounding."""
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert np.max(np.abs(got - ref)) <= ORACLE_RTOL * np.max(np.abs(ref))


@pytest.mark.parametrize("grid", GRIDS)
def test_forward_family_is_the_oracle_to_rounding(grid, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = rng.standard_normal(N)
    X = rng.standard_normal((3, N))
    pairs = [
        (T.forward_time(x, Nf, Nt, dt), oracle.forward_time(x, Nf, Nt, dt)),
        (T.forward(np.fft.rfft(x), Nf, Nt, dt), oracle.forward(np.fft.rfft(x), Nf, Nt, dt)),
        (T.forward_time(X, Nf, Nt, dt), np.stack([oracle.forward_time(r, Nf, Nt, dt) for r in X])),
        (T.forward(np.fft.rfft(X), Nf, Nt, dt),
         np.stack([oracle.forward(np.fft.rfft(r), Nf, Nt, dt) for r in X])),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref)


@pytest.mark.parametrize("grid", GRIDS)
def test_inverse_family_is_the_oracle_to_rounding(grid, rng):
    Nf, Nt, dt = grid
    w = rng.standard_normal((Nf + 1, Nt))          # junk in inactive pixels on purpose
    W = rng.standard_normal((3, Nf + 1, Nt))
    pairs = [
        (T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
        (T.inverse(W, Nf, Nt, dt), np.stack([oracle.inverse(r, Nf, Nt, dt) for r in W])),
        (T.inverse_time(W, Nf, Nt, dt), np.stack([oracle.inverse_time(r, Nf, Nt, dt) for r in W])),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref)


def test_leading_axes_are_the_loop_over_the_slices(rng):
    """Each kernel takes any leading axes: a (3, 2, ...) stack gives the loop over its
    slices, and an input one entry short on its last axis is refused."""
    Nf, Nt, dt = 8, 16, 0.5
    x = rng.standard_normal((3, 2, Nf * Nt))
    w = rng.standard_normal((3, 2, Nf + 1, Nt))
    for f, a in ((T.forward, np.fft.rfft(x)), (T.forward_time, x), (T.inverse, w), (T.inverse_time, w)):
        want = np.stack([np.stack([f(s, Nf, Nt, dt) for s in row]) for row in a])
        _assert_is_the_oracle_to_rounding(f(a, Nf, Nt, dt), want)
        with pytest.raises(ValueError, match="must have shape"):
            f(a[..., 1:], Nf, Nt, dt)


def test_inputs_are_not_mutated(rng):
    Nf, Nt, dt = 8, 16, 0.5
    w = rng.standard_normal((Nf + 1, Nt))
    w0 = w.copy()
    T.inverse(w, Nf, Nt, dt)
    assert np.array_equal(w, w0)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_transform_shim.py 2>&1 | grep -E "^E +ValueError: x_rfft|passed|failed"`

Expected:
```
E           ValueError: x_rfft must have shape (65,) for Nf=8, Nt=16; got (3, 2, 65)
17 failed, 1 passed in …
```
The line shown is the new test's failure. The 16 parametrised oracle tests fail too, on their stacked pairs, with the one-record shape messages (for example `x must have shape (128,) = (Nf*Nt,) for Nf=8, Nt=16; got (3, 128)` and `wmn must have shape (9, 16); got (3, 9, 16)`). `test_inputs_are_not_mutated` passes.

- [ ] **Step 2: Fold the four entry points and delete the others (`src/domainia/wilson/transform.py`)**

Two imports lose their last user: `index_array` (used only by `_check_interior_bands`) and `carrier_phase` (used only by `band_atoms` and `_band_phase`). The helper docstrings that named `forward_many` or `inverse_many` now say only that the helper is shape-generic. `base_atom_spectra`'s docstring no longer says "bit-identical" (the replay's gate has been 1e-12 since Task 1).

Edit 1, `src/domainia/wilson/transform.py`. Old:

```python
from ._checks import REAL_KINDS, index_array, require_dtype, require_int
from .grid import (
    active_mask,
    carrier_phase,
    n_samples,

```

New:

```python
from ._checks import REAL_KINDS, require_dtype, require_int
from .grid import (
    active_mask,
    n_samples,

```

Edit 2, `src/domainia/wilson/transform.py`. Old:

```python
__all__ = [
    "atom_spectrum",
    "band_atoms",
    "base_atom_spectra",
    "forward",
    "forward_bands",
    "forward_many",
    "forward_time",
    "forward_time_many",
    "inverse",
    "inverse_bands",
    "inverse_many",
    "inverse_time",
    "inverse_time_many",
]
```

New:

```python
__all__ = [
    "atom_spectrum",
    "base_atom_spectra",
    "forward",
    "forward_time",
    "inverse",
    "inverse_time",
]
```

Edit 3, `src/domainia/wilson/transform.py`. Old:

```python
    same index arrays under ``...`` indexing.  :func:`forward_many` used to
    call this once per row inside a list comprehension and ``np.stack`` the
    results -- which left the two edge bands unbatched and cost **85% of
    ``forward_many``'s runtime** at ``Nf=8, Nt=16, B=512``, the small-grid
    many-series case the batched path exists for.

```

New:

```python
    same index arrays under ``...`` indexing.

```

Edit 4, `src/domainia/wilson/transform.py`. Old:

```python
    Shape-generic in the leading axes, so :func:`inverse` and
    :func:`inverse_many` share it.

```

New:

```python
    Shape-generic in the leading axes.

```

Edit 5, `src/domainia/wilson/transform.py`. Old:

```python
    Shape-generic in the leading axes: ``X`` may be ``(N//2+1,)`` or
    ``(B, N//2+1)`` and ``wmn_work`` correspondingly, so :func:`inverse` and
    :func:`inverse_many` run the same code and cannot drift apart.

```

New:

```python
    Shape-generic in the leading axes: ``X`` is ``(..., N//2+1)`` and ``wmn_work``
    ``(..., Nf+1, Nt)``.

```

Edit 6, `src/domainia/wilson/transform.py`. Old:

```python
    The whole of the forward algorithm, shape-generic in the leading axes, so
    :func:`forward` and :func:`forward_many` are thin validation wrappers over
    one implementation rather than two copies of it.

```

New:

```python
    The whole of the forward algorithm, shape-generic in the leading axes;
    :func:`forward` is a thin validation wrapper over it.

```

Edit 7, `src/domainia/wilson/transform.py`. Old:

```python
    x_rfft : ndarray, complex, shape (N//2+1,)
        One-sided DFT of the real time series, as returned by
        ``np.fft.rfft(x)`` (no $\\delta t$ scaling).

```

New:

```python
    x_rfft : ndarray, complex, shape (..., N//2+1)
        One-sided DFT of the real time series, as returned by
        ``np.fft.rfft(x)`` (no $\\delta t$ scaling).  Any leading axes: each
        slice is transformed on its own, and a stack goes through one batched FFT.

```

Edit 8, `src/domainia/wilson/transform.py`. Old:

```python
    ndarray, float, shape (Nf+1, Nt)
        Real WDM coefficients.
```

New:

```python
    ndarray, float, shape (..., Nf+1, Nt)
        Real WDM coefficients.
```

Edit 9, `src/domainia/wilson/transform.py`. Old:

```python
        If ``Nf`` or ``Nt`` is not positive and even, if ``x_rfft``
        does not have length ``N//2 + 1``, or if it holds a ``nan`` or

```

New:

```python
        If ``Nf`` or ``Nt`` is not positive and even, if the last axis of
        ``x_rfft`` does not have length ``N//2 + 1``, or if it holds a ``nan`` or

```

Edit 10, `src/domainia/wilson/transform.py`. Old:

```python
    >>> bool(np.isclose(np.sum(w**2), dt * np.sum(x**2)))  # Parseval
    True
    """

```

New:

```python
    >>> bool(np.isclose(np.sum(w**2), dt * np.sum(x**2)))  # Parseval
    True
    >>> xs = np.random.default_rng(1).standard_normal((2, 3, Nf * Nt))   # leading axes
    >>> wilson.forward(np.fft.rfft(xs), Nf, Nt, dt).shape
    (2, 3, 9, 16)
    """

```

Edit 11, `src/domainia/wilson/transform.py`. Old:

```python
    x_rfft = np.asarray(x_rfft)
    if x_rfft.shape != (N // 2 + 1,):
        raise ValueError(
            f"x_rfft must have shape ({N // 2 + 1},) for Nf={Nf}, Nt={Nt}; "

```

New:

```python
    x_rfft = np.asarray(x_rfft)
    if x_rfft.shape[-1:] != (N // 2 + 1,):
        raise ValueError(
            f"x_rfft must have shape (..., {N // 2 + 1}) for Nf={Nf}, Nt={Nt}; "

```

Edit 12, `src/domainia/wilson/transform.py`. Old:

```python
    wmn : ndarray, float, shape (Nf+1, Nt)
        WDM coefficient grid.

```

New:

```python
    wmn : ndarray, float, shape (..., Nf+1, Nt)
        WDM coefficient grid, with any leading axes.

```

Edit 13, `src/domainia/wilson/transform.py`. Old:

```python
    ndarray, complex, shape (N//2+1,)
        Reconstructed one-sided DFT (compatible with ``np.fft.irfft``).

```

New:

```python
    ndarray, complex, shape (..., N//2+1)
        Reconstructed one-sided DFT (compatible with ``np.fft.irfft``).

```

Edit 14, `src/domainia/wilson/transform.py`. Old:

```python
    if wmn.shape != (Nf + 1, Nt):
        raise ValueError(f"wmn must have shape ({Nf + 1}, {Nt}); got {wmn.shape}")

```

New:

```python
    if wmn.shape[-2:] != (Nf + 1, Nt):
        raise ValueError(f"wmn must have shape (..., {Nf + 1}, {Nt}); got {wmn.shape}")

```

Edit 15, `src/domainia/wilson/transform.py`. Old:

```python
    X = np.zeros(N // 2 + 1, dtype=complex)

```

New:

```python
    X = np.zeros((*wmn.shape[:-2], N // 2 + 1), dtype=complex)

```

Edit 16, `src/domainia/wilson/transform.py`. Old:

```python
    x : ndarray, float, shape (N,)
        Real time series with $N = N_f N_t$ samples.

```

New:

```python
    x : ndarray, float, shape (..., N)
        Real time series with $N = N_f N_t$ samples, with any leading axes.

```

Edit 17, `src/domainia/wilson/transform.py`. Old:

```python
    ndarray, float, shape (Nf+1, Nt)

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if ``x`` does not have ``Nf*Nt`` samples, or
        if it holds a ``nan`` or ``inf``.  Gapped detector data often arrives

```

New:

```python
    ndarray, float, shape (..., Nf+1, Nt)

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if the last axis of ``x`` does not have ``Nf*Nt``
        samples, or if it holds a ``nan`` or ``inf``.  Gapped detector data often arrives

```

Edit 18, `src/domainia/wilson/transform.py`. Old:

```python
    if x.shape != (expected,):
        raise ValueError(
            f"x must have shape ({expected},) = (Nf*Nt,) for Nf={Nf}, Nt={Nt}; "

```

New:

```python
    if x.shape[-1:] != (expected,):
        raise ValueError(
            f"x must have shape (..., {expected}) = (..., Nf*Nt) for Nf={Nf}, Nt={Nt}; "

```

Edit 19, `src/domainia/wilson/transform.py`. Old:

```python
    wmn : ndarray, float, shape (Nf+1, Nt)
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, float, shape (N,)

```

New:

```python
    wmn : ndarray, float, shape (..., Nf+1, Nt)
    Nf, Nt : int
    dt : float
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, float, shape (..., N)

```

Edit 20, `src/domainia/wilson/transform.py`. Old:

```python
# Atoms and band-limited analysis

```

New:

```python
# Atoms

```

Edit 21, `src/domainia/wilson/transform.py`. Old:

```python
    even number of columns (:func:`atom_spectrum`).  ``dt * inverse_many(e)``:
    :func:`inverse` returns ``rfft(W_a^{-1} e) = rfft(atom) / dt``.  With the default
    window (:data:`domainia.wilson.MEYER`) these spectra and the Eq. 45 constants built from
    them are bit-identical to the Meyer baseline (pinned by
    ``tests/wilson/test_meyer_baseline.py``).

```

New:

```python
    even number of columns (:func:`atom_spectrum`).  ``dt * inverse(e)``:
    :func:`inverse` returns ``rfft(W_a^{-1} e) = rfft(atom) / dt``.  With the default
    window (:data:`domainia.wilson.MEYER`) these spectra and the Eq. 45 constants built from
    them are the Meyer baseline's (``tests/wilson/test_meyer_baseline.py``).

```

Edit 22, `src/domainia/wilson/transform.py`. Old:

```python
            out[m, a] = dt * inverse_many(e[None], Nf, Nt, dt, window=window)[0]

```

New:

```python
            out[m, a] = dt * inverse(e, Nf, Nt, dt, window=window)

```

In `src/domainia/wilson/transform.py`, delete these by the named-definition rule:
- `forward_many` (62 lines)
- `forward_time_many` (46 lines)
- `inverse_many` (58 lines)
- `inverse_time_many` (37 lines)
- `_check_interior_bands` (5 lines)
- `band_atoms` (35 lines)
- `_band_phase` (8 lines)
- `_band_windows` (10 lines)
- `forward_bands` (42 lines)
- `inverse_bands` (52 lines)

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -cE "_many|forward_bands|inverse_bands|band_atoms|_band_phase|_band_windows|_check_interior_bands|index_array|carrier_phase" src/domainia/wilson/transform.py`

Expected: `0`

- [ ] **Step 3: The callers in src**

`wilson/__init__.py` drops the four `_many` names. `evolutionary.py` and `_transform.py` call the folded names, as the table above shows. In `wrap_fractions` the comment column stays aligned with the next line's.

Edit 1, `src/domainia/wilson/__init__.py`. Old:

```python
* :func:`forward`, :func:`inverse`
* :func:`forward_time`, :func:`inverse_time` (time-domain wrappers)
* :func:`forward_many`, :func:`inverse_many` and their ``_time`` wrappers
  (batched over a stack of series sharing one grid -- one FFT for the whole
  stack, bit-identical to looping)

```

New:

```python
* :func:`forward`, :func:`inverse` (any leading axes: a stack of series sharing one
  grid goes through one FFT)
* :func:`forward_time`, :func:`inverse_time` (time-domain wrappers)

```

Edit 2, `src/domainia/wilson/__init__.py`. Old:

```python
from .transform import (
    forward,
    forward_many,
    forward_time,
    forward_time_many,
    inverse,
    inverse_many,
    inverse_time,
    inverse_time_many,
)

```

New:

```python
from .transform import forward, forward_time, inverse, inverse_time

```

Edit 3, `src/domainia/wilson/__init__.py`. Old:

```python
    "forward",
    "forward_many",
    "forward_time",
    "forward_time_many",
    "inverse",
    "inverse_many",
    "inverse_time",
    "inverse_time_many",

```

New:

```python
    "forward",
    "forward_time",
    "inverse",
    "inverse_time",

```

Edit 4, `src/domainia/wilson/evolutionary.py`. Old:

```python
    forward_time_many,
    inverse_time_many,

```

New:

```python
    forward_time,
    inverse_time,

```

Edit 5, `src/domainia/wilson/evolutionary.py`. Old:

```python
    X = np.atleast_2d(inverse_time_many(full, Nf, Nt, dt, window=window) * chi)
    return forward_time_many(X, Nf, Nt, dt, window=window)[:, mask]

```

New:

```python
    X = inverse_time(full, Nf, Nt, dt, window=window) * chi
    return forward_time(X, Nf, Nt, dt, window=window)[:, mask]

```

Edit 6, `src/domainia/wilson/evolutionary.py`. Old:

```python
            e = np.zeros((1, Nf + 1, Nt))
            e[0, m, a] = 1.0
            g2 = inverse_time_many(e, Nf, Nt, dt, window=window)[0] ** 2  # base atom, centre a Nf

```

New:

```python
            e = np.zeros((Nf + 1, Nt))
            e[m, a] = 1.0
            g2 = inverse_time(e, Nf, Nt, dt, window=window) ** 2        # base atom, centre a Nf

```

Edit 7, `src/domainia/wilson/evolutionary.py`. Old:

```python
    A = forward_time_many(np.ascontiguousarray(G.T), Nf, Nt, dt,
                          window=window)[:, active_mask(Nf, Nt)].T

```

New:

```python
    A = forward_time(np.ascontiguousarray(G.T), Nf, Nt, dt,
                     window=window)[:, active_mask(Nf, Nt)].T

```

Edit 8, `src/domainia/_transform.py`. Delete these lines:

```python
        lead = y.shape[:y.ndim - (2 if source == "wdm" else 1)]
```

Edit 9, `src/domainia/_transform.py`. Old:

```python
            R = wilson.inverse_many(y.reshape(-1, sg[0] + 1, sg[1]), *sg, dt,
                                    window=window).reshape(lead + (N // 2 + 1,))

```

New:

```python
            R = wilson.inverse(y, *sg, dt, window=window)

```

Edit 10, `src/domainia/_transform.py`. Old:

```python
            y = wilson.forward_many(R.reshape(-1, N // 2 + 1), *tg, dt,
                                    window=window).reshape(lead + (tg[0] + 1, tg[1]))

```

New:

```python
            y = wilson.forward(R, *tg, dt, window=window)

```

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "_many|forward_bands|inverse_bands|band_atoms|_band_phase|_band_windows|_check_interior_bands" src --include='*.py'`

Expected: no output.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_transform_shim.py tests/test_transform.py src/domainia/wilson 2>&1 | tail -1`

Expected: `78 passed in …`. That is 18 in the shim, 5 in `tests/test_transform.py` and 55 doctests (59 before, less the four `_many` doctests).

- [ ] **Step 4: The replay calls the folded kernels under the recorded names (`tests/wilson/meyer_cases.py`)**

Edit 1, `tests/wilson/meyer_cases.py`. Old:

```python
    Wm = np.stack([wl.forward(r, Nf, Nt, dt) for r in Xr])

```

New:

```python
    Wm = wl.forward(Xr, Nf, Nt, dt)

```

Edit 2, `tests/wilson/meyer_cases.py`. Old:

```python
        ("transform.forward_many", lambda kw: Tr.forward_many(Xr, Nf, Nt, dt, **kw)),
        ("transform.forward_time", lambda kw: Tr.forward_time(x, Nf, Nt, dt, **kw)),
        ("transform.forward_time_many", lambda kw: Tr.forward_time_many(X, Nf, Nt, dt, **kw)),
        ("transform.inverse", lambda kw: Tr.inverse(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_many", lambda kw: Tr.inverse_many(Wm, Nf, Nt, dt, **kw)),
        ("transform.inverse_time", lambda kw: Tr.inverse_time(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_time_many", lambda kw: Tr.inverse_time_many(Wm, Nf, Nt, dt, **kw)),

```

New:

```python
        # The stacks (Xr, X, Wm) keep the recorded names of the batched entry points that
        # the kernels' leading axes replaced.
        ("transform.forward_many", lambda kw: Tr.forward(Xr, Nf, Nt, dt, **kw)),
        ("transform.forward_time", lambda kw: Tr.forward_time(x, Nf, Nt, dt, **kw)),
        ("transform.forward_time_many", lambda kw: Tr.forward_time(X, Nf, Nt, dt, **kw)),
        ("transform.inverse", lambda kw: Tr.inverse(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_many", lambda kw: Tr.inverse(Wm, Nf, Nt, dt, **kw)),
        ("transform.inverse_time", lambda kw: Tr.inverse_time(w, Nf, Nt, dt, **kw)),
        ("transform.inverse_time_many", lambda kw: Tr.inverse_time(Wm, Nf, Nt, dt, **kw)),

```

The exact check compares every kept replay array with the recorded baseline by dtype and `np.array_equal`. It is stricter than the replay's 1e-12 gate.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')"
```

Expected: `0 of 1408 arrays differ from the baseline`. The prototype prints the same with the FLOOR interpreter (`$TMPDIR/dom3-envs/floor312/bin/python` in place of the PY one).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_meyer_baseline.py 2>&1 | tail -1`

Expected: `5 passed in …`

- [ ] **Step 5: The other tests**

Edit 1, `tests/wilson/test_window_threading.py`. Old:

```python
    "_gather_bands", "forward", "forward_many", "forward_time", "forward_time_many", "inverse",
    "inverse_many", "inverse_time", "inverse_time_many", "base_atom_spectra", "atom_spectrum",
    "band_atoms", "forward_bands", "inverse_bands",

```

New:

```python
    "_gather_bands", "forward", "forward_time", "inverse", "inverse_time", "base_atom_spectra",
    "atom_spectrum",

```

Edit 2, `tests/wilson/test_window_threading.py`. Old:

```python
    ("inverse_many(e, Nf, Nt, dt, window=window if a == 0 else MEYER)", True),

```

New:

```python
    ("inverse(e, Nf, Nt, dt, window=window if a == 0 else MEYER)", True),

```

Edit 3, `tests/wilson/test_window_threading.py`. Old:

```python
    many = wilson.forward_time_many(np.stack([x, -x]), Nf, Nt, dt, window=window)

```

New:

```python
    many = wilson.forward_time(np.stack([x, -x]), Nf, Nt, dt, window=window)

```

Edit 4, `tests/wilson/test_wrong_kinds.py`. Delete these lines:

```python
_BANDS = np.array([2, 3])
_W_BANDS = _T.forward_bands(_Z, 0, _BANDS, 8, 16, 1.0)
```

Edit 5, `tests/wilson/test_wrong_kinds.py`. Delete these lines:

```python
    "forward_bands k0": (lambda v: _T.forward_bands(_Z, v, _BANDS, 8, 16, 1.0), 0),
    "inverse_bands k0": (lambda v: _T.inverse_bands(_W_BANDS, v, 65, _BANDS, 8, 16, 1.0), 0),
    "inverse_bands n_bins": (lambda v: _T.inverse_bands(_W_BANDS, 0, v, _BANDS, 8, 16, 1.0), 65),
```

Edit 6, `tests/wilson/test_wrong_kinds.py`. Old:

```python
    (lambda: _T.forward_bands(_Z, 50, _BANDS, 8, 16, 1.0), r"^the bins \[50, 115\) do not cover"),
    (lambda: _T.inverse_bands(_W_BANDS, 0, 20, _BANDS, 8, 16, 1.0), r"^the bins \[0, 20\) do not cover"),
    (lambda: _T.atom_spectrum(2, 16, 8, 16, 1.0), r"^n must lie in 0\.\.15"),
    (lambda: wilson.empirical_variance(8, 16, 1.0, np.ones_like, 0), "^n_realizations must be >= 1, got 0"),
], ids=["forward_bands k0", "inverse_bands n_bins", "atom_spectrum n", "empirical_variance"])

```

New:

```python
    (lambda: _T.atom_spectrum(2, 16, 8, 16, 1.0), r"^n must lie in 0\.\.15"),
    (lambda: wilson.empirical_variance(8, 16, 1.0, np.ones_like, 0), "^n_realizations must be >= 1, got 0"),
], ids=["atom_spectrum n", "empirical_variance"])

```

Edit 7, `tests/wilson/test_wrong_kinds.py`. Delete these lines:

```python
    "inverse_many wmn": (lambda a: wilson.inverse_many(a, 8, 16, 1.0), _W[None], "wmn"),
```

Edit 8, `tests/wilson/test_wrong_kinds.py`. Delete these lines:

```python
    "inverse_time_many wmn": (lambda a: wilson.inverse_time_many(a, 8, 16, 1.0), _W[None], "wmn"),
    "inverse_bands w": (lambda a: _T.inverse_bands(a, 0, 65, _BANDS, 8, 16, 1.0), _W_BANDS, "w"),
```

Edit 9, `tests/wilson/test_wrong_kinds.py`. Delete these lines:

```python
        "forward_many": (lambda a: wilson.forward_many(a, 8, 16, 1.0), _Z[None]),
```

Edit 10, `tests/wilson/test_wrong_kinds.py`. Delete these lines:

```python
        "forward_time_many": (lambda a: wilson.forward_time_many(a, 8, 16, 1.0), x[None]),
        "forward_bands": (lambda a: _T.forward_bands(a, 0, _BANDS, 8, 16, 1.0), _Z),
```

Edit 11, `tests/wilson/test_grid_bands.py`. Old:

```python
"""Packing helpers, the Hz window, atoms and band-limited analysis."""
```

New:

```python
"""Packing helpers, the Hz window and the atoms."""
```

Edit 12, `tests/wilson/test_grid_bands.py`. Old:

```python
from domainia.wilson.transform import (
    atom_spectrum,
    band_atoms,
    base_atom_spectra,
    forward,
    forward_bands,
    forward_time,
    inverse,
    inverse_bands,
)

```

New:

```python
from domainia.wilson.transform import atom_spectrum, base_atom_spectra, forward_time

```

Edit 13, `tests/wilson/test_grid_bands.py`. Old:

```python
    """phi and atoms computed by the frozen galactic-binary code, from a temporary copy."""
```

New:

```python
    """phi computed by the frozen galactic-binary code, from a temporary copy."""
```

Edit 14, `tests/wilson/test_grid_bands.py`. Delete these lines:

```python
        "from geometry import band_atoms\n"
```

Edit 15, `tests/wilson/test_grid_bands.py`. Delete these lines:

```python
        "    res[f'atoms{i}'] = band_atoms(np.array([1, 2, Nf // 2, Nf - 1]), Nt, phi)\n"
```

Edit 16, `tests/wilson/test_grid_bands.py`. Old:

```python
def test_window_and_atoms_are_bitwise_the_gb_copies(tmp_path):
    ref = _gb_reference(tmp_path, ATOM_GRIDS)
    for i, (Nf, Nt, dt) in enumerate(ATOM_GRIDS):
        assert np.array_equal(phi_centered(Nf, Nt, dt), ref[f"phi{i}"])
        bands = np.array([1, 2, Nf // 2, Nf - 1])
        assert np.array_equal(band_atoms(bands, Nf, Nt, dt), ref[f"atoms{i}"])

```

New:

```python
def test_window_is_bitwise_the_gb_copy(tmp_path):
    ref = _gb_reference(tmp_path, ATOM_GRIDS)
    for i, (Nf, Nt, dt) in enumerate(ATOM_GRIDS):
        assert np.array_equal(phi_centered(Nf, Nt, dt), ref[f"phi{i}"])

```

Edit 17, `tests/wilson/test_wrong_kinds.py`. Old:

```python
    """``int()``, ``range()`` or a parity test took these silently (``k0="0"`` and
    ``carrier_phase(2.5, 4)`` ran), or ``True`` passed as 1."""
```

New:

```python
    """``int()``, ``range()`` or a parity test took these silently (``carrier_phase(2.5, 4)``
    ran), or ``True`` passed as 1."""
```

Edit 18, `tests/wilson/vendored/test_evolutionary.py`. Old:

```python
    np.testing.assert_allclose(g.forward_many(x), x @ Wa.T, atol=1e-12)
    np.testing.assert_allclose(g.forward(x[0]), forward_time(x[0], NF, NT, DT)[g.mask], atol=0)
    # round trip: W^{-1} = W^T / dt
    np.testing.assert_allclose(g.inverse_many(g.forward_many(x)), x, atol=1e-12)

```

New:

```python
    np.testing.assert_allclose(np.stack([g.forward(r) for r in x]), x @ Wa.T, atol=1e-12)
    np.testing.assert_allclose(g.forward(x[0]), forward_time(x[0], NF, NT, DT)[g.mask], atol=0)

```

In `tests/wilson/test_window_threading.py`, delete these by the named-definition rule:
- `test_batched_inverse_is_the_per_row_inverse_and_round_trips` (8 lines, with its decorators)
- `test_forward_bands_matches_its_atoms_and_forward` (14 lines, with its decorators)
- `test_inverse_bands_is_the_adjoint_of_forward_bands` (16 lines, with its decorators)

In `tests/wilson/test_wrong_kinds.py`, delete these by the named-definition rule:
- `_BAND_SLOTS` (5 lines, plus the `#:` line above it)
- `test_bands_of_the_wrong_dtype_raise_type_error` (9 lines, with its decorators)
- `test_bands_of_non_integral_floats_raise_value_error` (7 lines, with its decorators)
- `test_bands_take_integer_and_integral_float_arrays` (7 lines, with its decorators)

In `tests/wilson/test_grid_bands.py`, delete these by the named-definition rule:
- `test_forward_bands_is_the_dense_atom_analysis` (16 lines, with its decorators)
- `BASELINE_GRIDS` (1 line, plus the `#:` line above it)
- `test_forward_bands_is_the_transform_on_the_bands` (14 lines, with its decorators)
- `test_inverse_bands_is_the_inverse_on_its_bins` (14 lines, with its decorators)
- `test_inverse_bands_is_the_adjoint` (17 lines)
- `test_forward_bands_rejects_non_finite_bins` (10 lines)
- `test_inverse_bands_rejects_complex_and_non_finite_cells` (9 lines)

Run: `cd /Users/aaron/Documents/lisa/domainia && for f in test_window_threading test_wrong_kinds test_grid_bands vendored/test_evolutionary test_transform_shim; do PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/$f.py 2>&1 | tail -1; done`

Expected:
```
401 passed in …
117 passed in …
28 passed, 1 skipped in …
8 passed in …
18 passed in …
```

The same loop with the FLOOR interpreter prints the same five counts. The skip is `test_window_is_bitwise_the_gb_copy` (`DOMAINIA_ORACLE_ROOT is not set`), which skipped before this task too.

- [ ] **Step 6: Grep gate**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "_many|forward_bands|inverse_bands|band_atoms|_band_phase|_band_windows|_check_interior_bands" src tests --include='*.py' --exclude-dir=oracle | LC_ALL=C sort`

Expected: exactly these 6 lines, each holding recorded baseline key names: the four rewired case names, and the two `DROPPED` lines with the key prefixes of the band entry points that Task 1 set aside:
```
tests/wilson/meyer_cases.py:128:        ("transform.forward_many", lambda kw: Tr.forward(Xr, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:130:        ("transform.forward_time_many", lambda kw: Tr.forward_time(X, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:132:        ("transform.inverse_many", lambda kw: Tr.inverse(Wm, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:134:        ("transform.inverse_time_many", lambda kw: Tr.inverse_time(Wm, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:44:           "wilson::inner_product.", "wilson::transform.band_atoms::",
tests/wilson/meyer_cases.py:45:           "wilson::transform.forward_bands::", "wilson::transform.inverse_bands::")
```

- [ ] **Step 7: Gates**

The two suite runs are independent and can run in parallel. There is no NumPy lane: Task 5 removed the last JAX code.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1043 passed, 3 skipped`. It was `1186 passed, 3 skipped`. 144 items go and 1 comes: 60 in `test_window_threading.py`, 62 in `test_wrong_kinds.py`, 18 in `test_grid_bands.py` and 4 doctests go; the shim gains 1.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1043 passed, 3 skipped`.

Neither log contains `warning` or `error` (`grep -ciE "warning|error"` prints `0`).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. The FLOOR interpreter prints the same. `tests/wilson/test_meyer_baseline.py` passes inside both full runs.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 5219 lines / 1956 code
tests 4337 lines (+ oracle 4135)
```
That is −405 src lines (−169 code) and −191 test lines. src is now inside the spec's budget for before the docstring trim (at most about 5,400 lines and 2,150 code), and tests are inside theirs (at most about 5,700).

- [ ] **Step 8: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/wilson/transform.py src/domainia/wilson/__init__.py src/domainia/wilson/evolutionary.py src/domainia/_transform.py tests/wilson/meyer_cases.py tests/wilson/test_transform_shim.py tests/wilson/test_window_threading.py tests/wilson/test_wrong_kinds.py tests/wilson/test_grid_bands.py tests/wilson/vendored/test_evolutionary.py && git status --short | grep -vc "^M "`

Expected: `0`. Every status line is a staged modification, so nothing is unstaged or untracked.

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Fold the wilson kernels 8 -> 4; delete the band-limited entry points" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `10 files changed, 91 insertions(+), 687 deletions(-)`

Size after Task 7 (prototype): src 5219 lines / 1956 code; tests 4337 (+ oracle 4135).

---

### Task 8: `frequency_covariance` and the K×K model check and projection (spec §3.1, §3.2, §3.4, §6.2)

**Goal:** `domainia.frequency_covariance(S, *, N, dt)` evaluates a stationary K×K noise model as its `(K, K, N//2+1)` covariance per rfft bin. It rests on one private helper, `_covariance._hermitian`, which makes spec §3.1's per-point check and the projection `H = ½(S + Sᴴ)`. Task 9's `wdm_covariance` reuses that helper per chunk. Written test-first.

Assumes the tree after Task 7, whose commit is "Fold the wilson kernels 8 -> 4; delete the band-limited entry points".

**How it works** (spec §3.1, §3.2, §3.4).
- **`_hermitian(out, points, name)`**, the check and the projection. `out` is a model's raw output, `points` is the shape of the grid it was evaluated on, and `name` is used in messages (`"S(f)"` here, and `"S(f, t)"` in Task 9). The checks run in this order:
  - The output must be numeric, of dtype kind `i`, `u`, `f` or `c` (`require_dtype`). Bool, str and object outputs raise `TypeError`: `S(f) must be a numeric array, got dtype bool`.
  - The shape must be `(K, K, *p)` with K ≥ 1. Each axis of `p` equals the matching axis of `points` or is 1. Otherwise it raises `ValueError`: `S(f) must have shape (K, K, 65) with K >= 1 (an axis after the first two may be 1); got (2, 3, 65)`.
    - K is read from the output.
    - A bare `(K, K)` or a scalar is refused, because NumPy would align it with the trailing grid axes. A constant model returns `(K, K, 1)`.
  - Every entry must be finite, or it raises `ValueError`: `S(f) is not finite at (0, 0, 0)`. This check comes before the tolerance, because a NaN would make `tol` NaN, and every comparison with NaN is False.
    - `sanitize_psd`'s `+inf → 0` mapping does not apply (spec §3.1).
  - At each point, `tol = TOL · max_cd |S_cd|` there, with `TOL = 1e-12`. These are refused with `ValueError`, naming the first entry:
    - `|S_ab − conj(S_ba)| > tol`, for a ≠ b;
    - `|Im S_aa| > tol`;
    - `−Re S_aa > tol`.

    For example: `S(f) at point (0,): |S[0, 1] - conj(S[1, 0])| is 1e-09 of max |S_cd| there; at most 1e-12 is accepted`.
    - The comparisons are strict, so a point where every entry is 0 (tol = 0) passes without a special case.
    - The Hermitian rule leaves out a = b. There `|S_aa − conj(S_aa)| = 2|Im S_aa|`, so it would refuse a diagonal imaginary part between tol/2 and tol, which the diagonal rule accepts.
  - The result is `H = 0.5 * (S + conj(S.swapaxes(0, 1)))`, after a cast to complex128 for a complex output and to float64 otherwise.
    - Floating-point addition commutes exactly, so `H[a, b] == conj(H[b, a])` holds element for element, and `Im H[a, a]` is exactly 0.
    - `H` keeps the output's own shape: a size-1 axis is never expanded. This is what lets Task 9 keep a stationary model's size-1 time axis.
- **`frequency_covariance(S, *, N, dt)`**:
  - `S`: `require_callable` (`TypeError`: `S must be callable, got ndarray`).
  - `N`: `require_int` (`TypeError`), then `N >= 1` (`ValueError`).
  - `dt`: `validate_dt`.
  - `S` is called once, with `f = numpy.fft.rfftfreq(N, dt)`, which is `f_k = k/(N·dt)` and the frequencies Enchilada's `from_psd` uses. Then `H = _hermitian(S(f), f.shape, "S(f)")`.
  - The result is `C = (0.5 * N * dt) * H`, broadcast into a new complex128 `(K, K, N//2+1)` array. Its imaginary part is then set to 0 at DC, and at Nyquist when N is even, as `_transform.transform` does. That is the real part, so those two bins stay exactly Hermitian.
  - The orientation is `C[a, b, k] = E[X_a[k] conj(X_b[k])]`, with `X = dt·rfft(x)`.
- **Linearity** (§3.4) holds by construction. The check does not change values, and the projection and the scaling are linear.
- **What Task 9 reuses.** It reuses `_hermitian` (called once per chunk, with `points` the chunk's `(bands, Nt, J)` or similar, and `"S(f, t)"`) and the constant `TOL`. In `tests/test_covariance.py` it reuses:
  - the models `delayed`, `second`, `rounded` and `plus(a, b, rel)`. Each takes `f` of any shape. Task 9 wraps them as `lambda f, t: model(f)`.
  - `sigma(S, N, dt)`, spec §6.2's dense stationary Σ as `(K, K, N, N)`;
  - `error(got, want)`, spec §3.3's metric. It divides by `sqrt(want_aa·want_bb)`, so inactive WDM cells (zero diagonal) must be masked out before calling it;
  - the dict `ROUTES`, which the shared tests loop over. Task 9 adds a `"wdm"` entry.

**Choices the spec leaves open** (plan deviations):
- The Hermitian rule `|S_ab − conj(S_ba)| ≤ tol` is applied for a ≠ b only, and the diagonal has its own rule `|Im S_aa| ≤ tol`. Applied to a = b as well, the first rule would bound `|Im S_aa|` by tol/2.
- `f` is `numpy.fft.rfftfreq(N, dt)`, which is `k · (1/(N·dt))`, not the literal `k/(N·dt)`. The two differ by an ulp at most, and `rfftfreq` is what Enchilada's `from_psd` calls.
- The output must have both K axes and one axis per grid axis. So a scalar, or a bare `(K, K)` array, is refused even for K = 1, although §3.1's "broadcastable to `(K, K, *f.shape)`" would admit a scalar for K = 1.
- `H` is float64 for a real output and complex128 for a complex one, not complex128 always. Task 9 takes `Re H` either way, and a real evolutionary model's chunks stay half the size.
- The non-finite message is the module's own. `validate_finite`'s message is about data in a transform's FFT.
- The test model adds per-channel phases to the delayed cross-spectra of `kxk_stationary.py`. Without them, Im S at DC is exactly 0, and a mutant with no real part at DC passed every test.
- The rounding model is a complex `BᴴDB`, not the spec's real `BᵀDB`. So its diagonal also carries a rounding-level imaginary part, and the test checks that the projection removes it.

**Files:**
- Create:
  - `src/domainia/_covariance.py`: `TOL`, `_hermitian` and `frequency_covariance` (Step 2).
  - `tests/test_covariance.py`: three tests and the reusable helpers (Step 1).
- Modify:
  - `src/domainia/__init__.py`: the module docstring's Modules list, the import of `frequency_covariance` from `._covariance`, and `__all__` (Step 3).
  - `tests/wilson/test_window_threading.py`: `SCANNED` gains `"_covariance.py"` (Step 5).
  - `tests/test_package.py`: `PUBLIC` gains `"frequency_covariance"` (Step 6).
- Delete: none.
- No change: `README.md`, which Task 11 rewrites around the three functions, and `tests/wilson/meyer_cases.py`, because no kernel changes.
- Test: `tests/test_covariance.py`. Also affected: `tests/wilson/test_window_threading.py` and `tests/test_package.py`.

**Acceptance Criteria:**
- [ ] Step 1's run gives `2 errors` at collection, both `ImportError: cannot import name 'frequency_covariance' from 'domainia'`.
- [ ] `pytest -q tests/test_covariance.py` → `3 passed` in PY and in FLOOR.
- [ ] Step 4's probe prints `N=128 per-bin error 3.5e-15`, `N=75 per-bin error 2e-15` and `linearity 3.4e-16`, followed by the seven messages listed there, in PY. The FLOOR interpreter prints the same first four lines.
- [ ] `pytest -q tests/wilson/test_window_threading.py` → `402 passed` (it was 401), and `pytest -q tests/test_package.py` → `7 passed`.
- [ ] `grep -nEi "be=|float32|complex64|jax|solve|logdet|cholesky|inv\(|det\(" src/domainia/_covariance.py` prints nothing.
- [ ] The exact replay check prints `0 of 1408 arrays differ from the baseline`, and `meyer_cases.py --check` prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`, in PY and in FLOOR.
- [ ] Full suite (prototype): PY `1047 passed, 3 skipped`; FLOOR `1047 passed, 3 skipped`. No warnings and no collection errors.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1` → `3 passed in …`

**Steps:**

- [ ] **Step 1: Write the test (it fails first)**

Create `tests/test_covariance.py` with this text:

```python
"""Per-point K x K noise covariance from a model (domainia 3.0 spec, sections 3 and 6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import frequency_covariance

K, DT = 3, 5.0
#: Unequal levels: real, symmetric and positive definite.
LEVELS = np.array([[1.0, 0.6, 0.3], [0.6, 1.2, -0.4], [0.3, -0.4, 0.9]])
#: Unequal-arm delays [s] and phases [rad], which make the cross-spectra complex, at DC too.
DELAYS, PHASES = np.array([0.0, 7.3, -11.9]) * DT, np.array([0.0, 0.4, -1.1])
#: A complex mixing matrix for the loop-built model.
B = np.random.default_rng(20261114).standard_normal((K, K, 2)) @ [1.0, 1.0j]
#: The routes from a stationary model S(f) to a covariance.
ROUTES = {"frequency": lambda S, N=128: frequency_covariance(S, N=N, dt=DT)}


def psd(f):
    """A steep one-sided PSD [units^2/Hz]."""
    return 1e-2 * (1.0 + (2e-3 / (f + 1e-3)) ** 2) * (1.0 + (f / 0.05) ** 4)


def delayed(f, levels=LEVELS, delays=DELAYS, phases=PHASES, p=psd):
    """S_ab(f) = P(f) L_ab exp(i (theta_a(f) - theta_b(f))), theta_a(f) = 2 pi tau_a f + phi_a, for
    f of any shape: complex delayed cross-spectra with unequal levels, exactly Hermitian."""
    e = (slice(None), slice(None)) + (None,) * np.ndim(f)
    theta = 2 * np.pi * (delays[:, None] - delays)[e] * f + (phases[:, None] - phases)[e]
    return p(f) * levels[e] * np.exp(1j * theta)


def second(f):
    """Another such model, for linearity."""
    return delayed(f, np.array([[0.5, -0.2, 0.1], [-0.2, 0.7, 0.25], [0.1, 0.25, 0.4]]),
                   np.array([1.1, -4.0, 2.5]) * DT, np.array([0.3, 0.0, 0.7]),
                   lambda f: 3e-3 * (1.0 + (f / 0.02) ** 2))


def rounded(f):
    """S = B^H D(f) B summed entry by entry: Hermitian only to rounding, and its diagonal has
    a rounding-level imaginary part."""
    S = np.zeros((K, K) + np.shape(f), complex)
    for a in range(K):
        for b in range(K):
            S[a, b] = sum(np.conj(B[i, a]) * ((1.0 + i) * psd(f)) * B[i, b] for i in range(K))
    return S


def plus(a, b, rel):
    """``delayed`` with ``rel`` times each point's largest |S_cd| added to entry (a, b)."""
    def model(f):
        S = delayed(f)
        S[a, b] += rel * np.max(np.abs(S), axis=(0, 1))
        return S
    return model


def sigma(S, N, dt):
    """Spec 6.2's dense stationary reference, (K, K, N, N): Sigma[a, b, i, j] = R_ab((i - j) mod
    N), R_ab(tau) = Re[(1 / (N dt)) sum_k P_ab(k) e^{2 pi i k tau / N}], with P = S / 2 on the
    rfft bins, extended to all N bins by P_ab(N - k) = conj(P_ab(k))."""
    f = np.fft.rfftfreq(N, dt)
    P = 0.5 * np.asarray(S(f), complex)
    P = np.broadcast_to(P, P.shape[:2] + f.shape)
    R = np.fft.ifft(np.concatenate([P, np.conj(P[..., N - f.size:0:-1])], axis=-1)).real / dt
    i = np.arange(N)
    return R[..., (i[:, None] - i) % N]


def error(got, want):
    """Spec 3.3's metric: the largest |got_ab - want_ab| / sqrt(want_aa want_bb)."""
    d = np.abs(np.einsum("aa...->a...", want))
    return np.max(np.abs(got - want) / np.sqrt(d[:, None] * d[None]))


def test_frequency_covariance_is_the_per_bin_covariance_of_the_dense_reference():
    """E[X_a conj(X_b)] of X = dt rfft(x) under Sigma, at even and odd N, within 1e-12
    (measured: 3.5e-15 and 2.0e-15); real at DC and even-N Nyquist, where |Im S| is 33% and 49%
    of |S|; white noise has R(0) = S0 / (2 dt); a diagonal model gives Enchilada's from_psd."""
    white = lambda f: np.full((1, 1, 1), 4.0)
    np.testing.assert_allclose(sigma(white, 128, DT)[0, 0, :, 0],
                               np.r_[4.0 / (2 * DT), np.zeros(127)], rtol=1e-15, atol=1e-16)
    assert np.array_equal(ROUTES["frequency"](white), np.full((1, 1, 65), 128 * DT / 2 * 4.0 + 0j))
    level = np.array([1.0, 2.0, 0.5])
    diagonal = lambda f: np.eye(K)[:, :, None] * (level[:, None, None] * psd(f))
    for N in (128, 75):
        F = DT * np.fft.rfft(np.eye(N), axis=0)
        want = np.einsum("ki,abij,kj->abk", F, sigma(delayed, N, DT), F.conj(), optimize=True)
        got = ROUTES["frequency"](delayed, N)
        assert got.shape == (K, K, N // 2 + 1) and got.dtype == np.complex128
        assert error(got, want) < 1e-12
        assert not np.any(got[..., 0].imag) and np.any(got[..., -1].imag) == bool(N % 2)
        d = ROUTES["frequency"](diagonal, N)
        np.testing.assert_allclose(np.einsum("aak->ak", d), (N * DT / 2) * level[:, None]
                                   * psd(np.fft.rfftfreq(N, DT)), rtol=1e-15, atol=0)
        assert not np.any(d[~np.eye(K, dtype=bool)])


def test_linear_and_exactly_hermitian_and_a_zero_point_passes():
    """f(S1 + S2) = f(S1) + f(S2) to rounding (spec 3.4); a model Hermitian only to rounding, or
    off by less than the tolerance, gives an output that is exactly Hermitian with a real
    diagonal (3.1); a model that is 0 everywhere passes and gives 0."""
    S = rounded(np.fft.rfftfreq(128, DT))
    assert not np.array_equal(S, np.conj(np.swapaxes(S, 0, 1)))
    assert np.any(np.einsum("aak->ak", S).imag)
    for name, route in ROUTES.items():
        both = route(lambda f: delayed(f) + second(f))
        assert error(both, route(delayed) + route(second)) < 1e-14, name
        for C in (route(rounded), route(plus(0, 1, 5e-13)), route(plus(1, 1, 5e-13j))):
            assert np.array_equal(C, np.conj(np.swapaxes(C, 0, 1))), name
            assert not np.any(np.einsum("aa...->a...", C).imag), name
        assert not np.any(route(lambda f: 0.0 * delayed(f))), name


def test_every_refusal():
    """Each failure of the model check, made per point (spec 3.1), raises the right kind."""
    def dip(f):
        """``delayed``, 1e-6 as large at DC, with a 1e-16 asymmetry there: 1.7e-9 of that point's
        largest entry, though below 1e-15 of the model's largest."""
        S = delayed(f) * np.where(f == 0, 1e-6, 1.0)
        S[0, 1] += np.where(f == 0, 1e-16, 0.0)
        return S
    cases = [
        (np.eye(K), TypeError, "S must be callable"),
        (lambda f: delayed(f).real > 0, TypeError, "must be a numeric array, got dtype bool"),
        (lambda f: delayed(f).astype(str), TypeError, "numeric"),
        (lambda f: delayed(f).astype(object), TypeError, "numeric"),
        (psd, ValueError, r"must have shape \(K, K, "),
        (lambda f: 1.0, ValueError, r"got \(\)"),
        (lambda f: delayed(f)[:2], ValueError, r"got \(2, 3, "),
        (lambda f: delayed(f)[..., :-1], ValueError, "must have shape"),
        (plus(0, 0, np.nan), ValueError, "not finite"),
        (plus(2, 1, np.inf), ValueError, "not finite"),
        (plus(0, 1, 1e-9), ValueError, r"\|S\[0, 1\] - conj\(S\[1, 0\]\)\| is 1e-09 of"),
        (plus(1, 1, 1e-9j), ValueError, r"\|Im S\[1, 1\]\| is 1e-09 of"),
        (plus(2, 2, -3.0), ValueError, r"-Re S\[2, 2\] is 1 of"),
        (dip, ValueError, r"\|S\[0, 1\] - conj\(S\[1, 0\]\)\| is 1.7e-09 of"),
    ]
    for S, err, match in cases:
        for route in ROUTES.values():
            with pytest.raises(err, match=match):
                route(S)
    for kw, err, match in (({"N": 128.0}, TypeError, "N must be an integer"),
                           ({"N": 0}, ValueError, "N must be positive"),
                           ({"dt": 0.0}, ValueError, "dt must be")):
        with pytest.raises(err, match=match):
            frequency_covariance(delayed, **{"N": 128, "dt": DT, **kw})
```

What each test covers, against spec §6.2:
- `test_frequency_covariance_is_the_per_bin_covariance_of_the_dense_reference`:
  - "the stationary dense reference": for white noise S0 = 4, `R(0) = S0/(2·dt)` and every other lag is 0;
  - "`frequency_covariance`" bullet 1: the result equals `E[X_a conj(X_b)]`, the per-bin covariance of `X = dt·rfft(x)` under Σ, within 1e-12 by §3.3's metric. N is 128 (even) and 75 (odd). The model is K = 3 with unequal levels, delays and phases, from the probe `kxk_stationary.py`. The phases add an imaginary part at DC: |Im S|/|S| is 0.33 at DC and 0.49 at Nyquist. Without them, Im S is 0 at DC and the DC rule goes unchecked;
  - bullet 2: DC and even-N Nyquist entries are exactly real, and the odd-N last bin keeps its imaginary part;
  - bullet 3: a diagonal model gives `(T_obs/2)·S` on the diagonal (Enchilada's `from_psd`) to 1e-15, and exactly 0 off it;
  - a `(1, 1, 1)` model, a constant, is broadcast over the bins.
- `test_linear_and_exactly_hermitian_and_a_zero_point_passes` ("Shared", looping over `ROUTES`):
  - linearity, within 1e-14 by the metric;
  - a model that is Hermitian only to rounding (`rounded`, a loop-built `BᴴDB` with complex B) gives an output that is exactly Hermitian with an exactly real diagonal. The test first asserts that the model's output is neither exactly Hermitian nor exactly real on its diagonal. So do models off by less than the tolerance: `plus(0, 1, 5e-13)` and `plus(1, 1, 5e-13j)`;
  - an all-zero model passes and gives 0.
- `test_every_refusal` ("Shared": each validation failure gives the right exception kind):
  - one loop over 14 model cases, plus the `N` and `dt` checks;
  - `dip` checks that the tolerance is per point. It is 1e-6 as large at DC, with a 1e-16 asymmetry there. That is 1.7e-9 of the point's largest entry, so it is refused, though it is below 1e-15 of the model's largest entry, so a global tolerance would accept it.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1`

Expected: `2 errors in …`. Both are the collection error `ImportError: cannot import name 'frequency_covariance' from 'domainia'`. There are two because `--doctest-modules` (pyproject `addopts`) collects the file a second time, as a doctest module. Check with `… -m pytest -q tests/test_covariance.py 2>&1 | grep -c "^E   ImportError: cannot import name 'frequency_covariance'"`, which prints `2`.

- [ ] **Step 2: Create `src/domainia/_covariance.py`**

Create it with this text:

```python
"""Per-point K x K noise covariance from a model (domainia 3.0 spec, section 3).

A model returns the one-sided K x K cross-spectral density [data units^2/Hz], oriented so
that ``E[X_a conj(X_b)] = (T_obs / 2) S_ab(f_k)`` at an interior rfft bin, with
``S[b, a] = conj(S[a, b])``.  Each output is linear in the model, and exactly Hermitian with a
real diagonal at every point.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .wilson._checks import require_callable, require_dtype, require_int, validate_dt

__all__ = ["frequency_covariance"]

#: The model check's tolerance at a point, as a fraction of that point's largest |S_cd|.
TOL = 1e-12


def _hermitian(out: Any, points: tuple[int, ...], name: str) -> np.ndarray:
    """``H = (S + S^H) / 2`` at each point of a model's output ``S = out``, after the check.

    ``S`` must be numeric (else ``TypeError``), of shape ``(K, K, *p)`` with K >= 1 and ``p``
    equal to ``points`` up to axes of size 1, and finite (else ``ValueError``).  At each point,
    with ``tol = TOL * max_cd |S_cd|`` there, ``|S_ab - conj(S_ba)|`` (a != b), ``|Im S_aa|``
    and ``-Re S_aa`` must not exceed ``tol`` (else ``ValueError``, naming the first entry), so
    a point whose entries are all 0 passes.  ``H`` has ``S``'s shape, so a size-1 axis is never
    expanded, and is float64 for a real ``S``, complex128 otherwise.
    """
    S = np.asarray(out)
    require_dtype(name, S, "iufc", "a numeric array")
    if (S.ndim != 2 + len(points) or not 0 < S.shape[0] == S.shape[1]
            or any(s not in (1, p) for s, p in zip(S.shape[2:], points))):
        shape = ", ".join(map(str, points))
        raise ValueError(f"{name} must have shape (K, K, {shape}) with K >= 1 (an axis after "
                         f"the first two may be 1); got {S.shape}")
    if not np.isfinite(S).all():
        at = tuple(np.argwhere(~np.isfinite(S))[0].tolist())
        raise ValueError(f"{name} is not finite at {at}")
    S = S.astype(np.complex128 if S.dtype.kind == "c" else np.float64)
    peak = np.max(np.abs(S), axis=(0, 1))
    asym = np.abs(S - np.conj(np.swapaxes(S, 0, 1)))
    asym[np.diag_indices(S.shape[0])] = 0.0
    diag = np.moveaxis(np.diagonal(S), -1, 0)
    for what, value in (("|S[{0}, {1}] - conj(S[{1}, {0}])|", asym),
                        ("|Im S[{0}, {0}]|", np.abs(diag.imag)), ("-Re S[{0}, {0}]", -diag.real)):
        bad = np.argwhere(value > TOL * peak)
        if bad.size:
            at = tuple(bad[0].tolist())
            k = value.ndim - peak.ndim
            raise ValueError(f"{name} at point {at[k:]}: {what.format(*at[:k])} is "
                             f"{value[at] / peak[at[k:]]:.2g} of max |S_cd| there; at most "
                             f"{TOL:g} is accepted")
    return 0.5 * (S + np.conj(np.swapaxes(S, 0, 1)))


def frequency_covariance(S: Callable[[np.ndarray], Any], *, N: int, dt: float) -> np.ndarray:
    """The covariance per rfft bin of ``X = dt * numpy.fft.rfft(x)`` under a stationary model.

    ``S(f)`` is called once, with ``f = numpy.fft.rfftfreq(N, dt)`` (``f_k = k / (N dt)``), and
    returns shape ``(K, K, N // 2 + 1)``, or ``(K, K, 1)`` for a constant model; K is read from
    it.  It is checked and projected to ``H = (S + S^H) / 2`` per bin (:func:`_hermitian`).

    Returns ``C[a, b, k] = (N dt / 2) H_ab(f_k)``, complex128 of shape ``(K, K, N // 2 + 1)``:
    ``E[X_a[k] conj(X_b[k])]`` for a periodic record, the Whittle approximation otherwise.  It is
    real at DC and at even-``N`` Nyquist, where a real record has no imaginary part.

    Raises ``TypeError`` for a non-callable ``S``, a non-integer ``N`` or a non-numeric output,
    and ``ValueError`` for ``N < 1``, a bad ``dt`` or an output that fails the check.
    """
    require_callable("S", S)
    N = require_int("N", N)
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    validate_dt(dt)
    f = np.fft.rfftfreq(N, float(dt))
    H = _hermitian(S(f), f.shape, "S(f)")
    C = np.empty(H.shape[:2] + f.shape, np.complex128)
    C[...] = (0.5 * N * float(dt)) * H
    C[..., 0] = C[..., 0].real
    if N % 2 == 0:
        C[..., -1] = C[..., -1].real
    return C
```

- [ ] **Step 3: Re-export `frequency_covariance` from `src/domainia/__init__.py`**

Edit 1, the module docstring. Old:

```python
  (:mod:`domainia._transform`).
* Not imported here: :mod:`domainia.testing` (dense oracles and test windows).
```

New:

```python
  (:mod:`domainia._transform`).
* :func:`frequency_covariance`: a stationary K x K noise model as its covariance per rfft bin
  (:mod:`domainia._covariance`).
* Not imported here: :mod:`domainia.testing` (dense oracles and test windows).
```

Edit 2, the imports. Old:

```python
from . import wilson
from ._transform import transform
```

New:

```python
from . import wilson
from ._covariance import frequency_covariance
from ._transform import transform
```

Edit 3, `__all__`. Old:

```python
    "__version__",
    "transform",
```

New:

```python
    "__version__",
    "frequency_covariance",
    "transform",
```

- [ ] **Step 4: The new tests pass, and the measured values**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1`

Expected: `3 passed in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1`

Expected: `3 passed in …`

This probe prints the measured margins behind the tests' bounds (1e-12 per bin, 1e-14 for linearity), and the messages of seven refusals.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P - <<'EOF'
import numpy as np
import test_covariance as T
from domainia import frequency_covariance
for N in (128, 75):
    F = T.DT * np.fft.rfft(np.eye(N), axis=0)
    want = np.einsum("ki,abij,kj->abk", F, T.sigma(T.delayed, N, T.DT), F.conj(), optimize=True)
    print(f"N={N} per-bin error {T.error(frequency_covariance(T.delayed, N=N, dt=T.DT), want):.2g}")
r = T.ROUTES["frequency"]
print(f"linearity {T.error(r(lambda f: T.delayed(f) + T.second(f)), r(T.delayed) + r(T.second)):.2g}")
s = T.delayed(np.fft.rfftfreq(128, T.DT))
print("|Im S| / |S|: DC %.2f, Nyquist %.2f" % tuple(np.max(abs(s[..., k].imag)) / np.max(abs(s[..., k]))
                                                  for k in (0, -1)))
for S in (np.eye(3), lambda f: T.delayed(f).real > 0, lambda f: T.delayed(f)[:2], T.plus(0, 0, np.nan),
          T.plus(0, 1, 1e-9), T.plus(1, 1, 1e-9j), T.plus(2, 2, -3.0)):
    try:
        r(S)
    except (TypeError, ValueError) as e:
        print(f"{type(e).__name__}: {e}")
EOF
```

Expected:
```
N=128 per-bin error 3.5e-15
N=75 per-bin error 2e-15
linearity 3.4e-16
|Im S| / |S|: DC 0.33, Nyquist 0.49
TypeError: S must be callable, got ndarray
TypeError: S(f) must be a numeric array, got dtype bool
ValueError: S(f) must have shape (K, K, 65) with K >= 1 (an axis after the first two may be 1); got (2, 3, 65)
ValueError: S(f) is not finite at (0, 0, 0)
ValueError: S(f) at point (0,): |S[0, 1] - conj(S[1, 0])| is 1e-09 of max |S_cd| there; at most 1e-12 is accepted
ValueError: S(f) at point (0,): |Im S[1, 1]| is 1e-09 of max |S_cd| there; at most 1e-12 is accepted
ValueError: S(f) at point (0,): -Re S[2, 2] is 1 of max |S_cd| there; at most 1e-12 is accepted
```

The FLOOR interpreter prints the same first four lines.

Until Steps 5 and 6, the suite has two failures:
- `tests/wilson/test_window_threading.py::test_the_scan_covers_every_module_but_filter_and_windows`, because `_covariance.py` is a package module that is not in `SCANNED`;
- `tests/test_package.py::test_the_public_names`, because `__all__` is no longer `PUBLIC`.

The mutations below were each applied to `_covariance.py` in the prototype, and each makes `tests/test_covariance.py` fail:

| Mutation | Failing test |
|---|---|
| no real part at DC; none at even-N Nyquist; the real part also at odd N's last bin; the conjugate orientation `conj(H)`; `N·dt` in place of `N·dt/2` | `test_frequency_covariance_is_the_per_bin_covariance_of_the_dense_reference` |
| `return S` in place of the projection; `>=` in place of `>` (refuses an all-zero point) | `test_linear_and_exactly_hermitian_and_a_zero_point_passes` |
| a global tolerance in place of the per-point one; the diagonal not masked out of the Hermitian rule; no finite check; no negative-diagonal check; no diagonal-imaginary check | `test_every_refusal` |

- [ ] **Step 5: Scan the new module for the window (`tests/wilson/test_window_threading.py`)**

`test_the_scan_covers_every_module_but_filter_and_windows` requires every package module except `UNSCANNED` to be in `SCANNED`.

Old:

```python
           "testing.py", "__init__.py", "_transform.py")
```

New:

```python
           "testing.py", "__init__.py", "_transform.py", "_covariance.py")
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py 2>&1 | tail -1`

Expected: `402 passed in …`. It was 401. The new item is `test_no_kernel_call_drops_the_window[_covariance.py]`. `_covariance.py` calls no window consumer and names no window, so it passes.

- [ ] **Step 6: The public names (`tests/test_package.py`)**

Old:

```python
          "__version__", "transform", "wilson"}
```

New:

```python
          "__version__", "frequency_covariance", "transform", "wilson"}
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -1`

Expected: `7 passed in …`

- [ ] **Step 7: Gates**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -nEi "be=|float32|complex64|jax|solve|logdet|cholesky|inv\(|det\(" src/domainia/_covariance.py`

Expected: no output. The new module has no JAX, no float32 and no likelihood code.

The two suite runs are independent and can run in parallel. There is no NumPy lane: Task 5 removed the last JAX code.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1047 passed, 3 skipped`. That is Task 7's 1043, plus the 3 new tests and the new `SCANNED` item.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1047 passed, 3 skipped`.

Neither log contains `warning` or `error` (`grep -ciE "warning|error"` prints `0`).

The exact check compares every kept replay array with the recorded baseline, by dtype and `np.array_equal`. It is stricter than the replay's 1e-12 gate. No kernel changed, so it still holds.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')"
```

Expected: `0 of 1408 arrays differ from the baseline`. The FLOOR interpreter prints the same.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. The FLOOR interpreter prints the same. `tests/wilson/test_meyer_baseline.py` passes inside both full runs.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 5310 lines / 2005 code
tests 4486 lines (+ oracle 4135)
```
That is +91 src lines (+49 code) and +149 test lines.
- `_covariance.py` is 87 lines, and `__init__.py` gains 4 lines.
- `tests/test_covariance.py` is 149 lines.

src stays inside the spec's pre-trim budget (at most about 5,400 lines and 2,150 code), with 90 lines and 145 code lines to spare. Task 9's evaluator and chunking come next, so the line budget is tight until Task 10 deletes Eq. 39.

- [ ] **Step 8: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/_covariance.py src/domainia/__init__.py tests/test_covariance.py tests/test_package.py tests/wilson/test_window_threading.py && git status --short | grep -vc "^[AM] "`

Expected: `0`. Every status line is a staged addition or modification, so nothing is unstaged or untracked. (`grep -c` exits 1 when it prints 0, so this command does not continue with `&&`.)

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Add frequency_covariance and the K x K model check and projection" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `5 files changed, 242 insertions(+), 2 deletions(-)`

Size after Task 8 (prototype): src 5310 lines / 2005 code; tests 4486 (+ oracle 4135).

---

### Task 9: `wdm_covariance`, the K×K covariance per WDM cell (spec §3.3, §3.4, §6.2, Risks)

**Goal:** `domainia.wdm_covariance(S, *, Nf, Nt, dt, window=MEYER)` evaluates an evolutionary (or stationary) K×K noise model `S(f, t)` as its `(K, K, Nf+1, Nt)` float64 covariance per WDM cell. It has its own evaluator, which calls the model in chunks over bands and time columns and applies Task 8's check and projection to each chunk. Written test-first. This is the last task that adds code.

Assumes the tree after Task 8, whose commit is "Add frequency_covariance and the K x K model check and projection".

**How it works** (spec §3.3).
- **Arguments.** `S` must be callable (`require_callable`: `TypeError: S must be callable, got ndarray`). `validate_grid(Nf, Nt)` and `validate_dt(dt)` check the grid and `dt`. `sample_window(window, Nf, Nt, dt)` samples the window and checks it (`TypeError: window must be a domainia.wilson Window (Meyer, RampWindow, SampledWindow), got str`).
- **The frequency samples** are those of `noise.evolutionary_band_variance`, computed the same way:
  - `w = φ̂²` on the `J` slab bins where it is nonzero (`sup = np.flatnonzero(w)`). J is 25 on 16×32, 385 on 512×512 and 1537 on 2048×2048.
  - Band m samples `f = f_m + (j − Nt//2)/(N·dt)`. That is folded at DC by `abs`, and at Nyquist by `f → 2·f_Nyq − f` where `f > f_Nyq`. The result is `f`, of shape `(Nf+1, J)`.
  - `t_n = n·ΔT`, with `ΔT = Nf·dt` (`delta_T`).
- **The model's arguments.** `S` is called with `f[m0:m1, None]` of shape `(bands, 1, J)` and `t[None, n0:n1, None]` of shape `(1, columns, 1)`. So f varies along the last axis and t along the one before it (spec §3.1).
- **The first call.** `S` is first called on band 0 and the first two columns, and `_hermitian` checks the result against `points = (1, 2, J)`. That call gives K, and whether the model is stationary: it is if the output's time axis has size 1, as for `lambda f, t: S0(f)`. Nt ≥ 2 always, so two columns always exist.
- **Chunks.** `CHUNK = 2**16` is the most points (bands × columns × J) per call.
  - An evolutionary model gets `nc = min(Nt, max(1, CHUNK // J))` columns and `nb = max(1, CHUNK // (nc·J))` bands per call. On 512×512 that is 1 band × 170 columns, and on 2048×2048 it is 1 band × 42 columns.
  - A stationary model gets every column and `nb = max(1, CHUNK // J)` bands per call. Its output then has `nb·J` points, so it is never expanded along t. The cost is the number of frequency samples (spec §3.3): 2048×2048 takes 50 calls.
  - Each chunk is checked against `points = (bands, 1, J)` for a stationary model and `(bands, columns, J)` otherwise. So a model that returned a size-1 time axis on the first call and a full one later is refused by `_hermitian`'s shape rule (fail closed).
  - A chunk's K must equal the first call's K: `ValueError: S(f, t) [chunk from band 0, column 0] is 1 x 1, but was 3 x 3`.
  - Each chunk's `name` is `S(f, t) [chunk from band m0, column n0]`. A refusal's `at point (b, n, j)` counts from that band and column, and j indexes the band's J bins.
- **The entry rule** (§3.3). Entries are computed only for `a ≤ b` (`np.triu_indices(K)`), as `0.5 * (H.real[a, b] * w).sum(axis=-1) / w.sum()`. That is the expression of `evolutionary_band_variance` (`0.5 * (S * phi2).sum(axis=-1) / phi2.sum()`). The value is assigned to `C[a, b]` and `C[b, a]` in one statement, so each block is exactly symmetric.
  - A size-1 time axis gives a `(pairs, bands, 1)` value, which the assignment broadcasts over the chunk's columns. A size-1 bin axis broadcasts in the product with `w`.
  - Only `Re H_ab` enters. A same-cell covariance is real and symmetric, and `S_ba = conj(S_ab)`, so a linear rule can depend only on `Re S_ab` (spec §3.3).
- **Inactive cells.** At the end, `C[:, :, ~active_mask(Nf, Nt)] = 0.0`. Assignment, not multiplication, so no −0.0 appears. `wilson.active_mask` marks band 0 active at even n, and band Nf active where `Nf + n` is even. Nf is even, so the inactive cells are m ∈ {0, Nf} at odd n, as the task says.
- **Linearity** (§3.4) holds by construction: the check changes no value, and the projection, the band average and the mirror are linear. Measured on the WDM route: 3.7e-16.

**The chunk size** (measured with `tracemalloc` at K = 3, on 64×512, in the prototype; "beyond" is the peak minus the output's own `K²·(Nf+1)·Nt·8` bytes):

| CHUNK | real model `x = B(t) y`: time, beyond | complex model: time, beyond |
|---|---|---|
| 2^16 | 1.22 s, 24.8 MiB | 1.59 s, 45.7 MiB |
| 2^17 | 1.06 s, 44.8 MiB | 1.86 s, 82.2 MiB |
| 2^18 | 1.10 s, 74.0 MiB | 1.93 s, 137.1 MiB |
| 2^19 | 1.37 s, 147.7 MiB | 1.97 s, 273.9 MiB |
| 2^20 | 1.43 s, 368.8 MiB | 1.98 s, 684.5 MiB |

The wall time does not fall with a larger chunk, but the memory doubles at each step. So `CHUNK = 2**16`. The memory scales with K², so at K = 6 a chunk needs about 4× as much.

**Timing** (a plan note, not a test). The final function at K = 3 and `CHUNK = 2**16`, wdmrel (numpy 2.3.5), measured in the prototype:

| Model | Grid | Wall time | Calls | Model points | Peak = output + beyond |
|---|---|---|---|---|---|
| stationary, complex delayed | 512×512 | 0.05 s | 5 | 2.0e5 | 65 MiB = 18 + 47 |
| stationary, complex delayed | 2048×2048 | 0.69 s | 50 | 3.2e6 | 357 MiB = 288 + 69 |
| evolutionary, real (`x = B(t) y`) | 512×512 | 10.0 s | 2053 | 1.0e8 (99 ns each) | 44 MiB = 18 + 26 |
| evolutionary, complex (`delayed(f)·(1 + 0.3 sin 2πt/T)`) | 512×512 | 12.7 s | 2053 | 1.0e8 (125 ns each) | 65 MiB = 18 + 47 |
| evolutionary, real | 1024×1024 | 78.0 s | 13326 | 8.1e8 (97 ns each) | 103 MiB = 72 + 31 |

- 2048×2048 evolutionary was not run. Extrapolated from the per-point cost: 6.45e9 points (0.75·(Nf+1)·Nt², as spec Risks says), so about 10.5 min for a real model and 13.4 min for a complex one, in 100,402 calls. That is too slow inside a sampler, as the Risks section expects. The separable fast path is its follow-up.
- The stationary 2048×2048 peak beyond the output (69 MiB) also includes the folded frequency grid, `(Nf+1)·J` floats (25 MB there).
- In a first prototype without the first call, every chunk split the columns, and a stationary model was re-evaluated once per column chunk. At `CHUNK = 2**18` that was 26,637 calls and 14.4 s on 2048×2048 stationary, about 13× its frequency samples, against spec §3.3's "the number of frequency samples".

**Choices the spec leaves open** (plan deviations):
- The model gets one extra small call before the chunks: band 0 and two columns. This is what tells a stationary model from an evolutionary one before any memory is committed, and it gives K. So the model is called once per chunk plus once more, not strictly "once per chunk" (spec §3.3, task text).
- In stationary mode, a chunk is checked against a size-1 time axis. A model that returns a size-1 time axis on the first call and a full one later is refused (`ValueError: ... must have shape (K, K, 17, 1, 25) ...`), rather than accepted at a memory cost that is set by the grid.
- A K that changes between calls is refused with `ValueError`. The spec does not mention it. Without the check, a smaller K would raise `IndexError`, and a larger K would be silently cut to the first K×K.
- A refusal names the chunk (`S(f, t) [chunk from band 0, column 0] at point (0, 0, 0): ...`). The point is local to the chunk and is not mapped back to (m, n), because that would add a parameter to Task 8's `_hermitian`. The first call's refusals are named `S(f, t)`, with points that are already global (band 0, columns 0–1).
- `CHUNK` is a module constant, not a keyword. The interface contract fixes the signature. The test sets it with `monkeypatch.setattr("domainia._covariance.CHUNK", …)`.
- `wdm_covariance` joins `CONSUMERS` in `tests/wilson/test_window_threading.py`, because it samples the window. A future src caller that drops `window=` is then flagged by the AST scan. No src module calls it today, so no item count changes. `SCANNED` already holds `_covariance.py`.
- `ROUTES["wdm"]` returns the active cells only (`[..., MASK]`, shape `(K, K, N)`). This keeps `error()` unchanged and avoids a 0/0 at inactive cells, which would raise a `RuntimeWarning` and break the no-warnings gate. The inactive zeros are asserted in the two dedicated tests. The route passes a non-callable through unwrapped, so that `np.eye(K)` still gets `S must be callable`.
- The evolutionary test runs on 16×64, not 16×32, so that T_p = 64·ΔT divides the record. At 32·ΔT the error is 3.49e-2 on both grids (measured).
- The 64·ΔT bound is **1.5e-2**, set from the measurement: **9.33e-3** (PY and FLOOR), a margin of 1.6×. The 32·ΔT bound is 5e-2 against the measured 3.49e-2 (1.43×). The error falls by 3.7× as T_p doubles: second order, as in spec §3.3's K = 1 table. The 64·ΔT bound sits 2.3× below the 32·ΔT error, so the test pins the fall.
- The `wrap_cut_columns(16, 64, 5.0) = (6, 5)` columns are excluded, as §6.2 says. For this model, which is periodic over the record, excluding them changes nothing: the error over all columns is the same.
- The dense evolutionary reference computes only the same-cell blocks, as `Σ_c Σ_j (U_ac R)[p, j] U_bc[p, j]` with `U_ac[p, i] = W[p, i] B_ac(t_i)`. That is the same quantity as `kxk_evolutionary.py`'s `(I_K ⊗ W) Σ (I_K ⊗ W)ᵀ` (the two agree to 2.4e-15), but it avoids the `(K·N)²` products. `U` must be C-contiguous: on numpy 2.1.0 (FLOOR) a non-contiguous `U @ R` took 3.7 s per product, against 0.013 s (measured). With it, the evolutionary test took 22 s in FLOOR, against 0.17 s now.
- The diagonal equals `evolutionary_band_variance` bit for bit (measured, PY and FLOOR), because the reduction is the same expression. The test still asserts `rtol=1e-14`, as the task asks.
- The stationary case is measured at 3.3e-15 (interior bands) and 2.2e-15 (edge bands), not the spec's 1.9e-15 and 8.7e-16. The test model carries Task 8's per-channel phases, which `kxk_stationary.py` did not have. Both are far inside 1e-12.
- The window is threaded and checked by the stationary dense test, which runs in `MEYER` and in `testing.ALT_WINDOW`. Measured there: 4.1e-15. Meyer's output against the ALT_WINDOW reference is off by 1.4e-2, so the test tells the windows apart.

**Files:**
- Create: none.
- Modify:
  - `src/domainia/_covariance.py`: the imports, `__all__`, the new constant `CHUNK`, and the new function `wdm_covariance` after `frequency_covariance` (Step 2).
  - `src/domainia/__init__.py`: the module docstring's Modules list, the import from `._covariance`, and `__all__` (Step 3).
  - `tests/test_covariance.py`: the import; the constants `NF`, `NT`, `MASK`, `ROUTES["wdm"]`, `L0`, `M1` and `M2`; the helpers `mixing`, `modulated` and `same_cell`; the new tests `test_wdm_covariance_of_a_stationary_model_is_the_dense_reference` and `test_wdm_covariance_of_an_evolutionary_model`; and in `test_every_refusal`, the docstring and a WDM loop (Step 1).
  - `tests/test_package.py`: `PUBLIC` gains `"wdm_covariance"` (Step 5).
  - `tests/wilson/test_window_threading.py`: `CONSUMERS` gains `"wdm_covariance"` (Step 6).
- Delete: none.
- No change: `tests/wilson/meyer_cases.py` (no kernel changes) and `README.md` (Task 11).
- Test: `tests/test_covariance.py`. Also affected: `tests/test_package.py` and `tests/wilson/test_window_threading.py`.

**Acceptance Criteria:**
- [ ] Step 1's run gives `2 errors` at collection, both `ImportError: cannot import name 'wdm_covariance' from 'domainia'`.
- [ ] `pytest -q tests/test_covariance.py` → `5 passed` in PY and in FLOOR, each in well under 1 s.
- [ ] Step 4's probe prints the lines listed there, in PY and in FLOOR: stationary error `3.3e-15`, the 32·ΔT and 64·ΔT errors `0.035` and `0.0093`, diagonal `0`, WDM linearity `3.7e-16`, and three refusal messages.
- [ ] Each of the 17 mutations in Step 4's table makes `tests/test_covariance.py` fail.
- [ ] `pytest -q tests/test_covariance.py` also passes on `$TMPDIR/dom3-envs/cur313` and `cur314` (numpy 2.5.4), because its bit-for-bit assertions were checked there too (prototype: `5 passed`).
- [ ] `pytest -q tests/test_package.py` → `7 passed`, and `pytest -q tests/wilson/test_window_threading.py` → `402 passed`.
- [ ] `grep -nEi "be=|float32|complex64|jax|solve|logdet|cholesky|inv\(|det\(" src/domainia/_covariance.py` prints nothing.
- [ ] The exact replay check prints `0 of 1408 arrays differ from the baseline`, and `meyer_cases.py --check` prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`, in PY and in FLOOR.
- [ ] Full suite (prototype): PY `1049 passed, 3 skipped`; FLOOR `1049 passed, 3 skipped`. No warnings and no collection errors.
- [ ] Size: `src 5367 lines / 2037 code`, inside the pre-trim budget (at most about 5,400 / 2,150).

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1` → `5 passed in …`

**Steps:**

- [ ] **Step 1: Write the tests (they fail first)**

Make these seven edits to `tests/test_covariance.py`.

**Edit 1.1** (`tests/test_covariance.py`, the imports).

Old:

```python
import pytest

from domainia import frequency_covariance

K, DT = 3, 5.0
```

New:

```python
import pytest

from domainia import frequency_covariance, testing, wdm_covariance, wilson

K, DT = 3, 5.0
```

**Edit 1.2** (`tests/test_covariance.py`, the module constants: `NF`, `NT`, `MASK`, the `"wdm"` route, and the mixing matrices `L0`, `M1` and `M2` of spec §6.2's evolutionary reference).

Old:

```python
B = np.random.default_rng(20261114).standard_normal((K, K, 2)) @ [1.0, 1.0j]
#: The routes from a stationary model S(f) to a covariance.
ROUTES = {"frequency": lambda S, N=128: frequency_covariance(S, N=N, dt=DT)}


def psd(f):
```

New:

```python
B = np.random.default_rng(20261114).standard_normal((K, K, 2)) @ [1.0, 1.0j]
#: The WDM grid of the stationary case, and its active cells in numpy.nonzero order (the rows
#: of testing.dense_W).
NF, NT = 16, 32
MASK = wilson.active_mask(NF, NT)
#: The routes from a stationary model S(f) to a covariance; the WDM route keeps the active cells.
ROUTES = {"frequency": lambda S, N=128: frequency_covariance(S, N=N, dt=DT),
          "wdm": lambda S: wdm_covariance((lambda f, t: S(f)) if callable(S) else S, Nf=NF,
                                          Nt=NT, dt=DT)[..., MASK]}
#: x = B(t) y for K independent channels y of PSD psd (spec 6.2), with
#: B(t) = L0 (I + sin(2 pi t / Tp) M1 + cos(4 pi t / Tp) M2).
L0 = np.linalg.cholesky(LEVELS)
M1 = np.array([[0.3, 0.1, -0.2], [0.0, -0.25, 0.15], [0.2, 0.1, 0.3]])
M2 = np.array([[-0.1, 0.2, 0.0], [0.1, 0.2, -0.1], [0.0, -0.2, 0.15]])


def psd(f):
```

**Edit 1.3** (`tests/test_covariance.py`, new helpers `mixing` and `modulated` after `plus`).

Old:

```python
    return model


def sigma(S, N, dt):
```

New:

```python
    return model


def mixing(t, Tp):
    """B(t), of shape (*t.shape, K, K)."""
    t = np.asarray(t, float)[..., None, None]
    return L0 @ (np.eye(K) + np.sin(2 * np.pi * t / Tp) * M1 + np.cos(4 * np.pi * t / Tp) * M2)


def modulated(Tp):
    """S_ab(f, t) = psd(f) [B(t) B(t)^T]_ab for broadcasting f and t: real cross-spectra."""
    def model(f, t):
        Bt = mixing(t, Tp)
        return psd(f) * np.moveaxis(Bt @ np.swapaxes(Bt, -1, -2), (-2, -1), (0, 1))
    return model


def sigma(S, N, dt):
```

**Edit 1.4** (`tests/test_covariance.py`, new helper `same_cell` after `sigma`).

Old:

```python
    return R[..., (i[:, None] - i) % N]


def error(got, want):
```

New:

```python
    return R[..., (i[:, None] - i) % N]


def same_cell(Nf, Nt, Tp):
    """The same-cell blocks of spec 6.2's dense evolutionary reference on the active cells,
    (K, K, N): (I_K x W) Sigma (I_K x W)^T with Sigma_ab[i, j] = [B(t_i) B(t_j)^T]_ab R(i - j),
    where R is the stationary reference of psd."""
    N = Nf * Nt
    W = testing.dense_W(Nf, Nt, DT, wilson.active_mask(Nf, Nt))
    R = sigma(lambda f: psd(f)[None, None], N, DT)[0, 0]
    Bt = np.moveaxis(mixing(np.arange(N) * DT, Tp), 0, -1).copy()     # Bt[a, c, i] = B_ac(t_i)
    out = 0.0
    for c in range(K):
        U = Bt[:, c, None] * W              # U[a, p, i] = W[p, i] B_ac(t_i), C-contiguous for BLAS
        out = out + np.einsum("api,bpi->abp", U @ R, U)
    return out


def error(got, want):
```

**Edit 1.5** (`tests/test_covariance.py`, the two new tests after `test_frequency_covariance_is_the_per_bin_covariance_of_the_dense_reference`).

Old:

```python
        assert not np.any(d[~np.eye(K, dtype=bool)])


def test_linear_and_exactly_hermitian_and_a_zero_point_passes():
```

New:

```python
        assert not np.any(d[~np.eye(K, dtype=bool)])


def test_wdm_covariance_of_a_stationary_model_is_the_dense_reference():
    """The same-cell blocks of (I_K x W) Sigma (I_K x W)^T, active edge-band cells included,
    within 1e-12 (measured: 3.3e-15) in two windows; every block is 0 at the inactive cells; a
    size-1 time axis gives the explicitly broadcast output, bit for bit."""
    Sigma = sigma(delayed, NF * NT, DT)
    for window in (wilson.MEYER, testing.ALT_WINDOW):
        W = testing.dense_W(NF, NT, DT, MASK, window=window)
        want = np.einsum("pi,abij,pj->abp", W, Sigma, W, optimize=True)
        got = wdm_covariance(lambda f, t: delayed(f), Nf=NF, Nt=NT, dt=DT, window=window)
        assert got.shape == (K, K, NF + 1, NT) and got.dtype == np.float64
        assert error(got[..., MASK], want) < 1e-12
        assert not np.any(got[..., ~MASK])
    full = wdm_covariance(lambda f, t: delayed(f + 0 * t), Nf=NF, Nt=NT, dt=DT, window=window)
    assert np.array_equal(got, full)


def test_wdm_covariance_of_an_evolutionary_model(monkeypatch):
    """x = B(t) y on 16 x 64, outside the columns wrap_cut_columns names: the error is below 5e-2
    at T_p = 32 dT and below 1.5e-2 at 64 dT (measured: 3.5e-2 and 9.3e-3); each diagonal
    entry, also as a K = 1 model, is noise.evolutionary_band_variance to 1e-14 (measured: bit
    for bit) and 0 at the inactive cells; smaller chunks, with partial last chunks, change no
    bit and keep every call within CHUNK points."""
    Nf, Nt = 16, 64
    mask = wilson.active_mask(Nf, Nt)
    lead, trail = wilson.evolutionary.wrap_cut_columns(Nf, Nt, DT)
    n = np.nonzero(mask)[1]
    kept = (n >= lead) & (n < Nt - trail)
    for Tp, bound in ((32, 5e-2), (64, 1.5e-2)):
        S = modulated(Tp * Nf * DT)
        got = wdm_covariance(S, Nf=Nf, Nt=Nt, dt=DT)
        assert error(got[..., mask][..., kept], same_cell(Nf, Nt, Tp * Nf * DT)[..., kept]) < bound
        for a in range(K):
            want = wilson.noise.evolutionary_band_variance(
                Nf, Nt, DT, lambda f, t, S=S, a=a: S(f, t)[a, a])
            one = wdm_covariance(lambda f, t, S=S, a=a: S(f, t)[a:a + 1, a:a + 1],
                                 Nf=Nf, Nt=Nt, dt=DT)
            for v in (got[a, a], one[0, 0]):
                np.testing.assert_allclose(v[mask], want[mask], rtol=1e-14, atol=0)
                assert not np.any(v[~mask])
    points = []

    def counted(f, t):
        points.append(np.broadcast(f, t).size)
        return S(f, t)
    models = (counted, lambda f, t: delayed(f))
    want = [wdm_covariance(s, Nf=NF, Nt=NT, dt=DT) for s in models]
    # 25 bins per band on NF x NT: one call on 2 columns, then 1 band x 5 columns or 2 bands x
    # 32 columns per call.
    for chunk, calls in ((125, 1 + 17 * 7), (2000, 1 + 9)):
        monkeypatch.setattr("domainia._covariance.CHUNK", chunk)
        points.clear()
        for s, C in zip(models, want):
            assert np.array_equal(wdm_covariance(s, Nf=NF, Nt=NT, dt=DT), C)
        assert max(points) <= chunk and len(points) == calls


def test_linear_and_exactly_hermitian_and_a_zero_point_passes():
```

**Edit 1.6** (`tests/test_covariance.py`, `test_every_refusal`'s docstring).

Old:

```python
def test_every_refusal():
    """Each failure of the model check, made per point (spec 3.1), raises the right kind."""
    def dip(f):
```

New:

```python
def test_every_refusal():
    """Each failure of the model check, made per point (spec 3.1), raises the right kind, as do
    a bad grid, dt or window; a WDM model is checked in every chunk, and refused if its K, or
    its size-1 time axis, changes between calls."""
    def dip(f):
```

**Edit 1.7** (`tests/test_covariance.py`, `test_every_refusal`: a WDM loop after the frequency loop).

Old:

```python
            frequency_covariance(delayed, **{"N": 128, "dt": DT, **kw})
```

New:

```python
            frequency_covariance(delayed, **{"N": 128, "dt": DT, **kw})
    for kw, err, match in (({"Nf": 15}, ValueError, "Nf must be even"),
                           ({"Nt": 32.0}, TypeError, "Nt must be an integer"),
                           ({"dt": 0.0}, ValueError, "dt must be"),
                           ({"window": "meyer"}, TypeError, "window must be"),
                           ({"S": lambda f, t: plus(0, 1, 1e-9)(f) if t.size > 2 else delayed(f)},
                            ValueError, r"\[chunk from band 0, column 0\] at point \(0, 0, "),
                           ({"S": lambda f, t: delayed(f)[:1, :1] if t.size > 2 else delayed(f)},
                            ValueError, "is 1 x 1, but was 3 x 3"),
                           ({"S": lambda f, t: delayed(f + 0 * t) if t.size > 2 else delayed(f)},
                            ValueError, r"must have shape \(K, K, 17, 1, 25\)")):
        with pytest.raises(err, match=match):
            wdm_covariance(**{"S": lambda f, t: delayed(f), "Nf": NF, "Nt": NT, "dt": DT, **kw})
```

What each new test covers, against spec §6.2 and the task:
- `test_wdm_covariance_of_a_stationary_model_is_the_dense_reference` ("`wdm_covariance`, stationary"):
  - On 16×32 at dt = 5 s with K = 3, the model has complex delayed cross-spectra and unequal levels (`delayed`). The same-cell blocks match `(I_K ⊗ W) Σ (I_K ⊗ W)ᵀ` within 1e-12 by §3.3's metric. W is `testing.dense_W` on all active cells, edge-band cells included, and Σ is Task 8's `sigma`. This holds in two windows.
  - Shape `(K, K, 17, 32)`, float64, and every block is exactly 0 at the inactive cells.
  - A model that returns a size-1 time axis gives exactly (`np.array_equal`) the output of the same model broadcast explicitly (`delayed(f + 0 * t)`).
- `test_wdm_covariance_of_an_evolutionary_model` ("`wdm_covariance`, evolutionary", and the chunking):
  - The model is `x = B(t) y` on 16×64, with `B(t) = L₀(I + sin(2πt/T_p)M₁ + cos(4πt/T_p)M₂)`. L₀, M₁ and M₂ are those of `kxk_evolutionary.py`, and the cross-spectra are real. Excluding the columns `wrap_cut_columns` names, the error is below 5e-2 at T_p = 32·ΔT and below 1.5e-2 at 64·ΔT, against `same_cell`.
  - Each diagonal entry, and that entry as a K = 1 model, equals `noise.evolutionary_band_variance` to `rtol=1e-14` at active cells, and is 0 at inactive cells.
  - With `CHUNK` monkeypatched to 125 (1 band × 5 columns per call: 32 columns give 6 full chunks and one of 2) and to 2000 (2 bands × 32 columns: 17 bands give 8 full chunks and one of 1), the result is bit for bit the default one-chunk result. This holds for an evolutionary model and for a stationary one (5 bands per call at 125: chunks of 5, 5, 5 and 2). Every call has at most `CHUNK` points. The number of calls is 1 + 17·7 and 1 + 9, which pins the first call and one call per chunk.
- The shared tests now loop over both routes ("Shared"). `test_linear_and_exactly_hermitian_and_a_zero_point_passes` checks linearity on the WDM route, the exact symmetry of a model Hermitian only to rounding, and the all-zero model. `test_every_refusal` runs its 14 model cases through `wdm_covariance` too. Its new WDM loop covers a bad `Nf`, `Nt`, `dt` and window; a failure found by a later chunk, not the first call (`plus(0, 1, 1e-9)` only when `t.size > 2`); a K that changes; and a size-1 time axis that changes.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1`

Expected: `2 errors in …`. Both are the collection error `ImportError: cannot import name 'wdm_covariance' from 'domainia'`. There are two because `--doctest-modules` collects the file twice, as in Task 8. Check with `… -m pytest -q tests/test_covariance.py 2>&1 | grep -c "^E   ImportError: cannot import name 'wdm_covariance'"`, which prints `2`.

- [ ] **Step 2: Add `wdm_covariance` to `src/domainia/_covariance.py`**

**Edit 2.1** (`src/domainia/_covariance.py`, the imports, `__all__`, and the new constant `CHUNK` after `TOL`).

Old:

```python
from .wilson._checks import require_callable, require_dtype, require_int, validate_dt

__all__ = ["frequency_covariance"]

#: The model check's tolerance at a point, as a fraction of that point's largest |S_cd|.
TOL = 1e-12


def _hermitian(out: Any, points: tuple[int, ...], name: str) -> np.ndarray:
```

New:

```python
from .wilson._checks import require_callable, require_dtype, require_int, validate_dt
from .wilson.grid import active_mask, band_frequencies, delta_T, validate_grid
from .wilson.windows import MEYER, Window, sample_window

__all__ = ["frequency_covariance", "wdm_covariance"]

#: The model check's tolerance at a point, as a fraction of that point's largest |S_cd|.
TOL = 1e-12
#: The most points (bands x columns x frequency samples) per call of a WDM model.  At K = 3 a
#: call needs about 26 MiB (real model) or 47 MiB (complex) beyond the output (tracemalloc,
#: 512 x 512); a larger chunk is no faster.
CHUNK = 2**16


def _hermitian(out: Any, points: tuple[int, ...], name: str) -> np.ndarray:
```

**Edit 2.2** (`src/domainia/_covariance.py`: append `wdm_covariance` after `frequency_covariance`. The Old text is `frequency_covariance`'s last line, which is the only `    return C` in the file).

Old:

```python
    return C
```

New:

```python
    return C


def wdm_covariance(S: Callable[[np.ndarray, np.ndarray], Any], *, Nf: int, Nt: int, dt: float,
                   window: Window = MEYER) -> np.ndarray:
    """The covariance of each WDM cell's K coefficients under an evolutionary model ``S(f, t)``.

    Returns ``C[a, b, m, n] = BA_m[Re H_ab(., t_n)] / 2``, float64 of shape ``(K, K, Nf + 1, Nt)``,
    exactly symmetric in ``a, b`` and 0 at the inactive cells.  ``H = (S + S^H) / 2`` after the
    check (:func:`_hermitian`), ``t_n = n Nf dt``, and ``BA_m`` is the phi-hat^2-weighted mean over
    band ``m``'s frequency samples, folded at DC and Nyquist (WDM Paper II, Eq. 36), in the Wilson
    basis of ``window``.  For K = 1 this is ``wilson.noise.evolutionary_band_variance``.

    ``S`` gets broadcasting ``f`` (bands, 1, bins) [Hz] and ``t`` (1, columns, 1) [s from the
    record's first sample]: once on band 0 and two columns, then once per chunk of at most
    ``CHUNK`` points.  A model that returns a size-1 time axis on that first call (a stationary
    ``lambda f, t: S0(f)``) gets every column in each chunk and must keep that axis.  A refusal
    names its point from the chunk's first band and column.

    Raises ``TypeError`` for a non-callable ``S``, a non-integer ``Nf`` or ``Nt``, a bad window
    or a non-numeric output, and ``ValueError`` for a bad grid or ``dt``, or an output that fails
    the check or changes K between calls.
    """
    require_callable("S", S)
    validate_grid(Nf, Nt)
    validate_dt(dt)
    w = sample_window(window, Nf, Nt, dt) ** 2
    sup = np.flatnonzero(w)
    w, J, nyq = w[sup], sup.size, 0.5 / dt
    f = np.abs(band_frequencies(Nf, dt)[:, None] + (np.arange(Nt) - Nt // 2)[sup] / (Nf * Nt * dt))
    f = np.where(f > nyq, 2.0 * nyq - f, f)
    t = np.arange(Nt) * delta_T(Nf, dt)
    H = _hermitian(S(f[:1, None], t[None, :2, None]), (1, 2, J), "S(f, t)")
    K, stationary = H.shape[0], H.shape[3] == 1
    nc = Nt if stationary else min(Nt, max(1, CHUNK // J))
    nb = max(1, CHUNK // (J if stationary else nc * J))
    a, b = np.triu_indices(K)
    C = np.zeros((K, K, Nf + 1, Nt))
    for m in range(0, Nf + 1, nb):
        for n in range(0, Nt, nc):
            fm, tn = f[m:m + nb, None], t[None, n:n + nc, None]
            name = f"S(f, t) [chunk from band {m}, column {n}]"
            H = _hermitian(S(fm, tn), (fm.shape[0], 1 if stationary else tn.shape[1], J), name)
            if H.shape[0] != K:
                raise ValueError(f"{name} is {H.shape[0]} x {H.shape[0]}, but was {K} x {K}")
            C[a, b, m:m + nb, n:n + nc] = C[b, a, m:m + nb, n:n + nc] = (
                0.5 * (H.real[a, b] * w).sum(axis=-1) / w.sum())
    C[:, :, ~active_mask(Nf, Nt)] = 0.0
    return C
```

- [ ] **Step 3: Re-export `wdm_covariance` from `src/domainia/__init__.py`**

**Edit 3.1** (`src/domainia/__init__.py`, the module docstring's Modules list).

Old:

```python
  (:mod:`domainia._covariance`).
* Not imported here: :mod:`domainia.testing` (dense oracles and test windows).
```

New:

```python
  (:mod:`domainia._covariance`).
* :func:`wdm_covariance`: an evolutionary (or stationary) K x K noise model as its covariance
  per WDM cell (:mod:`domainia._covariance`).
* Not imported here: :mod:`domainia.testing` (dense oracles and test windows).
```

**Edit 3.2** (`src/domainia/__init__.py`, the import).

Old:

```python
from . import wilson
from ._covariance import frequency_covariance
from ._transform import transform
```

New:

```python
from . import wilson
from ._covariance import frequency_covariance, wdm_covariance
from ._transform import transform
```

**Edit 3.3** (`src/domainia/__init__.py`, `__all__`).

Old:

```python
    "transform",
    "wilson",
```

New:

```python
    "transform",
    "wdm_covariance",
    "wilson",
```

- [ ] **Step 4: The new tests pass, the measured values, and the mutations**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1`

Expected: `5 passed in …` (prototype: 0.33 s).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q tests/test_covariance.py 2>&1 | tail -1`

Expected: `5 passed in …` (prototype: 0.46 s).

This probe prints the measured values behind the tests' bounds, and three of the new refusal messages.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P - <<'EOF'
import numpy as np
import test_covariance as T
from domainia import testing, wdm_covariance, wilson
M = T.MASK
edge = np.zeros_like(M)
edge[[0, T.NF]] = M[[0, T.NF]]
Sigma = T.sigma(T.delayed, T.NF * T.NT, T.DT)
for w in (wilson.MEYER, testing.ALT_WINDOW):
    W = testing.dense_W(T.NF, T.NT, T.DT, M, window=w)
    want = np.einsum("pi,abij,pj->abp", W, Sigma, W, optimize=True)
    got = wdm_covariance(lambda f, t: T.delayed(f), Nf=T.NF, Nt=T.NT, dt=T.DT, window=w)
    print(f"{w.name}: stationary error {T.error(got[..., M], want):.2g} "
          f"(interior {T.error(got[..., M & ~edge], want[..., ~edge[M]]):.2g}, "
          f"edge {T.error(got[..., edge], want[..., edge[M]]):.2g})")
print(f"Meyer output against the ALT_WINDOW reference: {T.error(T.ROUTES['wdm'](T.delayed), want):.2g}")
r = T.ROUTES["wdm"]
print(f"WDM linearity {T.error(r(lambda f: T.delayed(f) + T.second(f)), r(T.delayed) + r(T.second)):.2g}")
Nf, Nt = 16, 64
mask = wilson.active_mask(Nf, Nt)
lead, trail = wilson.evolutionary.wrap_cut_columns(Nf, Nt, T.DT)
n = np.nonzero(mask)[1]
kept = (n >= lead) & (n < Nt - trail)
print(f"wrap_cut_columns(16, 64, 5.0) = {(lead, trail)}")
for Tp in (32, 64):
    S = T.modulated(Tp * Nf * T.DT)
    got = wdm_covariance(S, Nf=Nf, Nt=Nt, dt=T.DT)
    ref = T.same_cell(Nf, Nt, Tp * Nf * T.DT)
    diag = max(np.max(np.abs(got[a, a][mask] / wilson.noise.evolutionary_band_variance(
        Nf, Nt, T.DT, lambda f, t: S(f, t)[a, a])[mask] - 1)) for a in range(T.K))
    print(f"T_p = {Tp} dT: error {T.error(got[..., mask][..., kept], ref[..., kept]):.2g} "
          f"(all columns {T.error(got[..., mask], ref):.2g}); diagonal vs Eq. 36 kernel {diag:.2g}")
for S in (lambda f, t: T.plus(0, 1, 1e-9)(f) if t.size > 2 else T.delayed(f),
          lambda f, t: T.delayed(f)[:1, :1] if t.size > 2 else T.delayed(f),
          lambda f, t: T.delayed(f + 0 * t) if t.size > 2 else T.delayed(f)):
    try:
        wdm_covariance(S, Nf=T.NF, Nt=T.NT, dt=T.DT)
    except ValueError as e:
        print(f"ValueError: {e}")
EOF
```

Expected:
```
meyer(d=4): stationary error 3.3e-15 (interior 3.3e-15, edge 2.2e-15)
wide-smoothstep: stationary error 4.1e-15 (interior 4.1e-15, edge 2.3e-15)
Meyer output against the ALT_WINDOW reference: 0.014
WDM linearity 3.7e-16
wrap_cut_columns(16, 64, 5.0) = (6, 5)
T_p = 32 dT: error 0.035 (all columns 0.035); diagonal vs Eq. 36 kernel 0
T_p = 64 dT: error 0.0093 (all columns 0.0093); diagonal vs Eq. 36 kernel 0
ValueError: S(f, t) [chunk from band 0, column 0] at point (0, 0, 0): |S[0, 1] - conj(S[1, 0])| is 1e-09 of max |S_cd| there; at most 1e-12 is accepted
ValueError: S(f, t) [chunk from band 0, column 0] is 1 x 1, but was 3 x 3
ValueError: S(f, t) [chunk from band 0, column 0] must have shape (K, K, 17, 1, 25) with K >= 1 (an axis after the first two may be 1); got (3, 3, 17, 32, 25)
```

The FLOOR interpreter (`$TMPDIR/dom3-envs/floor312/bin/python` in place of the wdmrel one) prints the same lines.

Until Step 5, `tests/test_package.py::test_the_public_names` fails, because `__all__` is no longer `PUBLIC`.

Each mutation below was applied to `wdm_covariance` in the prototype, and each makes `tests/test_covariance.py` fail:

| Mutation | Failing tests |
|---|---|
| inactive cells kept (no `C[:, :, ~active_mask(Nf, Nt)] = 0.0`); no fold at Nyquist; no fold at DC (no `abs`); no `0.5`; no `/ w.sum()`; φ̂ in place of φ̂² | the stationary and the evolutionary test |
| `abs(H)` in place of `Re H`; the band chunk written one band off; the lower triangle not mirrored | those two and `test_linear_and_exactly_hermitian_and_a_zero_point_passes` |
| the column chunk read one column off | the stationary and the evolutionary test |
| `t_n = (n + 0.5)·ΔT`; every column in one call for an evolutionary model (`nc = Nt`) | `test_wdm_covariance_of_an_evolutionary_model` |
| Meyer sampled whatever the window | the stationary test and `test_every_refusal` |
| no `_hermitian` in the chunks; a stationary chunk checked against the full time axis; never stationary; no K check | `test_every_refusal` |

- [ ] **Step 5: The public names (`tests/test_package.py`)**

**Edit 5.1** (`tests/test_package.py`, `PUBLIC`).

Old:

```python
PUBLIC = {"Grid", "MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError",
          "__version__", "frequency_covariance", "transform", "wilson"}


def test_the_version_is_2_0_0_everywhere():
```

New:

```python
PUBLIC = {"Grid", "MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError",
          "__version__", "frequency_covariance", "transform", "wdm_covariance", "wilson"}


def test_the_version_is_2_0_0_everywhere():
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -1`

Expected: `7 passed in …`

- [ ] **Step 6: `wdm_covariance` is a window consumer (`tests/wilson/test_window_threading.py`)**

**Edit 6.1** (`tests/wilson/test_window_threading.py`, `CONSUMERS`).

Old:

```python
    "eq45_plan",
    "dense_W", "wdm_congruence", "eq45_dense",
})
```

New:

```python
    "eq45_plan",
    "dense_W", "wdm_congruence", "eq45_dense", "wdm_covariance",
})
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_window_threading.py 2>&1 | tail -1`

Expected: `402 passed in …`, unchanged. No scanned module calls `wdm_covariance`, and `_covariance.py`'s own `sample_window(window, …)` call passes the scan.

- [ ] **Step 7: Gates**

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -nEi "be=|float32|complex64|jax|solve|logdet|cholesky|inv\(|det\(" src/domainia/_covariance.py`

Expected: no output. There is no JAX, no float32 and no likelihood code.

The two suite runs are independent and can run in parallel. There is no NumPy lane: Task 5 removed the last JAX code.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1049 passed, 3 skipped`. That is Task 8's 1047 plus the 2 new tests.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: passes. Prototype: `1049 passed, 3 skipped`.

Neither log contains `warning` or `error` (`grep -ciE "warning|error"` prints `0`).

The exact check compares every kept replay array with the recorded baseline, by dtype and `np.array_equal`. No kernel changed, so it still holds.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')"
```

Expected: `0 of 1408 arrays differ from the baseline`. The FLOOR interpreter prints the same.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. The FLOOR interpreter prints the same. `tests/wilson/test_meyer_baseline.py` passes inside both full runs.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected:
```
src 5367 lines / 2037 code
tests 4596 lines (+ oracle 4135)
```
That is +57 src lines (+32 code) and +110 test lines against Task 8 (5310 / 2005; 4486).
- `_covariance.py` is now 141 lines (it was 87), and `__init__.py` gains 3 lines.
- `tests/test_covariance.py` is now 259 lines (it was 149).

**Budget.** src is inside the spec's pre-trim budget (at most about 5,400 lines and 2,150 code), with 33 lines and 113 code lines to spare. tests are inside theirs (at most about 5,700). No later task adds code: Task 10 deletes Eq. 39, and Tasks 11 and 12 delete dead code and trim docstrings.

- [ ] **Step 8: Timing (a plan note, not a test)**

This times `wdm_covariance` at K = 3 on the grids of the timing table. It takes about 1 min 45 s, 78 s of which is the 1024×1024 evolutionary run.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P - stationary:512:512 stationary:2048:2048 evo-real:512:512 evo-complex:512:512 evo-real:1024:1024 <<'EOF'
"""Wall time and tracemalloc peak of domainia.wdm_covariance at K = 3 (CHUNK = 2**16)."""
import sys, time, tracemalloc
import numpy as np
import test_covariance as T
from domainia import wdm_covariance
from domainia.wilson import sample_window, MEYER

for spec in sys.argv[1:]:
    which, Nf, Nt = spec.split(":")
    Nf, Nt = int(Nf), int(Nt)
    J = np.flatnonzero(sample_window(MEYER, Nf, Nt, T.DT)).size
    T_rec = Nf * Nt * T.DT
    S = {"stationary": lambda f, t: T.delayed(f),
         "evo-real": T.modulated(T_rec),                                 # x = B(t) y
         "evo-complex": lambda f, t: T.delayed(f) * (1 + 0.3 * np.sin(2 * np.pi * t / T_rec))}[which]
    calls = []
    def counted(f, t):
        calls.append(np.broadcast(f, t).size)
        return S(f, t)
    tracemalloc.start()
    t0 = time.perf_counter()
    C = wdm_covariance(counted, Nf=Nf, Nt=Nt, dt=T.DT)
    wall = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    out = C.nbytes
    pts = (Nf + 1) * (1 if which == "stationary" else Nt) * J
    print(f"{which:11s} {Nf}x{Nt}: {wall:7.2f} s, {len(calls)} calls, {pts:.2e} model points "
          f"({wall / pts * 1e9:.0f} ns each), peak {peak / 2**20:.0f} MiB = output {out / 2**20:.0f} MiB "
          f"+ {(peak - out) / 2**20:.0f} MiB", flush=True)
EOF
```

Expected (prototype; wall times vary with the machine, while the calls and model points do not):
```
stationary  512x512:    0.05 s, 5 calls, 1.98e+05 model points (238 ns each), peak 65 MiB = output 18 MiB + 47 MiB
stationary  2048x2048:    0.69 s, 50 calls, 3.15e+06 model points (219 ns each), peak 357 MiB = output 288 MiB + 69 MiB
evo-real    512x512:   10.00 s, 2053 calls, 1.01e+08 model points (99 ns each), peak 44 MiB = output 18 MiB + 26 MiB
evo-complex 512x512:   12.69 s, 2053 calls, 1.01e+08 model points (125 ns each), peak 65 MiB = output 18 MiB + 47 MiB
evo-real    1024x1024:   78.02 s, 13326 calls, 8.07e+08 model points (97 ns each), peak 103 MiB = output 72 MiB + 31 MiB
```

- [ ] **Step 9: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/_covariance.py src/domainia/__init__.py tests/test_covariance.py tests/test_package.py tests/wilson/test_window_threading.py && git status --short | grep -vc "^[AM] "`

Expected: `0`. Every status line is a staged modification, so nothing is unstaged or untracked. (`grep -c` exits 1 when it prints 0, so this command does not continue with `&&`.)

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Add wdm_covariance: the K x K covariance per WDM cell, chunked, with its own evaluator" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `5 files changed, 175 insertions(+), 8 deletions(-)`

Size after Task 9 (prototype): src 5367 lines / 2037 code; tests 4596 (+ oracle 4135).

---

### Task 10: Delete Eq. 39; `band_averaged_variance` returns its own grid; a float32 `dt` computes in float64 (spec Order step 5; §4 "`noise.py`, Eq. 39"; §5 "dtypes are fp64 throughout"; §6.3)

**Goal:** The four Eq. 39 functions of `domainia.wilson.noise` and their tests are gone, `band_averaged_variance` returns its grid without them (bit for bit the same), and a NumPy float32 `dt` gives every public function the result of `float(dt)`.

Assumes the tree after Task 9, whose commit is "Add wdm_covariance: the K x K covariance per WDM cell, chunked, with its own evaluator".

**What changes for a caller.**
- `whittle_variance`, `whittle_variance_from_array`, `evaluate_psd_on_band_grid` and `evolutionary_variance` (Eq. 39, the midpoint rule) leave `domainia.wilson.noise` and `domainia.wilson`. Eq. 36 is the only per-cell rule: `band_averaged_variance` and `evolutionary_band_variance` (spec decision 7).
- `band_averaged_variance` returns the same arrays, bit for bit.
- A NumPy float32 `dt` is computed in double at every public function that takes `dt`. Before this task, 21 public functions computed partly in float32, with relative drift from 7.3e-9 to 3.9e-6: the 19 of the drift table below, and `phi_fft_order` and `phi_squared_normalization`, which call `phi_centered`.
- `evolutionary.time_domain_factor`, and through it `time_domain_covariance`, `draw_noise` and `exact_covariance`, now validate `dt` with `validate_dt`, like every other entry point. A bool, str or `None` `dt` raises `TypeError`, and a non-positive or non-finite one raises `ValueError`. Before (measured), `dt=True` ran as 1.0, `dt=-1.0` returned a factor with NaN entries (the square root of a negative Δf), `dt=0.0` raised `ZeroDivisionError`, and a str or `None` raised a `TypeError` from inside the arithmetic (`unsupported operand type(s) for /: 'float' and 'str'`).

**Callers of the four names.** The grep `grep -rnwE "whittle_variance(_from_array)?|evaluate_psd_on_band_grid|evolutionary_variance" src` at Task 9's tree finds these callers, and only these:
- inside `noise.py`: `whittle_variance` calls `evaluate_psd_on_band_grid` and `whittle_variance_from_array`, and `band_averaged_variance`'s last line is `return whittle_variance_from_array(Nt, Sn_eff)`;
- `wilson/__init__.py` re-exports all four.

The rest are docstrings (`noise.py`, `wilson/__init__.py` and `grid.time_pixel_centers`' note). `tests/wilson/meyer_cases.py` calls none of the four, so no replay key changes, and `DROPPED` stays as it is.
- `noise._psd_at` stays (spec §5). It still has four callers: `noise.evolutionary_band_variance` and, in `evolutionary.py`, `time_domain_factor`, `eq45_coeffs_interp` and `eq45_direct`.
- `grid.time_pixel_centers` loses its only src caller, `evolutionary_variance`. It stays public in `domainia.wilson` and `noise.py` no longer imports it. Spec §5's `domainia.wilson` API does not list it, so Task 11's API trim decides.

**`band_averaged_variance`'s return.** `whittle_variance_from_array(Nt, Sn_eff)` did four things: `0.5 * sanitize_psd(Sn_eff)`, a 1-D check, `validate_grid(Sn_eff.size - 1, Nt)` and the broadcast copy. The new last line is spec §4's `np.broadcast_to(0.5 * Sn_eff[:, None], (Nf + 1, Nt)).copy()`, and the function's own grid check, its first line `validate_grid(Nf, Nt)`, stays. Nothing else is lost:
- `Sn_eff` is a float64 `(Nf + 1,)` array of finite non-negative values: the φ̂²-weighted mean of `Sn_vals`, which has already been through `sanitize_psd`. So the second `sanitize_psd` returned a copy, and the 1-D check and the second grid check could not fail.
- Measured: the 6 `noise.band_averaged_variance` replay arrays differ from the baseline by 0.0 (worst relative difference 0.0e+00). Every kept replay array is still bit for bit the baseline (`0 of 1408 arrays differ`), in PY and FLOOR.
- The one input where the two lines differ: a weighted mean that overflows to inf, for PSD values near 1.8e308. `sanitize_psd` mapped that inf to 0.0; the new line keeps it.

**A float32 `dt`: why it drifted, and the two options.** Under NumPy 2's promotion rules (NEP 50), a NumPy float32 scalar stays float32 in arithmetic with Python ints and floats. So `N * dt`, `0.5 / dt`, `1.0 / (2.0 * dt * Nf)` and `np.fft.rfftfreq(N, d=dt)` (which computes `1.0 / (N * d)` inside) all round to float32. When the scalar meets a float64 array instead (`dt * array`, `array / dt`), it is cast to float64 exactly and nothing drifts. This is why `forward`, `inverse`, `forward_time`, `inverse_time`, `atom_spectrum`, `domainia.transform`, `frequency_covariance` (it has used `float(dt)` since Task 8) and the window sampling (`sample` converts with `float(dt)`) were already exact.

The task offers two fixes:
- **(a)** `validate_dt` returns `float(dt)`, and every entry point assigns it (`dt = validate_dt(dt)`). After the Eq. 39 deletion that is 25 `validate_dt(dt)` call sites, plus the two public functions that take `dt` and never call it (`evolutionary_band_variance` and `time_domain_factor`), plus `validate_dt` itself: 28 edits.
- **(b)** `float(dt)` where a float32 `dt` meets scalar arithmetic: 10 functions, so 10 sites.

This task takes **(b)**. Each site converts after the function's validation, because `float()` of an unchecked value would accept `True` as 1.0 and `"0.5"` as 0.5. Where a function has one drifting expression, `float(dt)` goes inline. Where it has two, one `dt = float(dt)` line follows the validation. `time_domain_factor` validated nothing, so it gains `validate_dt(dt)` before its conversion.

| Site | Drifting expression(s) | Public functions it fixes |
|---|---|---|
| `grid.delta_F` | `1.0 / (2.0 * dt * Nf)` | `delta_F`, `band_frequencies` |
| `grid.delta_T` | `float(Nf * dt)` | `delta_T`, `time_pixel_centers` |
| `filter.phi_centered` | `T = Nf * Nt * dt` | `phi_centered` (also `phi_fft_order`, `phi_squared_normalization`) |
| `noise.band_averaged_variance` | `n_samples(Nf, Nt) * dt`, `0.5 / dt` | `band_averaged_variance` |
| `noise.colored_noise` | `rfftfreq(N, d=dt)` | `colored_noise`, `empirical_variance` |
| `noise.evolutionary_band_variance` | `N * dt`, `0.5 / dt` | `evolutionary_band_variance` |
| `matrix.build_C_time` | `rfftfreq(N, d=dt)` | `build_C_time`, `build_C_wdm`, `build_C_wdm_active` |
| `stationary._normalize_noise_to_sigma2` | `fftfreq(N, d=dt)` | `build_lambda_blocks_full_active`, `build_K_full_active` |
| `evolutionary.time_domain_factor` | `rfftfreq(N, d=dt)`, `1.0 / (N * dt)` | `time_domain_factor`, `time_domain_covariance`, `draw_noise`, `exact_covariance` |
| `_covariance.wdm_covariance` | `0.5 / dt`, `Nf * Nt * dt` | `wdm_covariance` |

`delta_T` and `phi_centered`'s `T` round only when `Nf·dt` or `Nf·Nt·dt` is not exact in float32. On 8×16 a power-of-two factor keeps them exact, so the test runs on 6×10 (N = 60) with `dt = np.float32(0.1)`, where every site drifts.

The drift before the fix, measured in the prototype (relative to the peak of the `float(dt)` result, dt = float32(0.1), 6×10):

```
delta_F                          float32  DIFFERS  8.9e-09
delta_T                          float64  DIFFERS  2.5e-08
band_frequencies                 float64  DIFFERS  8.9e-09
time_pixel_centers               float64  DIFFERS  2.5e-08
phi_centered                     float64  DIFFERS  5.1e-08
band_averaged_variance           float64  DIFFERS  7.3e-09
evolutionary_band_variance       float64  DIFFERS  9.9e-09
colored_noise                    complex128 DIFFERS  8.8e-09
empirical_variance               float64  DIFFERS  1.8e-08
build_C_time                     float64  DIFFERS  1.9e-08
build_C_wdm                      float64  DIFFERS  2.2e-08
build_C_wdm_active               float64  DIFFERS  2.2e-08
build_lambda_blocks_full_active  float64  DIFFERS  3.9e-08
build_K_full_active              complex128 DIFFERS  3.9e-08
time_domain_factor               float64  DIFFERS  2.8e-06
time_domain_covariance           float64  DIFFERS  1.6e-06
draw_noise                       float64  DIFFERS  3.9e-06
exact_covariance                 float64  DIFFERS  4.3e-07
wdm_covariance                   float64  DIFFERS  7.3e-09
```

`delta_F` also returned a float32 scalar. In the prototype, a sweep found no other drift. It ran every public function that takes `dt` (the 41 in `__all__` of `domainia`, `domainia.wilson` and its submodules, `validate_dt` aside), `Grid.dF` and `Grid.dT`, and `domainia.testing`'s `dense_W`, `wdm_congruence` and `eq45_dense`, for 6 values of `dt` on the grids 8×16, 6×10 and 12×6. After the fix every one equals its `float(dt)` run in dtype and value. Reverting any one of the 10 sites on its own makes the new test fail on that site's row (prototype mutation check).

**Choices the task leaves open** (plan deviations):
- The test has one row per affected public function, except `phi_fft_order` and `phi_squared_normalization`. Spec §4 deletes both in Task 11, and each is `phi_centered` on the same arguments, so the `phi_centered` row covers them.
- `Grid` stores `dt` as it is given, so `Grid(6, 10, np.float32(0.1)).dt` is still a float32. Every computation from a `Grid` goes through the converted functions (`dF` is `delta_F` and `dT` is `delta_T`), so it is left unchanged.
- `time_domain_factor` gains a `validate_dt` call (see above), a behaviour change that the task did not name.

**Which tests go** (spec §6.3).
- `tests/wilson/vendored/test_noise.py` has 10 tests in 136 lines. Their bodies are 102 lines, the spec's figure. Every test calls an Eq. 39 function:
  - 7 test Eq. 39 itself, and go: `test_whittle_variance_shape`, `test_whittle_variance_uniform_over_n`, `test_whittle_variance_value_flat_psd`, `test_whittle_from_array`, `test_evolutionary_variance_shape`, `test_evolutionary_reduces_to_whittle_for_stationary` and `test_evaluate_psd_on_band_grid`.
  - 3 test kept code against an Eq. 39 reference:
    - `test_band_averaged_matches_whittle_for_flat_psd` goes. Its property, that the band average of a flat PSD is S/2, is covered more strongly by `tests/wilson/test_window_threading.py::test_band_variances_are_the_diagonal_of_the_dense_covariance`, which compares with the dense `W C Wᵀ` diagonal in 5 windows on 4 grids, and by `band_averaged_variance`'s doctest.
    - `test_band_averaged_differs_for_steep_psd` goes. Its subject is the gap between Eq. 36 and Eq. 39, which spec §3.3 records (0.20 in interior bands, 1.5 in edge bands) as the reason Eq. 39 goes.
    - `test_empirical_variance_matches_whittle_flat_psd` stays, with `band_averaged_variance` as its reference, and is renamed `test_empirical_variance_matches_the_band_average_for_a_flat_psd`. It is the only test that the Monte-Carlo estimate (`colored_noise`, then `forward`) converges to the analytic variance. On its flat PSD `band_averaged_variance` is S/2, as `whittle_variance` was. This keeps 8 of the spec's 102 lines (plan deviation).
  - The file becomes 17 lines and 1 item. The vendored conftest's fixtures stay: `simple_grid`, `flat_psd` and `lisa_psd` are still used by the other vendored files.
- `tests/wilson/test_noise_stationary.py`: `test_variance_builders_are_the_oracle_to_rounding` loses its three Eq. 39 legs (`whittle_variance`; `evolutionary_variance`; `evaluate_psd_on_band_grid` with `whittle_variance_from_array`) and is renamed `test_band_averaged_variance_is_the_oracle_to_rounding`. `_psd_ft`, used only by the `evolutionary_variance` leg, goes. The module docstring no longer lists the Eq. 39 quantities. Its item count stays 36.
- The doctests of the four functions go: `src/domainia/wilson` goes from 55 to 51 doctests.
- No other test calls the four names. `tests/wilson/test_window_threading.py` needs no change: none of the four samples the window, so `CONSUMERS` and `SCANNED` do not name them.

**The oracle copies keep the four names.** `tests/wilson/oracle/vendored_2f7136c/noise.py`, `grid.py` and `transform.py` and `tests/wilson/oracle/wdm_dev_ev/evolutionary.py` mention them. They stay whole: their bytes are pinned by `tests/wilson/test_oracle_pins.py` and by `test_evolutionary_kernels.py`'s sha256 check, and spec §6.3 keeps them.

**Files:**
- Modify, source:
  - `src/domainia/wilson/noise.py`: delete `evaluate_psd_on_band_grid`, `whittle_variance_from_array`, `whittle_variance` and `evolutionary_variance`; the module docstring, the `.grid` import, `__all__`; in `band_averaged_variance` the docstring and the last line; the `empirical_variance` and `evolutionary_band_variance` docstrings (Step 1). The `dt` conversions in `band_averaged_variance`, `colored_noise` and `evolutionary_band_variance` (Step 5).
  - `src/domainia/wilson/__init__.py`: the docstring's noise list, the `.noise` import, `__all__` (Step 2).
  - `src/domainia/wilson/grid.py`: the `time_pixel_centers` note (Step 2); `delta_F` and `delta_T` (Step 5).
  - `src/domainia/wilson/filter.py` (`phi_centered`), `src/domainia/wilson/matrix.py` (`build_C_time`), `src/domainia/wilson/stationary.py` (`_normalize_noise_to_sigma2`), `src/domainia/wilson/evolutionary.py` (`time_domain_factor`) and `src/domainia/_covariance.py` (`wdm_covariance`) (Step 5).
- Modify, tests:
  - `tests/wilson/vendored/test_noise.py`, replaced whole (Step 3);
  - `tests/wilson/test_noise_stationary.py`: the module docstring, `_psd_ft` and `test_variance_builders_are_the_oracle_to_rounding` (Step 3);
  - `tests/wilson/test_wrong_kinds.py`: the imports and `test_float32_input_is_computed_in_double` (Step 4, test first).
- Create, delete: none.
- No change: `tests/wilson/meyer_cases.py`, `tests/wilson/test_window_threading.py`, `README.md`.
- Test: `tests/wilson/test_wrong_kinds.py::test_float32_input_is_computed_in_double`; `tests/wilson/test_meyer_baseline.py` (the replay, 5 items).

**Acceptance Criteria:**
- [ ] Step 2: the exact replay check prints `0 of 1408 arrays differ from the baseline` and `6 noise.band_averaged_variance arrays, worst relative difference 0.0e+00`.
- [ ] Step 3: `pytest -q` on `tests/wilson/vendored/test_noise.py tests/wilson/test_noise_stationary.py src/domainia/wilson` gives `88 passed` (1 + 36 + 51).
- [ ] Step 4's run gives `1 failed, 116 passed`, with `AssertionError: dt delta_F` (float32 against float64).
- [ ] Step 5: `tests/wilson/test_wrong_kinds.py` gives `117 passed`, in PY and in FLOOR.
- [ ] Step 6: the word grep prints nothing; the task's literal grep prints exactly the 7 lines listed there, all `band_averaged_evolutionary_variance`.
- [ ] Step 7: `tests/wilson/test_meyer_baseline.py` gives `5 passed`; `python tests/wilson/meyer_cases.py --check` prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`; the exact check prints `0 of 1408 arrays differ from the baseline`. All three in PY and in FLOOR.
- [ ] Full suite (prototype): PY `1036 passed, 3 skipped`; FLOOR `1036 passed, 3 skipped` (Task 9: 1049; −9 vendored tests, −4 doctests). No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 5188 lines / 1983 code` and `tests 4490 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1` → `1036 passed, 3 skipped in …`

**Steps:**

Edits are anchored to code text, never to line numbers. Each Old text occurs exactly once in its file when its step runs, and New replaces it. "New: nothing" means the Old lines go, newline included.

**Deleting a named function** (Step 1) follows one rule:
- Delete from its `def` line through its last line, with the two blank lines after it.
- None of the four has a decorator or a comment block directly above it, and none is the last thing in the file.
- Deleting the four in any order gives the same file, and so does making Step 1's Old/New edits before or after them.

Between Step 1 and Step 2, `import domainia` fails: `wilson/__init__.py` still imports the deleted names. Step 2 fixes it.

Step 8 is a git write in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: Delete Eq. 39 from `src/domainia/wilson/noise.py`**

Delete these four functions by the rule above: `evaluate_psd_on_band_grid`, `whittle_variance_from_array`, `whittle_variance` and `evolutionary_variance`.

Then make these edits. Edit 1 rewrites the module docstring's list of grids: Eq. 39's two entries go, and `evolutionary_band_variance` takes their place. Its last paragraph drops the stale "inner-product functions" and says which function zeroes the inactive cells (Task 9's `wdm_covariance`). Edit 2 drops `time_pixel_centers` from the import: its only user was `evolutionary_variance`. Edit 6 is spec §4's new last line of `band_averaged_variance`.

**Edit 1** (`src/domainia/wilson/noise.py`). Old:

```python
uniformly across all pixels.  This
module exposes three flavors of variance grid:

* :func:`whittle_variance` — point-evaluate the PSD at band centers
  (cheapest, exact when $S_n$ is constant within each band).
* :func:`band_averaged_variance` — filter-weighted mean of $S_n$ over
  each band; exact, and the right choice for steep PSDs.
* :func:`evolutionary_variance` — pixel-dependent variance for
  locally-stationary noise with a time-varying PSD $S_n(f, t)$.

A Monte-Carlo estimator (:func:`empirical_variance`) is provided for
verification.

All variance grids return shape ``(Nf+1, Nt)`` so they slot directly
into the inner-product functions.  The analytic grids fill the inactive edge pixels
($m = 0$ and $m = N_f$ at odd $n$) by the same formula as the active ones, so they hold
about $S_n/2$ there, not 0; nothing in domainia reads those pixels.
```

New:

```python
uniformly across all pixels.  Both variance grids here are the filter-weighted
band average of WDM Paper II, Eq. 36:

* :func:`band_averaged_variance` — filter-weighted mean of $S_n$ over
  each band; exact, and the right choice for steep PSDs.
* :func:`evolutionary_band_variance` — the same mean of a time-varying PSD
  $S_n(f, t)$ at each time pixel, for locally-stationary noise.

A Monte-Carlo estimator (:func:`empirical_variance`) is provided for
verification.

Both grids have shape ``(Nf+1, Nt)``.  They fill the inactive edge pixels
($m = 0$ and $m = N_f$ at odd $n$) by the same formula as the active ones, so they hold
about $S_n/2$ there, not 0; :func:`domainia.wdm_covariance` sets those cells to 0.
```

**Edit 2** (`src/domainia/wilson/noise.py`). Old:

```python
    sanitize_psd,
    time_pixel_centers,
    validate_dt,
```

New:

```python
    sanitize_psd,
    validate_dt,
```

**Edit 3** (`src/domainia/wilson/noise.py`). Old:

```python
__all__ = [
    "band_averaged_variance",
    "colored_noise",
    "empirical_variance",
    "evaluate_psd_on_band_grid",
    "evolutionary_band_variance",
    "evolutionary_variance",
    "whittle_variance",
    "whittle_variance_from_array",
]
```

New:

```python
__all__ = [
    "band_averaged_variance",
    "colored_noise",
    "empirical_variance",
    "evolutionary_band_variance",
]
```

**Edit 4** (`src/domainia/wilson/noise.py`). Old:

```python
    the band.  Strictly more accurate than :func:`whittle_variance`,
    which point-evaluates $S_n$ at the band center.
```

New:

```python
    the band.
```

**Edit 5** (`src/domainia/wilson/noise.py`). Old:

```python
       accurate than :func:`whittle_variance` in every interior band.
```

New:

```python
       accurate than $S_n(f_m)/2$ at the band centre in every interior band.
```

**Edit 6** (`src/domainia/wilson/noise.py`). Old:

```python
    return whittle_variance_from_array(Nt, Sn_eff)
```

New:

```python
    return np.broadcast_to(0.5 * Sn_eff[:, None], (Nf + 1, Nt)).copy()
```

**Edit 7** (`src/domainia/wilson/noise.py`). Old:

```python
    squared coefficients.  Useful for verifying the analytical
    :func:`whittle_variance` and :func:`band_averaged_variance`.
```

New:

```python
    squared coefficients.  Useful for verifying the analytical
    :func:`band_averaged_variance`.
```

**Edit 8** (`src/domainia/wilson/noise.py`). Old:

```python
    :func:`evolutionary_variance`).  ``Sn(f, t)`` is elementwise and broadcasting
```

New:

```python
    :func:`band_averaged_variance`).  ``Sn(f, t)`` is elementwise and broadcasting
```

- [ ] **Step 2: The exports and the `time_pixel_centers` note; the replay is unchanged**

**Edit 1** (`src/domainia/wilson/__init__.py`). Old:

```python
Noise variance grids:

* :func:`whittle_variance`, :func:`whittle_variance_from_array`
* :func:`band_averaged_variance`
* :func:`evolutionary_variance`
* :func:`empirical_variance`
* :func:`evaluate_psd_on_band_grid`
```

New:

```python
Noise variance grids:

* :func:`band_averaged_variance`
* :func:`empirical_variance`
```

**Edit 2** (`src/domainia/wilson/__init__.py`). Old:

```python
from .noise import (
    band_averaged_variance,
    empirical_variance,
    evaluate_psd_on_band_grid,
    evolutionary_variance,
    whittle_variance,
    whittle_variance_from_array,
)
```

New:

```python
from .noise import band_averaged_variance, empirical_variance
```

**Edit 3** (`src/domainia/wilson/__init__.py`). Old:

```python
    "empirical_variance",
    "evaluate_psd_on_band_grid",
    "evolutionary_variance",
    # Submodules
```

New:

```python
    "empirical_variance",
    # Submodules
```

**Edit 4** (`src/domainia/wilson/__init__.py`). Old:

```python
    "validate_grid",
    # Noise
    "whittle_variance",
    "whittle_variance_from_array",
    "windows",
```

New:

```python
    "validate_grid",
    "windows",
```

**Edit 5** (`src/domainia/wilson/grid.py`). Old:

```python
       inverse transform, the offset was exactly $-0.5\\,\\Delta T$ at every
       grid tested. The consumer inside the package is
       :func:`domainia.wilson.noise.evolutionary_variance`, which sampled a time-varying
       PSD half a pixel off.
```

New:

```python
       inverse transform, the offset was exactly $-0.5\\,\\Delta T$ at every
       grid tested.
```

The exact check compares every kept replay array with the recorded baseline by dtype and `np.array_equal`, which is stricter than the replay's 1e-12 gate. It also prints the worst relative difference of the `noise.band_averaged_variance` arrays.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
    ba = [k for k in kept if '::noise.band_averaged_variance::' in k]
    worst = max(float(np.max(np.abs(got[k] - z[k])) / np.max(np.abs(z[k]))) for k in ba)
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')
print(f'{len(ba)} noise.band_averaged_variance arrays, worst relative difference {worst:.1e}')"
```

Expected:
```
0 of 1408 arrays differ from the baseline
6 noise.band_averaged_variance arrays, worst relative difference 0.0e+00
```

The prototype prints the same with the FLOOR interpreter (`$TMPDIR/dom3-envs/floor312/bin/python` in place of the PY one).

- [ ] **Step 3: The Eq. 39 tests**

Replace the whole file `tests/wilson/vendored/test_noise.py` with this text:

```python
"""Tests for wdm.noise."""

from __future__ import annotations

import numpy as np

from domainia.wilson.noise import band_averaged_variance, empirical_variance


def test_empirical_variance_matches_the_band_average_for_a_flat_psd(simple_grid, flat_psd):
    """MC estimate of variance matches the band-averaged variance for flat PSD."""
    g = simple_grid
    var_mc = empirical_variance(g.Nf, g.Nt, g.dt, flat_psd, n_realizations=2000, seed=0)
    var_pred = band_averaged_variance(g.Nf, g.Nt, g.dt, flat_psd)
    # Compare interior bands; MC noise scales as 1/sqrt(n_realizations).
    rel_err = np.abs(var_mc[1 : g.Nf] - var_pred[1 : g.Nf]) / var_pred[1 : g.Nf]
    assert np.median(rel_err) < 0.05
```

Then edit `tests/wilson/test_noise_stationary.py`:

**Edit 1** (`tests/wilson/test_noise_stationary.py`). Old:

```python
arrays are scaled by 2 pi, a convention change).  Window-independent quantities
(``whittle_variance``, ``evolutionary_variance``, the band-grid PSD, ``colored_noise``,
the active-index lists) are still compared bit for bit.
```

New:

```python
arrays are scaled by 2 pi, a convention change).  Window-independent quantities
(``colored_noise``, the active-index lists) are still compared bit for bit.
```

**Edit 2** (`tests/wilson/test_noise_stationary.py`). Old:

```python
def _psd_ft(f, t):
    return np.outer(_psd(f), 1.0 + 0.1 * np.asarray(t) / (1.0 + np.max(t)))


```

New: nothing (delete the Old lines).

**Edit 3** (`tests/wilson/test_noise_stationary.py`). Old:

```python
def test_variance_builders_are_the_oracle_to_rounding(grid):
    """band_averaged_variance (a phi^2-weighted average) to rounding; the rest do not
    touch the window and are still the oracle bit for bit."""
    Nf, Nt, dt = grid
    got = noise.whittle_variance(Nf, Nt, dt, _psd)
    ref = onoise.whittle_variance(Nf, Nt, dt, _psd)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)
    got = noise.band_averaged_variance(Nf, Nt, dt, _psd)
    ref = onoise.band_averaged_variance(Nf, Nt, dt, _psd)
    assert got.dtype == ref.dtype
    assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 4.8e-16
    assert np.array_equal(noise.evolutionary_variance(Nf, Nt, dt, _psd_ft),
                          onoise.evolutionary_variance(Nf, Nt, dt, _psd_ft))
    band = noise.evaluate_psd_on_band_grid(Nf, dt, _psd)
    assert np.array_equal(band, onoise.evaluate_psd_on_band_grid(Nf, dt, _psd))
    assert np.array_equal(noise.whittle_variance_from_array(Nt, band),
                          onoise.whittle_variance_from_array(Nt, band))
```

New:

```python
def test_band_averaged_variance_is_the_oracle_to_rounding(grid):
    """band_averaged_variance (a phi^2-weighted average) to rounding."""
    Nf, Nt, dt = grid
    got = noise.band_averaged_variance(Nf, Nt, dt, _psd)
    ref = onoise.band_averaged_variance(Nf, Nt, dt, _psd)
    assert got.dtype == ref.dtype
    assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 4.8e-16
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/vendored/test_noise.py tests/wilson/test_noise_stationary.py src/domainia/wilson 2>&1 | tail -1`

Expected: `88 passed in …`. That is 1 in `vendored/test_noise.py`, 36 in `test_noise_stationary.py` and 51 doctests.

- [ ] **Step 4: The float32-`dt` test comes first (`tests/wilson/test_wrong_kinds.py`)**

`test_float32_input_is_computed_in_double` gains one row per affected public function in its loop. Each row takes a `dt`: the loop's `good` is `np.float64(0.1)`, `low = good.astype(np.float32)` is `np.float32(0.1)`, and the run on `low` is compared with the run on `low.astype(np.float64)`, the same value in double. The loop now wraps each result in `np.asarray`, because `delta_F` and `delta_T` return scalars; for an array result it changes nothing. The rows run on 6×10 (N = 60), where `Nf·dt` and `Nf·Nt·dt` round in float32. `domainia` is imported for `wdm_covariance`.

**Edit 1** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
from domainia import wilson
from domainia.wilson.filter import nu_d, phi_centered
```

New:

```python
import domainia
from domainia import wilson
from domainia.wilson.filter import nu_d, phi_centered
```

**Edit 2** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
def test_float32_input_is_computed_in_double():
    """Every array input is promoted to double before the first FFT: a float32 or complex64
    input gives the double run's result on the same values, dtype included.  forward_time's
    rfft would otherwise run in single precision."""
    x = np.fft.irfft(_Z, 128)
    calls = {slot: (call, good) for slot, (call, good, _) in _REAL_ARRAYS.items()} | {
        "forward": (lambda a: wilson.forward(a, 8, 16, 1.0), _Z),
        "forward_time": (lambda a: wilson.forward_time(a, 8, 16, 1.0), x),
    }
    for slot, (call, good) in calls.items():
        low = good.astype(np.complex64 if good.dtype.kind == "c" else np.float32)
        got, want = call(low), call(low.astype(good.dtype))
```

New:

```python
def test_float32_input_is_computed_in_double():
    """Every array input is promoted to double before the first FFT: a float32 or complex64
    input gives the double run's result on the same values, dtype included.  forward_time's
    rfft would otherwise run in single precision.  A float32 ``dt`` is computed in double
    too: ``N * dt`` or ``0.5 / dt`` on a float32 scalar rounds in float32 (6 x 10, N = 60)."""
    x = np.fft.irfft(_Z, 128)
    psd = lambda f: 1.0 / (1.0 + (4.0 * f) ** 2) + 0.1
    evo = lambda f, t: psd(f) * (1.0 + 0.3 * np.cos(t / 7.0))
    M, St, E, No = wilson.matrix, wilson.stationary, wilson.evolutionary, wilson.noise
    dts = {
        "delta_F": lambda dt: wilson.delta_F(6, dt),
        "delta_T": lambda dt: wilson.delta_T(6, dt),
        "band_frequencies": lambda dt: wilson.band_frequencies(6, dt),
        "time_pixel_centers": lambda dt: wilson.time_pixel_centers(6, 10, dt),
        "phi_centered": lambda dt: phi_centered(6, 10, dt),
        "band_averaged_variance": lambda dt: wilson.band_averaged_variance(6, 10, dt, psd),
        "evolutionary_band_variance": lambda dt: No.evolutionary_band_variance(6, 10, dt, evo),
        "colored_noise": lambda dt: No.colored_noise(60, dt, psd, np.random.default_rng(0)),
        "empirical_variance": lambda dt: wilson.empirical_variance(6, 10, dt, psd, 2, seed=0),
        "build_C_time": lambda dt: M.build_C_time(60, dt, psd),
        "build_C_wdm": lambda dt: M.build_C_wdm(6, 10, dt, psd),
        "build_C_wdm_active": lambda dt: M.build_C_wdm_active(6, 10, dt, psd),
        "build_lambda_blocks_full_active": lambda dt: St.build_lambda_blocks_full_active(
            6, 10, dt, psd)[0][1],
        "build_K_full_active": lambda dt: St.build_K_full_active(6, 10, dt, psd)[0][1],
        "time_domain_factor": lambda dt: E.time_domain_factor(60, dt, evo),
        "time_domain_covariance": lambda dt: E.time_domain_covariance(60, dt, evo),
        "draw_noise": lambda dt: E.draw_noise(60, dt, evo, np.random.default_rng(0)),
        "exact_covariance": lambda dt: E.exact_covariance(6, 10, dt, evo),
        "wdm_covariance": lambda dt: domainia.wdm_covariance(lambda f, t: psd(f)[None, None],
                                                             Nf=6, Nt=10, dt=dt),
    }
    calls = {slot: (call, good) for slot, (call, good, _) in _REAL_ARRAYS.items()} | {
        "forward": (lambda a: wilson.forward(a, 8, 16, 1.0), _Z),
        "forward_time": (lambda a: wilson.forward_time(a, 8, 16, 1.0), x),
    } | {f"dt {name}": (call, np.float64(0.1)) for name, call in dts.items()}
    for slot, (call, good) in calls.items():
        low = good.astype(np.complex64 if good.dtype.kind == "c" else np.float32)
        got, want = np.asarray(call(low)), np.asarray(call(low.astype(good.dtype)))
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py 2>&1 | grep -E "^E +AssertionError|passed|failed"`

Expected:
```
E           AssertionError: dt delta_F
1 failed, 116 passed in …
```
`delta_F` on a float32 `dt` returns a float32 scalar. The loop stops at the first failing row; every one of the 19 `dt` rows fails before Step 5 (the drift table above).

- [ ] **Step 5: `float(dt)` at the 10 sites**

Edits 1 and 2 are in `grid.py` (`delta_F`, `delta_T`), Edit 3 in `filter.py` (`phi_centered`), Edits 4 to 6 in `noise.py` (`band_averaged_variance`, `colored_noise`, `evolutionary_band_variance`), Edit 7 in `matrix.py` (`build_C_time`), Edit 8 in `stationary.py` (`_normalize_noise_to_sigma2`), Edit 9 in `evolutionary.py` (`time_domain_factor`) and Edit 10 in `_covariance.py` (`wdm_covariance`). In `evolutionary_band_variance` the conversion follows the `sample_window` line, because that call is the function's only `dt` check. `evolutionary.py` already imports `validate_dt`.

**Edit 1** (`src/domainia/wilson/grid.py`). Old:

```python
    width = 1.0 / (2.0 * dt * Nf)
```

New:

```python
    width = 1.0 / (2.0 * float(dt) * Nf)
```

**Edit 2** (`src/domainia/wilson/grid.py`). Old:

```python
    return float(Nf * dt)
```

New:

```python
    return float(Nf) * float(dt)
```

**Edit 3** (`src/domainia/wilson/filter.py`). Old:

```python
    T = Nf * Nt * dt
```

New:

```python
    T = Nf * Nt * float(dt)
```

**Edit 4** (`src/domainia/wilson/noise.py`). Old:

```python
    validate_grid(Nf, Nt)
    validate_dt(dt)
    phi = sample_window(window, Nf, Nt, dt)
```

New:

```python
    validate_grid(Nf, Nt)
    validate_dt(dt)
    dt = float(dt)
    phi = sample_window(window, Nf, Nt, dt)
```

**Edit 5** (`src/domainia/wilson/noise.py`). Old:

```python
    freqs = np.fft.rfftfreq(N, d=dt)
```

New:

```python
    freqs = np.fft.rfftfreq(N, d=float(dt))
```

**Edit 6** (`src/domainia/wilson/noise.py`). Old:

```python
    phi2 = sample_window(window, Nf, Nt, dt) ** 2
```

New:

```python
    phi2 = sample_window(window, Nf, Nt, dt) ** 2
    dt = float(dt)
```

**Edit 7** (`src/domainia/wilson/matrix.py`). Old:

```python
    freqs = np.fft.rfftfreq(N, d=dt)
```

New:

```python
    freqs = np.fft.rfftfreq(N, d=float(dt))
```

**Edit 8** (`src/domainia/wilson/stationary.py`). Old:

```python
        freqs_full = np.fft.fftfreq(N, d=dt)
```

New:

```python
        freqs_full = np.fft.fftfreq(N, d=float(dt))
```

**Edit 9** (`src/domainia/wilson/evolutionary.py`). Old:

```python
    if N % 2:
        raise ValueError("N must be even")
    f = np.fft.rfftfreq(N, d=dt)
```

New:

```python
    if N % 2:
        raise ValueError("N must be even")
    validate_dt(dt)
    dt = float(dt)
    f = np.fft.rfftfreq(N, d=dt)
```

**Edit 10** (`src/domainia/_covariance.py`). Old:

```python
    require_callable("S", S)
    validate_grid(Nf, Nt)
    validate_dt(dt)
```

New:

```python
    require_callable("S", S)
    validate_grid(Nf, Nt)
    validate_dt(dt)
    dt = float(dt)
```

Every edit is a no-op on a float64 or Python-float `dt`: `float(dt)` is the same value, and the arithmetic is the same IEEE operations, so the replay stays bit for bit (Step 7).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_wrong_kinds.py 2>&1 | tail -1`

Expected: `117 passed in …`. The prototype gives the same with the FLOOR interpreter.

- [ ] **Step 6: Greps**

The deleted names, as whole words, outside the oracle copies:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnwE "whittle_variance(_from_array)?|evaluate_psd_on_band_grid|evolutionary_variance" src tests README.md pyproject.toml | grep -v "^tests/wilson/oracle/"`

Expected: no output (grep exits 1).

The task's literal regex has no leading word boundary, so it also matches inside `band_averaged_evolutionary_variance`. That is wdm_dev's name for Eq. 36, the oracle function that `noise.evolutionary_band_variance` was ported from (`wdm_dev_ev.evolutionary`, imported as `ev`). It is kept code's reference, not Eq. 39.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "whittle_variance|evaluate_psd_on_band_grid|evolutionary_variance\b" src tests --include='*.py' | grep -v "^tests/wilson/oracle/" | LC_ALL=C sort`

Expected, exactly these 7 lines:
```
src/domainia/wilson/noise.py:316:    ``evolutionary.band_averaged_evolutionary_variance``.
tests/wilson/test_evolutionary_kernels.py:101:                ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo)) <= ORACLE_RTOL
tests/wilson/test_evolutionary_kernels.py:103:                ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF])
tests/wilson/test_evolutionary_kernels.py:361:                 "wdm_discarded", "band_averaged_evolutionary_variance"):
tests/wilson/vendored/test_evolutionary.py:78:    got = ev.band_averaged_evolutionary_variance(NF, NT, DT, lambda f, t: _S_stat(f) + 0 * t)
tests/wilson/vendored/test_evolutionary.py:84:    sep = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo)
tests/wilson/vendored/test_evolutionary.py:88:    sub = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF])
```
- `noise.py`: `evolutionary_band_variance`'s docstring names the wdm_dev function it was.
- `test_evolutionary_kernels.py` and `vendored/test_evolutionary.py`: oracle calls `ev.band_averaged_evolutionary_variance(...)`, the reference of the kept Eq. 36 kernel; and line 361, `test_evolutionary_has_no_classes_left`, which asserts that domainia's `evolutionary` module does not have that name.

- [ ] **Step 7: The replay**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_meyer_baseline.py 2>&1 | tail -1`

Expected: `5 passed in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`

Step 2's exact check again. Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
    ba = [k for k in kept if '::noise.band_averaged_variance::' in k]
    worst = max(float(np.max(np.abs(got[k] - z[k])) / np.max(np.abs(z[k]))) for k in ba)
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')
print(f'{len(ba)} noise.band_averaged_variance arrays, worst relative difference {worst:.1e}')"
```

Expected:
```
0 of 1408 arrays differ from the baseline
6 noise.band_averaged_variance arrays, worst relative difference 0.0e+00
```

The prototype gives the same three results with the FLOOR interpreter.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: the suite passes. Prototype: `1036 passed, 3 skipped in …` in PY, and the same in FLOOR (the Verify command). No warnings, no collection errors.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected (prototype):
```
src 5188 lines / 1983 code
tests 4490 lines (+ oracle 4135)
```
That is −179 src lines (−54 code) and −106 test lines against Task 9 (src 5367 / 2037; tests 4596). src is inside the spec's pre-trim budget (at most about 5,400 / 2,150) with 212 lines and 167 code lines to spare.

- [ ] **Step 8: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/_covariance.py src/domainia/wilson/__init__.py src/domainia/wilson/evolutionary.py src/domainia/wilson/filter.py src/domainia/wilson/grid.py src/domainia/wilson/matrix.py src/domainia/wilson/noise.py src/domainia/wilson/stationary.py tests/wilson/test_noise_stationary.py tests/wilson/test_wrong_kinds.py tests/wilson/vendored/test_noise.py && git status --short | grep -vc "^M "`

Expected: `0`. Every status line is a staged modification, so nothing is unstaged or untracked. (`grep -c` exits 1 when it prints 0, so this command does not continue with `&&`.)

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Delete Eq. 39; band_averaged_variance returns its grid itself; a float32 dt computes in double" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `11 files changed, 62 insertions(+), 347 deletions(-)`

Size after Task 10 (prototype): src 5188 lines / 1983 code; tests 4490 (+ oracle 4135).

---

### Task 11: Dead code inside the kept files; the public API; README; pyproject; version 3.0.0 (spec Order step 6; §4 "Inside the kept files"; §5 "Public API"; §7)

**Goal:** The dead helpers of the kept files are gone, `colored_noise` and `empirical_variance` live in `domainia.testing`, `domainia` and `domainia.wilson` export exactly spec §5's names, and the version, the description, the README and the module docstrings describe domainia 3.0.

Assumes the tree after Task 10, whose commit is "Delete Eq. 39; band_averaged_variance returns its grid itself; a float32 dt computes in double".

**What changes for a caller.**

`domainia.__all__` becomes `__version__`, `frequency_covariance`, `transform`, `wdm_covariance`, `wilson`. `domainia.wilson.__all__` becomes spec §5's list: `Grid`, `forward`, `inverse`, `forward_time`, `inverse_time`; `Window`, `Meyer`, `MEYER`, `RampWindow`, `SampledWindow`; `band_averaged_variance`, `evolutionary_band_variance`; and the submodules `evolutionary`, `stationary`, `matrix`, `noise`, `grid`, `filter`, `windows`. A name that leaves an `__all__` also leaves that package's namespace (it is no longer imported there), and it stays importable from its submodule:

| Name | Was | Now only at |
|---|---|---|
| `Grid`, `MEYER`, `Meyer`, `RampWindow`, `SampledWindow`, `Window` | `domainia` and `domainia.wilson` | `domainia.wilson` (and `.grid` / `.windows`) |
| `WindowError` | `domainia` and `domainia.wilson` | `domainia.wilson.windows` |
| `require_window`, `sample_window` | `domainia.wilson` | `domainia.wilson.windows` |
| `active_mask`, `band_frequencies`, `delta_F`, `delta_T`, `n_samples`, `time_pixel_centers`, `validate_dt`, `validate_finite`, `validate_grid` | `domainia.wilson` | `domainia.wilson.grid` |
| `phi_centered` | `domainia.wilson` | `domainia.wilson.filter` |
| `colored_noise` | `domainia.wilson.noise` | `domainia.testing` |
| `empirical_variance` | `domainia.wilson` and `domainia.wilson.noise` | `domainia.testing` |
| `count_active`, `pack`, `unpack`, `phi_fft_order` | `domainia.wilson` (and `.grid` / `.filter`) | deleted |
| `band_indices`, `carrier_phase`, `edge_columns`, `packed_cells` | `domainia.wilson.grid` | deleted |
| `phi_squared_normalization` | `domainia.wilson.filter` | deleted |
| `require_bool`, `require_str`, `index_array` | `domainia.wilson._checks` | deleted |

- `evolutionary_band_variance` and `grid` join `domainia.wilson.__all__`; until now only `wilson.noise.evolutionary_band_variance` existed.
- `domainia.wilson.transform` (the submodule) is still imported by `domainia.wilson`, so `wilson.transform.atom_spectrum` keeps working. Spec §5's submodule list does not name it, so it is not in `__all__`.
- `colored_noise` and `empirical_variance` move unchanged, code byte for byte. Only three things differ: their PSD annotation is `Callable[[Any], Any]` (testing.py's style; `PsdFunc` and `CArray` are `noise.py`'s aliases), `colored_noise` returns `NDArray[np.complex128]`, and `empirical_variance`'s docstring drops its Examples. That doctest would never run in testing.py, because pytest collects doctests only from `src/domainia/wilson`. `empirical_variance` calls `forward(..., window=window)`, so `test_window_threading`'s scan of `testing.py` (already in `SCANNED`, with `empirical_variance` in `CONSUMERS`) still checks it.
- The replay key `noise.empirical_variance` keeps its recorded name and is computed by `testing.empirical_variance`. Every kept replay array is still bit for bit the baseline (`0 of 1408 arrays differ`), in PY and FLOOR.
- Version 3.0.0 in `pyproject.toml` and `domainia.__version__`.

**The deletions have no remaining caller.** Spec §4's list was checked against Task 10's tree before deleting anything. `grep -rnw <name> src --include='*.py'` finds, for each of `carrier_phase`, `packed_cells`, `pack`, `unpack`, `edge_columns`, `band_indices`, `count_active`, `require_bool`, `require_str`, `index_array`, `phi_fft_order` and `phi_squared_normalization`, only its own definition, its doctest, its module's `__all__`, and, for the names `domainia.wilson` exports, `wilson/__init__.py`'s import and docstring. The only calls between them are `pack` and `unpack` calling `packed_cells`, and all three go. Nothing is kept for a caller.
- `testing.py`'s helpers of the deleted layers (`blocktri_dense`, `hhat`, `eq54_covariance`, `dense_lambda_generic`, `PointwiseOperator`) are already gone: all five left in Task 2's commit ("Delete the likelihoods, factorisations, covariance classes, operators and coordinates"). `grep -rnwE "blocktri_dense|hhat|eq54_covariance|dense_lambda_generic|PointwiseOperator" src tests` prints nothing at Task 10's tree.
- Kept on purpose: `grid.endpoint_weights` (`evolutionary.time_domain_factor` calls it) and `grid.time_pixel_centers` (not in §4's list; it only leaves `domainia.wilson`'s top level).

**The tests of deleted code** (spec §6.3). Whole test functions are deleted by the rule in the Steps. Item counts are PY collection counts.
- `tests/wilson/vendored/test_grid.py` (21 → 14 items): `test_count_active_equals_N` (4 items), `test_band_indices_unique_modN`, `test_carrier_phase_even_m`, `test_carrier_phase_odd_m`. That the active cells number `N` is still pinned by `active_mask`'s doctest (`int(mask.sum())` is 128).
- `tests/wilson/vendored/test_filter.py` (15 → 13): `test_phi_fft_order_is_ifftshift_of_centered`, `test_phi_fft_order_peaked_at_zero`, and the `phi_fft_order` / `phi_squared_normalization` lines at the end of `test_smoothness_order_d_changes_the_window`. That test keeps its 4 items.
- `tests/wilson/test_grid_bands.py` (29 → 24): `test_packed_cells_is_band_order` (4 items) and `test_pack_unpack_round_trip_and_active_W_order`. The second test's one check of kept code, that `active_W`'s rows are the active cells in `np.nonzero` order, is still made by `tests/test_dom_testing.py::test_dense_W_rows_are_the_atoms` (`dense_W` on the full mask is `active_W`; `W @ x` against `forward_time(x)[MASK]`) and by `test_atoms_are_the_rows_of_the_transform` (row `p` of `active_W` against the atom of the `p`-th `np.nonzero` cell). `test_edge_columns_and_endpoint_weights` loses its two `edge_columns` lines and becomes `test_endpoint_weights`. `test_atoms_are_the_rows_of_the_transform` takes its cells from `np.nonzero(active_mask(Nf, Nt))` in place of `packed_cells`.
- `tests/wilson/test_checks.py` (101 → 64): `test_require_bool_returns_a_python_bool`, `test_require_bool_refuses_every_other_kind`, `test_require_str_returns_the_string`, `test_require_str_refuses_every_other_kind`, `test_index_array_returns_int64`, `test_index_array_refuses_a_non_integral_or_non_finite_float_with_value_error`, `test_index_array_refuses_a_non_numeric_dtype_with_type_error`.
- `tests/wilson/test_wrong_kinds.py` (117 → 96): the `_INT_SLOTS` rows `carrier_phase m`, `edge_columns lead` and `edge_columns trail` (7 items each, across the two slot tests).
- `tests/wilson/test_window_threading.py` (402 → 400): `phi_fft_order` leaves `MEYER_HELPERS`, and the two scan cases `phi_fft_order(Nf, Nt, dt)` and `wilson.filter.phi_fft_order(Nf, Nt, dt)` go.
- The doctests of the deleted and moved functions: `src/domainia/wilson` goes from 51 to 40 doctests (`_checks` 10 → 7, `filter` 4 → 2, `grid` 15 → 10, `noise` 2 → 1).
- `tests/test_dom_testing.py::test_eq45_differs_from_exact_at_second_order_with_drift` excluded `wilson.grid.edge_columns(NT, 3, 1)`, the toy models' 3 leading and 1 trailing columns. It now excludes `np.r_[0:3, NT - 1]`, the same columns `(0, 1, 2, 15)`. Spec §3.3 says it "is rewritten with `wrap_cut_columns`", but `wrap_cut_columns(8, 16, 1/16)` is `(6, 5)`, which would exclude 11 of the 16 columns and change the ported test. So the explicit list is kept (plan deviation).

**Choices the task leaves open** (plan deviations):
- `WindowError` leaves both top levels, because spec §5 does not list it. It is raised by every window check, so a caller now catches `domainia.wilson.windows.WindowError`, or `ValueError`, its base class.
- The old namespace paths are rewired wherever they were used (code, tests, doctests, Sphinx references). Step 2's sweep makes 52 replacements, each `wilson.<grid name>` → `wilson.grid.<grid name>` (`wl.` in `meyer_cases.py`). Four explicit edits handle the rest: the two `WindowError` imports, one `phi_centered` doctest, and the two `is wilson.…` legs. No test is weakened. `test_checks.py::test_grid_re_exports_the_checks_under_their_old_names` loses only its `is wilson.validate_dt` and `is wilson.validate_finite` legs: those names no longer exist.
- `tests/test_package.py`: `test_the_public_names` now pins both `__all__` sets exactly. It checks that one leaving name per package (`domainia.MEYER`, `wilson.active_mask`) is gone, and that `wilson.MEYER` and `wilson.Grid` are the submodules' objects. The `pack`/`unpack` asserts go with the functions. The no-JAX subprocess test is unchanged.
- `pyproject.toml`: only `version` and `description` change. The `[jax]` extra and the `jax`/`slow` markers are already gone (Task 5): `optional-dependencies` is `{test = ["pytest>=8"]}` and there is no `markers` list. No other stale entry was found. `xfail_strict = true` stays although no test is marked xfail today: it is a guard for future xfails, not a leftover.
- README.md (117 → 83 lines) is rewritten around the three functions (spec §7). Its data block keeps the case Task 4 (Step 12) carried into the README: raw `wilson.forward_time` output (4.9e-17 in the inactive cells) and a full-FFT slice (1.3e-16 imaginary at Nyquist) on 6 × 12, `dt = 0.3`, seed 6, two channels. Step 7 measures both residues, which shows the block exercises the 1e-10 tolerance and is not vacuous.
- Module docstrings: fixed `domainia/__init__.py` (the 2.0 opening "data and covariance across the time, Fourier and time-frequency ... domains", and `:data:`MEYER`` / `:class:`Window`` now name `domainia.wilson`), `domainia/wilson/__init__.py` (the API lists named `count_active`, `pack`, `unpack`, `phi_fft_order` and `empirical_variance`, and top-level names that left), `testing.py` ("never on a likelihood path"; the new helpers), `wilson/noise.py` (`empirical_variance` moved) and `wilson/stationary.py` ("preferred when bias-free inner products matter (matched filtering ..., anything that consumes $\Lambda^{-1}$)", likelihood use). Checked and kept: `stationary.py`'s "Backward maps (Paper I Alg. 4)" and "operator-form ``lambda_to_psd``" (a paper section and a vendored function's form, not the deleted maps or operator layers), and `matrix.py`'s "linear map $w = W x$" (maths). Two function docstrings were edited, though function docstrings are otherwise left for Task 12: `grid.time_pixel_centers` ("time coordinates" → "times", the only hit of the second grep gate in src) and the Sphinx references `domainia.wilson.validate_grid` / `validate_finite` / `delta_F` in `transform.py` and `filter.py`, which this task's API change would otherwise break (the Step 2 sweep). `grid.sanitize_psd`'s history line about "the Gaussian log-likelihood" is a function docstring and is left for Task 12.

**Files:**
- Modify, source:
  - `src/domainia/__init__.py`: the module docstring, the imports, `__version__`, `__all__` (Step 4).
  - `src/domainia/wilson/__init__.py`: the module docstring's API lists, the imports, `__all__` (Step 4).
  - `src/domainia/testing.py`: the module docstring, the imports, `__all__`; new `colored_noise` and `empirical_variance` (Step 3); `_keep` and the docstring's `wilson.active_mask` (Step 2 sweep).
  - `src/domainia/wilson/grid.py`: delete `band_indices`, `carrier_phase`, `count_active`, `packed_cells`, `pack`, `unpack`, `edge_columns`, the packed-cells comment block, `IntArray`, and their `__all__` entries (Step 5); doctests (Step 2 sweep); `time_pixel_centers`' docstring (Step 6).
  - `src/domainia/wilson/_checks.py`: delete `require_bool`, `require_str`, `index_array` (Step 5); `validate_dt` and `validate_finite` doctests (Step 2 sweep).
  - `src/domainia/wilson/filter.py`: delete `phi_fft_order`, `phi_squared_normalization`, and their `__all__` entries (Step 5); `phi_centered`'s doctest and a reference (Step 2).
  - `src/domainia/wilson/noise.py`: delete `colored_noise` and `empirical_variance`; the module docstring, the imports, `CArray`, `__all__` (Step 5).
  - `src/domainia/wilson/stationary.py`: the module docstring (Step 6).
  - `src/domainia/_transform.py` (`_grid`, `_checked`, `transform`) and `src/domainia/wilson/transform.py` (docstring references) (Step 2 sweep).
  - `pyproject.toml`: `version`, `description` (Step 4).
  - `README.md`: replaced whole (Step 7).
- Modify, tests:
  - `tests/test_package.py`: the module docstring, `PUBLIC` and the new `WILSON_PUBLIC`, `test_the_version_is_3_0_0_everywhere` (renamed), `test_the_public_names` (Step 1, test first).
  - `tests/wilson/test_checks.py`, `tests/wilson/test_window_threading.py`, `tests/wilson/test_windows.py`, `tests/test_covariance.py`, `tests/test_dom_testing.py`, `tests/test_transform.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_wrong_kinds.py`, `tests/wilson/meyer_cases.py` (Step 2: namespace paths).
  - `tests/wilson/meyer_cases.py` (`wilson_cases`), `tests/wilson/test_window_threading.py` (the testing import, `test_empirical_variance_is_the_mean_square_of_the_same_draws`), `tests/wilson/test_wrong_kinds.py` (the imports, `_INT_SLOTS`, `test_integer_slots_out_of_range_stay_value_errors`, `test_float32_input_is_computed_in_double`), `tests/wilson/test_checks.py` (the import), `tests/wilson/test_noise_stationary.py` (the module docstring, the imports and five tests), `tests/wilson/vendored/test_noise.py` (the import) (Step 3: the move).
  - `tests/wilson/vendored/test_grid.py`, `tests/wilson/vendored/test_filter.py`, `tests/wilson/test_grid_bands.py`, `tests/wilson/test_checks.py`, `tests/wilson/test_wrong_kinds.py`, `tests/wilson/test_window_threading.py`, `tests/test_dom_testing.py`, `tests/wilson/test_noise_stationary.py` (`test_dropped_and_renamed_names`) (Step 5: tests of deleted code).
- Create, delete: none.
- No change: `tests/wilson/test_meyer_baseline.py`, `tests/test_readme.py`, `src/domainia/_covariance.py`, `src/domainia/wilson/matrix.py`, `src/domainia/wilson/evolutionary.py`, `src/domainia/wilson/windows.py`.
- Test: `tests/test_package.py` (7 items), `tests/test_readme.py` (2), `tests/wilson/test_meyer_baseline.py` (the replay, 5).

**Acceptance Criteria:**
- [ ] Step 1: `tests/test_package.py` gives `2 failed, 5 passed`; Step 4 makes it `7 passed`.
- [ ] Step 2: the sweep prints the 15 per-file counts listed there (52 replacements), and the namespace grep prints only `phi_squared_normalization`'s `wilson.phi_centered` doctest line, which Step 5 deletes.
- [ ] Step 3: the exact replay check prints `0 of 1408 arrays differ from the baseline` and `6 noise.empirical_variance arrays, 0 differ`.
- [ ] Step 5: the full suite (prototype) gives `951 passed, 3 skipped` in PY and in FLOOR.
- [ ] Step 7: `tests/test_readme.py` gives `2 passed`; README.md is 83 lines.
- [ ] Step 8: each grep prints exactly the lines listed there.
- [ ] Step 9: `tests/wilson/test_meyer_baseline.py` gives `5 passed`; `meyer_cases.py --check` prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`; the exact check prints `0 of 1408 arrays differ from the baseline`. All three in PY and in FLOOR.
- [ ] Full suite (prototype): PY `951 passed, 3 skipped`; FLOOR `951 passed, 3 skipped` (Task 10: 1036; −74 test items, −11 doctests). No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 4825 lines / 1849 code` and `tests 4346 lines (+ oracle 4135)`.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1` → `951 passed, 3 skipped in …`

**Steps:**

Edits are anchored to code text, never to line numbers. Each Old text occurs exactly once in its file when its step runs, and New replaces it. "New: nothing" means the Old lines go, newline included.

**Deleting a named function** (Step 5) follows one rule:
- Delete from its first decorator line (its `def` line if it has none) through its last line, with the two blank lines after it.
- A function at the end of its file has no blank lines after it: delete the two blank lines before it instead. Only `filter.py` ends with a deleted function (`phi_squared_normalization`, then `phi_fft_order` if you delete in that order). After Step 5 `filter.py` ends with `phi_centered`'s `return out` and a newline.
- No deleted function has a comment directly above it. The packed-cells comment block in `grid.py` is two blank lines above `packed_cells`, and Step 5 deletes it with its own edit.
- Deleting in any order gives the same file, and so does making the step's Old/New edits before or after the deletions.

The suite is green after Steps 2 and 3 apart from Step 1's two `test_package` failures. Between Step 4 and Step 5, four doctests of functions that Step 5 deletes fail (listed in Step 4).

Step 10 is a git write in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: The public names, test first (`tests/test_package.py`)**

**Edit 1** (`tests/test_package.py`). Old:

```python
"""domainia 2.0 identity: version, public names, no JAX on import, no release leftovers."""
```

New:

```python
"""domainia 3.0 identity: version, public names, no JAX on import, no release leftovers."""
```

**Edit 2** (`tests/test_package.py`). Old:

```python
PUBLIC = {"Grid", "MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError",
          "__version__", "frequency_covariance", "transform", "wdm_covariance", "wilson"}


def test_the_version_is_2_0_0_everywhere():
    assert domainia.__version__ == "2.0.0"
    assert tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"] == "2.0.0"
```

New:

```python
#: Spec 3.0, section 5.
PUBLIC = {"__version__", "frequency_covariance", "transform", "wdm_covariance", "wilson"}
WILSON_PUBLIC = {"Grid", "forward", "inverse", "forward_time", "inverse_time",
                 "Window", "Meyer", "MEYER", "RampWindow", "SampledWindow",
                 "band_averaged_variance", "evolutionary_band_variance",
                 "evolutionary", "stationary", "matrix", "noise", "grid", "filter", "windows"}


def test_the_version_is_3_0_0_everywhere():
    assert domainia.__version__ == "3.0.0"
    assert tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"] == "3.0.0"
```

**Edit 3** (`tests/test_package.py`). Old:

```python
def test_the_public_names():
    assert set(domainia.__all__) == PUBLIC
    assert all(hasattr(domainia, name) for name in domainia.__all__)
    assert all(hasattr(wilson, name) for name in wilson.__all__)
    assert {"MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError", "windows",
            "require_window", "sample_window", "pack", "unpack"} <= set(wilson.__all__)
    assert wilson.pack is wilson.grid.pack and wilson.unpack is wilson.grid.unpack
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid
```

New:

```python
def test_the_public_names():
    """A name that left a top level stays in its submodule (``wilson.grid.active_mask``,
    ``wilson.windows.WindowError``)."""
    assert set(domainia.__all__) == PUBLIC
    assert set(wilson.__all__) == WILSON_PUBLIC
    assert all(hasattr(domainia, name) for name in domainia.__all__)
    assert all(hasattr(wilson, name) for name in wilson.__all__)
    assert not hasattr(domainia, "MEYER") and not hasattr(wilson, "active_mask")
    assert wilson.MEYER is wilson.windows.MEYER and wilson.Grid is wilson.grid.Grid
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | grep -E "^FAILED|passed|failed" | sed "s/ - .*//"`

Expected:
```
FAILED tests/test_package.py::test_the_version_is_3_0_0_everywhere
FAILED tests/test_package.py::test_the_public_names
2 failed, 5 passed in …
```
The version is still 2.0.0, and `domainia.__all__` still holds the window names (Step 4 changes both).

- [ ] **Step 2: The namespace paths: every grid name through `wilson.` gains `grid.`**

The names that leave `domainia.wilson`'s namespace in Step 4 are rewired first. Each new path already works today, so the suite stays green. First, four explicit edits: the two `is wilson.…` legs go, the `WindowError` imports move to `domainia.wilson.windows`, and `phi_centered`'s doctest uses `wilson.filter`.

**Edit 1** (`tests/wilson/test_checks.py`). Old:

```python
    assert grid.validate_dt is validate_dt is wilson.validate_dt
    assert grid.validate_finite is validate_finite is wilson.validate_finite
```

New:

```python
    assert grid.validate_dt is validate_dt and grid.validate_finite is validate_finite
```

**Edit 2** (`tests/wilson/test_window_threading.py`). Old:

```python
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow, WindowError
```

New:

```python
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow
from domainia.wilson.windows import WindowError
```

**Edit 3** (`tests/wilson/test_windows.py`). Old:

```python
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow, Window, WindowError
from domainia.wilson.filter import nu_d, phi_centered
from domainia.wilson.windows import require_window, sample_window
```

New:

```python
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow, Window
from domainia.wilson.filter import nu_d, phi_centered
from domainia.wilson.windows import WindowError, require_window, sample_window
```

**Edit 4** (`src/domainia/wilson/filter.py`). Old:

```python
    >>> phi = wilson.phi_centered(8, 16, 1.0)
```

New:

```python
    >>> phi = wilson.filter.phi_centered(8, 16, 1.0)
```

Then the sweep. Every `wilson.X`, `domainia.wilson.X` included, with X one of the nine grid names, becomes `wilson.grid.X` in these 14 files, and `wl.active_mask` becomes `wl.grid.active_mask` in `meyer_cases.py`. The command prints the number of replacements per file.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P - <<'EOF'
import re
from pathlib import Path
GRID = r"\bwilson\.(active_mask|band_frequencies|delta_F|delta_T|n_samples|time_pixel_centers|validate_dt|validate_finite|validate_grid)\b"
FILES = [
    "src/domainia/_transform.py",
    "src/domainia/testing.py",
    "src/domainia/wilson/_checks.py",
    "src/domainia/wilson/filter.py",
    "src/domainia/wilson/grid.py",
    "src/domainia/wilson/transform.py",
    "tests/test_covariance.py",
    "tests/test_dom_testing.py",
    "tests/test_transform.py",
    "tests/wilson/test_checks.py",
    "tests/wilson/test_grid_bands.py",
    "tests/wilson/test_window_threading.py",
    "tests/wilson/test_windows.py",
    "tests/wilson/test_wrong_kinds.py",
]
SWEEP = [(GRID, r"wilson.grid.\1", FILES), (r"\bwl\.active_mask\b", "wl.grid.active_mask", ["tests/wilson/meyer_cases.py"])]
for rx, rep, paths in SWEEP:
    for path in paths:
        text, n = re.subn(rx, rep, Path(path).read_text())
        Path(path).write_text(text)
        print(path, n)
EOF
```

Expected:
```
src/domainia/_transform.py 5
src/domainia/testing.py 2
src/domainia/wilson/_checks.py 5
src/domainia/wilson/filter.py 1
src/domainia/wilson/grid.py 11
src/domainia/wilson/transform.py 5
tests/test_covariance.py 3
tests/test_dom_testing.py 1
tests/test_transform.py 1
tests/wilson/test_checks.py 2
tests/wilson/test_grid_bands.py 2
tests/wilson/test_window_threading.py 6
tests/wilson/test_windows.py 1
tests/wilson/test_wrong_kinds.py 5
tests/wilson/meyer_cases.py 2
```

The sweep's sites are doctests (`grid.py` 11, `_checks.py` 5), Sphinx references (`transform.py` 5, `filter.py` 1: `:func:`domainia.wilson.validate_grid`` becomes `:func:`domainia.wilson.grid.validate_grid``), code (`_transform.py` 5, `testing.py` 1, `meyer_cases.py` 2) and test code. One more site is a docstring: `testing.py`'s ``wilson.active_mask``.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "\b(wilson|wl)\.(WindowError|active_mask|band_frequencies|delta_F|delta_T|n_samples|phi_centered|require_window|sample_window|time_pixel_centers|validate_dt|validate_finite|validate_grid)\b" src tests | grep -v "^tests/wilson/oracle/"`

Expected, exactly this line: `phi_squared_normalization`'s doctest, which Step 5 deletes with its function.
```
src/domainia/wilson/filter.py:196:    >>> bool(np.isclose(total, np.sum(wilson.phi_centered(8, 16, 1.0) ** 2)))
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | grep -E "^FAILED|passed|failed" | sed "s/ - .*//"`

Expected (prototype):
```
FAILED tests/test_package.py::test_the_version_is_3_0_0_everywhere
FAILED tests/test_package.py::test_the_public_names
2 failed, 1034 passed, 3 skipped in …
```

- [ ] **Step 3: `colored_noise` and `empirical_variance` in `domainia.testing`; their callers**

The two functions are added to `testing.py` (code unchanged; see "What changes for a caller") and every caller moves to them. `noise.py` keeps its copies until Step 5, so nothing breaks in between. In `meyer_cases.wilson_cases`, the lazy `from domainia import testing` mirrors `domainia_cases`. `wilson_cases` receives only `wl` (`domainia.wilson`), and a module-level import would make `meyer_cases` import `domainia` at collection time.

**Edit 1** (`src/domainia/testing.py`). Old:

```python
"""Dense oracles for tests and small grids (NumPy only, never on a
likelihood path).  Every oracle here is ``O(N^2)`` memory or worse: keep ``N`` <~ 2048.
```

New:

```python
"""Dense oracles and noise helpers for tests and small grids (NumPy only).  Every dense
oracle here is ``O(N^2)`` memory or worse: keep ``N`` <~ 2048.
```

**Edit 2** (`src/domainia/testing.py`). Old:

```python
* :func:`eq45_dense`: Eq. (45) on a set of cells by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).
```

New:

```python
* :func:`eq45_dense`: Eq. (45) on a set of cells by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).
* :func:`colored_noise`, :func:`empirical_variance`: one draw of stationary noise (its rfft)
  and the Monte-Carlo WDM variance of such draws, which checks
  ``wilson.band_averaged_variance``.
```

**Edit 3** (`src/domainia/testing.py`). Old:

```python
:func:`dense_W`, :func:`wdm_congruence` and :func:`eq45_dense` take ``window=`` (default
Meyer: the WDM transform) and compute in the Wilson basis of that window.
```

New:

```python
:func:`dense_W`, :func:`wdm_congruence`, :func:`eq45_dense` and :func:`empirical_variance` take
``window=`` (default Meyer: the WDM transform) and compute in the Wilson basis of that window.
```

**Edit 4** (`src/domainia/testing.py`). Old:

```python
from domainia import wilson
from domainia.wilson._checks import require_callable, require_dtype
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
from domainia.wilson.windows import MEYER, SampledWindow, Window

__all__ = ["ALT_WINDOW", "dense_W", "eq45_dense", "smoothstep", "wdm_congruence", "wide_profile"]
```

New:

```python
from domainia import wilson
from domainia.wilson._checks import require_callable, require_dtype, require_int
from domainia.wilson.evolutionary import eq45_direct, eq45_plan
from domainia.wilson.grid import n_samples, sanitize_psd, validate_dt, validate_grid
from domainia.wilson.transform import forward
from domainia.wilson.windows import MEYER, SampledWindow, Window

__all__ = ["ALT_WINDOW", "colored_noise", "dense_W", "empirical_variance", "eq45_dense",
           "smoothstep", "wdm_congruence", "wide_profile"]
```

**Edit 5** (`src/domainia/testing.py`). Old:

```python
def smoothstep(x: Any) -> Array:
```

New:

```python
def colored_noise(
    N: int,
    dt: float,
    Sn_func: Callable[[Any], Any],
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Generate one realization of stationary noise with PSD $S_n(f)$.

    Public (it was the vendored private
    ``_generate_colored_noise``; same draws, same order: interior real parts,
    interior imaginary parts, DC, Nyquist).  $O(N\\log N)$, NumPy only.

    Returns the rfft of the time-domain noise, with the convention
    $E[|X_j|^2] = N\\,S_n(f_j)/(2\\,\\delta t)$ at **every** bin.

    The DC and Nyquist bins are real-valued (as they must be for a real
    time series) but carry their full variance, not half: for real
    Gaussian noise $E[|X_j|^2]$ is flat across the rfft grid.

    .. note::
       Before 2026-07-27 the DC and Nyquist draws were divided by
       $\\sqrt 2$, giving them half the correct power.  That biased
       :func:`empirical_variance` low at the two edge bands -- exactly
       where an analytic cross-check is hardest.

    Raises
    ------
    TypeError
        If ``N`` is not an integer (``bool`` included), or ``dt`` is not a real
        number.
    ValueError
        If ``N`` is not positive and even, or ``dt`` is not positive and
        finite.  Both are checked before the PSD is evaluated or ``rng`` is
        drawn from.  An odd ``N`` has no Nyquist bin, yet bin ``N//2`` would
        be drawn real-only as if it were one, and the noise would not be
        stationary (per-sample variance 0.79--1.22 of its mean at ``N = 9``).
    """
    # Public entry point: same checks as validate_grid/validate_dt, on N itself.
    # They only raise, so valid input takes the oracle's path bit for bit.
    require_int("N", N)
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if N % 2 != 0:
        raise ValueError(f"N must be even, got {N}")
    validate_dt(dt)
    freqs = np.fft.rfftfreq(N, d=float(dt))
    Sn = sanitize_psd(Sn_func(freqs), "colored_noise")
    sigma_full = np.sqrt(N * Sn / (2.0 * dt))

    n_rfft = N // 2 + 1
    out = np.zeros(n_rfft, dtype=complex)
    # Interior bins split their variance across the real and imaginary parts;
    # the real-only DC and Nyquist bins carry theirs in one component.
    interior = slice(1, N // 2)
    n_interior = (N // 2) - 1
    out[interior] = (
        sigma_full[interior]
        * (rng.standard_normal(n_interior) + 1j * rng.standard_normal(n_interior))
        / np.sqrt(2.0)
    )
    out[0] = sigma_full[0] * rng.standard_normal()
    out[N // 2] = sigma_full[N // 2] * rng.standard_normal()
    return out


def empirical_variance(
    Nf: int,
    Nt: int,
    dt: float,
    Sn_func: Callable[[Any], Any],
    n_realizations: int = 1000,
    seed: int | None = None,
    *,
    window: Window = MEYER,
) -> Array:
    """Monte-Carlo estimate of $\\langle w_{mn}^2 \\rangle$.

    Generates ``n_realizations`` independent noise realizations with the
    given PSD, transforms each to the WDM domain, and averages the
    squared coefficients.  Useful for verifying the analytical
    :func:`domainia.wilson.band_averaged_variance`.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    Sn_func : callable
    n_realizations : int, optional
        Number of realizations to average (default 1000).
    seed : int or None, optional
        Seed for the RNG.
    window : Window, optional
        The Wilson window (default :data:`domainia.wilson.MEYER`: the WDM transform).
        See :mod:`domainia.wilson.windows`.

    Returns
    -------
    ndarray, shape (Nf+1, Nt)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    if require_int("n_realizations", n_realizations) < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}")
    N = n_samples(Nf, Nt)
    rng = np.random.default_rng(seed)
    accum = np.zeros((Nf + 1, Nt))
    for _ in range(n_realizations):
        x_rfft = colored_noise(N, dt, Sn_func, rng)
        w = forward(x_rfft, Nf, Nt, dt, window=window)
        accum += w * w
    return accum / n_realizations


def smoothstep(x: Any) -> Array:
```

**Edit 6** (`tests/wilson/meyer_cases.py`). Old:

```python
    """The kernel window consumers of ``wl`` (``domainia.wilson``)."""
    N = Nf * Nt
```

New:

```python
    """The kernel window consumers of ``wl`` (``domainia.wilson``), and
    ``testing.empirical_variance`` under its recorded name."""
    from domainia import testing

    N = Nf * Nt
```

**Edit 7** (`tests/wilson/meyer_cases.py`). Old:

```python
        ("noise.empirical_variance", lambda kw: No.empirical_variance(Nf, Nt, dt, psd, 3, seed=11, **kw)),
```

New:

```python
        ("noise.empirical_variance",
         lambda kw: testing.empirical_variance(Nf, Nt, dt, psd, 3, seed=11, **kw)),
```

**Edit 8** (`tests/wilson/test_window_threading.py`). Old:

```python
from domainia.testing import ALT_WINDOW, smoothstep
```

New:

```python
from domainia.testing import ALT_WINDOW, colored_noise, empirical_variance, smoothstep
```

**Edit 9** (`tests/wilson/test_window_threading.py`). Old:

```python
    draws = [W @ np.fft.irfft(No.colored_noise(N, dt, psd, rng), n=N) for _ in range(3)]
    want = np.mean(np.square(draws), axis=0).reshape(Nf + 1, Nt)
    _close(No.empirical_variance(Nf, Nt, dt, psd, 3, seed=5, window=window), want, RTOL_KERNEL,
```

New:

```python
    draws = [W @ np.fft.irfft(colored_noise(N, dt, psd, rng), n=N) for _ in range(3)]
    want = np.mean(np.square(draws), axis=0).reshape(Nf + 1, Nt)
    _close(empirical_variance(Nf, Nt, dt, psd, 3, seed=5, window=window), want, RTOL_KERNEL,
```

**Edit 10** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
import domainia
from domainia import wilson
```

New:

```python
import domainia
from domainia import testing, wilson
```

**Edit 11** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
    "empirical_variance n_realizations": (lambda v: wilson.empirical_variance(8, 16, 1.0, np.ones_like, v,
                                                                              seed=0), 2),
```

New:

```python
    "empirical_variance n_realizations": (lambda v: testing.empirical_variance(8, 16, 1.0, np.ones_like,
                                                                               v, seed=0), 2),
```

**Edit 12** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
    (lambda: wilson.empirical_variance(8, 16, 1.0, np.ones_like, 0), "^n_realizations must be >= 1, got 0"),
```

New:

```python
    (lambda: testing.empirical_variance(8, 16, 1.0, np.ones_like, 0), "^n_realizations must be >= 1, got 0"),
```

**Edit 13** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
        "colored_noise": lambda dt: No.colored_noise(60, dt, psd, np.random.default_rng(0)),
        "empirical_variance": lambda dt: wilson.empirical_variance(6, 10, dt, psd, 2, seed=0),
```

New:

```python
        "colored_noise": lambda dt: testing.colored_noise(60, dt, psd, np.random.default_rng(0)),
        "empirical_variance": lambda dt: testing.empirical_variance(6, 10, dt, psd, 2, seed=0),
```

**Edit 14** (`tests/wilson/test_checks.py`). Old:

```python
from domainia.wilson.noise import colored_noise
```

New:

```python
from domainia.testing import colored_noise
```

**Edit 15** (`tests/wilson/test_noise_stationary.py`). Old:

```python
"""domainia.wilson.noise and domainia.wilson.stationary against the frozen oracle.
```

New:

```python
"""domainia.wilson.noise, domainia.wilson.stationary and the noise helpers of domainia.testing
against the frozen oracle.
```

**Edit 16** (`tests/wilson/test_noise_stationary.py`). Old:

```python
import domainia.wilson.noise as noise
import domainia.wilson.stationary as stat
```

New:

```python
import domainia.wilson.noise as noise
import domainia.wilson.stationary as stat
from domainia import testing
```

**Edit 17** (`tests/wilson/test_noise_stationary.py`). Old:

```python
    a = noise.colored_noise(Nf * Nt, dt, _psd, np.random.default_rng(7))
```

New:

```python
    a = testing.colored_noise(Nf * Nt, dt, _psd, np.random.default_rng(7))
```

**Edit 18** (`tests/wilson/test_noise_stationary.py`). Old:

```python
    got = noise.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3)
```

New:

```python
    got = testing.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3)
```

**Edit 19** (`tests/wilson/test_noise_stationary.py`). Old:

```python
    with pytest.raises(ValueError, match=match):
        noise.colored_noise(N, dt, psd, rng)
```

New:

```python
    with pytest.raises(ValueError, match=match):
        testing.colored_noise(N, dt, psd, rng)
```

**Edit 20** (`tests/wilson/test_noise_stationary.py`). Old:

```python
    with pytest.raises(TypeError, match=match):
        noise.colored_noise(N, dt, psd, rng)
```

New:

```python
    with pytest.raises(TypeError, match=match):
        testing.colored_noise(N, dt, psd, rng)
```

**Edit 21** (`tests/wilson/test_noise_stationary.py`). Old:

```python
    a = noise.colored_noise(2, 1.0, _psd, np.random.default_rng(5))
```

New:

```python
    a = testing.colored_noise(2, 1.0, _psd, np.random.default_rng(5))
```

**Edit 22** (`tests/wilson/test_noise_stationary.py`). Old:

```python
    c = noise.colored_noise(np.int64(16), 0.5, _psd, np.random.default_rng(5))
```

New:

```python
    c = testing.colored_noise(np.int64(16), 0.5, _psd, np.random.default_rng(5))
```

**Edit 23** (`tests/wilson/vendored/test_noise.py`). Old:

```python
from domainia.wilson.noise import band_averaged_variance, empirical_variance
```

New:

```python
from domainia.testing import empirical_variance
from domainia.wilson.noise import band_averaged_variance
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | grep -E "^FAILED|passed|failed" | sed "s/ - .*//"`

Expected (prototype):
```
FAILED tests/test_package.py::test_the_version_is_3_0_0_everywhere
FAILED tests/test_package.py::test_the_public_names
2 failed, 1034 passed, 3 skipped in …
```

The exact check compares every kept replay array with the recorded baseline by dtype and `np.array_equal`, which is stricter than the replay's 1e-12 gate. It also counts the `noise.empirical_variance` arrays, now computed by `testing.empirical_variance`.

Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
    ev = [k for k in kept if '::noise.empirical_variance::' in k]
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')
print(f'{len(ev)} noise.empirical_variance arrays, {len(set(ev) & set(bad))} differ')"
```

Expected:
```
0 of 1408 arrays differ from the baseline
6 noise.empirical_variance arrays, 0 differ
```

- [ ] **Step 4: The public API: the two `__init__.py` files and `pyproject.toml`**

**Edit 1** (`src/domainia/__init__.py`). Old:

```python
"""domainia: data and covariance across the time, Fourier and time-frequency (Wilson / WDM)
domains.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`MEYER`); every kernel that samples the window takes
  ``window=`` (any admissible :class:`Window`: ``Meyer(d)``, ``RampWindow``,
  ``SampledWindow``).
```

New:

```python
"""domainia: translate data arrays between time, frequency and WDM, and evaluate a noise
model's per-point K x K covariance on a frequency or WDM grid.

Modules
-------
* :mod:`domainia.wilson`: the Wilson-basis kernels.  The WDM transform is the Wilson basis
  with the Meyer window (:data:`domainia.wilson.MEYER`); every kernel that samples the window
  takes ``window=`` (any admissible :class:`domainia.wilson.Window`: ``Meyer(d)``,
  ``RampWindow``, ``SampledWindow``).
```

**Edit 2** (`src/domainia/__init__.py`). Old:

```python
* Not imported here: :mod:`domainia.testing` (dense oracles and test windows).
"""

from __future__ import annotations

from . import wilson
from ._covariance import frequency_covariance, wdm_covariance
from ._transform import transform
from .wilson import MEYER, Grid, Meyer, RampWindow, SampledWindow, Window, WindowError

__version__ = "2.0.0"

__all__ = [
    "MEYER",
    "Grid",
    "Meyer",
    "RampWindow",
    "SampledWindow",
    "Window",
    "WindowError",
    "__version__",
    "frequency_covariance",
    "transform",
    "wdm_covariance",
    "wilson",
]
```

New:

```python
* Not imported here: :mod:`domainia.testing` (dense oracles, test windows and the
  Monte-Carlo noise helpers).
"""

from __future__ import annotations

from . import wilson
from ._covariance import frequency_covariance, wdm_covariance
from ._transform import transform

__version__ = "3.0.0"

__all__ = ["__version__", "frequency_covariance", "transform", "wdm_covariance", "wilson"]
```

**Edit 3** (`src/domainia/wilson/__init__.py`). Old:

```python
Input validation (all applied automatically by the entry points):

* :func:`validate_grid`, :func:`validate_dt`, :func:`validate_finite`,
  :func:`grid.sanitize_psd`

Grid helpers:

* :func:`n_samples`, :func:`band_frequencies`, :func:`time_pixel_centers`
* :func:`delta_F`, :func:`delta_T`
* :func:`active_mask`, :func:`count_active`
* :func:`pack`, :func:`unpack` -- dense ``(Nf+1, Nt)`` <-> packed ``(N,)`` active cells
  in band order

Noise variance grids:

* :func:`band_averaged_variance`
* :func:`empirical_variance`
```

New:

```python
Input validation (in :mod:`domainia.wilson.grid`; all applied automatically by the entry
points):

* :func:`grid.validate_grid`, :func:`grid.validate_dt`, :func:`grid.validate_finite`,
  :func:`grid.sanitize_psd`

Grid helpers (in :mod:`domainia.wilson.grid`):

* :func:`grid.n_samples`, :func:`grid.band_frequencies`, :func:`grid.time_pixel_centers`
* :func:`grid.delta_F`, :func:`grid.delta_T`
* :func:`grid.active_mask`

Noise variance grids (WDM Paper II, Eq. 36):

* :func:`band_averaged_variance`, :func:`evolutionary_band_variance`
```

**Edit 4** (`src/domainia/wilson/__init__.py`). Old:

```python
* :func:`filter.phi_centered`, :func:`filter.phi_fft_order`

Windows (in :mod:`domainia.wilson.windows`):

* :class:`Window` (the protocol), :class:`Meyer`, :class:`RampWindow`,
  :class:`SampledWindow`, :data:`MEYER` (the WDM transform's window, ``Meyer(4)``),
  :class:`WindowError`
* :func:`sample_window` (the kernels' single entry point: validated, read-only samples of
  any :class:`Window`, a window of your own included), :func:`require_window`
"""

from __future__ import annotations

from . import (
    evolutionary,
    filter,
    matrix,
    noise,
    stationary,
    windows,
)
from .filter import phi_centered, phi_fft_order
from .grid import (
    Grid,
    active_mask,
    band_frequencies,
    count_active,
    delta_F,
    delta_T,
    n_samples,
    pack,
    time_pixel_centers,
    unpack,
    validate_dt,
    validate_finite,
    validate_grid,
)
from .noise import band_averaged_variance, empirical_variance
from .transform import forward, forward_time, inverse, inverse_time
from .windows import (
    MEYER,
    Meyer,
    RampWindow,
    SampledWindow,
    Window,
    WindowError,
    require_window,
    sample_window,
)

__all__ = [
    "MEYER",
    "Grid",
    "Meyer",
    "RampWindow",
    "SampledWindow",
    "Window",
    "WindowError",
    "active_mask",
    "band_averaged_variance",
    "band_frequencies",
    "count_active",
    "delta_F",
    "delta_T",
    "empirical_variance",
    # Submodules
    "evolutionary",
    "filter",
    # Transform
    "forward",
    "forward_time",
    "inverse",
    "inverse_time",
    "matrix",
    # Grid
    "n_samples",
    "noise",
    "pack",
    # Filter
    "phi_centered",
    "phi_fft_order",
    # Windows
    "require_window",
    "sample_window",
    "stationary",
    "time_pixel_centers",
    "unpack",
    "validate_dt",
    "validate_finite",
    "validate_grid",
    "windows",
]
```

New:

```python
* :func:`filter.phi_centered`

Windows (in :mod:`domainia.wilson.windows`):

* :class:`Window` (the protocol), :class:`Meyer`, :class:`RampWindow`,
  :class:`SampledWindow`, :data:`MEYER` (the WDM transform's window, ``Meyer(4)``),
  :class:`windows.WindowError`
* :func:`windows.sample_window` (the kernels' single entry point: validated, read-only
  samples of any :class:`Window`, a window of your own included),
  :func:`windows.require_window`
"""

from __future__ import annotations

from . import evolutionary, filter, grid, matrix, noise, stationary, windows
from .grid import Grid
from .noise import band_averaged_variance, evolutionary_band_variance
from .transform import forward, forward_time, inverse, inverse_time
from .windows import MEYER, Meyer, RampWindow, SampledWindow, Window

__all__ = [
    "MEYER",
    "Grid",
    "Meyer",
    "RampWindow",
    "SampledWindow",
    "Window",
    "band_averaged_variance",
    "evolutionary",
    "evolutionary_band_variance",
    "filter",
    "forward",
    "forward_time",
    "grid",
    "inverse",
    "inverse_time",
    "matrix",
    "noise",
    "stationary",
    "windows",
]
```

**Edit 5** (`pyproject.toml`). Old:

```toml
version = "2.0.0"
description = "Move data and covariance between the time, Fourier and Wilson (WDM) time-frequency domains."
```

New:

```toml
version = "3.0.0"
description = "Translate data arrays between time, frequency and WDM, and evaluate a noise model's per-point K x K covariance on a frequency or WDM grid."
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_package.py 2>&1 | tail -1`

Expected: `7 passed in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | grep -E "^FAILED|passed|failed" | sed "s/ - .*//"`

Expected (prototype):
```
FAILED src/domainia/wilson/filter.py::domainia.wilson.filter.phi_fft_order
FAILED src/domainia/wilson/filter.py::domainia.wilson.filter.phi_squared_normalization
FAILED src/domainia/wilson/grid.py::domainia.wilson.grid.count_active
FAILED src/domainia/wilson/noise.py::domainia.wilson.noise.empirical_variance
4 failed, 1032 passed, 3 skipped in …
```
These four doctests call `wilson.count_active`, `wilson.phi_fft_order`, `wilson.phi_centered` and `wilson.empirical_variance`, which left the namespace. Their functions go in Step 5.

- [ ] **Step 5: Delete the dead code and its tests**

Delete these functions by the rule above:

- From `src/domainia/wilson/grid.py`: `band_indices`, `carrier_phase`, `count_active`, `packed_cells`, `pack`, `unpack`, `edge_columns`.
- From `src/domainia/wilson/_checks.py`: `require_bool`, `require_str`, `index_array`.
- From `src/domainia/wilson/filter.py`: `phi_fft_order`, `phi_squared_normalization`.
- From `src/domainia/wilson/noise.py`: `colored_noise`, `empirical_variance`.
- From `tests/wilson/vendored/test_grid.py`: `test_count_active_equals_N`, `test_band_indices_unique_modN`, `test_carrier_phase_even_m`, `test_carrier_phase_odd_m`.
- From `tests/wilson/vendored/test_filter.py`: `test_phi_fft_order_is_ifftshift_of_centered`, `test_phi_fft_order_peaked_at_zero`.
- From `tests/wilson/test_grid_bands.py`: `test_packed_cells_is_band_order`, `test_pack_unpack_round_trip_and_active_W_order`.
- From `tests/wilson/test_checks.py`: `test_require_bool_returns_a_python_bool`, `test_require_bool_refuses_every_other_kind`, `test_require_str_returns_the_string`, `test_require_str_refuses_every_other_kind`, `test_index_array_returns_int64`, `test_index_array_refuses_a_non_integral_or_non_finite_float_with_value_error`, `test_index_array_refuses_a_non_numeric_dtype_with_type_error`.

Then make these edits. Edits 1 to 3 trim `grid.py`: `IntArray` was used only by `band_indices` and `packed_cells`. Edits 21 to 25 trim `noise.py`: its import of `require_int`, its import of `forward` and its `CArray` alias were used only by the two moved functions. Edit 20 is the `test_dom_testing.py` column list (see "The tests of deleted code").

**Edit 1** (`src/domainia/wilson/grid.py`). Old:

```python
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.integer]
```

New:

```python
type BoolArray = NDArray[np.bool_]
```

**Edit 2** (`src/domainia/wilson/grid.py`). Old:

```python
    "band_frequencies",
    "band_indices",
    "carrier_phase",
    "count_active",
    "delta_F",
    "delta_T",
    "edge_columns",
    "endpoint_weights",
    "n_samples",
    "pack",
    "packed_cells",
    "sanitize_psd",
    "unpack",
    "time_pixel_centers",
```

New:

```python
    "band_frequencies",
    "delta_F",
    "delta_T",
    "endpoint_weights",
    "n_samples",
    "sanitize_psd",
    "time_pixel_centers",
```

**Edit 3** (`src/domainia/wilson/grid.py`). Old:

```python
# ---------------------------------------------------------------------------
# Packed (band-order) cell vectors
# ---------------------------------------------------------------------------
#
# The N active cells of an (Nf+1, Nt) grid, listed in BAND order: DC with even
# n, then bands 1..Nf-1 with every n, then Nyquist with (Nf+n) even -- i.e.
# ``np.nonzero(active_mask(Nf, Nt))``.  This is the order of
# ``matrix.active_W`` and of every WDM Paper II covariance.  The Nyquist->DC fold of
# lisa/wdm 0.2.0 (and the manuscript's L1373 interleave) is not offered.


```

New: nothing (delete the Old lines).

**Edit 4** (`src/domainia/wilson/filter.py`). Old:

```python
    "phi_centered",
    "phi_fft_order",
    "phi_squared_normalization",
]
```

New:

```python
    "phi_centered",
]
```

**Edit 5** (`tests/wilson/vendored/test_grid.py`). Old:

```python
    band_frequencies,
    band_indices,
    carrier_phase,
    count_active,
    delta_F,
```

New:

```python
    band_frequencies,
    delta_F,
```

**Edit 6** (`tests/wilson/vendored/test_filter.py`). Old:

```python
from domainia.wilson.filter import nu_d, phi_centered, phi_fft_order
```

New:

```python
from domainia.wilson.filter import nu_d, phi_centered
```

**Edit 7** (`tests/wilson/vendored/test_filter.py`). Old:

```python
    """``d`` is a public parameter on four functions and must be plumbed.
```

New:

```python
    """``d`` is a public parameter on two functions and must be plumbed.
```

**Edit 8** (`tests/wilson/vendored/test_filter.py`). Old:

```python
    if d != 4:
        assert not np.allclose(phi, default)

    # phi_fft_order and phi_squared_normalization must honour the same d.
    assert np.allclose(phi_fft_order(g.Nf, g.Nt, g.dt, d), np.fft.ifftshift(phi))
    from domainia.wilson.filter import phi_squared_normalization

    assert np.isclose(
        phi_squared_normalization(g.Nf, g.Nt, g.dt, d), float(np.sum(phi**2))
    )
```

New:

```python
    if d != 4:
        assert not np.allclose(phi, default)
```

**Edit 9** (`tests/wilson/test_grid_bands.py`). Old:

```python
"""Packing helpers, the Hz window and the atoms."""
```

New:

```python
"""The grid, the Hz window and the atoms."""
```

**Edit 10** (`tests/wilson/test_grid_bands.py`). Old:

```python
from domainia.wilson.grid import (
    Grid,
    active_mask,
    edge_columns,
    endpoint_weights,
    pack,
    packed_cells,
    unpack,
)
```

New:

```python
from domainia.wilson.grid import Grid, active_mask, endpoint_weights
```

**Edit 11** (`tests/wilson/test_grid_bands.py`). Old:

```python
def test_edge_columns_and_endpoint_weights():
    assert edge_columns(256, 6, 5) == (0, 1, 2, 3, 4, 5, 251, 252, 253, 254, 255)
    assert edge_columns(16, 0, 0) == ()
    c = endpoint_weights(9)
```

New:

```python
def test_endpoint_weights():
    c = endpoint_weights(9)
```

**Edit 12** (`tests/wilson/test_grid_bands.py`). Old:

```python
    m, n = packed_cells(Nf, Nt)
    for p in range(m.size):
```

New:

```python
    m, n = np.nonzero(active_mask(Nf, Nt))
    for p in range(m.size):
```

**Edit 13** (`tests/wilson/test_checks.py`). Old:

```python
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
```

New:

```python
    REAL_KINDS,
    require_callable,
    require_dtype,
    require_instance,
    require_int,
    require_real,
    validate_dt,
```

**Edit 14** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
    "carrier_phase m": (lambda v: wilson.grid.carrier_phase(v, 4), 1),
    "edge_columns lead": (lambda v: wilson.grid.edge_columns(16, v, 1), 2),
    "edge_columns trail": (lambda v: wilson.grid.edge_columns(16, 2, v), 1),
```

New: nothing (delete the Old lines).

**Edit 15** (`tests/wilson/test_wrong_kinds.py`). Old:

```python
    """``int()``, ``range()`` or a parity test took these silently (``carrier_phase(2.5, 4)``
    ran), or ``True`` passed as 1."""
```

New:

```python
    """``int()``, ``range()`` or a parity test took these silently, or ``True`` passed as 1."""
```

**Edit 16** (`tests/wilson/test_window_threading.py`). Old:

```python
#: must never be scanned: they are where the Meyer samples come from (``phi_centered``,
#: ``phi_fft_order``) and where ``.sample`` is called, behind ``sample_window``'s validation.
```

New:

```python
#: must never be scanned: they are where the Meyer samples come from (``phi_centered``) and
#: where ``.sample`` is called, behind ``sample_window``'s validation.
```

**Edit 17** (`tests/wilson/test_window_threading.py`). Old:

```python
MEYER_HELPERS = frozenset({"phi_centered", "phi_fft_order"})
```

New:

```python
MEYER_HELPERS = frozenset({"phi_centered"})
```

**Edit 18** (`tests/wilson/test_window_threading.py`). Old:

```python
      ``phi_centered`` or ``phi_fft_order`` (they sample Meyer directly);
```

New:

```python
      ``phi_centered`` (it samples Meyer directly);
```

**Edit 19** (`tests/wilson/test_window_threading.py`). Old:

```python
    ("filter.phi_centered(Nf, Nt, dt, 8)", True),
    ("phi_fft_order(Nf, Nt, dt)", True),
    ("wilson.filter.phi_fft_order(Nf, Nt, dt)", True),
])
```

New:

```python
    ("filter.phi_centered(Nf, Nt, dt, 8)", True),
])
```

**Edit 20** (`tests/test_dom_testing.py`). Old:

```python
    interior = ~np.isin(np.nonzero(MASK)[1], wilson.grid.edge_columns(NT, 3, 1))
```

New:

```python
    interior = ~np.isin(np.nonzero(MASK)[1], np.r_[0:3, NT - 1])
```

**Edit 21** (`src/domainia/wilson/noise.py`). Old:

```python
A Monte-Carlo estimator (:func:`empirical_variance`) is provided for
verification.
```

New:

```python
A Monte-Carlo estimator, :func:`domainia.testing.empirical_variance`, checks them.
```

**Edit 22** (`src/domainia/wilson/noise.py`). Old:

```python
from ._checks import require_int
from .grid import (
```

New:

```python
from .grid import (
```

**Edit 23** (`src/domainia/wilson/noise.py`). Old:

```python
from .transform import forward
```

New: nothing (delete the Old lines).

**Edit 24** (`src/domainia/wilson/noise.py`). Old:

```python
type CArray = NDArray[np.complexfloating]
```

New: nothing (delete the Old lines).

**Edit 25** (`src/domainia/wilson/noise.py`). Old:

```python
__all__ = [
    "band_averaged_variance",
    "colored_noise",
    "empirical_variance",
    "evolutionary_band_variance",
]
```

New:

```python
__all__ = ["band_averaged_variance", "evolutionary_band_variance"]
```

**Edit 26** (`tests/wilson/test_noise_stationary.py`). Old:

```python
    assert "colored_noise" in noise.__all__
```

New:

```python
    assert "colored_noise" not in noise.__all__ and "colored_noise" in testing.__all__
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | grep -E "^FAILED|passed|failed" | sed "s/ - .*//"`

Expected (prototype): `951 passed, 3 skipped in …`. The prototype gives the same with the FLOOR interpreter (`$TMPDIR/dom3-envs/floor312/bin/python` in place of the PY one).

- [ ] **Step 6: Module docstrings**

Edit 1 removes `stationary.py`'s likelihood sentence. Edit 2 is the one-word change in `time_pixel_centers` (see the deviations).

**Edit 1** (`src/domainia/wilson/stationary.py`). Old:

```python
the dense $W_a\\,C_{\\rm time}\\,W_a^T$ at machine precision and is
preferred when bias-free inner products matter (matched filtering of
broadband signals, parameter estimation, anything that consumes
$\\Lambda^{-1}$).
```

New:

```python
the dense $W_a\\,C_{\\rm time}\\,W_a^T$ at machine precision.
```

**Edit 2** (`src/domainia/wilson/grid.py`). Old:

```python
    These are the time coordinates of the WDM lattice: the energy centroid of
```

New:

```python
    These are the times of the WDM lattice: the energy centroid of
```

- [ ] **Step 7: README.md**

Replace the whole file `README.md` with this text:

````markdown
# domainia

domainia does two things, in float64 NumPy. `transform` translates data arrays between the
time, frequency and WDM (Wilson-Daubechies-Meyer) domains. `frequency_covariance` and
`wdm_covariance` evaluate a K × K noise model as its covariance at each frequency bin or WDM
cell. It computes no likelihood: the caller does.

## Install and test

```bash
pip install -e ".[test]"
PYTHONDONTWRITEBYTECODE=1 python -P -m pytest -q
```

Three oracle tests compare against source trees outside this repository; they skip unless
`DOMAINIA_ORACLE_ROOT` names the directory that holds them (the skip message lists the
subdirectories each one expects).

## Data

```python
import numpy as np
from domainia import transform, wilson

Nf, Nt, dt = 6, 12, 0.3
N = Nf * Nt
x = np.random.default_rng(6).standard_normal((2, N))            # any leading axes

w = transform(x, "time", "wdm", N=N, dt=dt, target_wdm=(Nf, Nt))       # (2, 7, 12) float64
X = transform(w, "wdm", "frequency", N=N, dt=dt, source_wdm=(Nf, Nt))  # (2, 37): dt * rfft(x)
v = transform(X, "frequency", "wdm", N=N, dt=dt, target_wdm=(12, 6))   # another WDM grid
assert np.allclose(transform(v, "wdm", "time", N=N, dt=dt, source_wdm=(12, 6)), x)

# Raw kernel output (~1e-17 in the inactive cells) and a full-FFT slice (~1e-16 imaginary
# at Nyquist) are taken as they are; a residue above 1e-10 of the peak is refused.
assert np.allclose(transform(wilson.forward_time(x, Nf, Nt, dt), "wdm", "time", N=N, dt=dt,
                             source_wdm=(Nf, Nt)), x)
assert np.allclose(transform(dt * np.fft.fft(x)[:, : N // 2 + 1], "frequency", "time",
                             N=N, dt=dt), x)
```

## Covariance

```python
import numpy as np
from domainia import frequency_covariance, wdm_covariance

dt, N, Nf, Nt = 5.0, 1024, 16, 64

def S(f):                                       # one-sided 2 x 2 cross-spectral density
    p = 1e-2 / (1.0 + (f / 0.02) ** 2)
    c = 0.8 * np.exp(-2j * np.pi * f * 36.5)    # coherence 0.8, a 36.5 s delay: complex
    return p * np.array([[np.ones_like(c), c], [np.conj(c), np.ones_like(c)]])

C = frequency_covariance(S, N=N, dt=dt)                                  # (2, 2, 513) complex
D = wdm_covariance(lambda f, t: S(f), Nf=Nf, Nt=Nt, dt=dt)               # (2, 2, 17, 64) real
E = wdm_covariance(lambda f, t: S(f) * (1.3 + 0.3 * np.cos(2 * np.pi * t / (N * dt))),
                   Nf=Nf, Nt=Nt, dt=dt)                                  # evolutionary
assert np.array_equal(C, C.conj().transpose(1, 0, 2))
assert np.array_equal(D, D.transpose(1, 0, 2, 3)) and not D[:, :, 0, 1].any()
```

## Conventions

- Trailing-axis layouts: time is real `(N,)`; frequency is complex `(N // 2 + 1,)`,
  `dt * rfft(x)`; WDM is real `(Nf + 1, Nt)` with the edge bands, `Nf` and `Nt` even and
  `Nf * Nt == N`. The atoms have unit norm (`sum(w**2) == dt * sum(x**2)`), and the inactive
  WDM cells (bands 0 and `Nf` at odd `n`) are 0 in every output.
- A model `S(f)` or `S(f, t)` (`t` in seconds from the first sample) returns `(K, K, ...)`
  with `S[b, a] = conj(S[a, b])`, oriented so that `E[X_a conj(X_b)] = (N dt / 2) S_ab`.
- With `H = (S + S^H) / 2`, `frequency_covariance` is `(N dt / 2) H(f_k)`, real at DC and
  even-`N` Nyquist, and `wdm_covariance` is `½ BA_m[Re H(·, n Nf dt)]`, the band average of
  WDM Paper II, Eq. 36. Both are linear in `S`, exactly Hermitian (or symmetric), and fp64.
- For an evolutionary model Eq. 36 is a local-stationarity approximation. If the model is not
  periodic over the record, drop the edge columns `wilson.evolutionary.wrap_cut_columns` names.
- A wrong kind of argument raises `TypeError`, a wrong value `ValueError` (`WindowError` for
  an inadmissible window).

## Kernels

`domainia.wilson` holds the kernels: `forward`, `inverse`, `forward_time` and `inverse_time`
(any leading axes), the windows that `transform` and `wdm_covariance` take as `window=`
(`MEYER`, the default, `Meyer(d)`, `RampWindow`, `SampledWindow`), and the dense references.
````

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/test_readme.py 2>&1 | tail -1 && wc -l < README.md`

Expected:
```
2 passed in …
      83
```

The data block's second half is meant to exercise the input tolerance, so the residues it feeds `transform` must be nonzero. Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
Nf, Nt, dt = 6, 12, 0.3; N = Nf * Nt
x = np.random.default_rng(6).standard_normal((2, N))
w = wilson.forward_time(x, Nf, Nt, dt)
X = dt * np.fft.fft(x)[:, : N // 2 + 1]
print(f'raw forward_time, inactive cells: {np.abs(w[:, ~wilson.grid.active_mask(Nf, Nt)]).max():.1e} (peak {np.abs(w).max():.1f})')
print(f'full-FFT slice, imaginary part at DC: {np.abs(X[:, 0].imag).max():.1e}, at Nyquist: {np.abs(X[:, -1].imag).max():.1e} (peak {np.abs(X).max():.1f})')"
```

Expected:
```
raw forward_time, inactive cells: 4.9e-17 (peak 1.3)
full-FFT slice, imaginary part at DC: 0.0e+00, at Nyquist: 1.3e-16 (peak 5.6)
```

- [ ] **Step 8: Greps**

The task's literal gate, the deleted names as whole words in `.py` files, outside the oracle copies:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnwE "carrier_phase|packed_cells|pack|unpack|edge_columns|band_indices|count_active|require_bool|require_str|index_array|phi_fft_order|phi_squared_normalization" src tests --include='*.py' | grep -v "^tests/wilson/oracle/"`

Expected, exactly this line (the English verb in `matrix.py`'s module docstring):
```
src/domainia/wilson/matrix.py:34:Two utilities (:func:`rfft_to_real`, :func:`real_to_rfft`) pack the
```

The precise form: the names other than `pack` and `unpack` as words, and `pack`/`unpack` as code (a call, an attribute or a string), now also over README.md and pyproject.toml:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnwE "carrier_phase|packed_cells|edge_columns|band_indices|count_active|require_bool|require_str|index_array|phi_fft_order|phi_squared_normalization" src tests README.md pyproject.toml | grep -v "^tests/wilson/oracle/"; grep -rnE "\b(un)?pack\(|\.(un)?pack\b|\"(un)?pack\"" src tests README.md | grep -v "^tests/wilson/oracle/"`

Expected: no output.

The moved names through their old paths (`-I` skips the binary `tests/wilson/data/meyer_baseline.npz`, whose keys keep the recorded name):

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnIE "\b(noise|No|wilson|wl)\.(colored_noise|empirical_variance)\b" src tests README.md | grep -v "^tests/wilson/oracle/"`

Expected, exactly this line (the replay key's recorded name):
```
tests/wilson/meyer_cases.py:150:        ("noise.empirical_variance",
```

The names that left the two top levels:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE "\b(wilson|wl)\.(WindowError|active_mask|band_frequencies|delta_F|delta_T|empirical_variance|n_samples|phi_centered|require_window|sample_window|time_pixel_centers|validate_dt|validate_finite|validate_grid)\b|\bdomainia\.(Grid|MEYER|Meyer|RampWindow|SampledWindow|Window|WindowError)\b" src tests README.md | grep -v "^tests/wilson/oracle/"`

Expected: no output.

The task's second gate:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rniE "jax|float32|complex64|pytree|dict api|coordinates|TranslatedOperator|CellSet" src README.md`

Expected: no output.

- [ ] **Step 9: The replay, the suite and the size**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_meyer_baseline.py 2>&1 | tail -1`

Expected: `5 passed in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`

Step 3's exact check again. Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
    ev = [k for k in kept if '::noise.empirical_variance::' in k]
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')
print(f'{len(ev)} noise.empirical_variance arrays, {len(set(ev) & set(bad))} differ')"
```

Expected:
```
0 of 1408 arrays differ from the baseline
6 noise.empirical_variance arrays, 0 differ
```

The prototype gives the same three results with the FLOOR interpreter.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: the suite passes. Prototype: `951 passed, 3 skipped in …` in PY, and the same in FLOOR (the Verify command). No warnings, no collection errors (the prototype also ran both lanes with `-W error`).

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py`

Expected (prototype):
```
src 4825 lines / 1849 code
tests 4346 lines (+ oracle 4135)
```
That is −363 src lines (−134 code) and −144 test lines against Task 10 (src 5188 / 1983; tests 4490). This is the last measurement before the docstring trim. src is inside the spec's pre-trim budget (at most about 5,400 lines / 2,150 code) with 575 lines and 301 code lines to spare. tests are inside theirs (at most about 5,700) with 1,354 to spare. Nothing is over.

- [ ] **Step 10: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add README.md pyproject.toml src/domainia/__init__.py src/domainia/_transform.py src/domainia/testing.py src/domainia/wilson/__init__.py src/domainia/wilson/_checks.py src/domainia/wilson/filter.py src/domainia/wilson/grid.py src/domainia/wilson/noise.py src/domainia/wilson/stationary.py src/domainia/wilson/transform.py tests/test_covariance.py tests/test_dom_testing.py tests/test_package.py tests/test_transform.py tests/wilson/meyer_cases.py tests/wilson/test_checks.py tests/wilson/test_grid_bands.py tests/wilson/test_noise_stationary.py tests/wilson/test_window_threading.py tests/wilson/test_windows.py tests/wilson/test_wrong_kinds.py tests/wilson/vendored/test_filter.py tests/wilson/vendored/test_grid.py tests/wilson/vendored/test_noise.py && git status --short | grep -vc "^M "`

Expected: `0`. Every status line is a staged modification, so nothing is unstaged or untracked. (`grep -c` exits 1 when it prints 0, so this command does not continue with `&&`.)

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Trim the kept files and the public API to spec 3.0; testing holds the noise helpers; README; 3.0.0" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `26 files changed, 336 insertions(+), 877 deletions(-)`

Size after Task 11 (prototype): src 4825 lines / 1849 code; tests 4346 (+ oracle 4135).

---

### Task 12: The docstring trim (spec Order step 7; §7 "The docstring trim comes last")

**Goal:** Every docstring and comment in `src/domainia` says what the code computes and how to call it, and nothing about its history, its tests or this plan. The code, the doctests and the test tree do not change.

Assumes the tree after Task 11, whose commit is "Trim the kept files and the public API to spec 3.0; testing holds the noise helpers; README; 3.0.0".

**The rules.** They apply to every docstring and comment in `src/domainia`.
- Keep:
  - what the function computes;
  - the defining maths, conventions, units and orientation;
  - what a parameter means, and its shape, when its name and annotation don't say it;
  - the output's shape and dtype;
  - every Raises entry;
  - numerical caveats (precision, cost, memory).
- Remove:
  - history ("was", "used to", "now", "earlier", 2.0, the paper repo, the wdm_dev and lisa/wdm lineage, "vendored", "ported verbatim", "line for line", the JAX-era "host");
  - references to deleted code and to tests or tools;
  - "bit for bit" claims;
  - Parameters entries that only restate the signature;
  - text that repeats the module docstring;
  - measured-value anecdotes;
  - plan and spec cross-references;
  - line-number references.
- Shorten the prose that stays.
- Never delete a Raises entry or a units statement. A Raises entry may lose its rationale, never its condition.
- Comments: a stale or historical comment goes. A comment that explains a non-obvious step stays, shortened.

**How the prototype applied them.** These are the choices the rules leave open (plan deviations).
- **Doctests.** The doctests are tests: pytest collects 40 doctest items from `src/domainia/wilson`. Every Examples block stays verbatim. The AST check (Step 1) compares the doctest examples as well as the code.
- **Module docstrings.**
  - A module docstring states the conventions that the whole module shares: the window rule, the edge-band convention, the notes derivations.
  - The function docstrings lose their per-function `window : Window, optional` / "The Wilson window (default ...)" entries and their `Nf : int`-style entries. Step 1 counts 139 such lines at Task 11's tree; Step 8 lists what is left. One stays: `stationary.build_lambda_blocks_full_active`'s `rtol`, which now says what it is.
- **Derivation sources.** Each notes `.tex` path stays once, in the module docstrings of `wilson/transform.py` and `wilson/stationary.py`, which name where Algorithms 1–2 and the slab-IFFT recipe are derived. The paths leave the function docstrings. The WDM Paper I/II equation references stay.
- **Two numbers stay on purpose.**
  - `_covariance.CHUNK`'s "about 26 MiB (real model) or 47 MiB (complex)" stays, because it is what one call costs a caller.
  - The FFT-backend comment in `wilson/transform.py` goes from 18 lines to 2. It keeps the reason numpy's FFT is used, without the measured ratios.
- **Moved into public docstrings.** Three statements move, so nothing a caller needs is lost:
  - `transform.inverse` says "The inactive edge cells of ``wmn`` are ignored." Before, only a private helper said it.
  - `stationary.build_lambda_blocks_full_active` describes `rtol`.
  - `wilson/__init__.py`'s submodule list names `windows.sample_window` and `windows.WindowError`. The API list that named them goes, because they are not in `__all__`.
- **One Sphinx reference fixed.** `_checks.validate_finite` referred to :func:`sanitize_psd`, which `_checks` does not define. It becomes :func:`domainia.wilson.grid.sanitize_psd`. The prototype resolved every `:func:`/`:class:`/`:data:`/`:mod:`/`:meth:` target in src against the imported modules. Before this task this was the only target that did not resolve; after it, every target resolves.
- **Ordering.** `transform.forward` puts its Notes after Raises (numpydoc order).
- **Unchanged files.** `src/domainia/__init__.py` and `src/domainia/wilson/windows.py` keep their text: they hold nothing the rules remove.
- **The test tree does not change.** The task's one rule for tests is "fix a test docstring that names deleted code".
  - Step 8's grep finds two test docstrings that name deleted code, and both stay. `tests/wilson/meyer_cases.py::domainia_cases` names `EvolutionaryCovariance` and `BandAveragedVariance`. `tests/wilson/test_meyer_baseline.py::test_the_dropped_families_are_96_of_the_1504_keys` says "maps 54, inner_product 24".
  - Both explain the recorded baseline key names, and the baseline is never re-recorded.
  - Every other hit is a recorded key name, a `DROPPED` prefix or an absence assert.

**Files:**
- Modify (docstrings and comments only):
  - `src/domainia/_covariance.py`, `src/domainia/_transform.py`, `src/domainia/testing.py` (Step 2);
  - `src/domainia/wilson/__init__.py`, `src/domainia/wilson/_checks.py`, `src/domainia/wilson/filter.py`, `src/domainia/wilson/grid.py` (Step 3);
  - `src/domainia/wilson/matrix.py`, `src/domainia/wilson/noise.py` (Step 4);
  - `src/domainia/wilson/transform.py` (Step 5);
  - `src/domainia/wilson/evolutionary.py` (Step 6);
  - `src/domainia/wilson/stationary.py` (Step 7).
- No change: `src/domainia/__init__.py`, `src/domainia/wilson/windows.py`, everything under `tests/`, `README.md`, `pyproject.toml`.
- Create, delete: none in the repo. Step 1 writes the scratch script `$TMPDIR/dom3-ast.py`.
- Test: the full suite, with its 40 doctest items from `src/domainia/wilson`, and `tests/wilson/test_meyer_baseline.py` (the replay, 5 items).

**Acceptance Criteria:**
- [ ] Step 1, at Task 11's tree: the AST check prints `14 files: code and doctests unchanged; 160 doctest examples` against `HEAD`. The history grep prints 68 lines and the restated-parameter grep 139.
- [ ] Steps 2–7: after each step the AST check against `HEAD` still prints `14 files: code and doctests unchanged; 160 doctest examples`.
- [ ] Step 8:
  - the history grep prints exactly 2 lines, both error-message code;
  - the restated-parameter grep prints exactly 1 line;
  - the deleted-names grep prints nothing in src and exactly the 17 test lines listed there.
- [ ] Step 9: the AST check prints the 15 lines listed there. `pytest --collect-only -q src/domainia/wilson` collects 40 items.
- [ ] Step 9: `tests/wilson/test_meyer_baseline.py` gives `5 passed`; `meyer_cases.py --check` prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`; the exact check prints `0 of 1408 arrays differ from the baseline`. All three hold in PY and in FLOOR.
- [ ] Full suite (prototype): PY `951 passed, 3 skipped`; FLOOR `951 passed, 3 skipped`. These are the same as Task 11, because no test item is added or removed. No warnings and no collection errors.
- [ ] `$TMPDIR/dom3-size.py` prints `src 4059 lines / 1849 code` and `tests 4346 lines (+ oracle 4135)`. That is −766 src lines, with the code lines unchanged in every file.
- [ ] `git status --short` lists exactly the 12 modified src files.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P $TMPDIR/dom3-ast.py HEAD | tail -1 && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1` → `14 files: code and doctests unchanged; 160 doctest examples`, then `951 passed, 3 skipped in …`

**Steps:**

The edits are anchored to code names and code text, never to line numbers. Steps 2–7 use three kinds of edit:
- **"Replace the module docstring with:"** and **"Replace the docstring of `<dotted name>` with:"** replace the whole string literal with the block. That is every line from the line with the opening quotes (a `r` prefix included) through the line with the closing quotes. The block gives the complete new text, with its indentation and its `r` prefix where it has one. A dotted name names the function or class whose docstring it is: `domainia.wilson.evolutionary.Eq45Plan` is the class's own docstring.
- **"Replace the comment lines ... with ..."** is an Old/New pair. The Old text occurs exactly once in its file, both at Task 11's tree and after that file's docstring edits, so the edits of a file can be made in any order.
- **"Delete the comment lines ..."**: the Old lines go, newline included.

Each block is the prototype's text exactly. Copy it; don't retype it. A docstring's doctest lines are part of its block and are unchanged. The AST check after each step proves that only docstrings and comments changed and that the doctests are the same.

Step 10 is a git write in the main repo. git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

- [ ] **Step 1: The AST check, and the tree before the trim**

Write `$TMPDIR/dom3-ast.py`. It parses each src file at a git revision and in the working tree, removes every docstring (the first statement of a module, class or function, if it is a string), and compares `ast.dump` of the two trees, which carries no line numbers or comments. It also compares the doctest examples of the removed docstrings as a sorted list of (source, want) pairs:

```python
"""Docstring-only check for the 2026-11-12 plan.  Run from the repo root: ``python dom3-ast.py REV``.

For every src/**/*.py file, at REV and in the working tree: the AST with the docstrings
stripped (ast.dump, no attributes) must be equal, and so must the doctest examples."""
import ast
import doctest
import subprocess
import sys
from pathlib import Path

DOC = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def split(text):
    tree, docs = ast.parse(text), []
    for node in ast.walk(tree):
        body = getattr(node, "body", None) if isinstance(node, DOC) else None
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docs.append(body.pop(0).value.value)
    examples = sorted((e.source, e.want) for d in docs for e in doctest.DocTestParser().get_examples(d))
    return ast.dump(tree), examples


rev = sys.argv[1]
old = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, "src"], capture_output=True,
                     text=True, check=True).stdout.split()
old = sorted(p for p in old if p.endswith(".py"))
new = sorted(str(p) for p in Path("src").rglob("*.py"))
assert old == new, f"file sets differ: {sorted(set(old) ^ set(new))}"
bad = total = 0
for p in new:
    code0, ex0 = split(subprocess.run(["git", "show", f"{rev}:{p}"], capture_output=True,
                                      text=True, check=True).stdout)
    code1, ex1 = split(Path(p).read_text())
    ok = code0 == code1 and ex0 == ex1
    bad += not ok
    total += len(ex1)
    print(f"{p}: {'same' if code0 == code1 else 'CODE DIFFERS'}, {len(ex1)} doctest examples "
          f"{'same' if ex0 == ex1 else 'DIFFER'}")
print(f"{len(new)} files: {'code and doctests unchanged' if not bad else f'{bad} DIFFER'}; "
      f"{total} doctest examples")
sys.exit(1 if bad else 0)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD`

Expected: the 15 lines of Step 9, ending in `14 files: code and doctests unchanged; 160 doctest examples` (the working tree is `HEAD`). The prototype checked that the script can fail:
- a changed constant in one function prints `CODE DIFFERS` for that file, then `14 files: 1 DIFFER; 160 doctest examples`, and the script exits 1;
- a changed space in one doctest's expected output prints `7 doctest examples DIFFER` for `noise.py`.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py -v`

Expected (prototype), Task 11's size per file:
```
src 4825 lines / 1849 code
tests 4346 lines (+ oracle 4135)
      28      6  src/domainia/__init__.py
     142     79  src/domainia/_covariance.py
     134     88  src/domainia/_transform.py
     219     94  src/domainia/testing.py
     103     27  src/domainia/wilson/__init__.py
     213     50  src/domainia/wilson/_checks.py
     733    387  src/domainia/wilson/evolutionary.py
     137     34  src/domainia/wilson/filter.py
     428    111  src/domainia/wilson/grid.py
     426    123  src/domainia/wilson/matrix.py
     217     73  src/domainia/wilson/noise.py
    1051    387  src/domainia/wilson/stationary.py
     690    225  src/domainia/wilson/transform.py
     304    165  src/domainia/wilson/windows.py
```

The two text greps of Step 8 at Task 11's tree, as counts:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE --include='*.py' 'tests/|test_[a-z]+_[a-z_]+|tools/|lisa/wdm|wdm_dev|vendored|20[0-9]{2}-[0-9]{2}-[0-9]{2}|\b[Pp]orted|verbatim|bit-identical|bit for bit|bitwise|byte-|line for line|\bused to|\bwas\b|\bearlier\b|\bhost\b|domainia 3\.0 spec|benchmark|[Mm]easured[:;]' src | wc -l | tr -d ' '`

Expected: `68`

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE --include='*.py' 'window : Window|The Wilson window \(default|^ +(Nf|Nt|dt|N|Nf, Nt|m, n|Sn_func|Sn|rtol|tol|bands|m) : (int|float|callable)' src | wc -l | tr -d ' '`

Expected: `139`

- [ ] **Step 2: `_covariance.py`, `_transform.py`, `testing.py`**

#### `src/domainia/_covariance.py`

142 → 141 lines (79 code, unchanged); 1 docstring, 1 comment edit. The module docstring loses its spec cross-reference. The `CHUNK` comment loses how its figure was measured ("tracemalloc, 512 x 512") and "a larger chunk is no faster".

Replace the module docstring with:

```python
"""Per-point K x K noise covariance from a model.

A model returns the one-sided K x K cross-spectral density [data units^2/Hz], oriented so
that ``E[X_a conj(X_b)] = (T_obs / 2) S_ab(f_k)`` at an interior rfft bin, with
``S[b, a] = conj(S[a, b])``.  Each output is linear in the model, and exactly Hermitian with a
real diagonal at every point.
"""
```

Replace the comment lines

```python
#: call needs about 26 MiB (real model) or 47 MiB (complex) beyond the output (tracemalloc,
#: 512 x 512); a larger chunk is no faster.
```

with

```python
#: call needs about 26 MiB (real model) or 47 MiB (complex) beyond the output.
```

#### `src/domainia/_transform.py`

134 → 134 lines (88 code, unchanged); 1 docstring, 0 comment edits. The module docstring loses its spec cross-reference.

Replace the module docstring with:

```python
"""Data arrays between time, frequency and WDM.

The layouts, on the trailing axes of an array with any leading axes (typically ``(K, ...)``):

* ``"time"``: real ``(N,)``;
* ``"frequency"``: complex ``(N // 2 + 1,)``, ``dt * numpy.fft.rfft(x)``;
* ``"wdm"``: real ``(Nf + 1, Nt)``, the Wilson coefficients of ``window``, edge bands
  included.

Every route goes through the rfft ``numpy.fft.rfft(x)``.
"""
```

#### `src/domainia/testing.py`

219 → 183 lines (94 code, unchanged); 3 docstrings, 1 comment edit. The module docstring loses "(NumPy only)". `colored_noise` loses its provenance (the vendored private `_generate_colored_noise`) and its dated note. Its Raises keeps both conditions, with "(an odd ``N`` has no Nyquist bin)" in place of the measured variance. The "bit for bit" comment becomes one line. `empirical_variance` becomes a four-line summary: its shape, its draws, and what it checks.

Replace the module docstring with:

```python
"""Dense oracles and noise helpers for tests and small grids.  Every dense oracle here is
``O(N^2)`` memory or worse: keep ``N`` <~ 2048.

* :func:`dense_W`, :func:`wdm_congruence`: the WDM rows of a set of cells
  (``wilson.matrix.active_W``) and ``W_K Sigma W_K^T``.
* :func:`eq45_dense`: Eq. (45) on a set of cells by the direct sum over rfft bins
  (``wilson.evolutionary.eq45_direct``).
* :func:`colored_noise`, :func:`empirical_variance`: one draw of stationary noise (its rfft)
  and the Monte-Carlo WDM variance of such draws, which checks
  ``wilson.band_averaged_variance``.

The cells are a boolean ``mask`` of shape ``(Nf + 1, Nt)``, the WDM layout of
``wilson.grid.active_mask``, that selects active cells only.  Rows and columns follow
``numpy.nonzero(mask)``: band by band, time within a band.

:func:`dense_W`, :func:`wdm_congruence`, :func:`eq45_dense` and :func:`empirical_variance` take
``window=`` (default Meyer: the WDM transform) and compute in the Wilson basis of that window.

The oracles follow the package's rule: a wrong kind (a ``mask`` that is not boolean) raises
``TypeError``, a wrong value (a wrong shape, an inactive cell) ``ValueError``.

Test windows (``domainia.wilson.windows``):

* :func:`smoothstep`: a non-Meyer ramp for :class:`~domainia.wilson.RampWindow`.
* :func:`wide_profile`, :data:`ALT_WINDOW`: a non-Meyer admissible profile and the
  :class:`~domainia.wilson.SampledWindow` built from it.
"""
```

Replace the docstring of `domainia.testing.colored_noise` with:

```python
    """One realization of stationary noise with PSD $S_n(f)$, as the rfft of the time series.

    $E[|X_j|^2] = N\\,S_n(f_j)/(2\\,\\delta t)$ at **every** bin: DC and Nyquist are real
    but carry their full variance, not half.  ``rng`` draws the interior real parts, the
    interior imaginary parts, DC, then Nyquist.

    Raises
    ------
    TypeError
        If ``N`` is not an integer (``bool`` included), or ``dt`` is not a real
        number.
    ValueError
        If ``N`` is not positive and even (an odd ``N`` has no Nyquist bin), or ``dt``
        is not positive and finite.  Both are checked before the PSD is evaluated or
        ``rng`` is drawn from.
    """
```

Replace the docstring of `domainia.testing.empirical_variance` with:

```python
    """Monte-Carlo estimate of $\\langle w_{mn}^2 \\rangle$, shape ``(Nf + 1, Nt)``: the mean
    squared WDM coefficients of ``n_realizations`` draws of :func:`colored_noise` from
    ``numpy.random.default_rng(seed)``.  It checks
    :func:`domainia.wilson.band_averaged_variance`."""
```

Replace the comment lines

```python
    # Public entry point: same checks as validate_grid/validate_dt, on N itself.
    # They only raise, so valid input takes the oracle's path bit for bit.
```

with

```python
    # validate_grid's checks, on N itself.
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD | tail -1`

Expected: `14 files: code and doctests unchanged; 160 doctest examples`

- [ ] **Step 3: `wilson/__init__.py`, `wilson/_checks.py`, `wilson/filter.py`, `wilson/grid.py`**

#### `src/domainia/wilson/__init__.py`

103 → 57 lines (27 code, unchanged); 1 docstring, 0 comment edits. The module docstring keeps the WDM explanation and a compact list of `__all__`. The per-submodule function lists, which repeat the submodules' own docstrings, become one entry per submodule. "historical" and "the mathematicians whose work underlies the construction" go.

Replace the module docstring with:

```python
"""``domainia.wilson``: Wilson-basis kernels on the half-shifted lattice.  With the
default window :data:`MEYER` they are the WDM transform.  Every kernel that samples
the window takes ``window=`` (:mod:`domainia.wilson.windows`).

The WDM transform is a Wilson-basis time-frequency tiling with a Meyer
scaling-function window.  Despite the name, it is **not** a wavelet
transform: the frequency grid is uniform (not dyadic) and the basis
functions are Wilson-basis trigonometric atoms on a half-shifted
lattice.  The acronym stands for *Wilson--Daubechies--Meyer*.

Public API
----------
* :func:`forward`, :func:`inverse` (rfft <-> WDM) and :func:`forward_time`,
  :func:`inverse_time` (time <-> WDM), with any leading axes.
* :class:`Grid`: a validated ``(Nf, Nt, dt)`` that unpacks as ``(Nf, Nt, dt)``.
* :func:`band_averaged_variance`, :func:`evolutionary_band_variance`: noise variance
  grids (WDM Paper II, Eq. 36).
* :class:`Window` (the protocol), :class:`Meyer`, :class:`RampWindow`,
  :class:`SampledWindow`, :data:`MEYER` (``Meyer(4)``, the WDM transform's window).
* The submodules :mod:`domainia.wilson.grid` (grid helpers and the input checks the entry
  points apply), :mod:`domainia.wilson.filter` (the Meyer filter),
  :mod:`domainia.wilson.windows` (:func:`~domainia.wilson.windows.sample_window` and the
  :class:`~domainia.wilson.windows.WindowError` of a failed window check),
  :mod:`domainia.wilson.noise`, :mod:`domainia.wilson.matrix` (explicit matrices, small grids),
  :mod:`domainia.wilson.stationary` (fast stationary noise covariance) and
  :mod:`domainia.wilson.evolutionary` (kernels for evolutionary noise).
"""
```

#### `src/domainia/wilson/_checks.py`

213 → 179 lines (50 code, unchanged); 4 docstrings, 0 comment edits. `require_int` loses its rationale for `int()`, and `require_dtype` is reflowed. `validate_dt` loses its rationale and a Parameters block that restates the signature; its Raises stays. `validate_finite`'s essay (a measured 144-coefficient example, the benchmark remark, a Parameters block) becomes two paragraphs and its Raises; its Sphinx reference to `sanitize_psd` is fixed.

Replace the docstring of `domainia.wilson._checks.require_int` with:

```python
    """Return ``value`` as an ``int``; ``TypeError`` unless it is a Python or NumPy integer.

    ``bool``, ``np.bool_`` and every float (``2.0`` included) are refused.

    >>> from domainia.wilson._checks import require_int
    >>> require_int("n", 3)
    3
    >>> require_int("n", 3.0)
    Traceback (most recent call last):
    TypeError: n must be an integer, got float
    """
```

Replace the docstring of `domainia.wilson._checks.require_dtype` with:

```python
    """``TypeError`` unless the dtype kind of ``a`` is one of ``kinds``.

    Reads ``a.dtype`` when ``a`` has one, else ``numpy.asarray(a).dtype``.  ``expected`` is the
    phrase after "must be", article included (``"a real numeric array"``, with
    :data:`REAL_KINDS`).

    >>> import numpy as np; from domainia.wilson._checks import require_dtype, REAL_KINDS
    >>> require_dtype("x", np.zeros(2), REAL_KINDS, "a real numeric array")
    >>> require_dtype("x", np.zeros(2, complex), REAL_KINDS, "a real numeric array")
    Traceback (most recent call last):
    TypeError: x must be a real numeric array, got dtype complex128
    """
```

Replace the docstring of `domainia.wilson._checks.validate_dt` with:

```python
    """Raise unless the sampling interval ``dt`` is a positive finite real number.

    Raises
    ------
    TypeError
        If ``dt`` is not a real number (:func:`require_real`): ``None``, a ``bool``, a
        string, complex, a ``Decimal``, or an array of any shape, 0-d included.
    ValueError
        If ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.validate_dt(0.5)
    >>> wilson.grid.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    >>> wilson.grid.validate_dt(True)
    Traceback (most recent call last):
    TypeError: dt must be a real number, got bool
    """
```

Replace the docstring of `domainia.wilson._checks.validate_finite` with:

```python
    """Check that ``values`` holds no ``nan`` or ``inf`` (both parts of a complex array);
    ``context`` names it in the error message.

    One non-finite sample makes every coefficient of the transform ``nan``, silently, because
    each band goes through an FFT.  Stricter than
    :func:`domainia.wilson.grid.sanitize_psd`, which maps an infinite PSD sample to 0: an
    infinite *data* sample has no physical reading.

    Raises
    ------
    ValueError
        If any entry is ``nan`` or infinite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.validate_finite([1.0, 2.0], 'x')      # all finite: fine
    >>> wilson.grid.validate_finite([1.0, np.nan], 'x')   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: x: 1 of 2 entries are not finite (1 nan, 0 inf); first at index 1...
    """
```

#### `src/domainia/wilson/filter.py`

137 → 100 lines (34 code, unchanged); 3 docstrings, 1 comment edit. The module docstring loses a "Parameters" heading that has no parameters under it, the measured decay slopes and the lisa/wdm commit. It keeps the decay law $|t|^{-(d+1)}$. `nu_d` and `phi_centered` fold their Parameters and Returns into the summary. `phi_centered` loses a three-line comment about how `grid.delta_F` replaced an inline expression.

Replace the module docstring with:

```python
"""Meyer scaling function used as the WDM filter.

The Meyer scaling function $\\hat\\phi(f)$ has compact frequency support
on $[-3\\Delta F/4,\\,+3\\Delta F/4]$ and a smooth ($C^{d-1}$)
rolloff controlled by the regularized incomplete beta function $\\nu_d$.
Frequencies are in Hz throughout, with the Paper I (eq:Meyer-def)
normalisation: the flat value is $1/\\sqrt{\\Delta F}$.

It is the window that, paired with the half-shifted Wilson lattice and the
$C_{mn}$ phase factor, gives an orthonormal time-frequency basis at
critical density (the partition-of-unity property).

The shape parameter ``d`` (default ``4``) sets the smoothness of the rolloff; the basis
functions decay as $|t|^{-(d+1)}$ in time.
"""
```

Replace the docstring of `domainia.wilson.filter.nu_d` with:

```python
    """Regularized incomplete beta function $I_x(d, d)$, element-wise, with ``x`` clipped to
    ``[0, 1]``: the Meyer rolloff's ramp, $\\nu_d(0) = 0$, $\\nu_d(1) = 1$, $C^{d-1}$ smooth.

    Raises
    ------
    TypeError
        If ``d`` is not an integer (``bool`` included).
    ValueError
        If ``d < 1``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.filter.nu_d(np.array([0.0, 0.5, 1.0]))
    array([0. , 0.5, 1. ])
    """
```

Replace the docstring of `domainia.wilson.filter.phi_centered` with:

```python
    """Meyer filter $\\hat\\phi(f)$ (Paper I eq:Meyer-def), shape ``(Nt,)``, centered at
    index $N_t/2$, with rolloff smoothness ``d``.

    The grid spans frequencies $f \\in [-N_t/2,\\,N_t/2-1]/T$ with
    $T = N_f N_t \\delta t$.  In the flat region ($|f| < \\Delta F/4$)
    the value is $1/\\sqrt{\\Delta F}$; in the rolloff region
    ($\\Delta F/4 \\le |f| < 3\\Delta F/4$) it follows the Meyer
    cosine taper; otherwise it is zero.

    Raises
    ------
    TypeError
        If ``Nf``, ``Nt`` or ``d`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid or ``dt`` is invalid, or if $\\Delta F = 1/(2\\,\\delta t N_f)$
        is not a positive finite double (see :func:`domainia.wilson.grid.delta_F`).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> phi = wilson.filter.phi_centered(8, 16, 1.0)
    >>> phi.shape, bool(phi[8] == phi.max())   # flat top includes index Nt/2
    ((16,), True)
    """
```

Delete the comment lines

```python
    # grid.delta_F is the same expression, 1/(2 dt Nf), so the window is bitwise
    # unchanged; it raises when that width is 0 or inf (an unrepresentable dt*Nf),
    # where the inline form returned a window -- and a transform -- of exact zeros.
```

#### `src/domainia/wilson/grid.py`

428 → 330 lines (111 code, unchanged); 9 docstrings, 0 comment edits. `Grid`'s rationale is shortened. `validate_grid`, `sanitize_psd` and `active_mask` lose Parameters and Returns blocks that restate the signature. `sanitize_psd` also loses its history paragraph, the "Gaussian log-likelihood" line that Task 11 left for this task. `delta_F`'s Raises keeps its condition and loses the lisa/wdm commit. `time_pixel_centers` loses its 2026-07-27 note.

Replace the docstring of `domainia.wilson.grid.Grid` with:

```python
    """A validated ``(Nf, Nt, dt)`` triple, with the derived quantities.

    Unpacking one ``Grid`` at each call site rules out swapping ``Nf`` and ``Nt``, which no
    check can see (both are positive even ints with the same product).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> g = wilson.Grid(Nf=8, Nt=16, dt=1.0)
    >>> g.N, g.dF, g.dT
    (128, 0.0625, 8.0)
    >>> x = np.random.default_rng(0).standard_normal(g.N)
    >>> wilson.forward_time(x, *g).shape          # unpacks as (Nf, Nt, dt)
    (9, 16)
    >>> wilson.Grid(Nf=9, Nt=16, dt=1.0)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    """
```

Replace the docstring of `domainia.wilson.grid.validate_grid` with:

```python
    """Raise unless ``Nf`` and ``Nt`` are positive even integers.

    Raises
    ------
    TypeError
        If either argument is not an integer (``bool`` included).
    ValueError
        If either argument is not positive and even.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.validate_grid(8, 16)          # even and positive: fine
    >>> wilson.grid.validate_grid(9, 16)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    >>> wilson.grid.validate_grid(8.0, 16)
    Traceback (most recent call last):
    TypeError: Nf must be an integer, got float
    """
```

Replace the docstring of `domainia.wilson.grid.sanitize_psd` with:

```python
    """The package-wide policy for PSD samples ``values`` (``context`` names them in messages).

    An infinite sample means "this bin carries no measurable power" (sensitivity curves
    return ``inf`` at ``f = 0``), so it maps to ``0.0``; that keeps
    :func:`domainia.wilson.matrix.build_C_time` finite and positive semi-definite.  ``nan``
    has no physical reading and a negative PSD is a bug in the model, so both raise.

    Returns
    -------
    ndarray, float64
        Finite, non-negative PSD values with non-finite entries set to ``0.0``.

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
```

Replace the docstring of `domainia.wilson.grid.n_samples` with:

```python
    """Total time-domain sample count $N = N_f N_t$, an ``int``.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer (``bool`` included).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.n_samples(8, 16)
    128
    """
```

Replace the docstring of `domainia.wilson.grid.delta_F` with:

```python
    """Frequency-pixel width $\\Delta F = 1/(2\\,\\delta t\\,N_f)$ in Hz, a ``float``.

    Raises
    ------
    TypeError
        If ``Nf`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` is not positive and even, if ``dt`` is not positive
        and finite, or if $2\\,\\delta t N_f$ is so small or so large that the
        width is not a positive finite double (an ``inf`` or ``0.0`` width would make
        the transform return exact zeros).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.delta_F(8, 1.0)
    0.0625
    >>> wilson.grid.delta_F(0, 1.0)
    Traceback (most recent call last):
    ValueError: Nf must be positive, got 0
    >>> wilson.grid.delta_F(8, 5e-324)          # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: delta_F is not representable for Nf=8, dt=5e-324...
    """
```

Replace the docstring of `domainia.wilson.grid.delta_T` with:

```python
    """Time-pixel width $\\Delta T = N_f\\,\\delta t$, a ``float``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.delta_T(8, 1.0)
    8.0
    """
```

Replace the docstring of `domainia.wilson.grid.band_frequencies` with:

```python
    """Band-center frequencies $f_m = m\\,\\Delta F$ for $m = 0,\\ldots,N_f$, shape ``(Nf+1,)``:
    ``freqs[0] == 0`` (DC) and ``freqs[Nf] == 1/(2*dt)`` (Nyquist).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> f = wilson.grid.band_frequencies(8, 1.0)
    >>> f.shape, float(f[0]), float(f[-1])   # DC .. Nyquist
    ((9,), 0.0, 0.5)
    """
```

Replace the docstring of `domainia.wilson.grid.time_pixel_centers` with:

```python
    """Pixel-center times $t_n = n\\,\\Delta T$ for $n = 0,\\ldots,N_t-1$, shape ``(Nt,)``.

    These are the times of the WDM lattice: the energy centroid of
    the basis function $g_{mn}$ sits at exactly $n\\,\\Delta T$.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.grid.time_pixel_centers(8, 4, 1.0)
    array([ 0.,  8., 16., 24.])
    """
```

Replace the docstring of `domainia.wilson.grid.active_mask` with:

```python
    """Boolean mask of shape ``(Nf+1, Nt)`` marking active (nonzero) pixels.

    - Interior bands ($1 \\le m \\le N_f - 1$): all $N_t$ time bins active.
    - DC band ($m=0$): only even $n$ are active ($N_t/2$ pixels).
    - Nyquist band ($m=N_f$): only $n$ with $(N_f + n)$ even are active
      ($N_t/2$ pixels).

    Total active count equals $N = N_f N_t$.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> mask = wilson.grid.active_mask(8, 16)
    >>> mask.shape, int(mask.sum())   # total active == N
    ((9, 16), 128)
    """
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD | tail -1`

Expected: `14 files: code and doctests unchanged; 160 doctest examples`

- [ ] **Step 4: `wilson/matrix.py`, `wilson/noise.py`**

#### `src/domainia/wilson/matrix.py`

426 → 344 lines (123 code, unchanged); 8 docstrings, 2 comment edits. The module docstring folds `rfft_to_real` and `real_to_rfft` into its list. It also shortens the $O(N^2)$ paragraph and drops the `tools/benchmark.py` pointer. `build_W` loses its test references and says that it "agrees with the definition to rounding". The builders lose Parameters entries that restate the signature. Of the comments, the unit-norm remark goes and the lisa/wdm commit leaves the even-`N` comment.

Replace the module docstring with:

```python
"""Explicit WDM transform matrix and noise covariance.

The WDM forward transform is a linear map $w = W\\,x$ where $W$ is a
real $(N_f+1)N_t \\times N$ matrix.  Restricted to its $N$ active rows
(skipping the inactive edge-band entries), $W_a$ is square and
orthogonal **up to the sampling scale**:

$$W_a W_a^T = W_a^T W_a = \\delta t\\,I,$$

i.e. $W_a = \\sqrt{\\delta t}\\,Q$ with $Q$ orthogonal.  Invert with
$x = W_a^T w / \\delta t$.  The $\\delta t$ is the same factor that
appears in Parseval ($\\sum w^2 = \\delta t \\sum x^2$); $W_a$ is
exactly orthogonal only when $\\delta t = 1$.

Every builder takes ``window=`` (default :data:`domainia.wilson.MEYER`) and returns the
matrix of that window's Wilson basis; the identities above hold for every admissible
window (:mod:`domainia.wilson.windows`).

This module exposes:

* :func:`build_W` — full $(N_f+1)N_t \\times N$ matrix.
* :func:`active_W` — the $N \\times N$ restriction ($W_a W_a^T = \\delta t\\,I$).
* :func:`build_C_time` — Toeplitz time-domain noise covariance from
  a PSD (vectorized via ``irfft`` of the power spectrum).
* :func:`build_C_wdm` — WDM-domain noise covariance via the similarity
  transform $C^{(\\text{WDM})} = W\\,C^{(\\text{time})}\\,W^T$.
* :func:`rfft_to_real`, :func:`real_to_rfft` — the complex rfft as a real vector of
  length $N$ (the real-DFT representation).

These functions are $O(N^2)$ in memory and $O(N^3)$ in compute: references for
small $N$ (roughly $N \\lesssim 2048$); the FFT-based transforms are in
:mod:`domainia.wilson.transform`.
"""
```

Replace the docstring of `domainia.wilson.matrix.build_W` with:

```python
    """Build the full $(N_f+1)N_t \\times N$ WDM transform matrix for ``window``.

    Row $\\alpha = (m, n)$ is the basis function $g_{mn}$ sampled in time, so
    $W$ is *defined* by "column $k_0$ is the WDM transform of a delta at sample
    $k_0$".  It is *built* from the lattice's shift-covariance instead, which
    agrees with the definition to rounding.

    Returns
    -------
    ndarray, shape ((Nf+1)*Nt, N)
        Real matrix.  Rows are indexed by the flattened ``(m, n)``
        pixel index $\\alpha = m\\,N_t + n$; columns by the time
        sample $k_0 = 0, \\ldots, N - 1$.  Inactive edge-band rows are exactly zero.

    Notes
    -----
    Cost: $2(N_f+1)$ inverse transforms plus $O(N^2)$ of copying, and
    $O(N^2)$ memory.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> W = wilson.matrix.build_W(4, 8, 1.0)
    >>> W.shape
    (40, 32)
    """
```

Replace the docstring of `domainia.wilson.matrix.active_W` with:

```python
    """Restrict :func:`build_W` to its $N$ active rows.

    Returns the square $N \\times N$ matrix obtained by keeping only
    the rows corresponding to active pixels (interior bands and the
    parity-selected edge-band entries).  This matrix is orthogonal up to
    the sampling scale:
    $W_a W_a^T = W_a^T W_a = \\delta t\\,I$.  Invert with
    $x = W_a^T w / \\delta t$; the plain transpose is the inverse only
    at $\\delta t = 1$.

    Returns
    -------
    ndarray, shape (N, N)
        Square real matrix satisfying $W_a W_a^T = \\delta t\\,I$.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Wa = wilson.matrix.active_W(4, 8, 1.0)
    >>> Wa.shape, bool(np.allclose(Wa @ Wa.T, 1.0 * np.eye(32)))
    ((32, 32), True)
    """
```

Replace the docstring of `domainia.wilson.matrix.build_C_time` with:

```python
    """Toeplitz time-domain noise covariance of ``N`` samples from a one-sided PSD.

    For stationary noise with one-sided PSD ``Sn_func(f)`` $= S_n(f)$, the autocovariance
    is $R(\\tau) = E[n_k n_{k+\\tau}]$, and the covariance matrix is
    Toeplitz: $C[k, k'] = R(|k - k'|)$.  Built via the inverse rfft
    of the discrete two-sided power spectrum.

    Returns
    -------
    ndarray, shape (N, N)
        Real symmetric Toeplitz covariance matrix.

    Raises
    ------
    TypeError
        If ``N`` is not an integer (``bool`` included), or ``dt`` is not a real number.
    ValueError
        If ``N`` is not positive and even, or ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> C = wilson.matrix.build_C_time(32, 1.0, lambda f: 1e-3 * np.ones_like(f))
    >>> C.shape, bool(np.allclose(C, C.T))
    ((32, 32), True)
    """
```

Replace the docstring of `domainia.wilson.matrix.build_C_wdm` with:

```python
    """WDM-domain covariance via the similarity transform $W\\,C^{(\\text{time})}\\,W^T$.

    Returns the full $(N_f+1)N_t \\times (N_f+1)N_t$ covariance matrix
    of the WDM coefficients under stationary noise with PSD $S_n(f)$.
    Block-tridiagonal in the band index $m$ for every admissible ``window``
    (compact support inside one slab): only $|m - m'| \\le 1$ blocks are
    nonzero (up to numerical noise).

    Returns
    -------
    ndarray, shape ((Nf+1)*Nt, (Nf+1)*Nt)
        Symmetric real covariance matrix.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> C = wilson.matrix.build_C_wdm(4, 8, 1.0, lambda f: 1e-3 * np.ones_like(f))
    >>> C.shape
    (40, 40)
    """
```

Replace the docstring of `domainia.wilson.matrix.build_C_wdm_active` with:

```python
    """Active-pixel restriction of :func:`build_C_wdm`: the $N \\times N$ covariance of the
    active WDM coefficients, i.e. ``build_C_wdm`` on the active rows and columns.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> C = wilson.matrix.build_C_wdm_active(4, 8, 1.0, flat)
    >>> C.shape, float(np.round(np.diag(C)[0], 6))   # ~ S_n / 2
    ((32, 32), 0.0005)
    """
```

Replace the docstring of `domainia.wilson.matrix.rfft_to_real` with:

```python
    """Pack a complex rfft into a real vector of length $N$.

    The packing is

    $$\\bm{x}_R = (\\text{Re}\\,X_0,\\,\\text{Re}\\,X_1,\\,\\text{Im}\\,X_1,
    \\,\\ldots,\\,\\text{Re}\\,X_{N/2-1},\\,\\text{Im}\\,X_{N/2-1},
    \\,\\text{Re}\\,X_{N/2})^T,$$

    yielding $1 + 2(N/2 - 1) + 1 = N$ entries, from ``x_rfft`` of shape ``(N//2+1,)``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> x = np.random.default_rng(0).standard_normal(32)
    >>> packed = wilson.matrix.rfft_to_real(np.fft.rfft(x), 32)
    >>> packed.shape
    (32,)
    >>> wilson.matrix.rfft_to_real(np.fft.rfft(x[:5]), 5)
    Traceback (most recent call last):
    ValueError: N must be even, got 5
    """
```

Replace the docstring of `domainia.wilson.matrix.real_to_rfft` with:

```python
    """Inverse of :func:`rfft_to_real`: ``(N,)`` real to ``(N//2+1,)`` complex.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> x = np.random.default_rng(0).standard_normal(32)
    >>> packed = wilson.matrix.rfft_to_real(np.fft.rfft(x), 32)
    >>> back = wilson.matrix.real_to_rfft(packed, 32)
    >>> bool(np.allclose(back, np.fft.rfft(x)))
    True
    >>> wilson.matrix.real_to_rfft(np.zeros(5), 5)
    Traceback (most recent call last):
    ValueError: N must be even, got 5
    """
```

Delete the comment lines

```python
    #
    # The identity holds for the unit-norm basis, which is the only one there is.
```

Replace the comment lines

```python
    # strided ranges 1:N-2:2 and 2:N-1:2.  Even N only (lisa/wdm b5973b7): for
    # odd N the last stored bin is not Nyquist and its imaginary part would be
    # silently dropped, yet N//2+1 is still the right bin count.
```

with

```python
    # strided ranges 1:N-2:2 and 2:N-1:2.  Even N only: for odd N the last stored bin
    # is not Nyquist and its imaginary part would be silently dropped, yet N//2+1 is
    # still the right bin count.
```

#### `src/domainia/wilson/noise.py`

217 → 170 lines (73 code, unchanged); 3 docstrings, 3 comment edits. The module docstring loses its argument-order paragraph. `band_averaged_variance` loses its 2026-07-27 harmonic-mean note and its Parameters. `evolutionary_band_variance` loses "Was wdm_dev ..." and its Parameters. The fold and weights comments are shortened, losing the measured bin counts and the history of a removed fallback. The banner loses "moved from wdm_dev (verbatim)".

Replace the module docstring with:

```python
"""Per-pixel noise variance grids in the WDM domain.

For stationary Gaussian noise with one-sided PSD $S_n(f)$, the
expected variance of a normalized WDM coefficient $w_{mn}$ is
approximately

$$\\sigma_{mn}^2 \\approx S_n(f_m) / 2,$$

uniformly across all pixels.  Both variance grids here are the filter-weighted
band average of WDM Paper II, Eq. 36:

* :func:`band_averaged_variance` — filter-weighted mean of $S_n$ over
  each band; exact, and the right choice for steep PSDs.
* :func:`evolutionary_band_variance` — the same mean of a time-varying PSD
  $S_n(f, t)$ at each time pixel, for locally-stationary noise.

A Monte-Carlo estimator, :func:`domainia.testing.empirical_variance`, checks them.

Both grids have shape ``(Nf+1, Nt)``.  They fill the inactive edge pixels
($m = 0$ and $m = N_f$ at odd $n$) by the same formula as the active ones, so they hold
about $S_n/2$ there, not 0; :func:`domainia.wdm_covariance` sets those cells to 0.

Every kernel here that reaches the window takes ``window=`` (a
:class:`domainia.wilson.windows.Window`; default :data:`domainia.wilson.MEYER`) and
computes in the Wilson basis of that window.
"""
```

Replace the docstring of `domainia.wilson.noise.band_averaged_variance` with:

```python
    """Filter-weighted mean of $S_n$ over each band, divided by 2.

    For each band $m$, computes the effective PSD

    $$S_n^{\\text{eff}}(m)
    = \\frac{\\sum_j \\hat\\phi^2(f_j - f_m)\\,S_n(f_j)}
            {\\sum_j \\hat\\phi^2(f_j - f_m)}$$

    and returns $\\sigma_{mn}^2 = S_n^{\\text{eff}}(m)/2$ broadcast
    over $n$, shape ``(Nf+1, Nt)``.  This is exact: it reproduces the diagonal of the WDM
    covariance $W\\,C\\,W^T$ to machine precision, because the pixel
    variance is the $\\hat\\phi^2$-weighted average of the PSD across
    the band.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> steep = lambda f: 1e-3 * (1.0 + 1e2 / (f + 1 / (2 * dt * Nf)) ** 2)
    >>> var = wilson.band_averaged_variance(Nf, Nt, dt, steep)
    >>> var.shape
    (9, 16)
    >>> # exact: reproduces the diagonal of the WDM covariance
    >>> truth = np.diag(wilson.matrix.build_C_wdm_active(Nf, Nt, dt, steep))
    >>> bool(abs(var[1, 0] - truth[Nt // 2]) / truth[Nt // 2] < 1e-12)
    True
    """
```

Replace the docstring of `domainia.wilson.noise.evolutionary_band_variance` with:

```python
    r"""The filter-weighted band average of :math:`S(f, t_n)` at every cell centre,

    .. math::  \sigma^2_{mn} = \frac12\,
               \frac{\sum_j \tilde\phi^2(f_j - f_m)\,S(f_j, t_n)}{\sum_j \tilde\phi^2(f_j - f_m)},

    WDM Paper II, Eq. 36 (eq:wdm_variance_band_average; the edge-band factor is not
    applied -- inactive edge cells carry the same value, as in
    :func:`band_averaged_variance`).  ``Sn(f, t)`` is elementwise and broadcasting
    (argument order ``(f, t)``).  In the stationary limit this is
    :func:`band_averaged_variance`.  Returns ``(Nf + 1, Nt)``, or
    ``(len(bands), Nt)`` for the bands ``bands`` (all by default).
    """
```

Replace the comment lines

```python
    # spectrum actually aliases. Folding the top band about 0 (plain ``abs``)
    # samples the PSD at the wrong frequencies and biases band m=Nf by several
    # percent. One vectorised call also means one user-PSD invocation instead
    # of Nf+1, which matters for an expensive sensitivity curve.
```

with

```python
    # spectrum actually aliases.  One call of Sn_func covers every band.
```

Replace the comment lines

```python
    # Weight by every bin the window touches. A bin where S_n vanishes still
    # belongs in the denominator -- it contributes zero power, which is exactly
    # what pulls the band average down. Dropping such bins from both sums (as
    # an earlier ``Sn_vals > 0`` mask did) rescales the average by the surviving
    # weight and overstates a band with a spectral cutoff in it by up to 3.2x.
    # The Meyer window always has positive bins -- measured 3/4 of them at
    # Nf=2, Nt=4 and 1535/2048 at Nf=1024, Nt=2048, across dt from 1e-6 to 1e6 --
    # and so does every admissible window (its partition of unity makes phi(f) or
    # phi(f - dF) nonzero at every bin), so ``in_band`` is never empty. An earlier
    # revision carried an ``if not np.any(in_band): Sn_eff = inf`` fallback for
    # that case. It was unreachable, and a silent infinite variance is the last
    # thing this package should produce, so it is gone rather than merely untested.
```

with

```python
    # Weight by every bin the window touches.  A bin where S_n vanishes still
    # belongs in the denominator: it contributes zero power, which is exactly
    # what pulls the band average down.  ``in_band`` is never empty: an admissible
    # window's partition of unity makes phi(f) or phi(f - dF) nonzero at every bin.
```

Replace the comment lines

```python
# Evolutionary PSDs S(f, t) -- moved from wdm_dev ``evolutionary.py`` (verbatim)
```

with

```python
# Evolutionary PSDs S(f, t)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD | tail -1`

Expected: `14 files: code and doctests unchanged; 160 doctest examples`

- [ ] **Step 5: `wilson/transform.py`**

#### `src/domainia/wilson/transform.py`

690 → 539 lines (225 code, unchanged); 14 docstrings, 4 comment edits. The module docstring loses the comparison with "earlier versions" and the `normalize_edges` history paragraph. It keeps the notes `.tex` path and the Precision paragraph. The FFT-backend comment goes from 18 lines to 2. The docstrings lose the test and baseline references, the notes section pointers, "Host-side" and "used to be called twice". `forward` and `inverse` name their algorithm (1 and 2). Of the comments, the oracle-loop remark goes, a comment about strictly increasing indices is deleted, and the edge-basis comment is shortened.

Replace the module docstring with:

```python
"""Forward and inverse WDM transforms (RFFT-only, linear-centred slab).

These implementations follow Algorithms 1 and 2 of
``notes/Time_Frequency_Analysis/wdm_transform_algorithms.tex``: the
data lives only in the one-sided ``rfft`` $X[k]$, $k = 0, \\ldots, N/2$;
the two-sided spectrum is never materialised.  Out-of-range slab
indices are served by Hermitian symmetry on read (forward) or
discarded (inverse, where they are redundant by Hermitian symmetry of
the real WDM wavelets).

Every kernel here takes ``window=`` (a :class:`domainia.wilson.windows.Window`;
default :data:`domainia.wilson.MEYER`, the Meyer window of the WDM transform) and
computes in the Wilson basis of that window: $\\hat\\phi_c$ below is the window's
centred samples, from :func:`domainia.wilson.windows.sample_window`.  The formulas
and the edge normalisation are the same for every admissible window.

The slab is linear-centred:

$$\\ell_{m,k} \\equiv k + (m - 1)\\,N_t/2,\\qquad k = 0, \\ldots, N_t - 1.$$

Interior bands ($1 \\le m \\le N_f - 1$) read directly from the rfft;
only the DC band ($m = 0$) and the Nyquist band ($m = N_f$) ever take
the Hermitian-symmetry branch ($N_t$ slab fetches each).

Forward:

$$w_{mn} = \\frac{\\sqrt{2}}{N_f}\\,(-1)^{(m+1)n}\\,
\\Re\\!\\left[C_{mn}^{*}\\,\\mathrm{IFFT}\\!\\bigl(u_m\\bigr)[n]\\right],$$

with $u_m[k] = X_{\\ell_{m,k}}\\,\\hat\\phi_c[k]$ (centred filter,
peaked at $k = N_t/2$) and $C_{mn} = 1$ if $m+n$ even, $C_{mn} = i$
otherwise.

Inverse:

$$X[\\ell_{m,k}] \\mathrel{+}= \\frac{1}{\\sqrt{2}\\,\\delta t}\\,
\\hat\\phi_c[k]\\,\\mathrm{FFT}\\!\\bigl(w_{m\\cdot}\\,C_{mn}\\,
(-1)^{(m+1)n}\\bigr)[k],$$

with writes outside $[0, N/2]$ discarded.

The DC and Nyquist edge-band coefficients are divided by $\\sqrt{2}$ so
that all basis functions have unit norm.  This makes the WDM transform
matrix orthogonal up to the sampling scale ($W_a W_a^T = \\delta t\\,I$,
so invert with $x = W_a^T w/\\delta t$) and the Parseval identity exact:
$\\sum_{m,n} w_{mn}^2 = \\delta t \\sum_k x_k^2$.  This is the package's only
coefficient convention.

Precision
---------
Every entry point computes in double and returns float64 (complex128 for a
spectrum).  Input of another accepted dtype (a lower-precision float, an
integer) is promoted first.
"""
```

Replace the docstring of `domainia.wilson.transform._parity` with:

```python
    """``((m+n) even, (-1)^{(m+1)n})`` for one band.

    :func:`domainia.wilson.stationary._carriers` holds the 2-D form of the same rule.
    """
```

Replace the docstring of `domainia.wilson.transform._build_slab` with:

```python
    """Linear-centred slab $u_m[k] = X[\\ell_{m,k}]$ for $k \\in [0, N_t-1]$:
    ``(..., N//2+1)`` complex to ``(..., Nt)`` complex.

    Hermitian extension is applied inline: negative indices use the
    conjugate of the positive bin (only fires at $m = 0$, $k < N_t/2$);
    over-Nyquist indices use the conjugate of the mirror bin (only
    fires at $m = N_f$, $k > N_t/2$).  For interior bands every lookup
    is a direct rfft read.
    """
```

Replace the docstring of `domainia.wilson.transform._prepare_edges_for_inverse` with:

```python
    """Working copy of ``wmn`` with the edge bands put into ``inverse``'s form.

    Two things, both edge-only:

    1. **Zero the parity-killed entries.** They correspond to zero rows of $W$,
       so the adjoint must ignore them; otherwise junk there leaks into the
       active pixels of the reconstruction.
    2. **Undo the unit-norm $\\sqrt2$** that :func:`forward` applied.
    """
```

Replace the docstring of `domainia.wilson.transform._scatter_interior_bands` with:

```python
    """Accumulate the interior bands into the rfft accumulator ``X``; returns it.

    One batched length-$N_t$ FFT, then a contiguous slice-add per band. Within
    a band the slab indices are consecutive and duplicate-free, so ``X[a:b] +=``
    is exact and much cheaper than ``np.add.at``; consecutive bands overlap by
    $N_t/2$, which the sequential loop handles correctly.

    ``X`` is ``(..., N//2+1)`` and ``wmn_work`` ``(..., Nf+1, Nt)``.  There is always an
    interior band: :func:`domainia.wilson.grid.validate_grid` requires ``Nf`` even, so
    ``Nf >= 2``.
    """
```

Replace the docstring of `domainia.wilson.transform._scatter_edge_bands` with:

```python
    """Accumulate the DC and Nyquist bands into ``X``; returns it.

    These scatter partly outside $[0, N/2]$; those writes are redundant by
    Hermitian symmetry of the real WDM wavelets and are discarded.  ``ell`` is strictly
    increasing, so the surviving indices are duplicate-free and a plain ``+=`` is exact.
    """
```

Replace the docstring of `domainia.wilson.transform._gather_bands` with:

```python
    """Coefficient grid(s) from spectra: ``(..., N//2+1) -> (..., Nf+1, Nt)``.

    The whole of the forward algorithm; :func:`forward` validates and calls it.

    Interior bands $m = 1 \\ldots N_f-1$: every slab index
    $\\ell_{m,k} = k + (m-1)N_t/2$ lies in $[0, N/2]$
    (max $= (N_f-1)N_t/2 - 1 < N/2$), so the slab is a pure gather from the
    rfft with no Hermitian branch, and all of them go through one batched
    length-$N_t$ IFFT.  Edge bands $m \\in \\{0, N_f\\}$ take the
    Hermitian-symmetry slab path.
    """
```

Replace the docstring of `domainia.wilson.transform.forward` with:

```python
    """Forward WDM transform (Algorithm 1): $\\text{rfft}(x) \\to w_{mn}$.

    Parameters
    ----------
    x_rfft : ndarray, complex, shape (..., N//2+1)
        One-sided DFT of the real time series, as returned by
        ``np.fft.rfft(x)`` (no $\\delta t$ scaling).  Any leading axes: each
        slice is transformed on its own, and a stack goes through one batched FFT.

    Returns
    -------
    ndarray, float, shape (..., Nf+1, Nt)
        Real WDM coefficients.  Edge bands have $N_t/2$ active values;
        the inactive entries vanish by Hermitian symmetry of the slab,
        realized to floating-point rounding (~1e-16 relative to the
        peak) rather than bit-exactly.  Compare them with a tolerance.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If ``Nf`` or ``Nt`` is not positive and even, if the last axis of
        ``x_rfft`` does not have length ``N//2 + 1``, or if it holds a ``nan`` or
        ``inf`` (:func:`domainia.wilson.grid.validate_finite`).

    Notes
    -----
    Cost: $O(N \\log N_t)$.  The $N_f-1$ interior bands go through one batched
    length-$N_t$ IFFT, so the working set is $O(N_f N_t)$, not the $O(N_t)$ of a
    per-band loop.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward(np.fft.rfft(x), Nf, Nt, dt)
    >>> w.shape
    (9, 16)
    >>> bool(np.isclose(np.sum(w**2), dt * np.sum(x**2)))  # Parseval
    True
    >>> xs = np.random.default_rng(1).standard_normal((2, 3, Nf * Nt))   # leading axes
    >>> wilson.forward(np.fft.rfft(xs), Nf, Nt, dt).shape
    (2, 3, 9, 16)
    """
```

Replace the docstring of `domainia.wilson.transform.inverse` with:

```python
    """Inverse WDM transform (Algorithm 2): $w_{mn} \\to \\text{rfft}(x)$.

    Each band scatters into the one-sided rfft accumulator at its linear-centred slab
    indices.  The inactive edge cells of ``wmn`` are ignored.

    Parameters
    ----------
    wmn : ndarray, float, shape (..., Nf+1, Nt)
        WDM coefficient grid, with any leading axes.

    Returns
    -------
    ndarray, complex, shape (..., N//2+1)
        Reconstructed one-sided DFT (compatible with ``np.fft.irfft``).

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, ``dt`` is not a real number, or ``wmn``
        is not a real numeric array: a complex grid would have its imaginary part
        silently discarded, and a bool, str or object grid is refused too.
    ValueError
        If the grid is invalid, if ``wmn`` is wrongly shaped, or
        if it holds a ``nan`` or ``inf`` (see :func:`domainia.wilson.grid.validate_finite`).

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

Replace the docstring of `domainia.wilson.transform.forward_time` with:

```python
    """Time-domain forward transform, ``forward(np.fft.rfft(x), ...)``: a real ``x`` of shape
    ``(..., N)``, $N = N_f N_t$, to ``(..., Nf+1, Nt)``.

    Raises
    ------
    TypeError
        If ``Nf`` or ``Nt`` is not an integer, or ``dt`` is not a real number.
    ValueError
        If the grid is invalid, if the last axis of ``x`` does not have ``Nf*Nt``
        samples, or if it holds a ``nan`` or ``inf`` (gapped data often arrives
        ``nan``-filled: fill the gaps first).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> wilson.forward_time(x, Nf, Nt, dt).shape
    (9, 16)
    >>> x[5] = np.nan                      # one gap-filled sample
    >>> wilson.forward_time(x, Nf, Nt, dt)    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: forward_time: x: 1 of 128 entries are not finite...
    """
```

Replace the docstring of `domainia.wilson.transform.inverse_time` with:

```python
    """Time-domain inverse transform, ``np.fft.irfft(inverse(wmn, ...), n=N)``:
    ``(..., Nf+1, Nt)`` to ``(..., N)``.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> x = np.random.default_rng(0).standard_normal(Nf * Nt)
    >>> w = wilson.forward_time(x, Nf, Nt, dt)
    >>> bool(np.allclose(wilson.inverse_time(w, Nf, Nt, dt), x, atol=1e-12))
    True
    >>> w8 = wilson.forward_time(x, Nf, Nt, dt, window=wilson.Meyer(8))       # another window
    >>> bool(np.allclose(wilson.inverse_time(w8, Nf, Nt, dt, window=wilson.Meyer(8)), x,
    ...                  atol=1e-12))
    True
    """
```

Replace the docstring of `domainia.wilson.transform.base_atom_spectra` with:

```python
    """rfft of the atoms of the cells ``(m, a)``, ``a in {0, 1}`` (active only).

    Every other atom of band ``m`` is a circular shift of one of these by an
    even number of columns (:func:`atom_spectrum`).  ``dt * inverse(e)``:
    :func:`inverse` returns ``rfft(W_a^{-1} e) = rfft(atom) / dt``.
    """
```

Replace the docstring of `domainia.wilson.transform._check_base_spectrum` with:

```python
    """Reject a ``base`` that :func:`base_atom_spectra` did not build for ``(Nf, Nt, dt)``.

    An ``O(N)`` fingerprint: ``2 Nf`` entries pin ``Nf``; the length ``N//2+1`` of ``G``
    then pins ``Nt``; the energy of the atom pins ``dt``, because every active atom has
    energy ``dt`` to rounding (``W_a W_a^T = dt I``).  ``N = Nf Nt`` is even, so the last
    bin is the Nyquist bin.  The window is not pinned: the atoms of every admissible window
    have energy ``dt``.
    """
```

Replace the docstring of `domainia.wilson.transform.atom_spectrum` with:

```python
    """rfft of the atom of the active cell ``(m, n)`` (length ``N//2+1``); ``n`` must lie in
    ``0..Nt-1`` (no wrap-around).

    Pass ``base`` (:func:`base_atom_spectra` of the grid) to avoid recomputing it.  A
    ``base`` built for another ``(Nf, Nt, dt)`` is rejected; one built with another window
    is not detected.  With ``base`` given, ``window`` is unused, but it must still be a
    window (``TypeError`` otherwise).
    """
```

Replace the comment lines

```python
# ``scipy.fft`` was tried and reverted. Recorded here so it is not retried on
# the strength of a microbenchmark:
#
#   * Isolated, scipy IS faster: 1.64x single-threaded and 2.78x at
#     ``workers=8`` on a bare (255, 512) complex ifft.
#   * In this transform it is worth **0.93-1.14x** end to end, measured across
#     four grids from Nf=64 to Nf=1024. The FFT is not the bottleneck -- the
#     parity assembly (``np.where`` plus complex construction) and the scatter
#     dominate -- so Amdahl eats the gain.
#   * ``workers=8`` bought 1.00-1.23x for the same reason.
#
# The cost was real: scipy is not bit-identical to numpy (~3.7e-16 relative,
# same algorithm family, different rounding), and the Meyer outputs have stayed
# byte-for-byte stable across every change. Trading a verifiable property for
# measurement noise is a bad deal.
#
# Worth revisiting only if the parity assembly is ever vectorised further, at
# which point the FFT share of the runtime rises and the arithmetic changes.
```

with

```python
# ``scipy.fft`` is faster on a bare FFT but not end to end here (the parity assembly and
# the scatter dominate), and its rounding differs from numpy's.
```

Replace the comment lines

```python
    # Sequential slice-adds in band order, exactly the oracle's ``+=`` loop.
```

with

```python
    # Sequential slice-adds in band order.
```

Delete the comment lines

```python
        # The indices are strictly increasing, so the fancy-index add is one addition per bin.
```

Replace the comment lines

```python
    # Unit-norm edge basis functions. Not optional: see the module docstring.
```

with

```python
    # Unit-norm edge basis functions (module docstring).
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD | tail -1`

Expected: `14 files: code and doctests unchanged; 160 doctest examples`

- [ ] **Step 6: `wilson/evolutionary.py`**

#### `src/domainia/wilson/evolutionary.py`

733 → 654 lines (387 code, unchanged); 15 docstrings, 1 comment edit. The module docstring loses "NumPy" and "host", the sentence about porting the code, the wdm_dev lineage and the Numerics paragraph. The function docstrings lose the wdm_dev provenance (`PackedGrid.*`, `TriCovariance.*`, "verbatim", "line for line", "Ported verbatim"), `eq45_coeffs_bessel`'s "used to default" note, `wrap_cut_columns`' toy-model anecdote, and Parameters entries that restate the signature. `eq45_plan` keeps its Raises. The Eq. (45) banner loses "(host constants)".

Replace the module docstring with:

```python
r"""Stateless WDM Paper II kernels for evolutionary (nonstationary) noise.

Everything here is a pure function of ``(grid, spectrum, arrays)``.

* Time masks: :func:`energy_fraction`, :func:`apply_time_mask`
  (:math:`M = W\chi W^{\mathsf T}/\delta t`).
* The 0.1% wrap rule: :func:`wrap_fractions`, :func:`wrap_cut_columns`.
* The exact covariance (WDM Paper II, eq:wdm_generic_covariance, :math:`\theta = 0`):
  :func:`time_domain_factor`, :func:`time_domain_covariance`,
  :func:`exact_covariance`, :func:`draw_noise` (dense, small grids).
* WDM Paper II, Eq. (45), the nonstationary block tri-diagonal covariance
  :math:`\Lambda_{pq} = \frac1T\,\mathrm{Re}\sum_k \eta_k S(f_k, \bar t_{pq})
  \hat g_p(k)\hat g_q^*(k)`, :math:`\bar t = (t_n + t_{n'})/2` used literally:

  - :func:`eq45_plan` -- constants (one plan per grid and kept-cell set);
  - Chebyshev-in-time builders: :func:`eq45_coeffs_bessel` (exact Bessel-I
    coefficients of the log-linear family) or :func:`eq45_coeffs_interp` (any
    ``Sn(f, t)``, interpolated at ``P`` Chebyshev nodes), then
    :func:`eq45_blocks_cheb`;
  - the Gram form for the log-linear family:
    :func:`eq45_gram_plan`, :func:`eq45_gram_gather`, :func:`eq45_blocks_gram`;
  - the separable form :math:`S_c = s_I + g(t) s_G`:
    :func:`eq45_separable_kernels`, :func:`eq45_separable_assemble`;
  - :func:`eq45_direct` -- the direct sum, the oracle.

Every kernel here that reaches the window takes ``window=`` (a
:class:`domainia.wilson.windows.Window`; default :data:`domainia.wilson.MEYER`) and
computes in the Wilson basis of that window.  For Eq. (45) that is :func:`eq45_plan`:
the plan's atoms (``plan.base``) carry the window to every builder downstream.

PSDs are callables ``Sn(f, t)``: elementwise, broadcasting, argument order
``(f, t)`` (WDM Paper II writes :math:`S(t, f)`), samples through
:func:`domainia.wilson.grid.sanitize_psd`.  Cell times are :math:`t_n = n\Delta T`.
"""
```

Replace the docstring of `domainia.wilson.evolutionary.energy_fraction` with:

```python
    """``rho_p = sum_t w2(t) g_p(t)^2 / dt`` for every active cell ``p`` (band order), from the
    time-domain weight ``w2`` of shape ``(N,)``: one FFT correlation per base atom.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.apply_time_mask` with:

```python
    """``M y`` for each packed row ``y`` of ``Y``, ``M = W chi W^T / dt``: back to time,
    mask, forward again.

    ``Y`` is ``(N,)`` or ``(B, N)``: packed rows (active cells, band order).  ``chi`` is the
    time-domain mask, ``(N,)``.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.wrap_fractions` with:

```python
    r"""Fraction of each atom's energy that wraps across the record boundary.

    For the cell :math:`(m, n)`, centred on sample :math:`n N_f`, the fraction of
    :math:`\sum_j g_{mn}(t_j)^2` at samples whose unwrapped position (measured from
    the centre along the shorter way round) lies outside :math:`[0, N)`.
    Returns ``(Nf + 1, Nt)`` (or ``(len(bands), Nt)`` for the bands ``bands``), ``nan`` at
    inactive cells.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.wrap_cut_columns` with:

```python
    """``(leading, trailing)`` time columns to cut so that no kept atom of ``bands``
    (all by default) wraps more than ``tol`` of its energy (WDM Paper II: ``tol = 1e-3``).

    The rule is geometric (it ignores the noise).  Another window has other atoms, so other
    cuts.

    Examples
    --------
    >>> from domainia import wilson
    >>> wilson.evolutionary.wrap_cut_columns(64, 256, 2**-10)
    (6, 5)
    """
```

Replace the docstring of `domainia.wilson.evolutionary.exact_covariance` with:

```python
    r"""Exact :math:`\Lambda = W_a \Sigma W_a^{\mathsf T}` on the active cells (band order),
    built in Gram form :math:`(W_a G)(W_a G)^{\mathsf T}`, so PSD by construction.
    ``Sn(f, t)`` is elementwise and broadcasting.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.Eq45Plan` with:

```python
    """Constants of Eq. (45) on the kept cells of one grid.

    ``specs`` holds one :class:`Eq45BlockSpec` per band block ``(m, m2)``,
    ``m2 in {m, m+1}``.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.eq45_plan` with:

```python
    """Static plan of Eq. (45) on the cells ``keep`` (a boolean mask over the ``N`` packed
    cells, band order; all by default), Chebyshev order ``P`` in time on
    ``t_bar in [0, t_max]`` (default: this grid's ``T = N dt``).

    Pair mid-times of the kept cells must lie in ``[0, t_max]``.  A padded grid
    passes its extended ``t_max``; the spectrum's time dependence is then
    continued over the padding by the caller (``r = t_max / T_obs`` in
    :func:`eq45_coeffs_bessel`).

    ``window`` sets the atoms (``plan.base``); every Eq. (45) function downstream of the plan
    inherits it.

    Raises
    ------
    TypeError
        If ``Nf``, ``Nt`` or ``P`` is not an integer (``bool`` included), ``dt`` or
        ``t_max`` is not a real number, or ``keep`` is not a boolean mask.
    ValueError
        If the grid or ``dt`` is invalid, ``P < 1``, ``keep`` has the wrong shape, or a
        kept pair mid-time lies outside ``[0, t_max]``.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.eq45_coeffs_bessel` with:

```python
    r"""``(P, len(L))``: Chebyshev coefficients of :math:`e^{\alpha(\bar t) L}` in
    :math:`u = 2\bar t/t_{\max} - 1` for the log-linear family
    :math:`\alpha(t) = \alpha_c + \alpha_d (t/T_{\rm obs} - 1/2)`, exactly (Bessel I):
    with :math:`r = t_{\max}/T_{\rm obs}`,
    :math:`\alpha = \alpha_c + \alpha_d (r/2 - 1/2) + (\alpha_d r/2)\,u`.

    ``r`` is a required keyword: ``plan.t_max / T_obs``, i.e. ``1.0`` on the
    data grid and ``Nt' / Nt`` on a padded grid (``272/256`` for the sinusoid
    pad).  :math:`I_p(a)` is ``scipy.special.ive(p, a) * exp(|a|)``.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.eq45_coeffs_interp` with:

```python
    r"""``(P, len(f))``: Chebyshev coefficients :math:`c_p(f_k)` of a general
    :math:`S(f_k, \bar t)` in :math:`u = 2\bar t/t_{\max} - 1`, by interpolation at
    the ``P`` Chebyshev nodes; zero outside the bins ``k`` (all by default; pass
    ``plan.k_used``).
    """
```

Replace the docstring of `domainia.wilson.evolutionary.eq45_blocks_cheb` with:

```python
    """Diagonal (symmetrised) and upper band blocks of Eq. (45) from Chebyshev coefficients.

    ``coeffs`` is either a ``(P, K)`` array over every rfft bin
    (:func:`eq45_coeffs_interp`; read as ``coeffs[:, k]``) or a callable
    ``k -> (P, len(k))`` evaluated per block (for example
    ``lambda k: eq45_coeffs_bessel(L[k], ac, ad, P, r=r)``).  Each block costs
    one length-``Nt`` FFT per order.
    Returns ragged lists (``diag[m]``: kept cells of band ``m`` squared).
    """
```

Replace the docstring of `domainia.wilson.evolutionary.Eq45GramPlan` with:

```python
    """Constants of the Gram form, bands laid out ``(nb, B)`` with dummy slots.

    ``slot[m, i]`` is the kept-cell position of slot ``i`` of band ``m`` (``-1`` for
    a dummy); ``gG`` the cells' atom spectra on their band's support bins
    ``sup[m]``; ``gW = eta/T``; ``gT`` the cell times; ``uI1``/``uI2``/``uW`` the
    overlap bins of adjacent bands.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.eq45_blocks_gram` with:

```python
    """Eq. (45) blocks in the Gram form: diagonal ``(nb, B, B)`` and upper ``(nb-1, B, B)``.

    ``alpha_cells`` is ``alpha(t_p)`` per slot, ``(nb, B)``; the caller computes it
    from ``gplan.gT`` with the **data record's** ``T_obs`` (the log-linear family:
    ``ac + ad * (gT / T_obs - 0.5)``).  ``gL`` is :func:`eq45_gram_gather` of
    ``L(f)``.  Dummy slots give zero rows and columns.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.eq45_separable_kernels` with:

```python
    r""":math:`K^X = \tfrac12 \mathrm{Re}(U\,\mathrm{diag}(s)\,U^{\mathsf H})` in real arithmetic.

    ``U`` (cells x window bins) holds the unit-norm atoms; ``s`` a stationary
    spectrum on the window bins.  ``K^X`` is in the units of ``U``.
    """
```

Replace the docstring of `domainia.wilson.evolutionary.eq45_direct` with:

```python
    """Eq. (45) on the kept cells by the direct sum over rfft bins (dense; tests and
    small grids)."""
```

Replace the comment lines

```python
# WDM Paper II, Eq. (45): plans (host constants)
```

with

```python
# WDM Paper II, Eq. (45): plans
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD | tail -1`

Expected: `14 files: code and doctests unchanged; 160 doctest examples`

- [ ] **Step 7: `wilson/stationary.py`**

#### `src/domainia/wilson/stationary.py`

1051 → 896 lines (387 code, unchanged); 15 docstrings, 3 comment edits. The module docstring merges Scope and the edge-band convention into one paragraph and drops the "(was ...)" names. It keeps the backward-maps heading, the Inputs and the notes `.tex` paths. The function docstrings lose the "used to appear verbatim" history, the measured off-by-110% PSD, the `tools/benchmark.py` pointer, the table and the test names in `psd_from_K`, and the per-function window entries. `psd_from_K`'s Notes keep the cost and the DC-offset explanation. Three comments are shortened: the `eps` floor, the DC offset and the earliest candidate.

Replace the module docstring with:

```python
"""Fast stationary-noise WDM covariance via the slab-IFFT algorithm.

For stationary Gaussian noise, the frequency-domain covariance is
diagonal: $\\langle X_j X_{j'}^*\\rangle = \\sigma_j^2\\,\\delta_{jj'}$
with $\\sigma_j^2 = N\\,S_n(|f_j|)/(2\\,\\delta t)$.  In the WDM basis,
the resulting covariance $\\Lambda = W\\,\\Sigma_{\\rm time}\\,W^T$ is
block-tridiagonal in the band index $m$ — the window
$\\hat\\phi$ (Meyer by default; any admissible window) has compact frequency
support so only $|m - m'|\\le 1$ band pairs overlap.

Every kernel here that reaches the window takes ``window=`` (a
:class:`domainia.wilson.windows.Window`; default :data:`domainia.wilson.MEYER`) and
computes in the Wilson basis of that window.

This module implements the slab-IFFT algorithm from
``notes/Time_Frequency_Analysis/wdm_Lambda_stationary_fast.tex`` (and
the within-block-symmetry refinement in
``wdm_Lambda_stationary_implementation.tex``).  For each band pair
$(m, m')$ with $|m - m'|\\le 1$, the block is filled by one length-$N_t$
IFFT plus a vectorised Toeplitz expansion with the carrier
$(-1)^{(m+1)(n+n')}$ and Wilson parity factor $C^*_{mn}\\,C_{m'n'}$.

Total cost: $\\mathcal O((N_f+1)\\,N_t\\,\\log N_t + N_f\\,N_t^2)$,
versus $\\mathcal O(N^3)$ for the dense $W\\,C_{\\rm time}\\,W^T$.

:func:`build_lambda_blocks_full_active` covers **every** band pair,
including the parity-restricted edge bands $m \\in \\{0, N_f\\}$, via
half-slab Hermitian-extended IFFTs, with the edge $\\sqrt 2$ factors of the unit-norm
basis (:mod:`domainia.wilson.transform`) built in.  It matches
the dense $W_a\\,C_{\\rm time}\\,W_a^T$ at machine precision.

Backward maps (Paper I Alg. 4)
------------------------------
:func:`psd_from_K` recovers $S_n$ from the diagonal $K$ vectors;
:func:`psd_from_lambda_blocks` recovers it from stored blocks.

Inputs
------
Every input form expresses the **one-sided PSD** $S_n(f)$ — the
internal scaling $\\sigma_k^2 = N\\,S_n(|f_k|)/(2\\,\\delta t)$ is
applied uniformly. Three accepted forms (all stationary):

* a callable ``Sn_func(f) -> S_n(f)``;
* a 1D ndarray of $S_n$ values, length ``N`` (two-sided ``fftfreq``
  grid) or ``N//2+1`` (one-sided ``rfftfreq`` grid);
* a 2D diagonal ndarray of shape ``(N, N)`` or ``(N//2+1, N//2+1)``
  with $S_n$ values on the diagonal. Off-diagonal entries are checked
  against ``rtol`` and a non-diagonal input raises ``ValueError`` (the
  fast algorithm requires stationarity).
"""
```

Replace the docstring of `domainia.wilson.stationary.is_diagonal` with:

```python
    """Whether the square ``matrix`` is diagonal: every off-diagonal entry is at most
    ``rtol`` times the largest finite absolute diagonal entry (exactly 0 if that is 0).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> wilson.stationary.is_diagonal(np.diag([1.0, 2.0]))
    True
    >>> wilson.stationary.is_diagonal(np.array([[1.0, 0.5], [0.5, 2.0]]))
    False
    """
```

Replace the docstring of `domainia.wilson.stationary._mirror_to_two_sided` with:

```python
    """Expand a one-sided ``rfft``-grid PSD onto the full two-sided grid.

    Hermitian symmetry of a real process: $S(f_{N-k}) = S(f_{-k}) = S(|f_k|)$,
    so bins $1 \\ldots N/2-1$ mirror in reverse and DC and Nyquist -- which
    have no partner -- are left alone.
    """
```

Replace the docstring of `domainia.wilson.stationary._normalize_noise_to_sigma2` with:

```python
    """The per-bin variance $\\sigma_k^2 = N\\,S_n(|f_k|)/(2\\,\\delta t)$, shape ``(N,)``, on
    the two-sided ``fftfreq(N, dt)`` grid.

    Every accepted form of ``noise`` (module docstring) is the **one-sided PSD** $S_n(f)$,
    never the already-scaled covariance; the scaling is applied here, once.  ``rtol`` is the
    diagonality tolerance of a 2D input (:func:`is_diagonal`).

    Raises
    ------
    TypeError
        If the PSD is not a real numeric array in any form: complex (even with a
        zero imaginary part), bool, str or object.  It is rejected before any
        cast, rather than truncated to its real part, as
        :func:`domainia.wilson.grid.sanitize_psd` does.
    ValueError
        If a 2D matrix input is not diagonal (i.e., the noise is not
        stationary), or if the input shape is not recognized.
    """
```

Replace the docstring of `domainia.wilson.stationary._carriers` with:

```python
    """The four Wilson parity/carrier outer products, keyed by ``(m_l%2, m_r%2)``.

    ``carrier[n, n'] = a_n(m_l) * C*_{m_l,n} * a_{n'}(m_l) * C_{m_r,n'}`` where
    ``a_n(m) = (-1)^{(m+1)n}`` and ``C_{m,n} = i^{(m+n) mod 2}``.  It depends on
    the band indices only through their parities, so all ``2 N_f + 1`` blocks
    are covered by four matrices.
    """
```

Replace the docstring of `domainia.wilson.stationary._interior_slab_K` with:

```python
    """Batched interior-band slab IFFTs: ``(K_diag[1:Nf], K_off[1:Nf-1])``.

    The interior slab for band ``m`` starts at ``(m-1)*Nt/2`` and is a pure
    gather, so all ``Nf-1`` bands go through one length-``Nt`` IFFT.

    ``sigma2`` is the one-sided ``sigma2_rfft = sigma2_full / N`` grid, the
    module's single convention (``sigma2_full[:N//2+1] / N``).  Interior slabs
    index at most ``N//2 - 1``, so the one-sided array is always long enough.
    """
```

Replace the docstring of `domainia.wilson.stationary._block_from_K_with_carrier` with:

```python
    """Block expansion given a precomputed outer-product carrier matrix.

    The carrier $\\mathrm{carrier}[n, n'] = v_l[n]\\,v_r[n']$ encodes the
    band-dependent sign + Wilson parity factors.  It is built once per
    $(m_l\\%2, m_r\\%2)$ pattern (4 in all) and shared by every block of that pattern.
    """
```

Replace the docstring of `domainia.wilson.stationary._slab_K` with:

```python
    """Toeplitz-coefficient vectors $K^{(m,m)}$ and $K^{(m,m+1)}$ for every band.

    The one definition of the slab-IFFT recipe, for
    :func:`build_lambda_blocks_full_active` (which expands these into dense
    blocks) and :func:`build_K_full_active` (which returns them as-is).

    Interior bands read a full length-$N_t$ slab; the two edge bands read a
    half-slab and Hermitian-extend it, and the edge cross terms pick up the
    $\\sqrt 2$ from the unit-norm edge rescaling.  ``sigma2_rfft`` is the per-bin
    variance on the one-sided grid, already divided by $N$.

    Returns
    -------
    K_diag : list of $N_f+1$ complex arrays, each shape $(N_t,)$
    K_off : list of $N_f$ complex arrays, each shape $(N_t,)$
    """
```

Replace the docstring of `domainia.wilson.stationary.build_lambda_blocks_full_active` with:

```python
    r"""Stationary $\Lambda$ for the **full** active subspace, including edge bands.

    Returns variable-size diagonal and off-diagonal blocks, plus the
    per-band active-$n$ index lists.  Edge bands $m \in \{0, N_f\}$ are
    parity-restricted to half size $N_t/2$; interior bands are full
    $N_t$.  Off-diagonals between an edge and its interior neighbour
    have the edge dimension restricted accordingly.

    The slab-IFFT recipe per band pair $(m, m')$:

    - **Interior diag** ($1 \le m \le N_f-1$, $m'=m$):
      $V[k] = \hat\phi[k]^2\,\sigma^2_{\rm rfft}[k + (m-1)N_t/2]$,
      $K = N_t \cdot \mathrm{IFFT}(V)$.

    - **Interior offd** ($1 \le m \le N_f-2$, $m'=m+1$):
      $V[k] = \hat\phi[k]\,\hat\phi[k - N_t/2]\,\sigma^2_{\rm rfft}[k + (m-1)N_t/2]$
      for $k \in [N_t/2, N_t-1]$, else 0.

    - **Edge diag at m=0**: $V[k] = \hat\phi[k+N_t/2]^2\,\sigma^2_{\rm rfft}[k]$
      for $k \in [0, N_t/2)$, then Hermitian-extended around 0 with $V[N_t/2]=0$.

    - **Edge diag at m=N_f**: $V[k] = \hat\phi[k]^2\,\sigma^2_{\rm rfft}[k + N/2 - N_t/2]$
      for $k \in [0, N_t/2]$ (Nyquist included), then Hermitian-extended.

    - **Edge cross (0,1) and (N_f-1, N_f)**: same product
      $\sqrt 2\,\hat\phi[k]\,\hat\phi[k - N_t/2]\,\sigma^2$ on the upper half
      $k \in [N_t/2, N_t-1]$, with the appropriate slab indexing.

    All blocks share the same prefactor $1/N$ and the same
    block-from-K expansion (carrier $(-1)^{(m+1)(n+n')}$, Wilson parity
    $C^*_{m,n} C_{m',n'}$ with $C_{m,n} = i^{(m+n)\bmod 2}$).

    Parameters
    ----------
    noise : callable, 1D ndarray, or 2D ndarray
        The one-sided PSD: a callable ``f -> S_n(f)``, a 1D array on the
        ``fftfreq`` (length ``N``) or ``rfftfreq`` (length ``N//2+1``) grid, or
        a diagonal 2D array of either size.  Never pre-scaled.
    rtol : float, optional
        The diagonality tolerance of a 2D ``noise`` (:func:`is_diagonal`).

    Returns
    -------
    diag_blocks : list of ($N_f + 1$) ndarrays
        Diagonal blocks. Edges have shape ($N_t/2$, $N_t/2$); interior
        have shape ($N_t$, $N_t$).
    offd_blocks : list of $N_f$ ndarrays
        Upper off-diagonal blocks $\Lambda^{(m, m+1)}$. The first and
        last have an edge-restricted dimension; the middle $N_f - 2$
        are full ($N_t$, $N_t$).
    n_active : list of ($N_f + 1$) integer arrays
        Active-$n$ index arrays per band; edge entries have length
        $N_t/2$, interior entries are $\arange(N_t)$.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> psd = lambda f: 1e-3 * np.ones_like(f)
    >>> diag, offd, n_active = wilson.stationary.build_lambda_blocks_full_active(
    ...     Nf, Nt, dt, psd
    ... )
    >>> len(diag), len(offd)
    (9, 8)
    >>> diag[0].shape, diag[1].shape          # edge band, then interior
    ((8, 8), (16, 16))
    >>> Lam = wilson.stationary.assemble_lambda_full_active(
    ...     diag, offd, n_active, Nf, Nt
    ... )
    >>> Lam.shape
    (144, 144)
    """
```

Replace the docstring of `domainia.wilson.stationary._check_block_layout` with:

```python
    """Validate a block-tridiagonal layout against the grid it claims to be on: a list
    that is short by one would otherwise give a plausible but wrong answer.
    """
```

Replace the docstring of `domainia.wilson.stationary.build_K_full_active` with:

```python
    r"""Build the $K$ vectors that parameterise the stationary $\Lambda$.

    Each band pair $(m, m')$ with $|m - m'| \le 1$ carries a single
    length-$N_t$ complex Toeplitz-coefficient vector $K^{(m,m')}$ from
    which the dense block is reconstructible via the slab-IFFT recipe.
    Storage is $\sim 2 N_f N_t$ complex numbers, an $N_t$-fold reduction
    over storing the full $N_t \times N_t$ blocks.

    Mathematically (interior diagonal):
    $K^{(m,m)} = N_t \cdot \mathrm{ifft}(|\hat\phi_c|^2 \odot \sigma^2_{\rm rfft}[\mathrm{slab}_m])$,
    where slab$_m = [(m-1)N_t/2, (m-1)N_t/2 + N_t)$. Edge bands use the
    Hermitian-extended half-slab. Off-diagonals use the cross-band
    window product $\hat\phi_c \cdot \hat\phi_{c, \rm shift}$.  ``noise`` and ``rtol`` are
    as in :func:`build_lambda_blocks_full_active`.

    Returns
    -------
    K_diag : list of $N_f + 1$ ndarrays, each shape $(N_t,)$, complex
    K_off  : list of $N_f$ ndarrays, each shape $(N_t,)$, complex

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> K_diag, K_off = wilson.stationary.build_K_full_active(8, 16, 1.0, flat)
    >>> len(K_diag), len(K_off), K_diag[0].shape
    (9, 8, (16,))
    """
```

Replace the docstring of `domainia.wilson.stationary.psd_from_K` with:

```python
    r"""Slab-extraction backward: recover $\mathcal{S}_n$ from the
    diagonal $K$ vectors.

    For each band $m$, undo the slab-IFFT:
    $V_m = \mathrm{fft}(K^{(m,m)})/N_t = |\hat\phi_c|^2 \odot \sigma^2_{\rm rfft}[\mathrm{slab}_m]$,
    then divide by the window to extract $\sigma^2_{\rm rfft}[\mathrm{slab}_m]$.
    Stitch the slabs into a single rfft-length vector by assigning each
    bin to the band whose window value is largest there (which is
    numerically the most stable estimate).

    Edge bands $m \in \{0, N_f\}$ use the half-slab windows
    $\hat\phi_c[k + N_t/2]^2$ and $\hat\phi_c[k]^2$ respectively, matching
    the forward of :func:`build_K_full_active`.

    Parameters
    ----------
    K_diag : list of $N_f + 1$ ndarrays, each shape $(N_t,)$, complex
        As produced by :func:`build_K_full_active`, with the same ``window``.

    Returns
    -------
    ndarray, float, shape $(N/2 + 1,)$
        One-sided PSD evaluated on the ``np.fft.rfftfreq(N, dt)`` grid.

    Notes
    -----
    Cost: one length-$N_t$ FFT per band and a bin-assignment scan, $O(N \log N_t)$; it
    avoids the $O(N_f N_t^2)$ block-tridiag matvec of :func:`psd_from_lambda_blocks`.

    DC, the interior bands and Nyquist are handled apart because :func:`_slab_K` packs
    their slabs into $V$ differently.  Only DC pairs ``V[j]`` with the window bin
    ``half + j`` (the others with bin ``j``): its half-slab lies *below* DC and is packed
    from ``V[0]`` upward, while its window comes from the upper half of $\hat\phi$.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> K_diag, _ = wilson.stationary.build_K_full_active(8, 16, 1.0, flat)
    >>> wilson.stationary.psd_from_K(K_diag, 8, 16, 1.0).shape
    (65,)

    With another window, pass the same ``window=`` to both:

    >>> K8, _ = wilson.stationary.build_K_full_active(8, 16, 1.0, flat, window=wilson.Meyer(8))
    >>> S8 = wilson.stationary.psd_from_K(K8, 8, 16, 1.0, window=wilson.Meyer(8))
    >>> bool(np.allclose(S8, 1e-3, rtol=1e-12, atol=0))
    True
    """
```

Replace the docstring of `domainia.wilson.stationary.apply_lambda_from_psd` with:

```python
    r"""Apply $\Lambda$ to a WDM vector via the operator-form forward.

    Computes $\Lambda \bm x$ from the one-sided PSD via the three-step
    pipeline

    .. math::
        \Lambda \bm x \;=\;
            \mathtt{forward}\!\bigl((\mathcal{S}_n / 2)\odot
                          \mathtt{inverse}(\bm x)\bigr),

    i.e., inverse-WDM, pointwise scale each rfft bin by half the PSD,
    forward-WDM. No assembly of $\Lambda$ as a matrix is needed.  It is the operator
    dual of :func:`psd_from_lambda_blocks`.

    Parameters
    ----------
    x_wmn : ndarray, float, shape (Nf+1, Nt)
        Input WDM coefficient array.  Inactive (parity-killed) edge entries
        are ignored.
    Sn_rfft : ndarray, float, shape (N//2 + 1,)
        One-sided PSD evaluated on the ``rfftfreq(N, dt)`` grid.  ``inf`` samples are
        zero power (:func:`domainia.wilson.grid.sanitize_psd`).

    Returns
    -------
    ndarray, float, shape (Nf+1, Nt)
        $\Lambda \bm x$.

    Notes
    -----
    Cost is $O(N \log N_t)$ per call — dominated by the two fast WDM
    transforms — against $O(N_f N_t^2)$ time and storage to assemble the matrix
    (:func:`assemble_lambda_full_active`).  The result equals ``Lambda @ x.ravel()`` from
    :func:`build_lambda_blocks_full_active` plus
    :func:`assemble_lambda_full_active` to machine precision.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> N = Nf * Nt
    >>> x = wilson.forward_time(
    ...     np.random.default_rng(0).standard_normal(N), Nf, Nt, dt
    ... )
    >>> Sn = 1e-3 * np.ones(N // 2 + 1)
    >>> wilson.stationary.apply_lambda_from_psd(x, Nf, Nt, dt, Sn).shape
    (9, 16)
    """
```

Replace the docstring of `domainia.wilson.stationary.psd_from_lambda_blocks` with:

```python
    r"""Fast inverse of :func:`build_lambda_blocks_full_active`:
    recover the one-sided PSD $\mathcal{S}_n(f)$ from $\Lambda$.

    Implements the operator identity
    $\Lambda = \widetilde W\,\mathrm{diag}(\mathcal{S}_n/2)\,\widetilde W^{-1}$
    read backwards as

    .. math::
        \tfrac{1}{2}\mathcal{S}_n
          = \Re\!\bigl[\widetilde W^{-1}\bigl(\Lambda\,\widetilde W\mathbf{1}\bigr)\bigr]
          = \Re\!\bigl[\mathtt{inverse}\bigl(\Lambda\cdot\mathtt{forward}(\mathbf{1})\bigr)\bigr],

    where $\mathbf{1}$ is the all-ones rfft vector (= rfft of a time-domain
    impulse at $t=0$).  The pipeline is therefore three steps:

    1. Build the basis-only test wmn $\widetilde W\,\mathbf{1}$ via one
       :func:`domainia.wilson.transform.forward` call on the all-ones rfft.
    2. Apply $\Lambda$ to that real wmn vector (block-tridiag matvec).
    3. Apply :func:`domainia.wilson.transform.inverse` and take the real part.

    The round-trip with :func:`build_lambda_blocks_full_active` holds to
    machine precision: feeding the returned $\mathcal{S}_n$ back to the
    forward reproduces $\Lambda$.  ``diag_blocks``, ``offd_blocks`` and ``n_active`` are
    its output, with the same ``window``.

    Returns
    -------
    ndarray, float, shape (N//2 + 1,)
        One-sided PSD evaluated on the ``np.fft.rfftfreq(N, dt)`` grid.

    Notes
    -----
    Cost: one forward-WDM call (test vector) + one block-tridiag matvec
    ($O(N_f N_t^2)$) + one inverse-WDM call ($O(N \log N_t)$).

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> d, o, na = wilson.stationary.build_lambda_blocks_full_active(
    ...     8, 16, 1.0, flat
    ... )
    >>> Sn = wilson.stationary.psd_from_lambda_blocks(d, o, na, 8, 16, 1.0)
    >>> Sn.shape
    (65,)
    """
```

Replace the docstring of `domainia.wilson.stationary.assemble_lambda_full_active` with:

```python
    """Assemble the full-active blocks into a dense $(N_f+1)N_t \\times (N_f+1)N_t$ matrix.

    The blocks are :func:`build_lambda_blocks_full_active`'s.  Parity-killed edge entries
    stay exactly zero, and the lower off-diagonal is filled by transposition, so the result
    is symmetric: the dense $W\\,C_{\\rm time}\\,W^T$ at machine precision.

    Examples
    --------
    >>> import numpy as np; from domainia import wilson
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> d, o, na = wilson.stationary.build_lambda_blocks_full_active(
    ...     8, 16, 1.0, flat
    ... )
    >>> wilson.stationary.assemble_lambda_full_active(d, o, na, 8, 16).shape
    (144, 144)
    """
```

Replace the comment lines

```python
    # |phi|^2 scales with 1/dF = 2 dt Nf: an absolute 1e-30 dropped every bin once
    # 2 dt Nf fell below it (dt = 1e-33 gave an all-zero PSD).
```

with

```python
    # |phi|^2 scales with 1/dF = 2 dt Nf.
```

Replace the comment lines

```python
    # DC's window index is offset by `half` where the other two are not -- see
    # the table in this function's Notes for why, before trying to merge them.
```

with

```python
    # DC's window index is offset by `half` where the other two are not (see the Notes).
```

Replace the comment lines

```python
    # write per bin is the *earliest* candidate among equal weights -- matching
    # the strict ``w > best_w`` comparison of the original scan.
```

with

```python
    # write per bin is the *earliest* candidate among equal weights.
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD | tail -1`

Expected: `14 files: code and doctests unchanged; 160 doctest examples`

- [ ] **Step 8: Greps**

History, test and tool references, and provenance:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE --include='*.py' 'tests/|test_[a-z]+_[a-z_]+|tools/|lisa/wdm|wdm_dev|vendored|20[0-9]{2}-[0-9]{2}-[0-9]{2}|\b[Pp]orted|verbatim|bit-identical|bit for bit|bitwise|byte-|line for line|\bused to|\bwas\b|\bearlier\b|\bhost\b|domainia 3\.0 spec|benchmark|[Mm]easured[:;]' src | LC_ALL=C sort`

Expected, exactly these 2 lines (68 at Task 11's tree, Step 1):
```
src/domainia/_covariance.py:137:                raise ValueError(f"{name} is {H.shape[0]} x {H.shape[0]}, but was {K} x {K}")
src/domainia/wilson/transform.py:510:            f"base was not built for dt={dt}: its atoms have energy {energy:.17g}, "
```
Both hits are error-message code ("but was {K} x {K}", "base was not built for dt=..."), not docstrings or comments.

The prototype also searched for the other markers the rules name. It looked for `<file>.py:<n>` and `line <n>` references, "now", "2.0", "paper repo", "spec", "§", "Task <n>", "no longer", "previously", "renamed", "removed", "moved", and the measured-value words "MiB", "faster", "%" and "percent". Every remaining hit is code (`2.0 * dt`, a `spec` loop variable), the CHUNK and FFT-backend comments kept above, or the "0.1% wrap rule", a definition.

Parameters entries that restate the signature:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE --include='*.py' 'window : Window|The Wilson window \(default|^ +(Nf|Nt|dt|N|Nf, Nt|m, n|Sn_func|Sn|rtol|tol|bands|m) : (int|float|callable)' src | LC_ALL=C sort`

Expected, exactly this line (139 at Task 11's tree, Step 1):
```
src/domainia/wilson/stationary.py:393:    rtol : float, optional
```
The `rtol` entry stays because it now says what `rtol` is (Step 7).

Names of deleted code (spec §4), outside the oracle copies. The copies under `tests/wilson/oracle/` are frozen sources and still define these names, so they are excluded.

Run: `cd /Users/aaron/Documents/lisa/domainia && R='domainia\.(maps|covariance|linalg|translate|domains|operators|variance|spectra|data)\b|wilson\.(backend|special)\b|\b(inner_product|BlockTridiagonalCovariance|DenseCovariance|DiagonalCovariance|EvolutionaryCovariance|BandAveragedVariance|ChannelStack|CholFactor|BlockTriCovFactor|CellSet|GenericPSD|TranslatedOperator|TimeToWDM|WDMToTime|FourierToWDM|pcg|device_constants|bessel_i|chebyshev_basis|_working_dtypes|_set_rounded|_add_rounded|[a-z_]+_many|forward_bands|inverse_bands|band_atoms|carrier_phase|packed_cells|pack|unpack|edge_columns|band_indices|count_active|require_bool|require_str|index_array|phi_fft_order|phi_squared_normalization|whittle_variance[a-z_]*|evaluate_psd_on_band_grid|evolutionary_variance|blocktri_dense|hhat|eq54_covariance|dense_lambda_generic|PointwiseOperator|normalize_edges)\b' && grep -rnE --include='*.py' "$R" src; grep -rnE --include='*.py' --exclude-dir=oracle "$R" tests | LC_ALL=C sort`

Expected: nothing from src, then exactly these 17 lines from tests:
```
tests/test_package.py:84:    assert not {"domainia.wilson.backend", "domainia.wilson.special"} & set(info["modules"])
tests/wilson/meyer_cases.py:131:        ("transform.forward_many", lambda kw: Tr.forward(Xr, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:133:        ("transform.forward_time_many", lambda kw: Tr.forward_time(X, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:135:        ("transform.inverse_many", lambda kw: Tr.inverse(Wm, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:137:        ("transform.inverse_time_many", lambda kw: Tr.inverse_time(Wm, Nf, Nt, dt, **kw)),
tests/wilson/meyer_cases.py:175:    that recorded them are computed by the kernels they wrapped: ``EvolutionaryCovariance``
tests/wilson/meyer_cases.py:177:    ``eq45_direct``, ``BandAveragedVariance`` by ``noise.evolutionary_band_variance`` on the
tests/wilson/meyer_cases.py:195:        ("BandAveragedVariance", lambda kw: wl.noise.evolutionary_band_variance(Nf, Nt, dt, Sn, **kw)[m, n]),
tests/wilson/meyer_cases.py:196:        ("EvolutionaryCovariance.interp", interp),
tests/wilson/meyer_cases.py:203:            ("EvolutionaryCovariance.direct",
tests/wilson/meyer_cases.py:41:#: Baseline key prefixes of code that domainia 3.0 deletes: maps (54 keys), inner_product (24)
tests/wilson/meyer_cases.py:43:DROPPED = ("domainia::TimeToWDM.", "domainia::WDMToTime.", "domainia::FourierToWDM.",
tests/wilson/meyer_cases.py:44:           "wilson::inner_product.", "wilson::transform.band_atoms::",
tests/wilson/meyer_cases.py:45:           "wilson::transform.forward_bands::", "wilson::transform.inverse_bands::")
tests/wilson/test_meyer_baseline.py:26:    """maps 54, inner_product 24 and the band-limited entry points 18."""
tests/wilson/test_meyer_baseline.py:49:    assert mismatches({}, {k: w for k in ("domainia::TimeToWDM.apply::8x16::dt=1.0",
tests/wilson/test_meyer_baseline.py:50:                                          "wilson::inner_product.gaussian::8x16::dt=1.0")}) == []
```
- `test_package.py`: the assert that the no-JAX import loads neither deleted module.
- `meyer_cases.py`'s `DROPPED` tuple and its `#:` comment: the key prefixes of deleted code.
- The case names in `meyer_cases.wilson_cases` (`transform.*_many`, kept because the leading axes replaced the batched entry points) and `meyer_cases.domainia_cases` (`BandAveragedVariance`, `EvolutionaryCovariance.*`). They are recorded baseline key names, now computed by kept kernels.
- `meyer_cases.domainia_cases`' docstring and `test_meyer_baseline.py::test_the_dropped_families_are_96_of_the_1504_keys`' docstring. They explain those recorded names, and they stay because the baseline is never re-recorded.
- `test_meyer_baseline.py::test_the_comparator`: the check that a dropped key is not compared.

At Task 11's tree the same grep prints two src lines: `wilson/matrix.py`'s module docstring ("pack the", the English verb, in a sentence this task rewrites) and `wilson/transform.py`'s `normalize_edges=False` history line.

The spec's src gate still holds:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnE --include='*.py' 'jax|be=|float32' src`

Expected: no output (grep exits 1).

- [ ] **Step 9: The AST check, the replay, the suite and the size**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-ast.py HEAD`

Expected:
```
src/domainia/__init__.py: same, 0 doctest examples same
src/domainia/_covariance.py: same, 0 doctest examples same
src/domainia/_transform.py: same, 0 doctest examples same
src/domainia/testing.py: same, 0 doctest examples same
src/domainia/wilson/__init__.py: same, 0 doctest examples same
src/domainia/wilson/_checks.py: same, 20 doctest examples same
src/domainia/wilson/evolutionary.py: same, 2 doctest examples same
src/domainia/wilson/filter.py: same, 5 doctest examples same
src/domainia/wilson/grid.py: same, 32 doctest examples same
src/domainia/wilson/matrix.py: same, 27 doctest examples same
src/domainia/wilson/noise.py: same, 7 doctest examples same
src/domainia/wilson/stationary.py: same, 37 doctest examples same
src/domainia/wilson/transform.py: same, 27 doctest examples same
src/domainia/wilson/windows.py: same, 3 doctest examples same
14 files: code and doctests unchanged; 160 doctest examples
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q --collect-only src/domainia/wilson 2>&1 | tail -1`

Expected: `40 tests collected in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q tests/wilson/test_meyer_baseline.py 2>&1 | tail -1`

Expected: `5 passed in …`

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P tests/wilson/meyer_cases.py --check`

Expected: `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`

The exact check (Task 11, Step 9). Run:
```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import numpy as np; from domainia import wilson
from wilson.meyer_cases import BASELINE, DROPPED, compute
got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    kept = [k for k in z.files if not k.startswith(DROPPED)]
    bad = [k for k in kept if got[k].dtype != z[k].dtype or not np.array_equal(got[k], z[k])]
print(f'{len(bad)} of {len(kept)} arrays differ from the baseline')"
```

Expected: `0 of 1408 arrays differ from the baseline`

The prototype gives the same four results with the FLOOR interpreter.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q 2>&1 | tail -1`

Expected: the suite passes. Prototype: `951 passed, 3 skipped in …` in PY, and the same in FLOOR (the Verify command). No warnings and no collection errors (the prototype also ran both lanes with `-W error`). The counts are Task 11's, because no test item is added or removed.

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py -v`

Expected (prototype):
```
src 4059 lines / 1849 code
tests 4346 lines (+ oracle 4135)
      28      6  src/domainia/__init__.py
     141     79  src/domainia/_covariance.py
     134     88  src/domainia/_transform.py
     183     94  src/domainia/testing.py
      57     27  src/domainia/wilson/__init__.py
     179     50  src/domainia/wilson/_checks.py
     654    387  src/domainia/wilson/evolutionary.py
     100     34  src/domainia/wilson/filter.py
     330    111  src/domainia/wilson/grid.py
     344    123  src/domainia/wilson/matrix.py
     170     73  src/domainia/wilson/noise.py
     896    387  src/domainia/wilson/stationary.py
     539    225  src/domainia/wilson/transform.py
     304    165  src/domainia/wilson/windows.py
```

Per file, lines before → after (the code lines do not change):

| File | Lines (code) at Task 11 | After | Change |
|---|---|---|---|
| `src/domainia/__init__.py` | 28 (6) | 28 (6) | 0 |
| `src/domainia/_covariance.py` | 142 (79) | 141 (79) | -1 |
| `src/domainia/_transform.py` | 134 (88) | 134 (88) | 0 |
| `src/domainia/testing.py` | 219 (94) | 183 (94) | -36 |
| `src/domainia/wilson/__init__.py` | 103 (27) | 57 (27) | -46 |
| `src/domainia/wilson/_checks.py` | 213 (50) | 179 (50) | -34 |
| `src/domainia/wilson/evolutionary.py` | 733 (387) | 654 (387) | -79 |
| `src/domainia/wilson/filter.py` | 137 (34) | 100 (34) | -37 |
| `src/domainia/wilson/grid.py` | 428 (111) | 330 (111) | -98 |
| `src/domainia/wilson/matrix.py` | 426 (123) | 344 (123) | -82 |
| `src/domainia/wilson/noise.py` | 217 (73) | 170 (73) | -47 |
| `src/domainia/wilson/stationary.py` | 1051 (387) | 896 (387) | -155 |
| `src/domainia/wilson/transform.py` | 690 (225) | 539 (225) | -151 |
| `src/domainia/wilson/windows.py` | 304 (165) | 304 (165) | 0 |
| **src** | **4825 (1849)** | **4059 (1849)** | **-766** |

The trim removes 766 src lines and leaves the code lines at 1849. The spec sets no budget after the trim; before the trim the budget was at most about 5,400 lines (2,150 code). tests are unchanged at 4346 (budget at most about 5,700).

- [ ] **Step 10: Commit**

git writes in the main repo run with the sandbox disabled (dangerouslyDisableSandbox: true).

Run: `cd /Users/aaron/Documents/lisa/domainia && git add src/domainia/_covariance.py src/domainia/_transform.py src/domainia/testing.py src/domainia/wilson/__init__.py src/domainia/wilson/_checks.py src/domainia/wilson/evolutionary.py src/domainia/wilson/filter.py src/domainia/wilson/grid.py src/domainia/wilson/matrix.py src/domainia/wilson/noise.py src/domainia/wilson/stationary.py src/domainia/wilson/transform.py && git status --short | grep -vc "^M "`

Expected: `0`. Every status line is a staged modification, so nothing is unstaged or untracked. (`grep -c` exits 1 when it prints 0, so this command does not continue with `&&`.)

Run: `cd /Users/aaron/Documents/lisa/domainia && git commit -q -m "Trim the docstrings: history, test references and restated signatures go; code unchanged" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git show --stat --format= HEAD | tail -1`

Expected: `12 files changed, 215 insertions(+), 981 deletions(-)`

Size after Task 12 (prototype): src 4059 lines / 1849 code; tests 4346 (+ oracle 4135).

---

### Task 13: Final gates (spec "Size budget and gates"; §5 "Public API"; §7)

**Goal:** Run every gate of the spec on the final tree, in all four lanes (Python 3.12 to 3.14, at the dependency floors and at current versions), and record the size after every task against the budget. src and tests do not change.

Assumes the tree after Task 12, whose commit is "Trim the docstrings: history, test references and restated signatures go; code unchanged".

This task changes nothing in `src`, `tests`, `README.md` or `pyproject.toml`. If a gate fails, stop and report the failure to the coordinator. Do not fix it in this task.

There is no NumPy lane any more. It existed only while JAX code was in the package: Task 5 removed the `_HideJax` finders that read `WDM_TESTS_HIDE_JAX`, so that variable now changes nothing. Step 1 checks that nothing reads it.

**Files:**
- Create: none in the repo. Outside it, two scratch scripts: `$TMPDIR/dom3-drift.py` (Step 4) and `$TMPDIR/dom3-imports.py` (Step 6). Step 8 also makes the scratch directory `$TMPDIR/dom3-at` and removes it again.
- Modify / Delete: none.
- Test: none new. The gates are the full suite in four interpreters, `tests/wilson/test_meyer_baseline.py` (the replay), `tests/wilson/meyer_cases.py --check`, and `tests/test_transform.py::test_agrees_with_the_recorded_translate_transform`.

**Acceptance Criteria:**
- [ ] Step 1: the tree is clean, `HEAD`'s subject is Task 12's, `git log` lists the 12 task subjects of Step 1, and `grep -rnI HIDE_JAX` prints nothing.
- [ ] Step 3: the full suite passes in PY, FLOOR, cur313 and cur314, both as it is and with `-W error`. Prototype: `951 passed, 3 skipped` in each of the 8 runs. No summary line names a warning or an error. The 3 skips are `DOMAINIA_ORACLE_ROOT is not set`. In PY, the whole suite run in one process leaves `jax modules after the whole suite: []`.
- [ ] Step 4: in each lane, `$TMPDIR/dom3-drift.py` prints `0 mismatches of 1408 kept arrays at 1e-12`. `meyer_cases.py --check` prints `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`. `test_meyer_baseline.py` plus the recorded-`translate.transform` test give `6 passed`. The worst drift is reported per lane: 0 in the prototype, in all four lanes. The script's self-check reports the 2e-12 it injects.
- [ ] Step 5: `grep -rnIwE "jax|float32" src`, `grep -rnIE "be=|be\." src` and `grep -rnI "import jax\|from jax" src tests` each print nothing. There are no exceptions.
- [ ] Step 6: `$TMPDIR/dom3-imports.py` in PY prints `jax: []` and `other: []`.
- [ ] Step 7: `wilson.__all__` is spec §5's 19 names. `domainia.__all__` is spec §5's 4 names plus `__version__`. `domainia.__version__` and pyproject's version are both `3.0.0`.
- [ ] Step 8: `$TMPDIR/dom3-size.py` prints `src 4059 lines / 1849 code` and `tests 4346 lines (+ oracle 4135)`. The per-task table has the 13 rows of Step 8. The budget holds: at Task 11, the last tree before the trim, src was 4825 lines and 1849 code lines, against at most about 5,400 lines and 2,150 code lines. The tests are 4346 lines, against at most about 5,700.
- [ ] Step 9: `git diff --stat 3d4dbbc..HEAD -- src tests README.md pyproject.toml | tail -1` prints `99 files changed, 1689 insertions(+), 18900 deletions(-)`, and `git ls-files src/domainia` lists the 15 files of Step 9.
- [ ] `git status --short` prints nothing at the end, apart from a coordinator's own plan bookkeeping file under `docs/superpowers/plans/`, if it keeps one. There is no commit.

**Verify:** `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P $TMPDIR/dom3-drift.py && PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P -m pytest -q 2>&1 | tail -1` → `numpy 2.1.0: 0 mismatches of 1408 kept arrays at 1e-12; 1408 exactly equal; worst relative drift 0 (-)`, then `951 passed, 3 skipped in …`

**Steps:**

The interpreters:
- PY = `PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P`. That is Python 3.12.4, numpy 2.3.5 and scipy 1.16.3. jax 0.9.0.1 is installed there, so this lane is the one that proves nothing imports it. `JAX_PLATFORMS=cpu` is not needed now that nothing imports jax, so this task drops it.
- FLOOR = `PYTHONDONTWRITEBYTECODE=1 $TMPDIR/dom3-envs/floor312/bin/python -P`: Python 3.12.11, numpy 2.1.0, scipy 1.14.1. These are the dependency floors of `pyproject.toml`.
- cur313 and cur314 = the same prefix with `$TMPDIR/dom3-envs/cur313/bin/python` or `$TMPDIR/dom3-envs/cur314/bin/python`. They are Python 3.13 and 3.14 with current numpy and scipy, unpinned. The prototype saw numpy 2.5.4 and scipy 1.18.1 in both.

The loops below run the four lanes one after another in a single command, about 30 s per suite run. Give each such Bash call a timeout of 400000 ms or more. Do not start background runs and poll them: the sandbox blocks `sleep`.

No step writes inside `.git`. Step 8 reads old commits with `git archive`, not `git worktree add`, so it needs no sandbox override and leaves no worktree behind.

- [ ] **Step 1: The tree**

Run: `cd /Users/aaron/Documents/lisa/domainia && git branch --show-current && git status --short && git log -1 --format=%s && git log --reverse --format=%s 3d4dbbc..HEAD -- src tests`

Expected: `translation-core`, then no `git status` lines, then `HEAD`'s subject, then the 12 task subjects in order. The one allowed `git status` line is a coordinator's own plan bookkeeping file under `docs/superpowers/plans/` (such as a `.tasks.json`), if it keeps one:
```
translation-core
Trim the docstrings: history, test references and restated signatures go; code unchanged
Replay: one 1e-12 comparator; 96 keys of deleted code dropped; translate reference recorded
Delete the likelihoods, factorisations, covariance classes, operators and coordinates
Add domainia.transform: data arrays between time, frequency and WDM
Delete translate, maps, data, domains, spectra and variance; the testing oracles take a cell mask
Delete the JAX backend and special.py; the kernels call NumPy
Delete float32: the wilson kernels compute in and return float64
Fold the wilson kernels 8 -> 4; delete the band-limited entry points
Add frequency_covariance and the K x K model check and projection
Add wdm_covariance: the K x K covariance per WDM cell, chunked, with its own evaluator
Delete Eq. 39; band_averaged_variance returns its grid itself; a float32 dt computes in double
Trim the kept files and the public API to spec 3.0; testing holds the noise helpers; README; 3.0.0
Trim the docstrings: history, test references and restated signatures go; code unchanged
```
The `-- src tests` filter leaves out commits that touch only `docs/`, such as the plan's own. If execution added a fix-up commit to `src` or `tests`, it is listed here as one more subject. Step 8's table then has one more row.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnI "HIDE_JAX" src tests pyproject.toml README.md`

Expected: no output (grep exits 1). Nothing reads `WDM_TESTS_HIDE_JAX`, so there is no NumPy lane to run.

- [ ] **Step 2: The interpreters**

Run: `cd /Users/aaron/Documents/lisa/domainia && for P in ~/miniconda3/envs/wdmrel/bin/python $TMPDIR/dom3-envs/floor312/bin/python $TMPDIR/dom3-envs/cur313/bin/python $TMPDIR/dom3-envs/cur314/bin/python; do PYTHONDONTWRITEBYTECODE=1 $P -P -c "import sys, numpy, scipy, pytest; print(sys.version.split()[0], numpy.__version__, scipy.__version__, pytest.__version__)"; done`

Expected (prototype):
```
3.12.4 2.3.5 1.16.3 8.4.2
3.12.11 2.1.0 1.14.1 9.2.0
3.13.14 2.5.4 1.18.1 9.2.0
3.14.7 2.5.4 1.18.1 9.2.0
```
The second line must show numpy 2.1.0 and scipy 1.14.1 exactly. If an environment is missing, Task 0 Step 2 has the commands that create it.

- [ ] **Step 3: The full suite in four lanes**

Run: `cd /Users/aaron/Documents/lisa/domainia && for P in ~/miniconda3/envs/wdmrel/bin/python $TMPDIR/dom3-envs/floor312/bin/python $TMPDIR/dom3-envs/cur313/bin/python $TMPDIR/dom3-envs/cur314/bin/python; do PYTHONDONTWRITEBYTECODE=1 $P -P -m pytest -q 2>&1 | tail -1; done`

Expected: four summary lines with no `failed`, `error` or `warning` in them. Prototype:
```
951 passed, 3 skipped in 27.95s
951 passed, 3 skipped in 28.83s
951 passed, 3 skipped in 28.30s
951 passed, 3 skipped in 28.51s
```

The 3 skips are the oracle-root tests. pytest's `-ra` (in `addopts`) prints each one as `SKIPPED [1] tests/wilson/conftest.py:37: DOMAINIA_ORACLE_ROOT is not set: ...`. They skip in every lane, because `DOMAINIA_ORACLE_ROOT` is not set on this machine. A collection error would show in the summary line as `N errors`.

The same with warnings as errors:

Run: `cd /Users/aaron/Documents/lisa/domainia && for P in ~/miniconda3/envs/wdmrel/bin/python $TMPDIR/dom3-envs/floor312/bin/python $TMPDIR/dom3-envs/cur313/bin/python $TMPDIR/dom3-envs/cur314/bin/python; do PYTHONDONTWRITEBYTECODE=1 $P -P -m pytest -q -W error 2>&1 | tail -1; done`

Expected: the same four totals. Prototype: `951 passed, 3 skipped` in each lane, in 28 to 32 s.

jax is installed in PY. `tests/test_package.py::test_importing_every_module_resolves_to_src_and_does_not_import_jax` already imports every module in a fresh interpreter. The following checks that the whole suite does not import jax either:

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import sys, pytest
rc = pytest.main(['-q', '-W', 'error'])
print('pytest exit', int(rc), '; jax modules after the whole suite:', sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib')))
" 2>&1 | tail -2
```

Expected:
```
951 passed, 3 skipped in …
pytest exit 0 ; jax modules after the whole suite: []
```

- [ ] **Step 4: The replay gate and the recorded `transform` outputs, in four lanes**

Write `$TMPDIR/dom3-drift.py`. It recomputes every baseline case through `domainia.wilson`, using `tests/wilson/meyer_cases.py`'s own `compute`, and compares the kept keys with the replay's comparator, `mismatches`. It also prints the worst relative difference, `max |got - want| / s`, where `s` is the comparator's scale. It counts the arrays that are exactly equal, with dtype and values, so it also covers Task 12 Step 9's exact check (`0 of 1408 arrays differ from the baseline`).

```python
"""Worst replay drift for the 2026-11-12 plan.  Run from the repo root: ``python dom3-drift.py``.

Recomputes the Meyer baseline cases through ``domainia.wilson`` and prints, for the kept
keys, the mismatch count of ``meyer_cases.mismatches`` and the worst relative difference
``max |got - want| / s`` with ``s`` the comparator's scale (``max |want|``, or the family's
for an all-zero array)."""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path("src").resolve()), str(Path("tests").resolve())]
from domainia import wilson  # noqa: E402
from wilson.meyer_cases import BASELINE, DROPPED, _family, compute, mismatches  # noqa: E402

got = compute(wilson, {}, {})
with np.load(BASELINE) as z:
    want = {k: z[k] for k in z.files if not k.startswith(DROPPED)}
bad = mismatches(got, want)
scale: dict[str, float] = {}
for k, w in want.items():
    scale[_family(k)] = max(scale.get(_family(k), 0.0), float(np.max(np.abs(w), initial=0.0)))
worst, where, exact = 0.0, "-", 0
for k, w in want.items():
    if w.dtype.kind in "fc":
        s = float(np.max(np.abs(w), initial=0.0)) or scale[_family(k)]
        d = float(np.max(np.abs(got[k] - w), initial=0.0)) / s
        if d > worst:
            worst, where = d, k
    exact += got[k].dtype == w.dtype and np.array_equal(got[k], w)
print(f"numpy {np.__version__}: {len(bad)} mismatches of {len(want)} kept arrays at 1e-12; "
      f"{exact} exactly equal; worst relative drift {worst:.3g} ({where})")
```

Check that the script can see drift. The next command runs it with two arrays perturbed: one real array by a relative 3e-13, which is inside the gate, and one complex array by 2e-12, which is outside it:

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import os
s = open(os.environ['TMPDIR'] + '/dom3-drift.py').read()
old = 'got = compute(wilson, {}, {})\n'
assert s.count(old) == 1
s = s.replace(old, old + 'k0 = next(k for k, v in got.items() if v.dtype.kind == \"f\" and v.size > 1)\ngot[k0] = got[k0] * (1 + 3e-13)\nk1 = next(k for k, v in got.items() if v.dtype.kind == \"c\" and v.size > 1)\ngot[k1] = got[k1] * (1 + 2e-12)\n')
exec(compile(s, 'drift', 'exec'))
"
```

Expected: `numpy 2.3.5: 1 mismatches of 1408 kept arrays at 1e-12; 1406 exactly equal; worst relative drift 2e-12 (wilson::transform.inverse::8x16::dt=1.0)`. The 2e-12 array fails the gate and the 3e-13 array passes it. Both stop being exactly equal.

Now the gate, in each lane: the drift script, the replay as a script, then the replay test file together with the `transform` test against the outputs recorded from 2.0's `translate.transform` (Task 1's fixture `tests/data/translate_reference.npz`):

Run: `cd /Users/aaron/Documents/lisa/domainia && for P in ~/miniconda3/envs/wdmrel/bin/python $TMPDIR/dom3-envs/floor312/bin/python $TMPDIR/dom3-envs/cur313/bin/python $TMPDIR/dom3-envs/cur314/bin/python; do PYTHONDONTWRITEBYTECODE=1 $P -P $TMPDIR/dom3-drift.py; PYTHONDONTWRITEBYTECODE=1 $P -P tests/wilson/meyer_cases.py --check; PYTHONDONTWRITEBYTECODE=1 $P -P -m pytest -q tests/wilson/test_meyer_baseline.py "tests/test_transform.py::test_agrees_with_the_recorded_translate_transform" 2>&1 | tail -1; done`

Expected: in every lane, `0 mismatches of 1408 kept arrays at 1e-12`, then `baseline reproduces: 1408 arrays (96 DROPPED keys set aside)`, then `6 passed in …`. Prototype:
```
numpy 2.3.5: 0 mismatches of 1408 kept arrays at 1e-12; 1408 exactly equal; worst relative drift 0 (-)
baseline reproduces: 1408 arrays (96 DROPPED keys set aside)
6 passed in 0.91s
numpy 2.1.0: 0 mismatches of 1408 kept arrays at 1e-12; 1408 exactly equal; worst relative drift 0 (-)
baseline reproduces: 1408 arrays (96 DROPPED keys set aside)
6 passed in 1.04s
numpy 2.5.4: 0 mismatches of 1408 kept arrays at 1e-12; 1408 exactly equal; worst relative drift 0 (-)
baseline reproduces: 1408 arrays (96 DROPPED keys set aside)
6 passed in 0.94s
numpy 2.5.4: 0 mismatches of 1408 kept arrays at 1e-12; 1408 exactly equal; worst relative drift 0 (-)
baseline reproduces: 1408 arrays (96 DROPPED keys set aside)
6 passed in 0.93s
```

Worst relative drift per lane (prototype):

| Lane | Python | numpy | scipy | Mismatches at 1e-12 | Exactly equal | Worst relative drift |
|---|---|---|---|---|---|---|
| PY | 3.12.4 | 2.3.5 | 1.16.3 | 0 of 1408 | 1408 | 0 |
| FLOOR | 3.12.11 | 2.1.0 | 1.14.1 | 0 of 1408 | 1408 | 0 |
| cur313 | 3.13.14 | 2.5.4 | 1.18.1 | 0 of 1408 | 1408 | 0 |
| cur314 | 3.14.7 | 2.5.4 | 1.18.1 | 0 of 1408 | 1408 | 0 |

- On this machine every kept array, including the rewired families, is still bit-identical to the baseline recorded before the plan. So the 1e-12 headroom is not used here. The spec's largest drift across NumPy versions, 9.24e-14, was measured on NumPy 1.26.4.
- A worst drift that is nonzero but at most 1e-12 still passes the gate. Record it in the plan's execution errata (Step 10).
- `tests/wilson/data/meyer_baseline.npz` is read, never written.

- [ ] **Step 5: Greps**

Every grep in this task uses `-I`, which skips binary files. The first three below check the spec's gate, no `jax`, `be=` or `float32` in src, with `-I` added. Without it, a plain `/usr/bin/grep` prints `Binary file ... matches` for:
- `tests/wilson/data/meyer_baseline.npz`, whose compressed bytes happen to match the case-insensitive sweep `-iE "jax"` below;
- the untracked `__pycache__/*.pyc` files of deleted modules that the main repo still holds, for example `tests/__pycache__/test_dom_linalg_jax.cpython-312-pytest-9.2.0.pyc`. They are gitignored and never imported.

The Bash tool's `grep` is a shell function that already adds `-I` and skips gitignored files. With `-I` the results are the same under either grep.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnIwE "jax|float32" src`

Expected: no output (grep exits 1).

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnIE "be=|be\." src`

Expected: no output (grep exits 1). There are no exceptions. `be\.` would also match the English word "be" at the end of a sentence, as in "must be.". A future hit of that kind is prose, not a backend call, and would be listed as an exception with its line.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnI "import jax\|from jax" src tests`

Expected: no output (grep exits 1). This includes the oracle copies under `tests/wilson/oracle/`.

Wider sweeps, to back these up:

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnIiE "jax|float32|complex64|be=" src README.md pyproject.toml`

Expected: no output (grep exits 1). This is case-insensitive, so `JAX` in prose would show up too.

Run: `cd /Users/aaron/Documents/lisa/domainia && grep -rnIiE "jax" tests --exclude-dir=oracle | LC_ALL=C sort`

Expected, exactly these 6 lines, all in `tests/test_package.py`. They are its module docstring and the no-JAX test: its name, its docstring, and the check itself, which has to name the modules it looks for:
```
tests/test_package.py:1:"""domainia 3.0 identity: version, public names, no JAX on import, no release leftovers."""
tests/test_package.py:66:def test_importing_every_module_resolves_to_src_and_does_not_import_jax():
tests/test_package.py:68:    interpreter, leaves JAX unimported.  The JAX backend and its Bessel shim are gone."""
tests/test_package.py:76:            "jax = sorted(k for k in sys.modules if k.split('.')[0] in ('jax', 'jaxlib'))\n"
tests/test_package.py:77:            "print(json.dumps({'file': domainia.__file__, 'modules': mods, 'jax': jax}))")
tests/test_package.py:85:    assert info["jax"] == []
```

- [ ] **Step 6: What `import domainia` imports (fresh interpreter, PY)**

Write `$TMPDIR/dom3-imports.py`. It imports domainia in a fresh interpreter and sorts every module that the import adds to `sys.modules` by the file the module was loaded from. Sorting by file, not by top-level name, matters here. scipy's compiled extensions register top-level names such as `_csparsetools` and `_cyutility`, and Cython registers `_cython_3_1_6` and `cython_runtime`. A check on the first dotted component would report all of these as foreign modules.

```python
"""Import check for the 2026-11-12 plan.  Run from the repo root: ``python dom3-imports.py``.

Imports domainia in this fresh interpreter and sorts every module it adds to sys.modules by
the file it was loaded from: domainia's src, numpy, scipy, the stdlib, or no file (built-in
and Cython runtime modules).  Prints the counts, the no-file names, any jax module and any
module from anywhere else."""
import collections
import os
import sys
import sysconfig

before = set(sys.modules)
sys.path.insert(0, os.path.abspath("src"))
import domainia  # noqa: E402

std, site = sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"]
src = os.path.dirname(domainia.__file__)
kinds, nofile, other = collections.Counter(), [], []
for name in sorted(set(sys.modules) - before):
    f = getattr(sys.modules[name], "__file__", None)
    if f is None:
        nofile.append(name)
    elif f.startswith(src + os.sep):
        kinds["domainia"] += 1
    elif f.startswith(os.path.join(site, "numpy") + os.sep):
        kinds["numpy"] += 1
    elif f.startswith(os.path.join(site, "scipy") + os.sep):
        kinds["scipy"] += 1
    elif f.startswith(std + os.sep) and not f.startswith(site + os.sep):
        kinds["stdlib"] += 1
    else:
        other.append((name, f))
print(f"domainia from {src}")
print(dict(sorted(kinds.items())), f"no file: {len(nofile)}")
print("no file:", nofile)
print("jax:", sorted(m for m in sys.modules if m.split(".")[0] in ("jax", "jaxlib")))
print("other:", other)
```

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-imports.py`

Expected (prototype):
```
domainia from /Users/aaron/Documents/lisa/domainia/src/domainia
{'domainia': 13, 'numpy': 147, 'scipy': 142, 'stdlib': 145} no file: 14
no file: ['_ast', '_cython_3_1_6', '_cython_3_2_1', '_locale', '_sre', '_string', '_tokenize', '_typing', 'atexit', 'cython_runtime', 'errno', 'gc', 'typing.io', 'typing.re']
jax: []
other: []
```
- The gate is the last two lines: `jax: []` and `other: []`.
- The counts depend on the numpy and scipy versions. In the prototype the other three lanes also print `jax: []` and `other: []`.
- `domainia: 13` is the 14 src modules minus `domainia.testing`, which the package docstring says is not imported.
- The no-file names are interpreter built-ins plus the Cython runtime of scipy's compiled code.

- [ ] **Step 7: The public API and the version**

```bash
cd /Users/aaron/Documents/lisa/domainia && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P -c "
import tomllib, domainia
from domainia import wilson
print(sorted(domainia.__all__))
print(sorted(wilson.__all__))
v = tomllib.load(open('pyproject.toml', 'rb'))['project']['version']
print(domainia.__version__, v, domainia.__version__ == '3.0.0' == v)
print(sorted(set(dir(domainia)) - set(domainia.__all__) - {n for n in dir(domainia) if n.startswith('__')}))
print(sorted(set(n for n in dir(wilson) if not n.startswith('_')) - set(wilson.__all__)))
"
```

Expected:
```
['__version__', 'frequency_covariance', 'transform', 'wdm_covariance', 'wilson']
['Grid', 'MEYER', 'Meyer', 'RampWindow', 'SampledWindow', 'Window', 'band_averaged_variance', 'evolutionary', 'evolutionary_band_variance', 'filter', 'forward', 'forward_time', 'grid', 'inverse', 'inverse_time', 'matrix', 'noise', 'stationary', 'windows']
3.0.0 3.0.0 True
['_covariance', '_transform', 'annotations']
['annotations', 'transform']
```

Against spec §5:

| Namespace | Spec §5 | `__all__` | Difference |
|---|---|---|---|
| `domainia` | `transform`, `frequency_covariance`, `wdm_covariance`, `wilson` | the same four, plus `__version__` | `__version__`, kept by Task 11 and pinned by `tests/test_package.py`'s `PUBLIC` |
| `domainia.wilson` | `Grid`, `forward`, `inverse`, `forward_time`, `inverse_time`; `Window`, `Meyer`, `MEYER`, `RampWindow`, `SampledWindow`; `band_averaged_variance`, `evolutionary_band_variance`; the submodules `evolutionary`, `stationary`, `matrix`, `noise`, `grid`, `filter`, `windows` | the same 19 names | none; `tests/test_package.py`'s `WILSON_PUBLIC` pins this set |

The last two lines list names that are in the namespace but not in `__all__`:
- `_covariance`, `_transform` and `wilson.transform` are submodules, bound because the package imports from them.
- `annotations` is the binding that `from __future__ import annotations` leaves.

None of them is public API. Spec §5's submodule list does not include `transform`; its four kernels are exported by name.

- [ ] **Step 8: Size, per task and against the budget**

Run: `cd /Users/aaron/Documents/lisa/domainia && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py -v`

Expected (prototype):
```
src 4059 lines / 1849 code
tests 4346 lines (+ oracle 4135)
      28      6  src/domainia/__init__.py
     141     79  src/domainia/_covariance.py
     134     88  src/domainia/_transform.py
     183     94  src/domainia/testing.py
      57     27  src/domainia/wilson/__init__.py
     179     50  src/domainia/wilson/_checks.py
     654    387  src/domainia/wilson/evolutionary.py
     100     34  src/domainia/wilson/filter.py
     330    111  src/domainia/wilson/grid.py
     344    123  src/domainia/wilson/matrix.py
     170     73  src/domainia/wilson/noise.py
     896    387  src/domainia/wilson/stationary.py
     539    225  src/domainia/wilson/transform.py
     304    165  src/domainia/wilson/windows.py
```

The size at each task commit. `git archive` extracts each commit's `src` and `tests` into a scratch directory, and the size script runs there. This only reads `.git`. The rows are keyed by commit subject; the hashes are not compared:

Run: `cd /Users/aaron/Documents/lisa/domainia && for r in 3d4dbbc $(git rev-list --reverse 3d4dbbc..HEAD -- src tests); do rm -rf $TMPDIR/dom3-at && mkdir -p $TMPDIR/dom3-at && git archive $r src tests | tar -x -C $TMPDIR/dom3-at && echo "$(git log -1 --format='%<(44,trunc)%s' $r) | $(cd $TMPDIR/dom3-at && PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/wdmrel/bin/python -P $TMPDIR/dom3-size.py | paste -sd ';' -)"; done; rm -rf $TMPDIR/dom3-at`

Expected (prototype), 13 rows: the spec commit, which holds 9f53225's code, then the 12 task commits:
```
Spec: domainia 3.0 translation core for En.. | src 12252 lines / 5669 code;tests 13249 lines (+ oracle 4135)
Replay: one 1e-12 comparator; 96 keys of d.. | src 12252 lines / 5669 code;tests 13164 lines (+ oracle 4135)
Delete the likelihoods, factorisations, co.. | src 8592 lines / 3850 code;tests 8677 lines (+ oracle 4135)
Add domainia.transform: data arrays betwee.. | src 8731 lines / 3941 code;tests 8806 lines (+ oracle 4135)
Delete translate, maps, data, domains, spe.. | src 6560 lines / 2605 code;tests 5769 lines (+ oracle 4135)
Delete the JAX backend and special.py; the.. | src 5796 lines / 2174 code;tests 4760 lines (+ oracle 4135)
Delete float32: the wilson kernels compute.. | src 5624 lines / 2125 code;tests 4528 lines (+ oracle 4135)
Fold the wilson kernels 8 -> 4; delete the.. | src 5219 lines / 1956 code;tests 4337 lines (+ oracle 4135)
Add frequency_covariance and the K x K mod.. | src 5310 lines / 2005 code;tests 4486 lines (+ oracle 4135)
Add wdm_covariance: the K x K covariance p.. | src 5367 lines / 2037 code;tests 4596 lines (+ oracle 4135)
Delete Eq. 39; band_averaged_variance retu.. | src 5188 lines / 1983 code;tests 4490 lines (+ oracle 4135)
Trim the kept files and the public API to .. | src 4825 lines / 1849 code;tests 4346 lines (+ oracle 4135)
Trim the docstrings: history, test referen.. | src 4059 lines / 1849 code;tests 4346 lines (+ oracle 4135)
```

The same as a table, with the change from the row above:

| After task | src lines | src code | tests (non-oracle) | Change: src / code / tests |
|---|---|---|---|---|
| 0 (9f53225's code) | 12252 | 5669 | 13249 | — |
| 1 Replay gate | 12252 | 5669 | 13164 | 0 / 0 / −85 |
| 2 Likelihoods, factorisations, covariance classes, operators, coordinates | 8592 | 3850 | 8677 | −3660 / −1819 / −4487 |
| 3 `transform` | 8731 | 3941 | 8806 | +139 / +91 / +129 |
| 4 translate, maps, data, domains, spectra, variance | 6560 | 2605 | 5769 | −2171 / −1336 / −3037 |
| 5 JAX backend, `special.py` | 5796 | 2174 | 4760 | −764 / −431 / −1009 |
| 6 float32 | 5624 | 2125 | 4528 | −172 / −49 / −232 |
| 7 Fold the kernels 8 → 4 | 5219 | 1956 | 4337 | −405 / −169 / −191 |
| 8 `frequency_covariance` | 5310 | 2005 | 4486 | +91 / +49 / +149 |
| 9 `wdm_covariance` | 5367 | 2037 | 4596 | +57 / +32 / +110 |
| 10 Eq. 39 | 5188 | 1983 | 4490 | −179 / −54 / −106 |
| 11 Dead code, API, README, 3.0.0 | 4825 | 1849 | 4346 | −363 / −134 / −144 |
| 12 Docstring trim | 4059 | 1849 | 4346 | −766 / 0 / 0 |

The oracle copies are 4135 lines at every commit.

Tasks 3 and 4 are 2 src lines, 2 code lines and 1 test line above the figures their sections first reported (8729 / 3939 / 8805 and 6558 / 2603 / 5768). The difference is the masked-array refusal that the coordinator added to Task 3 when it rebased: two lines in `_transform._checked` (`if isinstance(x, np.ma.MaskedArray):` and its `raise TypeError`) and one case line in `test_transform.py::test_every_refusal`. The table above was measured from the commits. task-3.md and task-4.md already carry these figures.

**The budget holds.**
- src, before the docstring trim (Task 11): 4825 lines and 1849 code lines, against at most about 5,400 lines and 2,150 code lines. That is 575 lines and 301 code lines under. The spec expected about 5,300–5,450 lines (2,050–2,150 code) at this point.
- After the trim (Task 12), src is 4059 lines and 1849 code lines. The spec sets no budget after the trim.
- tests: 4346 non-oracle lines, against at most about 5,700. That is 1,354 lines under; the spec expected about 5,550–5,700.
- During the plan: src was over the budget at the start and through Task 6, while the cut was under way. Task 7 is the first commit inside it (5219 / 1956). The largest size after that is Task 9's, 5367 / 2037, which is 33 lines and 113 code lines under. tests are inside their budget from Task 5 on.
- Against the start (9f53225's code): src −8193 lines (−67%) and −3820 code lines (−67%); tests −8903 lines (−67%).
- The largest src files, as lines / code: `wilson/stationary.py` 896 / 387, `wilson/evolutionary.py` 654 / 387, `wilson/transform.py` 539 / 225, `wilson/matrix.py` 344 / 123, `wilson/grid.py` 330 / 111 and `wilson/windows.py` 304 / 165. Spec decision 11 offers cuts in four of them, none made: `stationary.py` whole, the Gram form in `evolutionary.py`, `RampWindow` and `SampledWindow` in `windows.py`, and `matrix.py`'s test-only helpers.

- [ ] **Step 9: The whole change, from the spec commit**

Run: `cd /Users/aaron/Documents/lisa/domainia && git diff --stat 3d4dbbc..HEAD -- src tests README.md pyproject.toml | tail -1 && git diff --stat 3d4dbbc..HEAD | tail -1 && git diff --name-status 3d4dbbc..HEAD -- src tests README.md pyproject.toml | cut -f1 | sort | uniq -c`

Expected (prototype):
```
 99 files changed, 1689 insertions(+), 18900 deletions(-)
 99 files changed, 1689 insertions(+), 18900 deletions(-)
   5 A
  61 D
  33 M
```
- The first line is the gate. It covers everything the plan changes in the package.
- The second, unscoped line also counts the plan's own files under `docs/superpowers/plans/`. In the prototype clone they are not committed, so the two lines are equal there. In the main repo the second line is larger by those files only.
- By path:
  - src: 35 files, +795 −8988;
  - tests: 62 files, +829 −9732;
  - `README.md` and `pyproject.toml`: 2 files, +65 −180.
- The 5 added files: `src/domainia/_covariance.py`, `src/domainia/_transform.py`, `tests/data/translate_reference.npz`, `tests/test_covariance.py` and `tests/test_transform.py`.

Run: `cd /Users/aaron/Documents/lisa/domainia && git ls-files src/domainia`

Expected, the 15 files of the end state:
```
src/domainia/__init__.py
src/domainia/_covariance.py
src/domainia/_transform.py
src/domainia/py.typed
src/domainia/testing.py
src/domainia/wilson/__init__.py
src/domainia/wilson/_checks.py
src/domainia/wilson/evolutionary.py
src/domainia/wilson/filter.py
src/domainia/wilson/grid.py
src/domainia/wilson/matrix.py
src/domainia/wilson/noise.py
src/domainia/wilson/stationary.py
src/domainia/wilson/transform.py
src/domainia/wilson/windows.py
```
Deleted from src since the spec commit (21 files): `covariance/` (`__init__`, `base`, `blocktri`, `dense`, `diagonal`, `evolutionary`, `stack`), `data.py`, `domains.py`, `linalg.py`, `maps/` (`__init__`, `base`, `fourier`, `time`), `operators.py`, `spectra.py`, `translate.py`, `variance.py`, `wilson/backend.py`, `wilson/inner_product.py` and `wilson/special.py`. This matches the plan's "File structure (the end state)".

Run: `cd /Users/aaron/Documents/lisa/domainia && find src/domainia -type d -not -name __pycache__ | LC_ALL=C sort`

Expected, exactly:
```
src/domainia
src/domainia/wilson
```
This lists the directories on disk, untracked ones included. Suppose a deleted package's directory were left holding only an untracked `__pycache__`, such as `src/domainia/maps/`. git would not show it, but `import domainia.maps` would still succeed, as a namespace package. Tasks 2 and 4 remove `covariance/` and `maps/` with `rm -rf` after `git rm`. If either directory is listed here, report it to the coordinator.

Run: `cd /Users/aaron/Documents/lisa/domainia && git status --short`

Expected: no output, apart from the coordinator's bookkeeping file allowed in Step 1. The scratch scripts live in `$TMPDIR`, and Step 8 removed `$TMPDIR/dom3-at`.

- [ ] **Step 10: No commit**

This task changes nothing in the repo, so it has no commit.

If the coordinator keeps an "Execution errata" section at the end of the plan file under `docs/superpowers/plans/`, the coordinator writes this task's results there and commits that one path by name. The results are the four lanes' summary lines, the worst drift per lane and the size table above. That commit belongs to the coordinator, not to this task. The prototype made no commit.

Size after Task 13 (prototype): src 4059 lines / 1849 code; tests 4346 (+ oracle 4135).

---

## Execution errata

Executed on 2026-11-13, subagent-driven, in the main checkout on `translation-core`. Nothing is pushed.

### How it ran

* Task 0 and Task 13 make no commit; they were verification runs, and every Expected held.
* Tasks 1–12 each went to one implementer subagent, chosen by the task's model tier: Sonnet for standard tasks, Opus for Tasks 3, 8 and 9. One reviewer then checked each task for spec compliance and code quality.
* Each task commit's tree was also compared byte for byte with the prototype clone's tree after the same task, with `docs/superpowers` excluded and `.npz` files compared by content. All twelve were identical, so every task reproduced the prototype exactly.
* Every review approved. Minor findings in code the plan dictates were deferred to the fix-ups below, because later tasks quote that code verbatim.
* Commits that implementer subagents made end with the Co-Authored-By line of the model that wrote them (Claude Sonnet 5.5 for the standard-tier tasks), not the Conventions' "Claude Opus 5.5" line. The coordinator's own commits keep the Opus line.
* Each task's completion was recorded in the tasks file by its own docs-only commit, "Plan: Task N completed".

| Task | Commit |
|---|---|
| 1 | b3e78f1 |
| 2 | a4023f6 |
| 3 | f58d1e6 |
| 4 | a4c1275 |
| 5 | b9dd185 |
| 6 | 8a9fd3c |
| 7 | 1c48285 |
| 8 | 4b27bdb |
| 9 | a4cde5c |
| 10 | 6d55cc6 |
| 11 | c513c3e |
| 12 | 63f0f06 |
| fix-ups | 526ebdc, 9544b38 |

### Plan text corrections

* **Task 2, Step 11.** The two `log_coordinate_scale` tests are not "the last two functions in the file"; they sit mid-file in `tests/test_translate.py`. They were deleted whole.
* **Task 3, Step 8.** The Expected stat is `5 files changed, 275 insertions(+), 5 deletions(-)`, not 272. The task's own texts add up to 275 (137 + 129 + 4 + 4 + 1).
* **Task 4, Step 12, Edit 4 (`pyproject.toml`).** The Old text is a substring that ends mid-line at "on sys.path.", and the rest of that line stays.
  * The first implementer read it as whole lines and stopped with nothing committed. A second finished the task from that edit.
  * From Task 5 on, the implementer prompt said that Old texts are exact substrings.

### Final review and fix-ups

After Task 13, an Opus review of the whole branch read the end state through four lenses: spec conformance, the new functions' correctness, the kept code and docs, and the tests. A separate skeptic tried to refute each Critical or Important finding.

* **No Critical findings.** Probes found the three public functions correct, fp64 throughout, exactly Hermitian and linear, and 0 at the inactive cells.
* **One verified Important finding.** `test_every_refusal` says a WDM model "is checked in every chunk", but it reached only chunk (0, 0).
* **Rejected as a defect:** `test_dom_testing`'s columns against §3.3. That is this plan's recorded decision, so it stays an open point.
* **Fix-up commit 526ebdc** (contracts, test first):
  * `wdm_covariance` hands the model read-only `f` and `t`. A model that wrote into them silently corrupted later chunks.
  * A masked model output is refused with `TypeError`.
  * `_hermitian` builds Sᴴ once.
  * `transform` checks `window` on every route, and refuses an `(Nf, Nt)` argument that is not a pair, naming the argument.
  * The Raises sections now say: a non-real `dt` is a `TypeError`, and an inadmissible window is a `WindowError` (a `ValueError`). The CHUNK bound now reads max(CHUNK, J).
  * New tests pin the 1e-12 and 1e-10 tolerances, a later chunk, the stationary chunk count, `time_domain_factor`'s `dt` check, and the replay comparator's own-peak scaling.
  * Each new test kills a mutant that the old tests let through.
* **Fix-up commit 9544b38** (cuts and stale text):
  * deleted `Eq45Plan.fold_has_duplicates`, the fixtures `lisa_grid` and `chirp_signal`, `flatten`'s `None` branch, and the oracle-only `test_wrap_fractions_match_the_atoms_directly`;
  * fixed the stale text in `eq45_coeffs_bessel`, `grid.py`'s summary and the vendored test file's header;
  * scoped two README sentences to the top-level functions.
* **Final state, measured after the fix-ups:**
  * 950 passed and 3 skipped under `-W error` in all four lanes (PY, floor312, cur313, cur314). The one test fewer is the oracle-only test that was cut.
  * The replay is bit-exact in all four lanes: 0 of 1408 arrays differ.
  * The public API and the version are unchanged, and `import domainia` loads no jax.
  * Size is src 4077 lines / 1859 code and tests 4335 (+ oracle 4135). The fix-ups added a net 7 lines.

### Open points left for the user

None of these was changed, because each is a semantics or scope choice:

1. **A constant model on the WDM route.** `frequency_covariance` accepts a `(K, K, 1)` output, but `wdm_covariance(lambda f, t: S0(f), …)` refuses it: `_hermitian` requires 2 + len(points) axes, while spec §3.1 says the output must be "broadcastable". Left-padding the point axes in `_hermitian` would be a one-line change.
2. **Spec §6.1's oracle comparisons.** §6.1 says to move the exact comparisons against oracle copies (`test_evolutionary_kernels.py`, `test_noise_stationary.py`) to the tolerance helper. They are not moved. They are bit-exact today on NumPy 2.1–2.5, and on 1.26.4 too.
3. **§3.3's columns.** `test_dom_testing`'s columns against §3.3's `wrap_cut_columns` sentence is the header's recorded decision. The spec text may want the edit.
4. **NumPy-integer grid sizes.** `validate_grid` admits a NumPy integer `Nf`/`Nt`, and arithmetic in its dtype can overflow (`np.int8`, `np.int16`). Converting to `int` in `validate_grid` would fix this package-wide.
5. **Argument kinds in `forward`/`forward_time`.** `wilson.forward`/`forward_time` compute on a bool array, and refuse complex or str input with NumPy's own message. `inverse` checks its kind. This predates the branch.
6. **The duplicated Eq. 36 sampling.** `wdm_covariance` repeats `evolutionary_band_variance`'s Eq. 36 sampling, as the spec specifies. A shared private helper is a candidate for the next cut.
7. **Minor review notes not acted on:**
   * `_block_spec`'s unused `Nf`;
   * `transform` reports a non-finite entry by flat index;
   * `noise.py`'s two elementwise refusals are untested;
   * `test_window_threading`'s `CONSUMERS` lacks `transform`;
   * the vendored `test_evolutionary.py` module docstring still names `wdm.evolutionary`;
   * `band_averaged_variance` no longer maps an overflowed `inf` mean to 0 (Task 10 accepts this).

The Known open points above about the historical comment in the vendored test file and the stale `maps/` and `covariance/` directories are resolved: 9544b38 rewrote the comment, and the directories are gone.
