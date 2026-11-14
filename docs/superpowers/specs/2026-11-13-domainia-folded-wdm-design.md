# domainia: the folded WDM layout

**Status:** design approved section by section on 2026-11-13. This written spec awaits the
user's review.

**Branch:** `folded-wdm`, stacked on `translation-core` at 8931b99. Its PR targets
`translation-core`. PR #1 (`translation-core` → `main`) stays as reviewed, and both merge
before any 3.0.0 tag.

**Version:** 3.0.0 is unreleased, so the version stays 3.0.0. The layout change lands before
the tag, not as a break after it.

## User decisions

All on 2026-11-13, in order:

1. "We should be using the packed representation though." This came after the
   per-point covariance discussion (domainia keeps the covariance, and the likelihood takes
   the inverse).
2. On which packed form: "Both should be supported. Why don't we use these?"
3. On scope: "Kernels too", not only the public API.
4. On the flat order: "Rectangle, raveled". So flat is `rect.reshape(..., N)`, and there is one
   cell order for vectors, masks and N×N matrices.
5. On the API: "Rect in and out". Every entry point takes and returns `(..., Nf, Nt)`, and
   there is no layout keyword.
6. On padded arrays: "Public fold/unfold".
7. On the approach: "Fold at the boundary".
8. Sections 1–5 of the design: "yes, looks right" / "yes, next section".
9. On delivery: "Stacked branch".

## Why domainia used the padded layout

This records the background to decision 2's question.
- The 3.0 spec chose `(Nf+1, Nt)` because Enchilada uses it. The mapping between the two
  packages was then the identity (3.0 spec, "Enchilada's layouts match domainia's").
- It also follows the paper's `w_mn` with m = 0..Nf.
- 2.0 had a flat packed form (`wilson.pack`/`unpack`, band order) for its maps and covariance
  classes. 3.0 deleted those classes, and with them the helpers.

The padded layout costs only Nt cells, but every consumer has to know about them:
- likelihoods need masks;
- `transform` checks the inactive cells' rounding;
- `wdm_covariance` zeroes them;
- the test oracles convert masks.

## 1. The folded layout

Every WDM array in domainia has trailing shape `(Nf, Nt)`: exactly N = Nf·Nt reals per
channel, with any leading axes. Slot `(i, j)` holds the padded cell `(m, n)`:

| Slot | Cell (m, n) | Meaning |
|---|---|---|
| i ≥ 1 | (i, j) | interior band i at time j, unchanged |
| i = 0, j even | (0, j) | band 0 (DC) at time j |
| i = 0, j odd | (Nf, j − 1) | band Nf (Nyquist) at time j − 1 |

- Nf is even, so both edge bands are active at the same (even) times. The fold is therefore
  a shift of band Nf into band 0's odd slots, with no parity interleave.
- **The flat form is `w.reshape(..., N)`**, so slot (i, j) has flat index `i·Nt + j`. N×N
  matrices index their rows and columns in this order, and cell masks are boolean
  `(Nf, Nt)`.
- **Compatibility with other packages:**
  - **The user's local `wdm` package** (unreleased, `packed_wdm_indices`, `data.py:88-104`):
    the same slots and the same values, because it uses the same normalisation (atoms of
    unit norm, Σw² = dt·Σx², edge bands ÷√2). Its `build_W` and folded `Covariance` use
    the same flat order.
  - **pywavelet** `(Nf, Nt)` and Digman's `WDMWaveletTransforms` `(Nt, Nf)`: the same slots,
    but different values. They differ by √dt (√(dt/2) for pywavelet) and by a sign on the
    interior cells where m+n is odd. domainia does not try to match those values.
- **Nothing in the folded layout is structurally zero.** Every inactive-cell rule goes: the
  residue checks, the zeroing, the NaN fills and the mask refusals.
- **Band structure in this order:** row 0 couples to row 1 through its DC slots, and to row
  Nf−1 through its Nyquist slots. The stationary covariance in flat order is therefore
  block-tridiagonal plus a (0, Nf−1) corner pair. The user's `wdm` has the same structure
  ("the `(0, Nf - 1)` wrap pair").
- **Known blind spot:** on a square grid (Nf = Nt), a transposed array passes every shape
  check. No check can see this, the same as today (`grid.py` already notes that an Nf/Nt
  swap is invisible).

## 2. `wilson.grid`: `cells`, `fold`, `unfold`

The module gains three public names, and `active_mask` leaves `wilson.grid.__all__` and
becomes private (`_active_mask`).

- **`cells(Nf, Nt) -> (m, n)`.**
  - It returns two integer arrays of shape `(Nf, Nt)`: the band and time of each folded slot,
    as in the table in §1.
  - It is the one index that `fold`, `unfold` and every kernel boundary use.
  - Users take each cell's frequency and time from it: `band_frequencies(Nf, dt)[m]` and
    `time_pixel_centers(Nf, Nt, dt)[n]`.
- **`fold(P, Nf, Nt)`** maps padded `(..., Nf+1, Nt)` to folded `(..., Nf, Nt)`, as
  `P[..., m, n]`. It keeps the dtype, so a boolean padded mask folds too.
  - It refuses (`ValueError`, naming the first bad cell and its leading index) an inactive
    cell whose magnitude exceeds `RESIDUE = 1e-10` of the slice's peak (`max |P|` over the
    last two axes), or that is non-finite. For a boolean array, the refusal is any `True`.
  - This is the rule `transform` applies to WDM input today. It means a padded array whose
    inactive cells hold data, such as a variance grid filled with S/2, a mask over inactive
    cells, or the wrong array, is refused rather than silently truncated.
  - `TypeError` for a masked or non-numeric (and non-bool) array; `ValueError` for a wrong
    trailing shape or a bad grid.
- **`unfold(R, Nf, Nt)`** maps folded `(..., Nf, Nt)` to padded `(..., Nf+1, Nt)`, with R's
  dtype and the inactive cells exactly 0 (`False` for a boolean array). It raises the same
  kinds for a wrong kind or shape.
- **`RESIDUE`** moves from `_transform.py` into `wilson.grid`, as one private module
  constant. `fold` uses it, and `transform` imports it for its frequency check.

## 3. The kernels (`domainia.wilson`)

The approach is to fold at the boundary.
- **The kernels keep computing on a private padded grid**, where row = band and column = time
  hold everywhere.
- **Each public function folds its WDM output and unfolds its WDM input**, through `cells`.
  The values are unchanged; only the slots move.
- **Each transform kernel keeps a private padded twin** (`_forward`, `_inverse`,
  `_forward_time`, `_inverse_time`). Internal callers use the twins, so no padded ↔ folded
  round trip happens inside the package. Those internal callers are:
  - `wrap_fractions`, which probes with a padded unit vector;
  - `base_atom_spectra`;
  - `apply_lambda_from_psd`;
  - `testing.empirical_variance`;
  - the matrix builders.
- **A padded array passed to a public kernel is refused** (`ValueError`), and the message
  names `wilson.grid.fold`. A flat `(..., N)` array is refused with a message that names
  `.reshape(..., Nf, Nt)`.

### `transform.py`

- **Changed:**
  - `forward` and `forward_time` return `(..., Nf, Nt)`.
  - `inverse` and `inverse_time` take it. `inverse` no longer zeroes inactive cells itself,
    because `unfold` writes the zeros.
- **Unchanged:**
  - `base_atom_spectra`, keyed by `(m, a)`;
  - `atom_spectrum(m, n, ...)`.

  Both use band and time coordinates, not slots.

### `matrix.py`

This is the same change the user's `wdm` made.
- **`build_W`** returns `(N, N)`, with rows in folded flat order and W Wᵀ = dt·I. Today it
  returns `((Nf+1)·Nt, N)` with zero rows at the inactive cells.
- **`active_W` is removed:** it would be identical to `build_W`.
- **`build_C_wdm`** returns the folded `(N, N)` covariance.
- **`build_C_wdm_active` is removed** for the same reason.
- **Unchanged:** `build_C_time`, `rfft_to_real` and `real_to_rfft`.

### `noise.py`

- **`band_averaged_variance` and `evolutionary_band_variance` return `(Nf, Nt)` folded.** Row
  0's odd slots get the Nyquist band's value. Today the inactive cells carry the band value
  instead, so these arrays are the ones a strict `fold` refuses.
- **`evolutionary_band_variance(bands=)` is removed.** Nothing in `src` uses it.
  - `test_evolutionary_kernels.py:102` compares rows of the full result with the oracle's
    `bands=` output instead.
  - `test_window_threading.py:529` compares rows of the full result.

### `stationary.py`

- **`apply_lambda_from_psd`** takes and returns folded arrays. Inside it calls the padded
  twins.
- **Unchanged band blocks:** `build_lambda_blocks_full_active`, `build_K_full_active` and
  `psd_from_K`.
  - These are per band m = 0..Nf, with edge blocks over the Nt/2 active times. They are a band
    structure, not a cell layout.
  - The block order is also unchanged: it is band order.
- **`psd_from_lambda_blocks`** takes the same blocks, and its internal matrix-vector product
  stays padded.
- **`assemble_lambda_full_active`** returns `(N, N)` in folded flat order, including the
  corner pair (§1). Today it returns `((Nf+1)·Nt)²`.

### `evolutionary.py`

- **Now folded:**
  - `energy_fraction` returns `(Nf, Nt)`; today it returns band-order `(N,)`.
  - `apply_time_mask(Y, chi, ...)` takes and returns `(..., Nf, Nt)`; today it takes
    band-order `(B, N)`. Its internal pad and gather become `unfold` and `fold` through the
    twins.
  - `wrap_fractions` with `bands=None` returns `(Nf, Nt)`, with no NaN, since every slot is a
    cell.
- **Now in folded flat order:**
  - `exact_covariance` returns `(N, N)`.
  - `eq45_direct` returns the kept × kept matrix.
- **`eq45_plan(keep)`** takes a boolean `(Nf, Nt)` folded mask. Its cell arrays (`m`, `n`,
  `t`, `cells`, `pos`) follow the folded flat order of the kept cells.
- **Unchanged:**
  - **`wrap_fractions(bands=...)`** keeps returning band rows `(len(bands), Nt)`, with NaN at
    an edge band's odd columns. That is a band view, not a cell layout.
  - **`wrap_cut_columns`** keeps returning `(lead, trail)`. It computes its column maxima on
    the band rows, never on the folded slots, because a folded column mixes times n and
    n − 1 in row 0.
  - The Eq. 45 block outputs: `eq45_blocks_cheb`, the Gram plan family and `_block_spec`.
  - `time_domain_factor`, `time_domain_covariance` and `draw_noise`.

### `grid.py`

`band_frequencies` (per band) and `time_pixel_centers` (per column) are unchanged.
Per-cell values come from `cells` (§2).

## 4. `transform`, `wdm_covariance` and `testing`

### `transform`

- **The `"wdm"` layout is `(..., Nf, Nt)` folded.** `source_wdm` and `target_wdm` stay
  `(Nf, Nt)` pairs.
- **Every route behaves as today:** regridding from WDM to WDM, the validated copy when source
  equals target, and the window threading.
- **Removed:**
  - the WDM residue check (`_checked`'s inactive-cell branch) and its error text;
  - the zeroing of WDM output.
- **`RESIDUE` keeps its frequency job.** The imaginary part at DC and at even-N Nyquist may
  hold up to 1e-10 of the slice's peak, and the output sets it to exactly 0.
- **New refusals**, both `ValueError`:
  - a padded `(..., Nf+1, Nt)` WDM input, with a message naming `wilson.grid.fold`;
  - a flat `(..., N)` WDM input, with a message naming `.reshape(..., Nf, Nt)`.
- **The WDM output is the kernel's folded output as is.**

### `wdm_covariance`

- **It returns `(K, K, Nf, Nt)` folded**:
  `C[a, b, i, j] = ½ · BA_m[Re H_ab(·, t_n)]`, where `(m, n) = cells(Nf, Nt)[:, i, j]` and
  `t_n = n·Nf·dt`. Row 0's odd slots therefore hold the Nyquist band's K×K at time j − 1.
- **The model contract is unchanged.** It is still called with band frequencies and column
  times (`f` of shape `(bands, 1, J)`, `t` of shape `(1, columns, 1)`), so a model never sees
  a folded slot.
- **The computation is unchanged inside.** The band × column chunk loop still fills a padded
  `(K, K, Nf+1, Nt)` and folds once at the end. The final zeroing of inactive cells goes.

### `frequency_covariance`

Unchanged.

### `domainia.testing`

- **`dense_W`, `wdm_congruence` and `eq45_dense`** take a boolean `(Nf, Nt)` folded mask.
  - Their rows follow `numpy.nonzero(mask)` on that mask, which is the folded flat order of
    the selected cells.
  - The private `_keep` converter goes, because the mask is already in `eq45_plan`'s order.
  - The refusals of an inactive cell and of a `(Nf+1, Nt)` mask go too, because no folded slot
    is inactive. A wrong mask shape is still a `ValueError`, and a non-boolean mask is still
    a `TypeError`.
- **`empirical_variance`** returns `(Nf, Nt)` folded.
- **Unchanged:** `colored_noise`, `smoothstep`, `wide_profile` and `ALT_WINDOW`.

### Public names

- `domainia.__all__` is unchanged.
- `wilson.__all__` is unchanged.
- `wilson.grid` adds `cells`, `fold` and `unfold`, and loses `active_mask`.
- `wilson.matrix` loses `active_W` and `build_C_wdm_active`.

### README

- **The Data example:**
  - The WDM shape becomes `(2, 6, 12)`.
  - The inactive-cell half of the residue comment and its `forward_time` assertion go. The
    full-FFT half (the imaginary part at Nyquist) stays.
- **The Covariance example:**
  - The WDM shape becomes `(2, 2, 16, 64)`.
  - `not D[:, :, 0, 1].any()` goes.
- **Conventions:** the WDM bullet describes the folded slots, `.reshape(..., N)` for the flat
  form, `wilson.grid.cells` for each cell's band and time, and `fold`/`unfold` for padded
  arrays. The sentence about inactive cells being 0 goes.
- **Kernels:** the section names `cells`, `fold` and `unfold`.

## 5. Tests and the recorded fixtures

### The Meyer replay (`tests/wilson/data/meyer_baseline.npz`)

The npz is never re-recorded and never edited.

- **A new `load_baseline()` in `tests/wilson/meyer_cases.py`** converts the recorded arrays as
  they are loaded. The `meyer` fixture in `test_meyer_baseline.py` and `--check` both use it.
  It is not inside `mismatches`, because `test_the_comparator` feeds that function synthetic
  arrays.
- **It gathers the 58 arrays recorded in padded shapes into folded slots**, with a raw gather
  through `cells`, not through the checking `fold`. These are:
  - `forward`, `forward_time`, `forward_many` and `forward_time_many`;
  - `apply_lambda_from_psd`;
  - `empirical_variance`;
  - `band_averaged_variance` and `evolutionary_band_variance`;
  - `wrap_fractions`;
  - `build_W` and `build_C_wdm` (rows, and columns for the latter).

  A raw gather is needed because the recorded variance grids hold the band value in inactive
  cells, and `wrap_fractions` holds −1 there. The checking `fold` would refuse both.
- **It permutes the 30 arrays recorded in band-order flat into folded flat order**, and
  reshapes the vectors to `(..., Nf, Nt)` where the kernel now returns that shape. These are:
  - `energy_fraction`;
  - `apply_time_mask`;
  - `exact_covariance`;
  - `eq45_direct`;
  - `testing.dense_W`, `testing.wdm_congruence` and `testing.eq45_dense`;
  - `domainia::BandAveragedVariance`;
  - `domainia::EvolutionaryCovariance.direct`.
- **Key families:**
  - The 4 keys of the removed `active_W` and `build_C_wdm_active` join `DROPPED`, which goes
    from 96 to 100. 1,404 arrays stay under the gate.
  - The band-block families are layout-free and unchanged: the Λ blocks, `n_active`, the K
    vectors, the Eq. 45 interp blocks and the `(m, a)`-keyed spectra.
- **The case generators** that pass a padded mask or index padded rows are edited to the
  folded index (`meyer_cases.py` around L123, L154-157, L186-187, L195 and L205-207). Key
  names do not change.
- **The tolerance stays 1e-12** relative to each array's peak (or the family's peak for an
  all-zero array). `wrap_fractions`' scale changes because its −1 fill is no longer part of
  the array.

### `tests/data/translate_reference.npz`

This fixture holds 30 cases recorded from 2.0's `translate.transform`. That function was
deleted in 3.0, so the file cannot be re-recorded. Its 10 padded WDM inputs `x` and 10
padded WDM outputs `y` are gathered into folded slots at load, using the grids in
`z["grids"]`. Their inactive cells are exactly 0.

### The paper oracles

`tests/wilson/oracle/` is untouched, padded and byte-pinned (`test_oracle_pins`,
`test_evolutionary_kernels.py:364-376`). Every comparison gathers the oracle's padded output
into folded slots with a raw gather. Our output is never unfolded for the comparison, because
the oracle's noise grids also hold values in inactive cells.

### Tests that go

These exist only because of the padded layout:
- the inactive-cell zero checks: `test_covariance` (stationary and evolutionary),
  `test_transform` and `vendored/test_transform::test_inactive_pixels_zero`;
- the WDM rows of `test_transform`'s residue test;
- `test_dom_testing`'s refusals of a `(Nf+1, Nt)` mask and of an inactive cell;
- the "junk in inactive cells is ignored" clauses (`test_transform_shim`,
  `test_window_threading`);
- the NaN-pattern equalities (`test_evolutionary_kernels`, `test_window_threading`);
- the four `active_mask` tests in `vendored/test_grid.py`.

### Tests that change

- WDM shapes become `(..., Nf, Nt)`. `ROUTES["wdm"]` in `test_covariance` drops `[..., MASK]`.
- A check that needs "row = band" or "column = time" unfolds first, or uses `cells`. Examples:
  - column maxima of `wrap_fractions` (`test_evolutionary_kernels`, `test_window_threading`);
  - band sums and band slices that reach row 0 (`vendored/test_transform`,
    `vendored/test_noise`, `vendored/test_matrix`).
- `test_package` expects the `wilson.grid` and `wilson.matrix` name changes in §4.
- The doctests show folded shapes:
  - `grid.py` (`Grid`, the old `active_mask` example);
  - `transform.py` (three shapes);
  - `stationary.py` (`(144, 144)` → `(128, 128)`, and the padded input);
  - `matrix.py` (`(40, 32)` → `(32, 32)`, `(40, 40)` → `(32, 32)`);
  - `noise.py` (`(9, 16)` → `(8, 16)`, and the `var[1, 0]` comparison).

### New tests (small)

- **`cells`:** the slot table for Nf = Nt = 4, written out by hand. It is the same map as
  `wdm`'s `packed_wdm_indices`.
- **`fold` and `unfold`:**
  - they round-trip exactly;
  - `unfold` writes exact zeros (`False` for a boolean array);
  - `fold` refuses an inactive value above 1e-10 of the peak, and a non-finite one, naming the
    cell;
  - both refuse a wrong shape and a wrong kind.
- **`transform` and `wilson.inverse`** refuse a padded input and a flat input, with the named
  fix in each message.
- **Norm preservation:** Σw² over the folded array equals dt·Σx².

## 6. Enchilada (a hand-off; no edits from domainia)

Enchilada is read-only from domainia sessions. It is on branch `covariance-simplification`
at a203721.

- **It uses the padded layout throughout:**
  - `WDMGrid.array_shape = (Nf+1, Nt)` and `active_mask` (`domains.py:23-57`);
  - `check_wdm_coefficients`, which requires exact zeros at inactive cells
    (`domains.py:192-217`);
  - `L1Data`, which derives Nf as `shape[0] - 1` (`data.py:207-212`);
  - `Covariance`'s rule that blocks are zero at inactive cells (`covariance.py:165-173`);
  - `_transforms.py`, which zeroes inactive cells after `wdm.forward_many`
    (`_transforms.py:96-105`).
- **Its WDM backend breaks on the next `wdm` release anyway.** The local `wdm` removes
  `forward_many` and `inverse_many` in favour of `Transform(grid)`.
- **It has two paths.** The choice is Enchilada's.
  - **(a) Adopt the fold**, like `wdm` and domainia. All three packages then share one WDM
    layout.
    - `array_shape` becomes `(Nf, Nt)`.
    - `L1Data` reads Nf = `shape[0]`.
    - `active_mask`, the inactive-zero checks and `Covariance`'s inactive rule go.
    - `Covariance` stays `(K, K, *S)` with nothing special.
    - The sites:
      - `domains.py`: 27-30, 42-45, 52-57, 111-126, 192-217;
      - `data.py`: 28-30, 189, 207-212, 250-253;
      - `block_state.py`: 118-126;
      - `covariance.py`: 106-109, 165-173;
      - `_transforms.py`: 78-105;
      - `tests/conftest.py`: 69;
      - about 20 test lines;
      - `examples/domain_translation.py`: 96-97;
      - two notebooks;
      - the README.
  - **(b) Stay padded and convert at the boundary** with `domainia.wilson.grid.fold` and
    `unfold`. The conversions sit at the two backend calls in `_transforms.py`, and where the
    Wheel wraps `wdm_covariance` output in a `Covariance`.
- **Recommendation: (a).**
- The 3.0 spec's §8 hand-off text about the WDM layout and `check_wdm_coefficients` is
  superseded by this section.

## 7. Gates and size

- **Tests:** `-W error`, with no collection errors, in all four lanes: `wdmrel` (Python 3.12,
  numpy 2.3.5), `floor312` (numpy 2.1.0, scipy 1.14.1), `cur313` and `cur314`. The 3 oracle
  skips remain.
- **Replay:** 0 mismatches over the 1,404 kept arrays at 1e-12, in every lane.
- **The two recorded fixtures are never re-recorded:** `meyer_baseline.npz` and
  `translate_reference.npz`.
- **Doctests** pass in `src/domainia/wilson`.
- **Size**, reported before and after.
  - The baseline is 8931b99: src 4,080 lines / 1,861 code, and tests 4,344 lines (the oracle
    copies apart).
  - **Removed:** `active_W`, `build_C_wdm_active`, `evolutionary_band_variance(bands=)`, the
    inactive-cell checks and zeroing in `transform`, `inverse` and `wdm_covariance`, the
    `testing._keep` converter and its refusals, and the padded-only tests.
  - **Added:** `cells`/`fold`/`unfold`, about 16 one-line boundary calls, the private padded
    twins, the replay loader and the new tests.
  - The expected result is a small net cut. Growth in src code lines or in test lines is
    flagged.
- **No jax** anywhere.

## 8. Order of work (for the plan)

The replay gate must pass at the end of every task. `load_baseline()` therefore converts a key
family in the same task that changes that family's kernel.

1. **`wilson.grid`:** `cells`, `fold`, `unfold`, `RESIDUE`, `active_mask` made private, and
   their tests. The replay loader is added with no families converted yet.
2. **`wilson.transform`:** the padded twins and the public folded kernels, their callers inside
   the package, `transform`, the README Data example, and the replay families for the
   transforms and `translate_reference.npz`.
3. **`wilson.matrix`:** `build_W` and `build_C_wdm` folded, and `active_W` and
   `build_C_wdm_active` removed (DROPPED 96 → 100).
4. **`wilson.noise` and `wilson.stationary`.**
5. **`wilson.evolutionary` and `domainia.testing`** (`eq45_plan`'s mask and the oracles'
   masks).
6. **`wdm_covariance`**, the README Covariance example and Conventions.
7. **Docstrings and doctests swept, and the final gates** in all four lanes.

## 9. What this supersedes

- **In the 3.0 spec** (`2026-11-12-domainia-translation-core-design.md`), this supersedes the
  WDM layout statements:
  - "WDM is real `(Nf+1, Nt)` ... Inactive cells ... are exactly 0" (§2 and the
    survey);
  - §3.3's output shape `(K, K, Nf+1, Nt)` and its inactive-cell rule;
  - the residue rule's WDM half;
  - §8's WDM layout hand-off.
- **In the 3.0 plan's header decisions**, the `testing` oracles' padded `(Nf + 1, Nt)` mask is
  replaced by the folded mask (§4).

## 10. Open points and risks

- **Square grids:** an Nf/Nt transpose is invisible on a square grid (§1).
- **Enchilada's choice** between (a) and (b) in §6 is Enchilada's, and until it is made, the
  two packages disagree on the WDM layout. Enchilada does not import domainia today.
- **Digman and pywavelet:** value compatibility is out of scope (§1). A future
  converter would need the √dt scale (√(dt/2) for pywavelet), the sign on the interior cells
  where m+n is odd, and Digman's transpose.
