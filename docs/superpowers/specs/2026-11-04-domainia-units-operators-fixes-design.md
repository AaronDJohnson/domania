# domainia 2.0 follow-ups: units in place, covariance operators, fixes

Date: 2026-11-04. Status: approved design (approach A, 2026-11-04). Spec review
(2026-11-07) folded in; awaiting the user's review of the written spec.
Branch `domainia-2.0` at 73ec04a, not pushed. Nothing in this spec pushes.

**Test pin lists.** Each section names the tests it is known to break. The lists are a
starting point, not a census: the plan greps for every pattern a section changes (`.scale`
on a `WDMDomain`, `"sigma0"`, `forward_bands`, `inverse_bands`, `FourierToWDM(`,
`push_rows`, the moved error messages) and updates every hit.

## Goal

Three changes to the unreleased 2.0 branch:

1. **Units.** Every WDM value domainia produces is a Wilson coefficient of the time record,
   whichever path produced it. Every Fourier value carries units, and every map honours them.
2. **Covariance operators.** domainia's covariances, and explicit covariance matrices, can
   be moved between domains with `translate_operator`.
3. **The rest of the review list.** Single precision, wrong-kind errors, smaller defects,
   dead code, and a text sweep (last).

## Decisions (user)

* "For this package, I want to keep the units in their place." Approach A ("Go with A"):
  the maps convert, and a WDM value always means a Wilson coefficient. Approach B (keep the
  raw numbers and label them with `WDMDomain.scale`) is rejected.
* The covariance bridge (§2) and the fix list (§3–§7) were approved together with A.
* No push. The finishing menu (merge, PR, keep, discard) comes back after execution.

## Context (state at 73ec04a, measured)

* Wilson coefficients are `w = W x` with `W Wᵀ = dt I`. `transform`, `TimeToWDM`,
  `wilson.forward` and `wilson.forward_time` produce them. `WDMToTime`, the WDM variances
  and `EvolutionaryCovariance` assume them.
* `FourierToWDM` applies unit-norm atoms to its input `z` whatever `src.units` says. For
  `z = rfft(x)` it returns `sqrt(N/(2 dt)) · w` (≤ 2e-15 relative on three grids).
* `wilson.forward_bands(rfft(x), …)` is `sqrt(N/(2 dt)) · forward(rfft(x))[bands]`
  (measured ratio 11.3137… and 128.0, constant). `inverse_bands(w, …)` is
  `sqrt(2 dt/N) · inverse(w)` on its bins (≤ 8e-17).
* `FourierZeroPad.apply(z)` is `sqrt(N/N′) · rfft_N′(x zero-padded)` (≤ 2e-15), although
  `dst.units` says `"rfft"`. The factor kept the old galactic-binary model's unit-norm
  cells comparable across record lengths.
* `pad_maps` and `crop_maps` count self-conjugate source bins twice. With a non-zero DC and
  Nyquist, pad is off by 0.37 and crop by 0.73 (relative). With them zero, both match the
  FFT reference to 2e-15.
* `FourierDomain(units="sigma0")` stores no σ0 value.
* `WDMDomain.scale` is refused by every map and by `translate`, except `Identity`, which
  passes it through.
* **float32 on NumPy:**
  * `forward_time`, `forward`, `inverse`, `inverse_time` and their `_many` forms already
    return float32/complex64.
  * Their intermediates are not all single precision (measured in spec review with a
    recording backend):
    * `forward_time`, `forward_time_many`, `forward` and `forward_many` already feed
      `rfft` float32 and `ifft` complex64. The leak is after the FFT: float64 `kappa` and
      sign arrays multiply the output (`_gather_bands`, `transform.py:425` and `:431`), so
      that product is float64 and is rounded once when stored.
    * `inverse`, `inverse_many`, `inverse_time`, `inverse_time_many`, `forward_bands` and
      `inverse_bands` run complex128 FFTs.
  * `forward_bands` returns float64 for complex64 input, and `inverse_bands` returns
    complex128 for float32 input.
  * The docstrings at `wilson/transform.py:65-66` and `:166-167` claim "entirely in single
    precision".
  * The float32 comparisons against the frozen oracle are already tolerance checks
    (`ORACLE_RTOL[float32] = 2e-6` in `tests/wilson/test_transform_shim.py`), and no
    vendored test pins float32 bits.
* **Wrong kinds in `domainia.wilson` raise `ValueError`:**
  * `grid.py:112` (`_require_int`), `:152` (`validate_grid`) and `:346` (`n_samples`).
  * `noise.py:370` (`colored_noise`'s `N`).
  * `filter.py:65` (`nu_d`'s `d`, which mixes the kind and range checks).
  * `validate_dt` calls `float(dt)`.
  * The rest of domainia raises `TypeError` for a wrong kind (`domains._int`).
* **Covariances:**
  * `factor(nu)` returns a `Factor` (`solve`, `quad`, `logdet`, `ok`), and `dense(nu)`
    works through `_matrix`. There is no multiply.
  * `ChannelStack`'s factor takes `(C, n_c)` rows. The others take `(n,)`.
  * `PushedCovariance` looks up a duck-typed `matvec(X, be)` attribute, which has a
    fixed-noise-state signature.

## 1. Units

### 1.1 The rule

A WDM value is the Wilson coefficient `w = W x` of the time record, in the domain's window.
This holds on every path:
* `transform`;
* `TimeToWDM`;
* `FourierToWDM` (both implementations, and `after()`);
* the `domainia.wilson` band kernels;
* the covariances.

A Fourier value `z` has the units its `FourierDomain` declares:

| `units`     | `z`                    | `u` with `numpy.fft.rfft(x) = u · z` |
|-------------|------------------------|--------------------------------------|
| `"rfft"`    | `numpy.fft.rfft(x)`    | `1`                                  |
| `"dt_rfft"` | `dt · rfft(x)`         | `1 / dt`                             |
| `"sigma0"`  | `rfft(x) / σ0`         | `σ0`                                 |

Every map honours this label. No map rescales silently.

### 1.2 `FourierDomain.sigma0` and `rfft_scale`

* New kw-only field `sigma0: float | None = None`.
  * `units="sigma0"` requires a positive finite `sigma0`. Other units require
    `sigma0=None`. Both cases raise `ValueError` naming the field.
  * `sigma0` is part of equality and the hash.
* New property `rfft_scale`, the `u` of the table.
* Every place that derives a `FourierDomain` from another uses `dataclasses.replace`, so
  `units` and `sigma0` travel with it:
  * `BinSelect` (`maps/base.py:403`): `replace(src, k0=…, n=…)`;
  * `FourierToWDM.window` (`maps/fourier.py:284-285`):
    `replace(self.src, N=g.N, dt=g.dt, k0=geo["g0"], n=geo["G"])`. The record comes from
    `dst`'s grid, not from `src`: for a map returned by `after()`, `src` is on the outer
    record (`src.N ≠ g.N`). `replace(src, k0=…, n=…)` alone would break
    `tests/test_dom_maps_fourier_wdm.py:137-140`, and in the pad case (`:151`) `k0 + n`
    would exceed `N//2 + 1` and the constructor would raise;
  * any other site that `grep -n "FourierDomain(" src` finds, by the same rule.
* The "same units" checks in `FourierZeroPad`, `FourierCrop` and `BinSelect` compare
  `(units, sigma0)`.
* `TabulatedSpectrum.on` and `.at` stay unit-blind, because a PSD does not depend on the
  data's Fourier normalisation. Their docstring says so.
* **Tests to update** (they build `FourierDomain(..., "sigma0")` without `sigma0`):
  `tests/test_dom_domains.py:28`, `tests/test_dom_spectra.py:13-29`,
  `tests/test_dom_maps_base.py:112`.

### 1.3 The band kernels return Wilson units (`domainia.wilson.transform`)

* **`forward_bands(z, k0, bands, Nf, Nt, dt, …)`**
  * It returns `forward(z)[bands]`: the Wilson coefficients of rfft-unit bins, ≤ 1e-12
    relative to the full `forward`.
  * Implementation: today's algorithm times `sqrt(2 dt/N)`, or equivalently with the
    unit-norm normalisation removed where it enters.
* **`inverse_bands(w, k0, n_bins, bands, …)`**
  * It returns `inverse(w_full)[k0:k0 + n_bins]`, where `w_full` is the dense grid holding
    `w` on `bands` and zeros elsewhere.
  * Today's output is `sqrt(2 dt/N)` times this.
* The existing adjoint identity between the two kernels picks up the factor `2 dt / N`:
  `Σ forward_bands(z) · w = (2 dt/N) · Re⟨z, inverse_bands(w)⟩`, in the inner product
  that `test_inverse_bands_is_the_adjoint` uses today.
* `band_atoms` stays unit-norm: it is the spectrum of a basis function, not a
  coefficient. Its docstring states the factor `sqrt(2 dt/N)` that turns it into
  coefficients.
* **Tests to update** in `tests/wilson/test_grid_bands.py`:
  * `test_forward_bands_is_the_dense_atom_analysis` (around line 226);
  * `test_forward_bands_is_the_transform_times_sqrt_N_over_2dt` (line 243), renamed
    `…_is_the_transform_on_the_bands`;
  * `test_inverse_bands_is_the_adjoint` (line 256).
* **Also** in `tests/wilson/test_window_threading.py`: `:457-468`
  (`forward(z)[bands] / sqrt(2 dt/N)`) and `:473-487` (the adjoint identity, which picks
  up `2 dt/N`).
* `FourierToWDM.fft` calls `forward_bands`, so this section changes the fft
  implementation while §1.4 changes the dense one. The two land in one task (see Order),
  or `test_fft_impl_matches_dense` is red in between.

### 1.4 `FourierToWDM`

* `κ = u · sqrt(2 dt/N)`, with `N` and `dt` taken from the grid record. The constructor
  already checks that they equal `src`'s.
* **Dense.** `C = κ · U`, where `U` holds the unit-norm atoms (truncated when
  `restrict=True`). `apply`, `adjoint` and `rows` then carry `κ`, and `push_rows` carries
  `κ²`, with no further change.
* **fft.** `u · forward_bands(z)`, since §1.3 puts `forward_bands` in Wilson units.
* **`after(inner)`.** `C = κ · (the analysis of A and B)`. There is no `inner.scale` term
  any more (§1.5).
* **Docstring.** The sentence "`sqrt(N/(2 dt))` times `TimeToWDM.apply(x)`" becomes
  "equals `TimeToWDM.apply(x)` on the same cells". The `WDMDomain.scale` remark goes.
* **Acceptance tests:**
  * For each units value, and for `impl="dense"` and `"fft"`,
    `FourierToWDM(src, wdm).apply(z)` equals `TimeToWDM(time, wdm).apply(x)` to 1e-12
    relative.
  * `restrict=True` gives `C = κ ·` the truncated atoms.
  * The white-noise push: for `M′ = N σ² I` on the rfft bins (rfft units, unit-variance
    white noise when σ² = 1), `push_rows(M′, all)` equals `σ² dt I` on the cells to 1e-13.
    This was measured at 6e-16 for bands 1..Nf−1 and 2..Nf−2. It is the covariance of the
    Wilson coefficients of white noise, `W σ² I Wᵀ`.
* **Pins to update** (old record scale): `tests/test_dom_windows.py:128-135` and
  `:160-186` (rows, and the push `(N/(2 dt)) W Σ Wᵀ`);
  `tests/test_dom_maps_fourier_wdm.py:114-121` (push `M′ = 2I → I`) and `:155`.

### 1.5 `FourierZeroPad` and `FourierCrop`

* **No scale.** `FourierZeroPad` computes `z′ = A z + B conj(z)`, which is
  `rfft_N′(x zero-padded)` in the same units.
  * The `scale` field leaves `FourierZeroPad`, the `scale = 1.0` class attribute leaves
    `FourierCrop`, and `_WidelyLinear`'s docstring stops mentioning `.scale`.
* **Units.** Both maps require the same units, including the same `sigma0`. σ0 is a
  constant of the data, so the padded or cropped record keeps it.
* **Self-conjugate bins.**
  * `pad_maps` and `crop_maps` halve both the `A` and `B` columns of the absolute source
    bin `j_in = 0`, and of `j_in = N/2` for even `N`, where `N` is the source record's
    length.
  * Then `z′` is the exact rfft of the padded or cropped real record, whatever bins `src`
    covers.
  * The output DC and Nyquist bins are real to rounding. They are not forced to be exactly
    real.
* **Acceptance tests:**
  * Pad and crop applied to full-record bins (DC and Nyquist non-zero) equal the rfft of
    the padded or cropped series to 1e-13 relative.
  * The headline test checks
    `FourierToWDM(window′, wdm′).after(FourierZeroPad(store, window′)).apply(z)` and
    `Compose(FourierToWDM(…), FourierZeroPad(…)).apply(z)`. Both must equal
    `TimeToWDM(time′, wdm′).apply(concat(x, zeros))` to 1e-12 relative.
* **Pins to update:**
  * `tests/test_dom_maps_dirichlet.py:31-33` and `:56-58` (`Z.scale`);
  * `tests/test_dom_maps_fourier_wdm.py:18`, `:56-59`, `:125` and `:147` (the record-scale
    cases).

### 1.6 `WDMDomain.scale` is removed

* The signature becomes `WDMDomain(cells, window=MEYER)`.
* **Code to remove or update:**
  * `require_unit_scale` and its calls: `maps/time.py:61`, `:118`;
    `maps/fourier.py:248`; `maps/base.py:362-363`;
  * `translate.py:62-64`;
  * the `Identity` docstring (`maps/base.py:272-274`);
  * the docstrings at `domains.py:14` and `:327-337`.
* **Tests to update:**
  * `tests/test_translate.py:173` (`"wdm-scale"` case);
  * `tests/wilson/test_window_threading.py:267`;
  * `tests/test_dom_maps_base.py:204`;
  * the scale cases in `tests/test_dom_domains.py`;
  * `tests/test_dom_maps_base.py:63-74` (`WDMDomain(cut.cells, 2.0)`, positional: under
    the new signature `2.0` would land in `window`) and `:188-198`;
  * `tests/test_translate.py:64`;
  * any `.scale` read on a `WDMDomain` that `grep` finds in `tests/wilson/test_windows.py`.

### 1.7 The Meyer baseline

The recording made at d991ac4 is not re-recorded. Five case families change, and every
other key stays `np.array_equal`:

| case                                   | `got = f · recorded`, `f =` |
|----------------------------------------|-----------------------------|
| `domainia::FourierToWDM.dense`         | `sqrt(2 dt/N)`              |
| `domainia::FourierToWDM.fft`           | `sqrt(2 dt/N)`              |
| `domainia::FourierToWDM.adjoint`       | `sqrt(2 dt/N)` (κ once)     |
| `wilson::transform.forward_bands`      | `sqrt(2 dt/N)`              |
| `wilson::transform.inverse_bands`      | `1` (tolerance, not bits)   |

The baseline cases use `"rfft"` units, so `u = 1`.

`inverse_bands` gets `f = 1` because its input is not recorded: `tests/wilson/meyer_cases.py:80`
recomputes `wb = forward_bands(xr, …)` on every run. After §1.3, `wb` carries `sqrt(2 dt/N)`
and `inverse_bands` carries `sqrt(N/(2 dt))`, so the two cancel. Spec review emulated the
new kernels against the npz: with `f = 1`, four of the six grid/`dt` pairs are bit-equal
and the 32x8 pairs differ by ≤ 2e-16. The case therefore stays in `RESCALED` with `f = 1`
(compared by tolerance), not in the bit-exact set.

* `tests/wilson/test_meyer_baseline.py` gets a table `RESCALED = {(layer, case): f(N, dt)}`.
* `_mismatches` compares those keys for the same dtype and for
  `max|got − f·want| ≤ 1e-13 · max|f·want|`. It parses `N = Nf·Nt` and `dt` from the
  key's `"{Nf}x{Nt}::dt={dt!r}"` tag.
* All three baseline tests and `meyer_cases.main --check` use this one comparator.
* The test docstring records the reason: units, this spec §1.

## 2. Covariance operators

### 2.1 Why there was none

* `translate_operator` translates an `Operator`: one fixed covariance on a full-record
  domain, acting on `{channel: array}` dicts.
* `domainia.covariance` was built for likelihoods instead. Its covariances depend on a
  noise state `nu`, act on packed cell vectors that are often a subset of the grid, and
  expose `factor(nu)`, with no multiply.
* The 2.0 spec only planned the enchilada-side adapter, so nothing connected the two.
* The same missing multiply is why `PushedCovariance` refuses the WDM → time push.

### 2.2 Public real coordinates (`domainia.translate`)

* **`to_coordinates(channel_data, domain, channel_names=None) -> ndarray`**
  * It returns `(K, N)` float64.
  * Row order: `channel_names` when given, else the dict's order. With `channel_names`,
    the dict's channels must be exactly that set (`ValueError` otherwise), and the rows
    follow `channel_names` whatever order the dict was built in.
  * It validates its input exactly as `transform` does, with the same errors, on the same
    full-record domains.
* **`from_coordinates(R, domain, channel_names) -> dict`**
  * It returns the canonical layout: inactive WDM cells exactly 0, and Fourier end bins
    exactly real. The dict's order is `channel_names`.
* **Channel order is pinned to `channel_names`.** `DenseOperator`, `CovarianceOperator`
  and `operator_matrix` always pass their `channel_names` to `to_coordinates`. Today
  `TranslatedOperator` checks only the channel set (`translate.py:354`, `:366`) and
  `transform` keeps the caller's insertion order. Without the pin, `{"E": …, "A": …}`
  passed to an `("A", "E")` operator would meet the wrong matrix blocks. A test passes a
  reversed dict to each operator class and to a translated one.
* **Coordinates:**
  * Time: the samples.
  * Fourier, with `X = dt · rfft(x)` and `h = (N − 1) // 2`:
    `[Re X_0, √2 Re X_1 … √2 Re X_h, √2 Im X_1 … √2 Im X_h]`, then `Re X_{N/2}` for
    even `N` only.
  * WDM: the active cells in `wilson.grid.packed_cells` order. A test asserts that this
    equals the `CellSet.all_active` order.
* These are the coordinates the module docstring already describes. A test checks that
  each domain's coordinates are `exp(log_coordinate_scale(domain))` times an orthogonal
  map of the time samples.
* `tests/test_translate_operator.py` replaces its private `_pack`/`_unpack` with these two
  functions.

### 2.3 `DenseOperator` (`domainia.operators`)

`DenseOperator(matrix, domain, channel_names, *, rtol=None)` is an explicit covariance
matrix as an `Operator`.

* **Layout.** `matrix` is `(K·N, K·N)` on the stacked coordinates. The stacking is
  channel-major in `channel_names` order: `r` concatenates the rows of
  `to_coordinates(values, domain, channel_names)`.
* **Validation:**
  * the shape, and finite entries;
  * symmetry to 1e-12 relative. It then stores `(C + Cᵀ)/2`, read-only.
* **Decomposition.**
  * `numpy.linalg.eigh` decomposes the matrix, and `tol = max|λ| · K N · eps` (numpy's
    `matrix_rank` default) unless `rtol` is given.
  * An eigenvalue below `−tol` raises `ValueError` (the matrix is not PSD).
  * The eigenpairs with `λ > tol` are kept: `V`, `Λ`.
* **Operator methods:**
  * `degrees_of_freedom` is the number of kept eigenpairs;
  * `project` is `V Vᵀ`;
  * `solve` is `V Λ⁻¹ Vᵀ`;
  * `quadratic_form(r)` is `rᵀ solve(r)`;
  * `log_determinant` is `Σ log λ`.
* **Input and output.** Inputs go through `to_coordinates`. Channels other than
  `channel_names` raise `ValueError`. Outputs come from `from_coordinates`, so they meet
  the `Operator` output contract exactly.

### 2.4 `operator_matrix(op)` (`domainia.operators`)

* It returns the `(K·N, K·N)` matrix of `op.apply` in stacked coordinates, for any
  `Operator`, native or translated.
* Column `j` is the stacked `to_coordinates(op.apply(from_coordinates(e_j)))`.
* It costs `K·N` applies, and the docstring says it is for small `N`.
* **Translating a matrix:**
  `operator_matrix(translate_operator(DenseOperator(C, time, ("A",)), wdm))` is
  `W C Wᵀ`. The acceptance test builds the reference from `wilson.matrix.active_W`.

### 2.5 `Covariance.multiply`

`Covariance.multiply(nu, b, be=None)` returns `Σ(nu) b`. `b` uses the layout that the
covariance's factor `solve` takes: `(n,)` for a single covariance, `(C, n_c)` for
`ChannelStack`.

* **Per class:**
  * `DiagonalCovariance`: `v · b`.
  * `DenseCovariance`: `matrix @ b`.
  * `BlockTridiagonalCovariance`: from `band_blocks` in the ragged layout,
    `D_b x_b + U_b x_{b+1} + U_{b−1}ᵀ x_{b−1}`. It never forms the dense matrix.
  * `EvolutionaryCovariance`: delegates to its `impl`.
  * `ChannelStack`: each child, with `split(nu)`.
  * Base class: `_matrix @ b`. A class with neither raises `TypeError`.
* **`ChannelStack._matrix` (new).** Today `ChannelStack(...).dense(nu)` raises
  `NotImplementedError` (`stack.py` has no `_matrix`). It gains one: the block-diagonal of
  the children's `_matrix`, each at its own `split(nu)` slice, in channel order. The
  `(C, n_c)` layout of `b` flattens row-major (channel-major), so
  `multiply(nu, b).reshape(-1) == dense(nu) @ b.reshape(-1)`. The children must share `n_c`
  for the `(C, n_c)` layout, as `StackFactor.solve` (`xp.stack`) already requires.
* It is not named `matvec`: `PushedCovariance` duck-types `matvec(X, be)`, a
  fixed-noise-state signature.
* NumPy is required. A JAX `be` is passed through, but this spec does not test it.
* **Acceptance:** `multiply` equals `dense(nu) @ b` to 1e-12 for every class above
  (flattened as above for `ChannelStack`).

### 2.6 `CovarianceOperator` (`domainia.operators`)

`CovarianceOperator(cov, nu, support, channel_names)` fixes a covariance at `nu` and
presents it as an `Operator`.

* **Arguments.**
  * `support` is the `WDMDomain` (cells and window) that the covariance lives on. It may
    be partial.
  * The operator's `domain` is `WDMDomain.full(support.cells.grid, support.window)`.
* **Validation.** A wrong kind raises `TypeError`; a wrong value raises `ValueError`.
  * `cov` is a domainia `Covariance`, and `support` is a `WDMDomain`.
  * `channel_names` follows the `ChannelAxis` rules.
  * `cov.n == K · support.size`. For a `ChannelStack`, `len(cov.covs) == K` and every
    child has `n == support.size`.
  * `cov.window`, when it is not `None`, equals `support.window`.
  * `cov.cells`, when the covariance has one, equals `support.cells`.
  * `cov.n_u == 0`.
  * `cov.factor(nu)` is computed eagerly on NumPy at construction, and `ok` false raises
    `ValueError` ("not positive definite at nu").
* **Layout.**
  * Values go to `to_coordinates(values, domain, channel_names)` `(K, N)`, then to
    `R[:, support.cells.packed_positions()]`, giving `(K, s)`.
  * That array is flattened channel-major for a single covariance and kept as rows for a
    `ChannelStack`.
  * Results are scattered back with exact zeros off the support, then passed through
    `from_coordinates`.
* **Methods:**
  * `apply`: `multiply`.
  * `solve`: `F.solve`.
  * `project`: zeros off the support (the identity for a full support).
  * `quadratic_form`: `F.quad`.
  * `log_determinant`: `F.logdet`.
  * `degrees_of_freedom`: `K · s`.
  * With `P` the selection of the support, this is the Moore–Penrose structure of
    `Pᵀ Σ P` that the `Operator` contract describes.
* **Acceptance tests:**
  * `operator_matrix(op) = Pᵀ Σ P`, using `cov.dense(nu)`, for each of these:
    * a diagonal covariance;
    * a block-tridiagonal covariance;
    * a small-grid `EvolutionaryCovariance` (blocktri and dense assembly);
    * a `DenseCovariance`;
    * a two-channel `ChannelStack`;
    * a band-subset support;
    * a `drop_columns` support.
  * Translated to time, `operator_matrix` equals `Wᵀ Pᵀ Σ P W / dt²`.
  * Translated to Fourier, it equals the same reference mapped by the Fourier coordinates.
  * The translated `quadratic_form` and `log_determinant` match the dense references,
    including the pseudo-determinant for a partial support.
* `PushedCovariance._matvec`'s error message names `CovarianceOperator` as the route.

### 2.7 Taking the window from a `WDMDomain`

* `EvolutionaryCovariance`, `BandAveragedVariance` and `MidpointVariance` accept a
  `WDMDomain` where they take `cells`, and use its cells and window. `MidpointVariance`
  does not depend on the window, so it takes only the cells.
* `window=` defaults to `None`, which means `MEYER` for a `CellSet`. A `WDMDomain` together
  with a different explicit `window` raises `ValueError`.
* `EvolutionaryCovariance` drops the error "for a WDMDomain pass domain.cells".
* A `CellSet` with the default window keeps today's numbers, so the Meyer baseline is
  unchanged.
* The `WDMDomain` form is for callers. Call sites in `src/` keep passing `window=`
  explicitly, so the window scan in `tests/wilson/test_window_threading.py`, which flags a
  consumer call without `window=`, stays as it is.

### 2.8 Package surface for §2

* `domainia.translate.__all__` gains `to_coordinates` and `from_coordinates`.
* The new module `domainia.operators` has
  `__all__ = ["CovarianceOperator", "DenseOperator", "operator_matrix"]`.
* `domainia.__all__` gains the module `operators` and those five names, following how it
  already re-exports `translate`'s names. `tests/test_package.py` pins `PUBLIC` exactly
  and is updated with them.
* `import domainia` still does not import `domainia.covariance`, as the package docstring
  says (`src/domainia/__init__.py:25-27`). `domainia.operators` imports
  `domainia.covariance` lazily, inside `CovarianceOperator`'s constructor, where its
  `isinstance` checks need it. A test asserts that a fresh `import domainia` leaves
  `domainia.covariance` out of `sys.modules`.
* `operators.py` joins `SCANNED` in `tests/wilson/test_window_threading.py`, or
  `test_the_scan_covers_every_module_but_filter_and_windows` fails.

## 3. Single precision throughout

* **Rule.**
  * A float32 or complex64 input is transformed in single precision. Every constant
    multiplied into a working array is cast to the working dtype first: sign patterns,
    `kappa`, `lam`, `phi_c`, `1/√2` and the Hermitian slab arrays.
  * Every FFT then receives a single-precision array, on NumPy and on JAX.
  * float64 inputs are bit for bit unchanged: the casts are no-ops there, and no float64
    expression is regrouped. The Meyer baseline is the gate.
* **Dtypes by entry point:**

  | entry point                         | input → output | today                                      |
  |-------------------------------------|----------------|--------------------------------------------|
  | `forward_time`, `forward_time_many` | f32 → f32      | f32; FFTs single, f64 product after the FFT |
  | `forward`, `forward_many`           | c64 → f32      | f32; FFTs single, f64 product after the FFT |
  | `inverse`, `inverse_many`           | f32 → c64      | c64, c128 FFTs                             |
  | `inverse_time`, `inverse_time_many` | f32 → f32      | f32, c128 FFTs                             |
  | `forward_bands`                     | c64 → f32      | **f64**, c128 FFTs                         |
  | `inverse_bands`                     | f32 → c64      | **c128**, c128 FFTs                        |

* For the forward family the FFTs are already single precision. The change is the
  post-FFT products (`transform.py:425`, `:431`) and the edge-band divides by `np.sqrt(2)`
  (`:437-438`), which are float64 in a float32 run today.

* **Tests:**
  * A recording backend wraps the NumPy backend's `fft` namespace and asserts that every
    FFT input in a float32 run is single precision. It is already green for the forward
    family; it starts red only for the inverse family and the band kernels.
  * The forward family therefore needs its own red test, on the post-FFT product: the
    dtype of the array handed to `_set_rounded` in a float32 run. The plan picks the hook.
  * A test checks the dtype table above.
  * `ORACLE_RTOL[float32] = 2e-6` still holds against the oracle. The float32 bits move,
    at about 1e-7.
  * The shim tests compare against domainia's dtype table, not the oracle's dtype.
* **Text.** The docstrings and the `_add_rounded`/`_set_rounded` comments say what is now
  true: float32 bits differ from the oracle at rounding, and float64 bits are identical.

## 4. Wrong kinds raise `TypeError` in `domainia.wilson`

* **Sites:**
  * `grid.py:112`, `:152` and `:346`, and `noise.py:370`: a non-integer or `bool` raises
    `TypeError`, with the same message.
  * `filter.py:65` splits into a kind check (`TypeError`: "d must be an integer, got
    float") and a range check (`ValueError`: "d must be >= 1, got 0").
  * `validate_dt`: a `bool`, `str`, `None` or complex `dt` raises `TypeError`.
* **Tests and docs to update:**
  * `tests/wilson/test_noise_stationary.py:100-101` move to a `TypeError` parametrisation.
  * In `tests/wilson/vendored/test_filter.py:106`, `2.5` moves to a `TypeError` case;
    `0` and `-1` stay `ValueError`, and their `match="d must be an integer"` becomes the
    range message (`"d must be >= 1"`), which the old pattern would not match.
  * The "Raises" sections of the touched docstrings.
* **Unchanged:**
  * `tests/wilson/vendored/test_grid.py:114-116` passes only ints.
  * The frozen oracle `tests/wilson/oracle/vendored_2f7136c` is not edited.

## 5. Smaller fixes

1. **`psd_from_K`** (`wilson/stationary.py:756`): `eps = 1e-30 · max(phi2)`, a relative
   threshold.
   * The new test uses a `dt` small enough that `max φ² < 1e-30` (for example 1e-33, with
     a flat PSD). There `psd_from_K(build_K_full_active(psd))` must recover the PSD as it
     does at `dt = 1`. Today every bin is dropped.
   * The Meyer entry `stationary.psd_from_K` stays bit-exact.
2. **`CellSet.__reduce__`** rebuilds the set from `(grid, m, n)`. Then `pickle`, `copy`
   and `deepcopy` give read-only `m_array`/`n_array` and an equal hash.
3. **The `built_for` guard.** Seven maps carry constants built in `__post_init__`. Each gets
   a kw-only meta field `built_for` (`repr=False`) holding the fields that its constants
   derive from:

   | map                                                       | `built_for`                   |
   |-----------------------------------------------------------|-------------------------------|
   | `TimeToWDM`, `WDMToTime`, `Restrict`, `FourierZeroPad`    | `(src, dst)`                  |
   | `FourierCrop`                                             | `(src, dst, start)`           |
   | `FourierToWDM`                                            | `(src, dst, impl, restrict)`  |
   | `TimeToFourier` (its derived `dst`)                       | `(src, fmin)`                 |

   Construction routes:
   * a normal build sets `built_for`;
   * `after()` passes the key of the pair it built;
   * pytree unflatten passes, because the meta carries the key, and under `jax.jit` the
     check reads meta only;
   * `dataclasses.replace` changing a key field raises `ValueError` ("built for another
     src/dst: construct a new map");
   * constants supplied without `built_for` raise `ValueError`;
   * internal constructions that supply constants pass `built_for` too. Known site:
     `WDMToTime.adjoint` (`maps/time.py:144`) builds
     `TimeToWDM(self.dst, self.src, m_idx=…, n_idx=…)`, so it passes
     `built_for=(self.dst, self.src)`. The plan greps `src/` for every other constructor
     call that passes constants (`m_idx=`, `pos=`, …).

   The guard runs after each map's existing `src`/`dst` validation, so a mismatched
   window or grid keeps its current message. `tests/test_dom_windows.py:241`
   (`Restrict(..., pos=…)` with a window mismatch) keeps passing unchanged.
   (`tests/test_dom_maps_base.py:67`, the scale case, changes with §1.6.)
4. **Spec wording.** In the 2.0 spec, line 229 "domainia's own outputs make the endpoints
   exactly real" becomes "`transform`'s outputs make the endpoints exactly real".
5. **sdist.** Add `[tool.hatch.build.targets.sdist] exclude = ["docs/superpowers"]`.
   hatchling is not installed in the test env, so a `tomllib` test asserts the exclude.

## 6. Dead code and duplicates

Remove each item only after a grep shows no reader in `src/`, `tests/` or `README.md`.
Otherwise keep it, and record why in the plan errata. A test whose only job is to pin the
item itself is not a reader: it is removed with the item.

* `Covariance.batchable`;
* the `Covariance.complete_constants` hook, if no subclass overrides it;
* `GenericPSD.capabilities` (`spectra.py:79`). Its only reader is the pin at
  `tests/test_dom_spectra.py:37`, which goes with it;
* `BandAveragedVariance.t` and `MidpointVariance.t`, unused pytree data leaves. Both
  classes' `on()` methods stay: `as_evaluator` (`covariance/base.py:153`) recognises the
  variance classes by `hasattr(obj, "on")`;
* `covariance.evolutionary.keep_mask` (`evolutionary.py:44`), replaced by
  `CellSet.keep_mask()`. Its extra checks (active cells, band order) are already
  guaranteed by `CellSet.__post_init__` (`domains.py:192-197`), and
  `EvolutionaryCovariance` only reaches it after `isinstance(cells, CellSet)`
  (`evolutionary.py:150-152`). Spec review found the two equal on 8 cell sets over two
  grids.
  * `evolutionary.py:165` calls `cells.keep_mask()`.
  * The name leaves `covariance.evolutionary.__all__` and `covariance/__init__.py`.
  * Its pins go with it: the `"keep_mask"` entry of the export table at
    `tests/test_dom_cov_api.py:28`, and the import at `tests/test_dom_evolutionary_cov.py:10`
    with its test at `:105`. `CellSet.keep_mask()` stays covered by
    `tests/test_dom_domains.py:44` and `:84-85`;
* `_oracle_paths`, duplicated in `tests/wilson/test_evolutionary_kernels.py:44` and
  `test_grid_bands.py:46`: move it to `tests/wilson/conftest.py`.

## 7. Text sweep (last task)

* **Shorthand.** About 124 WT/WG/GB/"WDM II" hits across 19 files become plain words.
  * Say what the code does, and spell out paper references ("WDM Paper II, Eq. 36").
  * Mention a port from an oracle that does not ship once per module, described ("ported
    from the galactic-binary toy model, not shipped"), or drop it.
  * This includes `maps/fourier.py:3`.
* **1.x spec references.** About 26 "spec §X" references to the 1.x spec are replaced by
  what they cite, or removed.
* **Doctest alias.** 63 doctests that use `from domainia import wilson as wdm` become
  `from domainia import wilson`, with `wilson.` calls.
* **Gate.** The plan's regexes find zero hits outside the frozen oracle and the vendored
  tests. The full suite and every README block pass.

## Gates for the whole change

* The full suite passes, the NumPy lane (`WDM_TESTS_HIDE_JAX=1`) passes, and every README
  python block runs.
* The Meyer baseline is bit-exact except the five §1.7 families, which match the factor in
  the table.
* Each section's acceptance tests pass.

## Order

* **Units:** §1.2 → §1.3 and §1.4 together, with §1.7 → §1.5 → §1.6.
* **Operators:** §2.2 → §2.3/§2.4 → §2.5 → §2.6 with §2.8 → §2.7.
* **Then:** §3 → §4 → §5 → §6 → §7.

## Non-goals

* The enchilada adapter, and any push or PR.
* Re-recording the Meyer baseline.
* JAX support for the operator classes.
* Changing `band_atoms`' normalisation.
* Units for `TabulatedSpectrum`.
