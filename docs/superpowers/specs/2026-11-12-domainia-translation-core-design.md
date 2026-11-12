# domainia 3.0: a translation core for Enchilada's Wheel

Date: 2026-11-12. Status: approved by the user on 2026-11-12 ("create a branch and let's
go"). The numbered decisions below stand as written: decision 3 takes option (a), and none
of decision 11's extra cuts is made. Written against `main` at 9f53225 (= origin/main) and
committed on branch `translation-core`.

Supersedes:
- spec 2026-10-31, §3 (the `{channel: array}` translate API), §4 (the Enchilada adapter) and
  its decision "translate API shape: dicts";
- spec 2026-11-08, §1's "the wilson kernels keep float32 when called directly", and its
  section "Parked for the Enchilada spec".

Enchilada's spec 2026-11-11 (`covariance-simplification`, at d1e0419) needs the changes
listed in §8.

**Sources.** The facts come from read-only work on 2026-11-12:
- four surveys: Enchilada's array layouts, per-point covariance from a noise model, the
  data-translation route, and the tests;
- three reviews of the first draft of this spec: code, decisions and Enchilada, and maths.

The probe scripts are in the session scratchpad. Other sources are the 2026-11-09
maintainability review and a full test run on NumPy 1.26.4. `file:line` references are to
domainia 9f53225 unless marked "Enchilada" (d1e0419).

## Goal

domainia becomes a small fp64 NumPy library with two jobs:

1. **Translate data** arrays between time, frequency and WDM, in both directions.
2. **Evaluate a noise model** as a per-point K×K covariance on a frequency or WDM grid.

Enchilada's Wheel calls domainia for each block, in the domain that block prefers. Blocks
compute their own likelihoods. Everything else in domainia goes.

## Decisions (user)

* 2026-11-09: domainia is costly to maintain and a large cut is coming. The no-regret stage
  of the maintainability review is folded into this spec.
* 2026-11-10: "the objects in enchilada are planned to be classes" (`L1Data.X`,
  `covariance.XY`); "Typically likelihoods will run in XYZ, but we can also call AET … these
  channels are not orthogonal". "For a first pass, we will likely be using diagonal
  likelihoods within the framework, but it needs to remain general if possible".
* 2026-11-12, in order:
  * "I don't actually care if we can jit anything here … Right now working fp64 is the goal."
  * "enchilada should not contain any of the likelihood infrastructure"
  * "delete the JAX backend for sure!"
  * "also drop domainia's channel layer and dict API"
  * "domainia doesn't need to compute the log likelihood right now. That can be done at the
    block level. The only thing that domainia needs to do is translate data and covariance
    between domains and provide it to the blocks when requested."
  * "The Wheel absolutely asks domainia. Each block will specify its preferred domain and
    the wheel will pass it data in its preferred format."
  * "keep the evolutionary noise parts. We absolutely need these pieces"
  * "The wheel must hold the total noise model. But this will be farmed out to be
    calculated at several points in the wheel too."
  * "yes, per-point KxK is fine for now. write the spec"

## Decisions in this spec (approved with the spec, 2026-11-12)

1. **Noise components.**
   - "The total noise model" is a sum of components, each owned by one block.
   - A component is either a K×K model, or a fixed per-point array on one grid (as
     Enchilada's `from_variance` and `from_psd` make today).
   - domainia evaluates models, one component at a time, and guarantees linearity (§3.4).
   - The Wheel sums the results per block grid and caches them.
2. **Model-to-grid routes.** domainia evaluates:
   - a stationary model → frequency, per bin;
   - a stationary or evolutionary model → WDM, per cell.

   No function moves a covariance array between domains (§1). This has two consequences:
   - A time-preferring block gets no covariance from a model. It uses array components on
     the time grid, as Enchilada's toy fit and benchmark do today (§8).
   - If the total includes an evolutionary model, a frequency-preferring block cannot be
     served. Enchilada's GB example is frequency-domain. The Wheel refuses the request,
     naming the component and the domain.

   Two routes are not taken now:
   - **A per-sample time route.** For a stationary model it is a single K×K matrix,
     `R_ab(0)`, for the whole record, and it is exact for same-sample blocks. For an
     evolutionary model it evaluates S at every sample, which is `(N/2+1)·N` points.
   - **An evolutionary → frequency route.** It gives the time-averaged S per bin, under the
     same local-stationarity rule as Eq. 36.

   Each is about 10–20 code lines. Add one when a block needs it.
3. **Enchilada's core stays NumPy-only.** This is option (a), recommended.
   - Enchilada keeps its own time ↔ frequency data path. That is `dt·rfft`, behind
     `L1Data.to_frequency` and `to_time`, which are public API (Enchilada `data.py:401-422`).
   - Enchilada also keeps its array constructors (`from_psd`, `from_variance`).
   - The Wheel imports domainia lazily, for WDM data and for every model component.
   - domainia is an optional extra, as `wdm` is today. This keeps Enchilada's decision
     "Optional and private; … the core stays NumPy-only" (Enchilada spec, decision table).
   - Cost: the formula `(T/2)·S(f_k)` exists twice, in `from_psd` and in
     `frequency_covariance`. Each copy is a few lines.
   - This departs from the quoted decision "The Wheel absolutely asks domainia" in two ways:
     - time ↔ frequency data is the one route the Wheel does not ask domainia for;
     - without domainia installed, only array components exist, because model evaluation
       (§3) is domainia's.

   Option (b) makes domainia a hard Enchilada dependency. Then:
   - the core lanes need scipy, but the core-minimum lane asserts that scipy is absent
     (Enchilada `ci.yml:105`);
   - domainia's NumPy floor must drop to 1.26.4, which is feasible (see Context);
   - every Enchilada install needs a route to a private package whose PyPI name is taken.
4. **One data route in domainia.** `translate.py` and `maps/` are deleted. A single
   `transform` (§2) sits on four folded kernels.
5. **Canonical outputs, tolerant inputs.**
   - Outputs are exactly canonical. Covariances are also exactly Hermitian, with a real
     diagonal (§3.1).
   - Data inputs may carry a rounding residue up to 1e-10 of their peak. Anything larger is
     refused (§2).
6. **Tolerance baseline.** The Meyer baseline is compared at 1e-12 relative, no longer bit
   for bit (§6.1).
7. **Eq. 39 goes.** The midpoint variance functions go, and Eq. 36 is the only per-cell
   rule.
8. **Grids and coordinates go.** `domains.py`, `data.py`, `spectra.py`, `variance.py` and the
   real-coordinate and Jacobian helpers go. Enchilada owns grids, and blocks own Jacobians.
9. **"The evolutionary noise parts" means the kernels.**
   - These stay:
     - `wilson/evolutionary.py`: the Eq. 45 builders, the exact time-domain references and
       the wrap rules;
     - `noise.evolutionary_band_variance`;
     - the evolutionary route of `wdm_covariance`.
   - The class `covariance/evolutionary.py` (`EvolutionaryCovariance`) goes with
     `covariance/`. It is a factorising covariance object, which serves likelihoods.
   - Its 232 baseline keys replay through the kernels, to 6.1e-14.
10. **Packaging.** The NumPy floor stays `>=2.1`, scipy stays, and the version becomes
    3.0.0.
11. **Extra cuts are offered, not made.** Each is made only if the user says so:

    | Cut | Lines / code | Note |
    |---|---|---|
    | `wilson/stationary.py` | 1,068 / 394 | the stationary band builder, for the band form that comes later; no caller in src |
    | Gram form, `evolutionary.py:600-716` | about 117 lines | the file calls it "the JAX build"; 68 test lines, 12 items |
    | `RampWindow`, `SampledWindow` | about 84 lines | used only by `testing.ALT_WINDOW`; `window=` stays with `Meyer(d)` |
    | `matrix.py`'s test-only helpers | 122 / 37 | `build_C_wdm_active`, `rfft_to_real`, `real_to_rfft` |

    git keeps everything that is cut.

## Context (measured at 9f53225)

- src is 12,252 lines (5,669 code). Tests are 17,384 lines and 3,156 items, including 4,135
  lines of oracle copies.
- Every concept exists two or three times: three transform routes, five stationary
  covariance builders, three covariance interfaces and three data models (review
  2026-11-09).
- **Enchilada's layouts match domainia's.**
  - `domainia.translate.transform` matches Enchilada's `transform_channels`
    (Enchilada `_transforms.py:35-111`) on all six domain pairs to 7.5e-16.
  - Four grids were checked, including 6×10 and 32×8, and odd N for time ↔ frequency.
  - Both packages use the same layouts:
    - time is real `(N,)`;
    - frequency is complex `(N//2+1,)` with `X = dt·rfft(x)`. It is real at DC and at
      even-N Nyquist.
    - WDM is real `(Nf+1, Nt)`, edge bands included. Inactive cells (m ∈ {0, Nf} at odd n)
      are exactly 0, and the atoms have unit norm, so Σw² = dt·Σx².
  - The mapping between the two is the identity.
  - Enchilada's `wdm` package needs Python ≥3.13 (Enchilada `_transforms.py:12-13`);
    domainia needs 3.12.
- **NumPy 1.26.4.**
  - 2,804 tests pass and 85 fail. None of the failures is a wrong answer:
    - 40 are float32;
    - 27 are only NumPy 1.x's `bool_` type name in error messages;
    - 9 are bit-exact or exact-zero comparisons;
    - 8 are `pcg`, which uses `np.vecdot`;
    - 1 is the lane meta-test.
  - Every fp64 baseline array agrees with NumPy 2 to 9.2e-14 relative.

## 1. The boundary

| | Owns |
|---|---|
| Enchilada | `L1Data` (`.X`, `.A`), and its time ↔ frequency data path (decision 3) |
| | `Covariance`, as a container of the per-point `(K, K, *grid)` array, with `.XY`, `.AE` and `in_basis` |
| | channel names and the XYZ ↔ AET rotation |
| | grids |
| | the Wheel, which holds the noise components, has domainia evaluate the models, caches the results and sums them |
| | blocks |
| domainia | `transform` (§2) |
| | `frequency_covariance` and `wdm_covariance` (§3) |
| | the Wilson kernels under them |
| Blocks | likelihoods: solves, determinants, quadratic forms and any Jacobian between domains |

- Neither package imports the other when it is imported. The Wheel imports domainia lazily,
  as it imports `wdm` today.
- domainia contains no solve, log-determinant, quadratic form, coordinate Jacobian,
  factorisation or likelihood.
- Every covariance is built from a model on the target grid, or comes from an array
  component on its own grid.
- No domainia function takes a covariance array in one domain and returns it in another.
  That rule keeps "approximate last", because an approximated covariance is never
  re-approximated. It also keeps the lazy-operator layer from coming back.

## 2. Data translation

```python
domainia.transform(x, source, target, *, N, dt,
                   source_wdm=None, target_wdm=None, window=MEYER) -> ndarray
```

**Arguments.**
- `source` and `target` are each `"time"`, `"frequency"` or `"wdm"` (Enchilada's names).
- `N` is always required, because the frequency length `N//2+1` does not fix N's parity.
- `source_wdm` and `target_wdm` are each an `(Nf, Nt)` pair.
  - Each is required when its side is `"wdm"`, and refused with `ValueError` otherwise.
  - Nf and Nt must be positive even integers with `Nf·Nt == N`.
  - The two WDM grids may differ. WDM → WDM then regrids through the rfft, as Enchilada's
    `transform_channels` does (Enchilada `_transforms.py:42-43`).
- When the source equals the target, including the WDM grid, `transform` returns a validated
  canonical copy.

**Shapes.** `x` may have any leading axes, typically `(K, …)`. The trailing axes must be:
- time: real `(N,)`;
- frequency: complex `(N//2+1,)`, in `dt·rfft` units;
- WDM: real `(Nf+1, Nt)`.

**Route and output.**
- Every route goes through the rfft, as both packages do today.
- The output is float64 or complex128, and exactly canonical:
  - inactive WDM cells are 0;
  - the imaginary parts at DC and even-N Nyquist are 0.

**Input checks.**
- Kinds:
  - A non-numeric input (bool, str, object) raises `TypeError`.
  - A complex input on the time or WDM side raises `ValueError`, as it does today
    (`translate.py:136-137`). Enchilada's spec gives the reason (Enchilada spec,
    `Covariance` rule 5): complex is valid in another domain, so it is a wrong value, not a
    wrong kind.
  - A real input on the frequency side is taken as complex.
- A wrong trailing shape, a non-finite value or a bad grid raises `ValueError`.
- Canonical form:
  - An inactive WDM cell, or the imaginary part at DC or even-N Nyquist, may hold up to
    1e-10 of the slice's peak. The peak is max |x| over the domain axes, for each leading
    index.
  - Anything larger is refused, with a message naming the cell or bin. A slice that is zero
    apart from such a value is therefore refused.
  - This check only validates the input. The kernels ignore those values exactly (measured:
    +iε at DC changes no active cell, `irfft` drops it, and `inverse` ignores 1e3 placed in
    inactive cells).
  - The measured residues are 1.4e-17 to 5.3e-17 of the peak on NumPy 1.26, on grids from
    6×10 to 100×50, so 1e-10 leaves a wide margin.
  - Today's exact-zero refusals (`translate.py:149-176`) go.

**Kernels.**
- The eight entry points in `wilson/transform.py` fold into four, each taking any leading
  axes:
  - `forward(X, Nf, Nt, dt, *, window)` maps a raw rfft `(…, N//2+1)` to `(…, Nf+1, Nt)`;
  - `inverse(w, Nf, Nt, dt, *, window)` maps it back;
  - `forward_time` and `inverse_time` put an `rfft` or `irfft` around them, a few lines
    each.
- The `_many` forms go (`forward_many`, `inverse_many`, `forward_time_many`,
  `inverse_time_many`).
- The internals already broadcast over leading axes. This was probed for `_gather_bands`
  and the scatter helpers on `(3, 2, …)` stacks.
- Kept modules call the time forms:
  - `evolutionary.py`: imports at :69-70; calls at :163-164 (`apply_time_mask`), :204
    (`wrap_fractions`) and :313 (`exact_covariance`);
  - `matrix.py`: import at :56; `build_W` calls it at :145.
  These move to the folded names.
- The band-limited entry points go with their private helpers: `forward_bands`,
  `inverse_bands`, `band_atoms`, `_band_phase`, `_band_windows` and
  `_check_interior_bands`. The second `_band_phase` (`maps/fourier.py:196`) goes with
  `maps/`.
- `_parity` stays, because `_scatter_edge_bands` and `_gather_bands` use it.

## 3. Noise covariance per point

### 3.1 The model

A noise model is a callable that returns the one-sided K×K cross-spectral density, in data
units²/Hz:

- **stationary:** `S(f)`, returning an array broadcastable to `(K, K, *f.shape)`;
- **evolutionary:** `S(f, t)`, returning an array broadcastable to
  `(K, K, *broadcast(f, t).shape)`.
  - t is in seconds from the record's first sample.
  - domainia passes f and t as broadcasting arrays: f varies along the last axis and t
    along the one before it.
  - A model that ignores t may return a size-1 time axis, and it is then never expanded
    along t.
- The Wheel shifts a model to the record's start if the model uses absolute time.

**Orientation.**
- `S[a, b]` is defined so that `E[X_a conj(X_b)] = (T_obs/2)·S_ab(f_k)` at an interior
  frequency bin. The conjugate convention is wrong by O(1) (measured).
- `S[b, a] = conj(S[a, b])`.
- Off-diagonal entries may be complex; for example, unequal arms give a complex S_XY. They
  may also be negative or zero.

**Validation.** Each check is made per point (frequency, or frequency and time). Its
tolerance is `tol = 1e-12·max_cd |S_cd|` at that point.
- A non-callable model, or a non-numeric output, raises `TypeError`.
- These raise `ValueError`:
  - a leading shape other than `(K, K)`;
  - a non-finite value. The `+inf → 0` mapping in `sanitize_psd` (`grid.py:232`) does not
    apply here: applied per component, it broke linearity by 5.7% in band 0 (measured).
  - `|S_ab − conj(S_ba)| > tol`;
  - a diagonal entry with an imaginary part above `tol`, or a real part below `−tol`.
- A point where every entry is 0 passes.

**Projection.**
- domainia uses `H = ½(S + Sᴴ)` at each point, so the outputs are exactly Hermitian and their
  diagonal is exactly real.
- Enchilada's `Covariance` requires both, element for element (Enchilada spec, `Covariance`
  rules 4 and 5).
- Natural model forms miss this by 2.3e-16 to 2.8e-14 (measured: a loop-built BᵀDB and a
  delayed cross-spectrum). Without the projection, they pass the tolerance check but fail
  Enchilada's constructor.

Enchilada checks positive-definiteness on the summed total, because a single component may
be zero somewhere. `sanitize_psd` keeps its role for scalar kernel inputs.

### 3.2 `frequency_covariance(S, *, N, dt) -> (K, K, N//2+1)` complex

- `C[a, b, k] = (N·dt/2)·H_ab(f_k)`, with `f_k = k/(N·dt)`.
- At DC and at even-N Nyquist, the route takes the real part. A real record carries no
  imaginary part at those two bins, and Enchilada's `Covariance` requires those entries to
  be real.
- Measured against the circulant reference (§6.2):
  - the variance matches to 3.3e-15;
  - different bins are uncorrelated (4e-16);
  - the pseudo-covariance `E[X_a X_b]` at interior bins is zero (1.6e-16).

  So a Hermitian K×K per interior bin describes the data completely.
- At DC and even-N Nyquist, `E[X_a X_b]` equals `E[X_a conj(X_b)]` (measured: difference
  0.0). So the K×K is the covariance of real Gaussian variables. In a log-likelihood it has
  half the weight of a complex bin.
  - 9f53225's `gaussian` does this today. It goes with `inner_product.py`, so blocks own it
    (§3.5, §8).
- For a record that is not periodic, this is the Whittle approximation.
- It matches Enchilada's `from_psd` (Enchilada `covariance.py:420-468`), which stores
  `(T_obs/2)·S(f_k)` on the diagonal at every bin.

### 3.3 `wdm_covariance(S, *, Nf, Nt, dt, window=MEYER) -> (K, K, Nf+1, Nt)` real

`S` is an evolutionary model `S(f, t)`. A stationary model `S0` is passed as
`lambda f, t: S0(f)`. Its output has a size-1 time axis, so its cost is the number of
frequency samples, not that number times Nt.

**Entry rule.** `C[a, b, m, n] = ½·BA_m[Re H_ab(·, t_n)]`.
- `BA_m` is Eq. 36's φ̂²-weighted average over the band around `f_m`, folded at DC and
  Nyquist, and `t_n = n·ΔT`.
- The ½ makes `Var w = S/2` for unit-norm atoms (Σw² = dt·Σx²).
- For K = 1 this is `noise.evolutionary_band_variance` (`noise.py:472`), which applies the ½
  at :516.
- Entries are computed for a ≤ b and mirrored, so each block is exactly symmetric.

**Inactive cells** are zero for every K×K block (Enchilada's spec 2026-11-11, L132).
- Today's kernel fills them (its docstring at `noise.py:480-482`).
- Measured: it returns 354.25 where the dense diagonal is 0.

**Evaluation.** `wdm_covariance` has its own K×K evaluator.
- It does not use `_psd_at` (`noise.py:462-469`), for two reasons:
  - `_psd_at` refuses an output with a size-1 time axis and refuses negative values;
  - it is shared with `evolutionary.py:273`, `:549` and `:786`, which keep it unchanged.
- The model is called once per chunk for all K² entries, not once per pair.
- Chunks cover bands and time columns, so peak memory is set by the chunk size and not by
  the grid. The plan picks the chunk size.
  - One whole band is already too large: at Nf = Nt = 2048 and K = 3 it is about 0.45 GB of
    complex values.

**Why only `Re S_ab`.**
- A same-cell covariance is a real scalar, symmetric in a ↔ b, and `S_ba = conj(S_ab)`. So
  a linear rule can depend only on `Re S_ab`.
- Measured for stationary noise: `Im S_ab` changes same-cell entries by 8.2e-16, while it
  changes off-cell entries by 45%.

**Accuracy against the dense reference** `(I_K ⊗ W) Σ (I_K ⊗ W)ᵀ` (Σ as defined in §6.2).
The metric is, per cell, the largest `|C_ab − C_dense,ab| / sqrt(C_dense,aa·C_dense,bb)`
over a and b.
- **Stationary:** exact to rounding.
  - The test case is K = 3 with complex delayed cross-spectra, on a 16×32 grid at dt = 5 s.
    The errors are 1.9e-15 in interior bands and 8.7e-16 in active edge-band cells.
  - With no time dependence, `evolutionary_band_variance` equals `band_averaged_variance`
    to 4e-16.
- **Evolutionary, periodic over the record, with one common modulation or real
  cross-spectra:** Eq. 36's local-stationarity error applies. It is second order in
  ΔT/period:

  | Case | Period of variation | Max error |
  |---|---|---|
  | K = 1, 30% modulation | 64·ΔT | 1.7e-3 |
  | | 32·ΔT | 6.6e-3 |
  | | 16·ΔT | 2.4e-2 |
  | | 8·ΔT | 7.0e-2 |
  | K = 3, `x = B(t)·y`, real cross-spectra | 32·ΔT | 3.5e-2 |

- **Evolutionary, with channel-dependent modulation and complex cross-spectra:** there is
  an extra first-order term.
  - The test case is K = 2, with 30% modulation, a 7.3·dt delay and coherence 0.8, on 16×64.
  - At a period of 64·ΔT the error is 7.9e-3. That compares with 2.3e-3 for a common
    modulation with the delay, and 1.4e-3 for different modulations without it.
  - As the period halves, the error grows by a factor of about 1.2–1.7, against about 4
    in the other cases.
  - The Wigner–Ville midpoint term `((a₁′a₂ − a₁a₂′)/2)·Re[(i/2π)∂_f S₁₂]` removes most of
    it: 7.85e-3 → 1.17e-3 at 64·ΔT. That term is not in scope.
- **Evolutionary, not periodic over the record:** WDM atoms wrap around the record, so the
  first and last few columns mix S(f, 0) and S(f, T).
  - The error there is first order in S(T) − S(0), however slow the drift.
  - Measured with K = 1 on 16×64, with S ∝ 1 + drift·t/T:
    - drift 10%: 4.8e-2 at n = 0, 7.5e-3 at n = 1, 9.1e-3 at n = Nt−1, and 1.8e-8 at
      n = Nt/2;
    - drift 60%: 0.23 at n = 0.
  - Blocks that need those columns drop the ones `evolutionary.wrap_cut_columns` names.
    That function is WDM Paper II's geometric rule, and it stays.
- The diagonal is the same quantity as the Eq. 45 diagonal (equal to 6e-16).
  - Today's pin is `tests/test_dom_testing.py:62-70` (<5%).
  - It excludes 3 leading columns and 1 trailing column with `grid.edge_columns(NT, 3, 1)`
    (:68).
  - It is rewritten with `wrap_cut_columns` when `edge_columns` goes.
- **Comparison with Eq. 39:** on the same steep stationary model, the Eq. 39 midpoint rule
  is off by 0.20 in interior bands and 1.5 in edge bands. That is why Eq. 39 goes.

### 3.4 Linearity

- Both functions are linear in S: `f(S1 + S2) = f(S1) + f(S2)` to rounding. The projection
  `½(S + Sᴴ)` is linear too.
- Measured: ≤6e-16 on every route.
- The Wheel's per-component cache depends on this, so it is a tested contract.

### 3.5 What a block does with it

- The Wheel wraps the summed array as an Enchilada `Covariance` on the block's grid.
- A diagonal likelihood reads the diagonal entries (`cov.XX` or `cov.AA`) and ignores the
  cross terms.
- Rotating XYZ → AET is `R C Rᵀ` at each point. It is exact, and it is Enchilada's
  `in_basis`.
- The 3-channel toy compares each structure with the exact WDM covariance, by KL divergence
  per degree of freedom:
  - per-point K×K costs 0.0029 in either basis;
  - diagonal costs 0.19 in XYZ and 0.008 in AET.
- At DC and even-N Nyquist, a frequency block uses the real-Gaussian form with half weight
  (§3.2).
- A block that uses an evolutionary model that is not periodic over the record drops the
  wrap columns (§3.3).
- A route that does not exist (decision 2) is refused by the Wheel. The exception is
  Enchilada's choice: its spec removes `NotImplementedError` from the package (§8).

## 4. What goes

Whole files (total lines / code lines):

| Module | Lines | Why |
|---|---|---|
| `maps/` | 1,203 / 756 | duplicate data route; pushes covariance arrays between domains |
| `covariance/` | 1,162 / 645 | covariance classes with factorisations, including `EvolutionaryCovariance` (decision 9) |
| `wilson/inner_product.py` | 1,008 / 305 | likelihoods |
| `linalg.py` | 620 / 379 | factorisations, `pcg`, `cblock_*`, `device_constants` |
| `wilson/backend.py` | 582 / 354 | the JAX shim |
| `translate.py` | 530 / 308 | dict API, coordinates, `Operator`, `TranslatedOperator` |
| `domains.py` | 457 / 276 | Enchilada owns grids |
| `operators.py` | 329 / 194 | operator layer |
| `variance.py` | 113 / 56 | duplicates `noise.py:287` and `:472` |
| `spectra.py` | 94 / 49 | `GenericPSD(f, t, nu)`; the Wheel's components replace it |
| `data.py` | 93 / 53 | JAX pytree data model |
| `wilson/special.py` | 49 / 18 | `bessel_i`, the backend Bessel wrapper; `chebyshev_basis`, used only by `tests/wilson/test_evolutionary_kernels.py:452` |
| **Total** | **6,240 / 3,393** | |

Two notes on the table:
- `testing.py`'s use of `domains.CellSet` (`testing.py:67-89`) becomes a bool mask, which
  `eq45_plan` already takes.
- The test at `test_evolutionary_kernels.py:452` builds the Chebyshev basis itself.

Inside the kept files:
- `wilson/transform.py`:
  - the float32 machinery: `_working_dtypes`, `_check_single_output`, `_set_rounded`,
    `_add_rounded`;
  - the four `_many` wrappers, the band-limited entry points and their helpers (§2).
- `wilson/grid.py`: `carrier_phase`, `packed_cells`, `pack`, `unpack`, `edge_columns`,
  `band_indices`, `count_active`.
- `_checks`: `require_bool`, `require_str`, `index_array`.
- `filter`: `phi_fft_order`, `phi_squared_normalization`.
- `noise.py`, Eq. 39: `whittle_variance`, `whittle_variance_from_array`,
  `evaluate_psd_on_band_grid`, `evolutionary_variance`.
  - `band_averaged_variance` returns through `whittle_variance_from_array` today
    (`noise.py:284`).
  - Its last line becomes `np.broadcast_to(0.5 * Sn_eff[:, None], (Nf + 1, Nt)).copy()`,
    and its grid check stays.
- `testing.py`: the helpers of the deleted layers (`blocktri_dense`, `hhat`,
  `eq54_covariance`, `dense_lambda_generic`, `PointwiseOperator`).

`colored_noise` and `empirical_variance` are used only by tests, so they move to
`testing.py`.

## 5. What stays, and how it changes

- `wilson/transform.py`, with the four folded entry points (§2).
- `grid.py`, `filter.py` (scipy's `betainc`) and `_checks.py`.
- `windows.py`. `window=` stays, with `Meyer(d)`; decision 11 offers the cut of the other
  two windows.
- `wilson/noise.py` keeps Eq. 36: `band_averaged_variance` and `evolutionary_band_variance`.
  It also keeps `_psd_at`, for its current callers.
- `wilson/evolutionary.py` stays, with these edits:
  - `@pytree_dataclass` (:324, :349, :604) goes; the classes stay frozen dataclasses;
  - `eq45_coeffs_bessel` calls `scipy.special.ive` directly, in place of `be.bessel_i`
    (:531). Today that call is in `backend.py:317`.
  - the time-form imports (:69-70) move to `forward_time` and `inverse_time`;
  - every `be.` call becomes NumPy.

  The file keeps the Eq. 45 builders (the Gram form too, unless decision 11 cuts it), the
  exact time-domain references and the wrap rules. In src only `testing.eq45_dense` calls
  it (`eq45_plan` and `eq45_direct`; `testing.py:44`, `:82-89`). It is the evolutionary
  band builder for later and the dense reference for §3.3.
- `wilson/stationary.py` stays, unless decision 11 cuts it.
- `wilson/matrix.py` stays as the dense-reference module: `build_W` (now calling
  `inverse_time`), `active_W`, `build_C_time` and `build_C_wdm`, plus the test-only
  helpers unless decision 11 cuts them.
- Every `be.` call becomes plain NumPy: `np.fft`, and direct slicing for `be.set` and
  `be.add_at`. dtypes are fp64 throughout.
- New code: `transform`, `frequency_covariance`, and `wdm_covariance` with its evaluator and
  chunking, plus the §3.1 check and projection. Together they come to about 200–300 code
  lines.
- **Public API:**
  - `domainia`: `transform`, `frequency_covariance`, `wdm_covariance`, `wilson`;
  - `domainia.wilson`:
    - `Grid`, `forward`, `inverse`, `forward_time`, `inverse_time`;
    - `Window`, `Meyer`, `MEYER`, `RampWindow`, `SampledWindow`;
    - `band_averaged_variance`, `evolutionary_band_variance`;
    - the submodules `evolutionary`, `stationary`, `matrix`, `noise`, `grid`, `filter` and
      `windows`.

## 6. Tests

### 6.1 Replay gate first

Order step 1 does all of the following before anything is deleted.

**One comparator.** `meyer_cases.mismatches` (`tests/wilson/meyer_cases.py:104-109`) becomes
one comparator:
- dtype and shape must match exactly;
- integer arrays must match exactly;
- float and complex arrays pass when `max|got − want| ≤ 1e-12·s`. Here s is max |want| over
  the array.
  - For an all-zero recorded array, s is the largest |want| recorded for the same function.
    There are 128 such arrays; all are still exactly 0 on NumPy 1.26.4.
- A NaN fails.

The worst drift measured across NumPy versions is 9.24e-14, so 1e-12 leaves about 11×
headroom.

**Keys.** There are 1,504 recorded keys.
- 96 keys of code that goes are dropped:
  - maps: 54;
  - inner_product: 24, of which 12 are already retired;
  - `forward_bands`, `inverse_bands` and `band_atoms`: 18.
- The other 1,408 stay under their old names. These are rewired inside `compute()` to kept
  code:
  - `EvolutionaryCovariance.interp/.direct` (232) → `eq45_plan`, `eq45_coeffs_interp`,
    `eq45_blocks_cheb` with P = 24, and `eq45_direct`;
  - `BandAveragedVariance` (6) → `noise.evolutionary_band_variance`;
  - `transform.forward_many`, `inverse_many`, `forward_time_many` and `inverse_time_many`
    (24) → the folded entry points;
  - `noise.empirical_variance` (6) → `testing.empirical_variance`;
  - `testing.dense_W`, `wdm_congruence` and `eq45_dense` (6) → their bool-mask signatures.

  The first two families replay to 6.1e-14 through the kept kernels (measured).

**Decoupling.** Collection must survive every later step.
- `test_meyer_baseline.py:18-23` imports `domainia.covariance`, `domains`, `maps`,
  `spectra` and `variance` at module level.
- `meyer_cases` builds inputs eagerly with names that go:
  - `wilson_cases`, :146-151: `forward_time_many`, `transform.forward_bands` and
    `inner_product`;
  - `domainia_cases`, :217-227: `CellSet`, `TimeDomain`, `FourierDomain`,
    `spectra.GenericPSD` and `covariance.EvolutionaryCovariance`.
- Step 1 rewrites both files:
  - the inputs are built with `np.fft.rfft` plus `forward`, and `wb` is dropped;
  - `GenericPSD.bind(nu)` becomes a plain lambda, and `CellSet` becomes a mask;
  - the module-level imports of modules that go are removed.

**The other changes.**
- `RESCALED` goes. Its five families are the three maps families and `forward_bands` and
  `inverse_bands`, and all of them are dropped. The assertion of that set
  (`test_meyer_baseline.py:39-50`) goes too.
- `RETIRED` empties, and `THREADED` loses `inner_product.`.
- The "bit_for_bit" tests are renamed, and the docstrings at `meyer_cases.py:1` and
  `:16-23` are fixed.
- Step 1 also records `translate.transform`'s outputs on the old suite's inputs, for the
  `transform` agreement test in §6.2.
- The other exact comparisons against oracle copies move to the same tolerance helper as
  their files are rewritten. The tests survey lists the sites; for example,
  `test_evolutionary_kernels.py:159`.

### 6.2 New tests (the dense reference is the definition)

**The dense references.**
- **Stationary:**
  - `Σ_ab[i, j] = R_ab((i − j) mod N)`, with
    `R_ab(τ) = Re[(1/(N·dt))·Σ_{k=0}^{N−1} P_ab(k)·e^{2πikτ/N}]`.
  - `P = S/2` is taken on the rfft bins and extended to all N bins by
    `P_ab(N−k) = conj(P_ab(k))`.
  - So DC and Nyquist use Re S, which is §3.2's convention.
  - Check: for white noise S0, R(0) = S0/(2·dt).
- **Evolutionary:**
  - `x = B(t)·y(t)`, where y has K independent channels, each with the stationary PSD P.
    So `Σ_ab[i, j] = [B(t_i)·B(t_j)ᵀ]_ab·R_P(i − j)` and `S_ab(f, t) = P(f)·[B(t)·B(t)ᵀ]_ab`.
    The cross-spectra are real.
  - Use the measured construction of §3.3's K = 3 row:
    `B(t) = L₀·(I + sin(2πt/T_p)·M₁ + cos(4πt/T_p)·M₂)`, with L₀, M₁ and M₂ fixed in the
    test, and T_p dividing the record.
- The metric is §3.3's.

**`wdm_covariance`, stationary.**
- The same-cell blocks match the dense reference within 1e-12.
  - The case is K = 3, with complex delayed cross-spectra and unequal levels.
  - It includes the active edge-band cells, and the inactive blocks are 0.
- A model that returns a size-1 time axis is accepted. It gives the same result as its
  explicitly broadcast output.

**`wdm_covariance`, evolutionary.**
- For K = 1, and for the diagonal entries, the result equals `evolutionary_band_variance`
  to 1e-14 at active cells. Inactive cells are zero.
- On the dense evolutionary reference, excluding the columns `wrap_cut_columns` names:
  - the error is below 5e-2 at T_p = 32·ΔT, where 3.5e-2 was measured;
  - it falls as T_p doubles to 64·ΔT. The plan's first measurement sets that bound.

**`frequency_covariance`.**
- It equals the per-bin covariance of `dt·rfft` under the stationary reference, within
  1e-12, with the `E[X_a conj(X_b)]` orientation.
- DC and Nyquist entries are real.
- For a diagonal model it matches `(T_obs/2)·S`, as Enchilada's `from_psd` stores it.

**Shared.**
- Linearity on both routes.
- A model that is Hermitian only to rounding, such as a loop-built BᵀDB, gives an output
  that is exactly Hermitian with a real diagonal.
- Each validation failure gives the right exception kind.
- A point whose entries are all 0 passes.

**`transform`.**
- Every one of the six pairs agrees with the `translate.transform` outputs recorded in step 1.
- Round trips work.
- Inputs with leading axes `(K, B, …)` work.
- Outputs are canonical.
- A residue of 1e-12 of the peak is accepted, and one of 1e-6 is refused.
- WDM → WDM between two grids equals the chain through frequency.
- `source == target` returns a canonical copy.
- Complex time data raises `ValueError`, and non-numeric input raises `TypeError`.

The new tests add about 300–450 lines.

### 6.3 What happens to today's tests

The tests survey classified every test function. The figures below correct it in two ways:
- `domains.py` goes. `test_dom_domains.py` and `test_wrong_kinds_domains.py` therefore go
  whole: 487 lines and 204 items, of which 472 lines and 198 items were Keep or Rewrite.
- Kept tests of code that goes inside the kept files are reclassified:
  - Delete: the Eq. 39 tests (`vendored/test_noise.py`, 102 lines; the Eq. 39 leg of
    `test_noise_stationary.py:60`); `count_active`, `band_indices` and `carrier_phase` in
    `vendored/test_grid.py`; `phi_fft_order` in `vendored/test_filter.py`; `packed_cells`,
    `pack` and `edge_columns` in `test_grid_bands.py`.
  - Rewrite: the `CellSet` tests in `test_dom_testing.py`; the `_many` callers in
    `vendored/test_evolutionary.py`.

The corrections come from a regex over function bodies, so they are approximate.

| Fate | Lines | Items |
|---|---|---|
| Delete | about 7,770 | about 1,820 |
| Rewrite | about 1,600 | about 330 |
| Keep | about 3,880 | about 1,000 |

- **Oracle copies** (4,135 lines) stay whole. They back kept code, and their bytes are
  pinned.
- **Vendored ports:**
  - these go: `test_inner_product`, the 151 lines that call only the oracle, and the Eq. 39,
    grid and filter tests above;
  - about 950 lines remain.
- **Infrastructure:**
  - both `_HideJax` finders, `_jax_or_skip`, `jax_be`, the `backend` fixture,
    `dom_db_inputs` and `test_lanes` go;
  - the two `rng` fixtures keep their own seeds (20261012 and 20261011), so seeded inputs
    don't change.
- **Window-threading test:** the AST meta-test in `test_window_threading.py` is rewritten
  for the kept modules.
- **Expected size:** about 5,250 non-oracle lines and 1,320 items, plus the new tests. That
  makes about 5,550–5,700 lines in all.

## 7. Packaging and text

- Version 3.0.0, because the dict API, JAX and the covariance classes go.
- The `[jax]` extra and the `jax` and `slow` markers go.
- The dependencies stay `numpy>=2.1,<3` and `scipy>=1.14.1,<2`.
- The README is rewritten around the three functions, and `test_readme` follows.
- Module docstrings drop the JAX, float32 and dict-API text.
- The docstring trim comes last.

## 8. Hand-off to Enchilada's spec 2026-11-11

Enchilada's spec needs these changes. They are recorded here because they follow from the
decisions above. Enchilada itself is changed only in an Enchilada session, which decides the
details.

**Sections to drop or reverse.**
- Drop:
  - §2.8: the operator methods and `w_p` weights on `Covariance`. These are likelihood
    infrastructure.
  - §3: `_domainia.py`, `NativeOperator`, `TranslatedCovariance`, `NoiseModel` and the
    covariance `transform`. §3.4's `transform(block_state)` and `reference_data` rules still
    hold.
  - §3.5: the domainia CI lane as written.
- Reverse two decision-table rows:
  - "What an off-grid block receives: An operator it queries". Instead, it receives a plain
    `Covariance` on its grid.
  - "Where the Wheel keeps the covariance: One native `Covariance` on its owner's grid".
    Instead, it keeps components.
- Keep the row "domainia dependency: Optional and private; … the core stays NumPy-only"
  (decision 3).
- Reverse §8's non-goals:
  - "Non-stationary or evolutionary noise models";
  - "Moving the data transforms … to domainia", for WDM.

**The Wheel and blocks.** Enchilada's spec must define each of these:
- **Components.** `BlockState.covariance` is `Covariance | None` today
  (`block_state.py:36`, `:58-59`). It needs a field for noise components. Each has a name, a
  kind (stationary or evolutionary), a time origin (the record's start or absolute), and
  either a model or a per-point array on one grid. The publishing block declares the kind
  and the time origin, because a bare callable carries neither.
- **Ownership.** Each component has one owner. This replaces the single-owner rule
  (`NoiseOverwrittenWarning`, `wheel.py:506`) and `_adopt`'s owner rule
  (`wheel.py:455-471`).
- **`Wheel.covariance`** (`wheel.py:143-149`): which grid it refers to.
- **Delivery.** `Block.sample` takes one covariance today (`block.py:56-63`). A noise block
  also needs the sum of the other components, which changes that signature (proposed).
- **Caching.**
  - Results are cached per component and grid, and invalidated on adoption events.
  - The key is not the callable. The Wheel deep-copies adopted states (`wheel.py:435`), and
    a deep-copied bound method is a new object that compares unequal (measured). An
    identity or equality key would never hit.
- **Time origin.** The Wheel shifts a model that uses absolute time to the record's start.
- **Array components.** An array component is handed only to blocks on its own grid. That
  keeps these working:
  - `examples/toy_fit.py:102-110` (`WhiteNoiseBlock`, `from_variance`);
  - `benchmarks/benchmark_wheel.py:35`;
  - the six xfails that Enchilada's spec clears without domainia (its testing section),
    including the toy time blocks, whose "noise block publishes on the time grid".
- **Missing routes.** The Wheel refuses them with an error naming the component and the
  domain. Not `NotImplementedError`, which Enchilada's spec removes from the package.

**Data and covariance.**
- `_transforms.py` keeps its NumPy time ↔ frequency path (decision 3).
  - Its WDM branch and the `wdm` package are replaced by `domainia.transform`. `np.stack` and
    its inverse are the adapter.
  - WDM → WDM regrids use the two WDM grids of `transform`. Today these are
    `_transforms.py:42-43` and `:58-59`, `tests/test_translation.py:186-198`, and
    `wheel.py:274-278`.
  - Python 3.12 suffices.
- `from_psd` stays as the core frequency constructor, with its per-channel protocol
  `model.psd(frequencies, channel)`. For a diagonal model it agrees with
  `frequency_covariance`. K×K model components need domainia.
- At DC and even-N Nyquist, blocks use the real-Gaussian form with half weight. The 9f53225
  `gaussian` fix goes with domainia's `inner_product.py`.
- Enchilada's exact checks can stay, because domainia's outputs are exactly canonical. These
  are `check_rfft_endpoints` (`block_state.py:85`, `:89`) and `check_wdm_coefficients`
  (`domains.py:131`). domainia's own inputs tolerate a 1e-10 residue (§2).

**Errors, tests, CI and changelog.**
- §5 Errors: the `ImportError` seam stays for the optional extra. The removal of
  `NotImplementedError` stands.
- §6 Testing: these go:
  - the operator-method tests;
  - "weights … pinned natively", which uses `quadratic_form` and `log_determinant`;
  - `tests/test_translated_covariance.py`.

  The xfail plan's "time-domain block that calls `quadratic_form`" needs a block-level
  replacement.
- §7 CHANGELOG: drop `TranslatedCovariance`, `NoiseModel` and the operator methods.
- CI: `wdm.yml` is the only lane that runs WDM translation and the notebooks
  (`wdm.yml:78-79`). It becomes the domainia lane, and it needs an install route for the
  private domainia repo.

## Size budget and gates

**src.**
- The kept files are 6,012 lines (2,276 code).
- §4's removals inside them come to 1,077 lines (430 code), net of the two functions that
  move to `testing.py`. That leaves about 4,935 lines (1,846 code) before new code.
- The new code and the two time wrappers add about 350–500 lines (200–300 code).
- So the result is about 5,300–5,450 lines (2,050–2,150 code). Removing the `be.` calls
  should recover some of that; the amount is not measured.
- **Budget:** at most about 5,400 lines (2,150 code) before the docstring trim.
  - This is above the first draft's 4,900 (1,850), which left out the new code and the
    functions that move.
  - Decision 11's cuts lower it by up to about 1,400 lines; `stationary.py` alone is 1,068
    (394 code).
- The plan reports the size after each stage and flags any overrun.

**tests.** At most about 5,700 non-oracle lines.

**Gates for the whole change:**
- the suite is green at the dependency floors (numpy 2.1.0, scipy 1.14.1) and at current
  versions, on Python 3.12–3.14;
- the replay gate passes at 1e-12;
- `grep` finds no `jax`, `be=` or `float32` in src;
- `transform` agrees with the recorded `translate.transform` outputs;
- the size budget holds.

## Order

1. Replay gate at a tolerance (§6.1), before any deletion:
   - drop the 96 keys of code that goes;
   - decouple the replay from modules that go;
   - record `translate.transform`'s outputs.
2. Delete the likelihoods, factorisations, covariance classes, operators and coordinates,
   with their tests.
3. Delete the JAX backend, `special.py` and float32. Move the kernels to plain NumPy, with
   §5's edits to `evolutionary.py`, under the replay gate.
4. Fold the kernels (eight entry points into four) and add `transform`.
   - Delete `translate.py`, `maps/`, `data.py` and `domains.py`, and rewrite their tests.
   - `testing.py`'s `CellSet` becomes a bool mask.
5. Add `frequency_covariance`, `wdm_covariance` and the K×K check and projection, with their
   tests.
   - Inline `band_averaged_variance`'s return.
   - Delete `spectra.py`, `variance.py` and Eq. 39.
6. Remove the dead code inside the kept files, make any cuts the user chose under decision
   11, and update the README, pyproject and version.
7. Trim the docstrings.

## Risks

- **Evolutionary cost at full size.**
  - Eq. 36 evaluates S at about 0.75·(Nf+1)·Nt² points, for all K² entries at once. At
    Nf = Nt = 2048 that is about 6×10⁹ points, which is too slow to call inside a noise
    block's sampler.
  - A stationary model costs only about 0.75·(Nf+1)·Nt.
  - For a separable model, `S_ab(f, t) = P_ab(f)·A_ab(t)`, Eq. 36 factorises exactly,
    because it averages over f at a fixed t_n.
    - The entry is `½·Re[BA(P_ab)(m)·A_ab(t_n)]`, with BA taken over the folded |f|. This
      matches the full kernel to 3e-16.
    - The Re is needed: without it the product is wrong when P_ab or A_ab is complex.
  - That fast path is the first follow-up. The plan times `wdm_covariance` on a realistic
    grid.
- **Evolutionary models that are not periodic over the record.** The wrap columns carry a
  first-order error (§3.3). Blocks must drop them.
- **Channel-dependent modulation with complex cross-spectra.** The error has a first-order
  term (§3.3).
- **The convention that maps a continuous model to the discrete grid.**
  - At DC and Nyquist, a real record carries only Re S, so a model's Im S there is dropped.
  - Two real processes with the same interior cross-spectral density can differ at Nyquist.
    For a real-filter fractional delay, measured:
    - 0.43% of the largest entry in frequency (diagonal ratios 0.345 and 0.905 against
      (T/2)·S_aa);
    - 4.0% in the WDM Nyquist band at 16×32.
  - §6.2's Σ fixes the convention: the Hermitian extension of S/2.
- **Size.** The budget is above the first draft's target. Decision 11 lists the cuts.

## Non-goals

- The band (block-tridiagonal) covariance output.
- A time-domain covariance and evolutionary → frequency, until a block needs them
  (decision 2).
- The Wigner–Ville correction for channel-dependent modulation.
- Likelihoods, Jacobians and coordinates.
- jit, JAX and float32.
- Partial and band-limited domains.
- The separable fast path (see Risks).
- A PyPI rename or publication.
- Any change to Enchilada's code; that needs its own spec.
