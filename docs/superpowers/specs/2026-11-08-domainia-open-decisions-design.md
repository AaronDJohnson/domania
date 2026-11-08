# domainia 2.0 follow-ups II: the open decisions

Date: 2026-11-08. Status: approved design (three parts approved 2026-11-08). Spec review
(2026-11-08: four reviewers against the code, each finding checked by a skeptic) folded in;
awaiting the user's review of the written spec. `main` at 779550d, pushed to origin; this
spec is on branch `domainia-open-decisions`. This spec pushes nothing.

**Test pin lists.** Each section names the tests it is known to change. The lists are a
starting point, not a census: the plan greps for every pattern a section changes (the
moved error messages, `"; got"`, `pytest.raises(ValueError` next to a kind or complex
check, `.rtol`, `dtype=`, `exclude`) and updates every hit.

**Sources.** The facts below come from a read-only survey of `main` at 779550d
(2026-11-08): one agent per area, each probing the code, a completeness check that re-ran
the key claims, and the spec review. `file:line` references are to 779550d.

## Goal

Settle the decisions left open by the Nov 7 plan
(`docs/superpowers/plans/2026-11-07-domainia-units-operators-fixes.md`, its "Known open
points" and "Execution errata") and the new problems the survey found:

1. **Precision.** The maps compute in double (§1).
2. **Wrong kinds.** "Wrong kind → `TypeError`, wrong value → `ValueError`" holds across the
   whole package, as the specs and README already promise (§2).
3. **`DenseOperator`.** `rtol` keeps both jobs, and its fail-open bugs are fixed (§3).
4. **Packaging.** The sdist lists what it includes; the name stays (§4).
5. **Test gaps, text, record** (§5–§7).

## Decisions (user)

* Scope: "One domainia spec". The Enchilada adapter gets its own spec later (see
  "Parked for the Enchilada spec").
* Precision: "Maps compute in double". The wilson kernels keep float32 when called
  directly.
* Validation: "Whole package". One set of shared helpers in `domainia.wilson`.
* Array dtype: "dtype is kind: TypeError". A dtype that can never be valid for an argument
  is a wrong kind.
* `rtol`: "One rtol, both jobs" (scipy.stats' convention).
* Distribution: stay private. Name: "I hate all of these keep the name and we'll rework it
  once the package is complete." The distribution stays `domainia`.
* The three design parts (§1 + §3, §2, §4–§7) were approved as presented, including the
  items marked *(my call)*. One of those calls was reversed after approval, on evidence
  (§3.4).

## Context (state at 779550d, measured)

* **Maps' dtypes.** No rule is written down. Output dtype is the NumPy/JAX result type of
  the input and the map's constants, which are host float64/complex128
  (`maps/base.py:9-10`). For single-precision input (complex64 bins, float32 records or
  cells):
  * `FourierToWDM(impl="fft")` returns float32; `impl="dense"`, `restrict=True` and
    `after()` maps return float64 (`fourier.py:322`, `z @ self.C.T` with complex128 `C`).
  * Every `FourierToWDM.adjoint`, `FourierZeroPad`, `FourierCrop` and their `Compose`
    return complex128/float64.
  * `TimeToWDM`, `WDMToTime` and `TimeToFourier.apply` follow float32 input. They refuse
    complex input only by accident, and differently per backend: NumPy's rfft ufunc error
    (`TypeError`), wilson's `"wmn must be real"` (`ValueError`,
    `wilson/transform.py:707-711`, `:771-775`), or JAX's `ValueError`. The crop, pad,
    restrict and select maps follow the input.
  * Four methods (three causes) break even the result-type pattern: `BinSelect.adjoint`
    (`base.py:450`, `dtype=complex`) and `TimeToFourier.adjoint` (`time.py:208`) promote;
    `TimeToWDM.adjoint` and `WDMToTime.adjoint` go to float64 when `wilson.Grid` is given
    `dt=np.float64(...)` (`time.py:90`, `:146`), because `Grid` keeps the raw `dt` type.
  * A push through `Identity` or `BinSelect` only selects (`base.py:322-323`, `:453-455`),
    so it keeps the dtype of the covariance's `dense()`; every other push is double.
  * NumPy and JAX (x64 on, eager or jit) agree row for row. With x64 off, every map that
    resolves a backend raises `RuntimeError` (`wilson/backend.py:525-540`).
  * `translate` and `operators` are float64 by contract (`tests/test_translate.py:130-140`).
* **Float32 overflow.** With `dt=1e4` and complex64 bins of amplitude 1e36,
  `FourierToWDM(impl="fft").apply` returns NaN with only `RuntimeWarning`s, although the
  true cells (max 1.7e33) fit float32. The overflow is at `wilson/transform.py:1155`
  (`phi.astype(rdtype) * z[..., ell]`); `max|phi|` scales with `sqrt(dt)` (400 at
  `dt=1e4` on an 8×16 grid). `forward_time` overflows in its own rfft for a float32 record
  of amplitude 1e37, and `forward`'s input check then reports it as a gap ("Fill or excise
  the gaps"). `inverse_time` can overflow in its irfft after `inverse` returned finite
  values. The kernels check their input for finiteness, not their output.
* **Wrong kinds.** The rule is promised package-wide: spec 2026-10-31 §5 (lines 295-299),
  `README.md:165-167`; spec 2026-11-04 line 73 and `tests/wilson/test_wrong_kinds.py:1-2`
  say the rest of domainia already follows it. The probes show:
  * about 24 constructors and functions outside `wilson` (about 30 argument slots) raise a
    bare `AttributeError` on a wrong-kind domain or object;
  * about 26 scalar sites outside `wilson` convert silently (`int()` truncation, bare
    `float()` parsing strings, `operator.index` taking `True`, truthy flags), and more
    inside `wilson` (§2.3);
  * `Compose`, `Identity` and `GenericPSD` accept anything;
  * `CellSet` index arrays, `CellSet.from_mask` and `TabulatedSpectrum.at` report wrong
    kinds as `ValueError`; `TabulatedSpectrum.values` silently drops imaginary parts;
  * `EvolutionaryCovariance` checks `P` only at its first `factor()` and only for
    `builder="interp"`; `t_max` is only parsed by `float()` then (`t_max="300"` is
    accepted);
  * no existing test pins this leniency except the two integral-float index-array cases
    (§2.3) and `tests/wilson/test_wrong_kinds.py:66`, which accepts `dt=np.array(0.5)`.
* **The integer check is written out seven times:** `wilson/grid.py:111` (`_require_int`),
  `:156` (`validate_grid`), `:363` (`n_samples`), `filter.py:70`, `noise.py:372`,
  `windows.py:176` (`Meyer.d`, its own message), and `domains.py:53` (`domains._int`, its
  own message). The import direction is core → `wilson` only. Kind messages use two forms:
  33 `TypeError` messages in `src` say `"...; got X"`, 7 say `"..., got X"`.
* **`DenseOperator`** (`operators.py:48-154`):
  * one tolerance: refuse if `w[0] < -tol`, keep `w > tol`; `tol = top * n * eps`
    (`eps` float32's for a float32 matrix, float64's otherwise) or `rtol * top`;
  * an `rtol` of 1 or more accepts `-I` with `dof = 0`;
  * an `rtol` below round-off refuses a rank-deficient PSD matrix as "not positive
    semi-definite" whenever round-off makes a zero eigenvalue negative (always with several
    zero eigenvalues, about half the time at rank `n-1`); otherwise the matrix is accepted,
    and with `rtol=0` a round-off eigenvalue may be kept as a direction. That is accepted
    behaviour for an explicit `rtol`;
  * `top * n * eps` overflows to inf for `top > DBL_MAX / n`: `-1.2e307 * I` is accepted
    with `dof = 0`;
  * `0.5 * (C + C.T)` overflows for entries above about 8.99e307 after `validate_finite`
    passed: `diag(1e308)` is accepted with `dof = 0`;
  * a finite matrix whose largest eigenvalue exceeds `DBL_MAX` (`np.full((4, 4), 5e307)`)
    makes `eigh` return inf, and it is accepted with `dof = 0`;
  * the float32 `eps` is not stored: `dataclasses.replace(op)` on a float32-built
    rank-deficient operator raises "not positive semi-definite";
  * the symmetry check is `max|C - C^T| <= 1e-12 * max|C|` whatever the dtype, so an
    unsymmetrised float32 product `B @ D @ B.T` (asymmetry 5.2e-8) is refused;
  * scipy.stats `_PSD` shares one tolerance between its PSD check and its cut in the same
    way.
* **Packaging.** hatchling 1.27 builds the sdist and wheel cleanly and reproducibly (the
  wheel from the tree and the wheel from the sdist are byte-identical); `twine check
  --strict` passed on both (twine 7.0.0, scratch install); the suite run against the
  installed wheel passes (1768 + 1 probe, 3 oracle skips). hatchling is installed only in
  `~/miniconda3/envs/sksparse_test` (Python 3.13.5), not in `wdmrel`. The sdist uses a deny
  list (`exclude = ["docs/superpowers"]`), so any untracked folder `.gitignore` does not
  cover is packed (`.superpowers/` was). The PyPI name `domainia` belongs to an unrelated
  tool (2.2.0, Python ≥ 3.13).
* **Enchilada** (`../enchilada` at 8a55ddb) does not import domainia. It has no JAX, always
  converts to float64 before its WDM backend, and keeps complex64 on time → frequency on
  NumPy ≥ 2 (`src/enchilada/_transforms.py:66-76`). domainia already matches it on the WDM
  layout, Wilson units and the Meyer transform (≤ 7.8e-16 against its pinned `wdm` 0.2.0).

## 1. Precision of the maps

### 1.1 The rule

A map that computes returns float64 or complex128, whatever precision it is given
(integers included). A map that only selects, slices or zero-fills returns its input's
dtype. In one sentence for `maps/base.py`'s module docstring:

> Maps that compute (`TimeToWDM`, `WDMToTime`, `TimeToFourier`, `FourierToWDM`,
> `FourierZeroPad`, `FourierCrop`), and a push through one of them, return
> float64/complex128 whatever the input's precision; maps that only select, slice or
> zero-fill (`Identity`, `CropTime`, `ZeroPadTime`, `Restrict`, `BinSelect`), and their
> adjoints, return the input's dtype (`BinSelect.adjoint`: the input's complex
> counterpart), and a push through `Identity` or `BinSelect` returns the dtype of the
> covariance's `dense()`. `Compose` returns what its parts return.

### 1.2 Sites

* **Real-taking methods.** `TimeToWDM.apply` (`x`), `TimeToWDM.adjoint` (`y`),
  `WDMToTime.apply` (`y`), `WDMToTime.adjoint` (`x`) and `TimeToFourier.apply` (`x`)
  (`maps/time.py:69-200`) take real arrays. Each first refuses a complex, bool, str or
  object dtype with `TypeError` through §2's dtype helper — the cast below would otherwise
  drop an imaginary part with only a `ComplexWarning`, bypassing today's accidental
  refusals. Only then does it convert with `be.xp.asarray(x, dtype=np.float64)` (integer
  input converts). Under JAX x64 is already required.
* `TimeToFourier.apply_complex` converts complex input to complex128 and real input to
  float64.
* `TimeToFourier.adjoint` (`time.py:202-210`): its input is the complex kept bins; it
  converts `y` to complex128 (a real `y` converts too), never to float64. The spectrum is
  allocated complex128 as now, which now follows the rule.
* `FourierToWDM(impl="fft")` (`fourier.py:325-327`): convert `z` to complex128 before
  `forward_bands`. The dense path, `restrict`, `after()` and every adjoint already return
  double through `C`; no change beyond the docstring.
* `FourierZeroPad`, `FourierCrop` (including `apply_data`): already double through their
  complex128 constants; no change.
* `BinSelect.adjoint` (`base.py:450`): allocate in the input's complex counterpart, with the
  dtype computed on the host so both backends agree:
  `dtype=np.result_type(np.dtype(y.dtype), np.complex64)` passed to `be.xp.zeros`
  (complex64 stays complex64; float32 gives complex64; float64 and integers give
  complex128).
* The `Grid.dt` leak at `time.py:90` and `:146` goes away with the conversion; `wilson.Grid`
  is not changed.
* `push` is unchanged: through a computing map it is double already; through `Identity` or
  `BinSelect` it keeps the covariance's dtype (§1.1). domainia's own covariances do not
  reach that path (their `dense` takes `nu`); only a user-supplied `SupportsDense` does.
* Docstrings: `maps/base.py:9-10` (the rule); `FourierToWDM` (`fourier.py:254-257`: both
  impls return float64; the single-precision fft route is gone); the module docstring of
  `wilson/transform.py` ("Where it reaches, and where it stops": the maps compute in
  double); the `maps/time.py` module docstring (`time.py:3-5`) and the `TimeToFourier`
  docstring (`time.py:169-171`) qualify their identities with "for float64 input (other
  inputs are converted to float64 first)".

### 1.3 What does not change

* float64 results are bit for bit what they are now (converting float64 to float64 is a
  no-op). See the bit gate under "Gates".
* The wilson kernels keep single precision when called directly (spec 2026-11-04 §3).

### 1.4 The single-precision kernel guard *(my call)*

A single-precision run of a `domainia.wilson` transform entry point (the ten in spec
2026-11-04 §3's table, `forward_time` … `inverse_bands`, `_many` forms included) that
produces a non-finite value from finite input raises `ValueError` naming the entry point
and suggesting float64 input. Each entry point checks the arrays it computes itself:

* `forward`, `forward_many`: the result of `_gather_bands` (`transform.py:397`);
* `forward_time`, `forward_time_many`: their rfft result, before delegating to
  `forward`/`forward_many` (`transform.py:895`, `:648`), so the message names
  `forward_time(_many)` instead of `forward`'s "Fill or excise the gaps" input message;
* `inverse`, `inverse_many`: their results (`:651`, `:728`);
* `inverse_time`, `inverse_time_many`: the irfft result (`:830`, `:945`);
* `forward_bands`, `inverse_bands`: their results (`:1113`, `:1160`).

The check runs only on concrete arrays (skipped for JAX tracers, as `_check_finite` is,
`transform.py:193-200`), and only when the working dtype is single (`_working_dtypes`), so
float64 runs gain no extra pass. The guard does not change `np.errstate`. wilson functions
that call these entry points (`inner_product`, `stationary`, `evolutionary`) inherit the
guard. The variance builders are not covered.

### 1.5 Safeguard shared with §2

`FourierDomain` keeps converting `dt` and `sigma0` with `float()` after its new kind check
(§2.4). If it kept the raw type, an `np.float64` `dt` would make `rfft_scale` a NumPy scalar
again. Under §1.2 the fft path converts to complex128 anyway, but `rfft_scale` stays a Python
float so that no other weak-type promotion depends on it.

### 1.6 Tests

* A new `tests/test_dom_map_dtypes.py`: one table over every public map method that takes
  data (forward, adjoint, `apply_complex`, `apply_data`, `after()` maps, `Compose`,
  `FourierToWDM(restrict=True)`, `Restrict`, `push(...).rows`) × input dtype → output dtype,
  on NumPy and on JAX with x64 on. Input dtypes: float32, float64 and an integer array where
  the method takes reals; complex64 and complex128 where it takes complex. Complex64 and
  complex128 into the five real-taking methods (§1.2) expect `TypeError` on both backends.
* A large-`dt` complex64 case: `FourierToWDM(impl="fft")` returns finite float64 cells
  matching the dense path.
* The kernel guard: `forward_bands` with complex64 bins of amplitude 1e36 at `dt=1e4`
  raises `ValueError`, and the same call with complex128 succeeds; `forward_time` with a
  float32 `np.full(N, 1e37)` raises `ValueError` matching `forward_time` and `float64`;
  `inverse_time` with float32 cells scaled so `inverse` is finite but the irfft overflows
  raises `ValueError` naming `inverse_time`. NumPy emits overflow `RuntimeWarning`s before
  the guard runs, so these tests wrap the call in
  `np.errstate(over="ignore", invalid="ignore")`, as `tests/test_translate.py:277` does.
* Known to change: any test pinning float32 from the fft path, or a complex input reaching
  a real-taking map (grep `complex64` in `tests/test_dom_maps_fourier_wdm.py`,
  `tests/test_dom_maps_fourier_time.py` and `tests/wilson/test_single_precision.py`).

## 2. Wrong kinds raise `TypeError` across the package

### 2.1 The rule

`TypeError` when an argument's type, or an array's dtype, can never be valid in that place.
`ValueError` when the type is right but the value is wrong: range, shape, finiteness,
integrality, or a string that is not one of the choices.

### 2.2 Helpers *(my call)*

A new private module `src/domainia/wilson/_checks.py` holds the integer, real-scalar, flag,
string-option, `isinstance` (against the class passed in) and index-array helpers, plus
`validate_dt` and `validate_finite`. It imports only NumPy, so both `wilson` and the core can
use it without a cycle. `require_window`, `transform._check_finite` and `sanitize_psd` stay
where they are (they need `Window`, `is_tracer`, and their doctested names).

* The seven copies of the integer check (Context) are replaced by calls into it.
  `domains._int` becomes a thin wrapper that adds its range check.
* `validate_dt`, `validate_finite` and `_require_int` keep their names in
  `wilson/grid.py`, re-exported from `_checks` (`matrix.py:49`, `evolutionary.py:57` and
  `tests/wilson/test_wrong_kinds.py:11` import `_require_int` from `grid`).
* Add `"wilson/_checks.py"` to `SCANNED` in `tests/wilson/test_window_threading.py:31-37`
  (its `test_the_scan_covers_every_module_but_filter_and_windows` requires every module to
  be listed; `_checks.py` calls no window consumer).
* One message format, `wilson`'s: `"{name} must be an integer, got {type}"`, and the same
  shape for the other kinds (`"{name} must be a real number, got {type}"`,
  `"{name} must be a bool, got {type}"`, `"{name} must be a {Expected}, got {type}"`).
  Every kind (`TypeError`) message moves to the `", got"` form, the 33 `"; got"` ones
  included. Known to change: `tests/test_dom_evolutionary_cov.py:101` and
  `tests/test_dom_variance_generic.py:77` (anchored `match`), and whatever else the grep for
  `"; got"` next to `TypeError` finds.

### 2.3 Integers, index arrays and value arrays

* **Integers.** Python `int` and NumPy integers are accepted. `bool`, `np.bool_`, any float
  (2.0 included), `str`, `None` (where `None` is not a documented default) and arrays raise
  `TypeError`. Sites:
  * the four `operator.index` sites that take `True` today (`CropTime` `start`/`n`,
    `time.py:243-244`; `ZeroPadTime` `n_extra`, `:273`; `FourierCrop` `start`,
    `fourier.py:153`);
  * `pad_maps`' `N` and `Np`, and `crop_maps`' `N`, `start` and `Nc` (`fourier.py:54`,
    `:70`);
  * the `int()` sites: `DiagonalCovariance` `n` and `channel`
    (`covariance/diagonal.py:40-41`), `DenseCovariance` `n` (`dense.py:40`), `ChannelStack`
    `params_per_channel` (`stack.py:65`), `BinSelect`'s `(k0, n)` (`base.py:434`),
    `cblock_factor`'s `b` (`linalg.py:444`), `pcg`'s `maxit` (`linalg.py:507`, `:542`),
    `empirical_variance`'s `n_realizations` (`noise.py:444-447`);
  * `testing.eq54_covariance`'s `margin` (`testing.py:109`; `margin=True` is accepted today
    and `1.0` raises `IndexError`), which `dense_lambda_generic` passes through;
  * in `wilson`: `forward_bands` and `inverse_bands` `k0`, `atom_spectrum` `m` and `n`,
    `grid.carrier_phase` `m`, `grid.edge_columns` `lead` and `trail` (their `src` callers
    already pass `int(...)`: `maps/time.py:99`, `fourier.py:193`, `evolutionary.py:662`,
    `transform.py:1083`).
* **Sequences of counts.** `BlockTridiagonalCovariance`'s sizes (`blocktri.py:147`) and
  `BandLayout.from_sizes`' sizes (`linalg.py:109`) take any sequence, an integer ndarray
  included; the integer rule applies to each element, so `(2.0,)` raises `TypeError`.
* **Index arrays.** Integer dtypes are accepted; integral float arrays such as `[2.0]` stay
  accepted (pinned by `tests/test_dom_domains.py:149` and `tests/test_dom_spectra.py:24`).
  Non-integral floats and non-finite values raise `ValueError`. bool, str, object and
  complex arrays raise `TypeError`. One shared helper, used by:
  * `CellSet` `m`/`n` (`domains.py:161-175`) and `CellSet.bands`' band list
    (`domains.py:305`, an `np.asarray` cast like `m`/`n`);
  * `TabulatedSpectrum.at` bins (`spectra.py:66-68`);
  * the `rows` `idx` of `LinearMap` (`maps/base.py:177`), `TimeToWDM` (`time.py:96`),
    `TimeToFourier` (`time.py:216`), `FourierToWDM` (`fourier.py:339`) and
    `PushedCovariance` (`base.py:268`) — each casts on its own today;
  * `pad_maps`' and `crop_maps`' `k_out` and `j_in`;
  * the bands of `band_atoms`, `forward_bands` and `inverse_bands`, through
    `_check_interior_bands` (`transform.py:1047`).
* **Value arrays** (the dtype is the kind):
  * `TabulatedSpectrum.values` (`spectra.py:37`): real numeric dtypes only; complex, str,
    object and bool raise `TypeError`;
  * `CellSet.from_mask`'s `keep` (`domains.py:323`): a non-bool dtype raises `TypeError`
    (as `eq45_plan` already does, `wilson/evolutionary.py:477-478`); a wrong shape stays
    `ValueError`;
  * `testing.eq54_covariance`'s `s_I` and `s_G_src` (`testing.py:120`, `:122`): complex
    raises `TypeError`;
  * the "never complex" checks of §3.5.
* Known to change: `tests/test_dom_domains.py:145-148` (`(True,)` and the other wrong-dtype
  cases move from `ValueError` to `TypeError`; the non-integral and NaN cases stay
  `ValueError`).

### 2.4 Real scalars

`dt`, `t0`, `sigma0`, `fmin`, `rtol`, `tol`, `t_max`, `pcg`'s `true_rtol`: accepted if
`isinstance(x, numbers.Real)` and not a `bool`/`np.bool_` (so `int`, `float`, every NumPy
real scalar and `Fraction` pass; `bool` is a `numbers.Real`, hence the explicit exclusion).
`str`, complex, `Decimal`, `None` (where `None` is not a documented default) and arrays of
any shape, 0-d included, raise `TypeError` *(my call: `validate_dt` accepts `Decimal`, 0-d
and size-1 arrays today, the last with a NumPy `DeprecationWarning`)*. Range and finiteness
stay `ValueError`. Accepted values are stored as Python `float` (§1.5). Sites:
`domains._positive_float` (`domains.py:60-64`), `t0` (`domains.py:78`), `TimeToFourier`'s
`fmin` (`time.py:153`, `:177`; stored raw today, so `fmin="2"` gives a different `built_for`
key), `eq45_plan`'s `t_max` (`wilson/evolutionary.py:481`), `validate_dt` itself. Known to
change: `tests/wilson/test_wrong_kinds.py:66` (`np.array(0.5)` moves from
`test_real_dt_kinds_still_pass` to `test_dt_of_the_wrong_kind_raises_type_error`, kind
`ndarray`).

### 2.5 Flags and string options

* **Flags**: `bool` or `np.bool_` only, stored as `bool`; anything else (`"no"`, `1`, `0.5`,
  and `None` where it is not a documented default) raises `TypeError`. Sites:
  * `FourierToWDM` `restrict` (`fourier.py:264`), checked before the `built_for` key is
    formed;
  * `DenseCovariance` `symmetrize` (`dense.py:42`) and `dense_factor`'s `symmetrize`
    (`dense.py:21`);
  * `EvolutionaryCovariance` `symmetrize`, checked in `__init__`
    (`covariance/evolutionary.py:123-126`) for both assemblies (today it is read only for
    `assembly="dense"`, `:155`);
  * `Covariance.factor`'s `jit` (`covariance/base.py:382`; `None` or a bool);
  * `include_logdet` of the `wilson.inner_product` loglike functions (grep
    `include_logdet: bool`).
* **String options**: a non-string raises `TypeError`; an unknown string raises `ValueError`
  listing the choices (as today). Sites: `units`, `impl`, `chol_impl`, `layout`, `builder`,
  `assembly`, `on_fail`, `cells` (`wilson.inner_product`, `:82`), `backend`
  (`wilson.backend.get_backend`, `:503`; a `Backend` instance and `None` are also accepted)
  and `form` (the factors' `quad` and `quad_form`, `covariance/base.py:172`).

### 2.6 Domains, maps and other objects

* Every entry point that takes a domain, a `CellSet`, a map or a covariance checks
  `isinstance` against the concrete class and raises `TypeError` naming the expected kind,
  instead of the bare `AttributeError` at about 24 entry points today: the map constructors
  (`src`/`dst`, and `FourierToWDM.after`'s `inner`), `Restrict`, `BinSelect`, the `CellSet`
  class methods taking `grid` or `sub` (`domains.py:283`, `:297-326`, `:349`),
  `WDMDomain.full` (`:385`), `TabulatedSpectrum` `domain` and `.on` (`spectra.py:36`, `:53`),
  `ChannelStack`'s `covs`, `BandLayout.from_cells` (`linalg.py:125`), and
  `domainia.testing`'s `dense_W`, `wdm_congruence` and `eq45_dense` (`testing.py:76`)
  arguments.
* `Compose` requires two `LinearMap`s; `Identity` requires a domain; `GenericPSD` requires a
  callable.
* Channel names: the data classes' `channels` (`data.py:26-36`) must be a `ChannelAxis`;
  anything else — a tuple, a str or a NumPy scalar (accepted today as a one-name axis)
  included — raises `TypeError`. `ChannelAxis(names)` refuses a bare string with
  `TypeError` *(my call: `ChannelAxis("AE")` splits it into `("A", "E")` today)*.

### 2.7 JAX rebuilds

`jax.tree_util` rebuilds a map by calling its constructor, with placeholder values (`None`,
`object()`) for the arrays the map carries (`wilson/backend.py:454`,
`register_dataclass`). So the checks apply only to static fields the caller supplies
(counts, flags, options, domains) and to map arguments, never to the arrays a map carries;
derived static fields that default to the `_BUILD` sentinel are checked only after they are
filled. On every map class the fields §2 checks are static meta, and `Compose`'s `outer` and
`inner` are data whose rebuilds pass real maps (verified: `tree_map(lambda x: None, m)` and
`tree_map(lambda x: object(), m)` work on all 12 map classes and an `after()` map). A
`Compose` built from non-map parts (a JAX prefix-tree pattern with `is_leaf`) is refused; no
test or caller builds one.

### 2.8 The shape of `b`

* `Covariance.multiply` (`covariance/base.py:339-348`) compares `b`'s leading axes with
  `cov._b_shape()`: `(n,)`, or `(len(covs), n_c)` for a `ChannelStack` whose channels share
  `n_c`. A `ChannelStack` whose channels differ in `n` raises `ValueError` in `multiply` and
  in `StackFactor`'s methods, not at construction (`tests/test_operators_covariance.py:232`
  builds one and expects `CovarianceOperator`'s own message).
* A factor holds no covariance (`Covariance.factor` only attaches constants,
  `base.py:175-179`, `:390-391`), so each factor derives its own expected leading shape:
  `DiagFactor` `(v.shape[0],)`, `CholFactor` `(L.shape[0],)`, a ragged `BlockTriCovFactor`
  the sum of its blocks' sizes, a padded `BlockTriCovFactor` `(layout.n,)` from its
  constants (`BandLayout.n` is static meta). `StackFactor` checks
  `b.shape[0] == len(channels)` and passes each `b[c]` to its channel factor's own check.
* The check runs in `solve`, `whiten` and `quad` (every form: `"solve"`, `"whiten"`,
  `DiagFactor`'s `"elementwise"`, and `StackFactor.quad`, which loops over its channels
  without calling `solve`). A mismatch raises `ValueError`.
* This fixes three silent wrong answers: a `(1,)` `b` broadcast by `DiagonalCovariance`, an
  `(n+1,)` `b` truncated by `BlockTridiagonalCovariance`, a third row dropped by a
  two-channel `ChannelStack` — on the multiply path and on the likelihood path.
* `(n, k)` column batches stay supported (only the leading axes are compared); spec
  2026-11-04 §2.5's contract gains them. Shapes are static under `jit`, so the check also
  runs there.

### 2.9 Late and missing checks

* `EvolutionaryCovariance` checks `P` (integer, ≥ 1) and `t_max` (real, §2.4) at
  construction, for every builder, `builder="direct"` included.
* `to_coordinates` raises `ValueError` when the `sqrt(2)` scaling of a real or imaginary
  part overflows (`|X_k| > 1.27e308`; `translate.py:270-282`), instead of returning inf
  *(my call)*.
* `domainia.testing` follows the rule like the rest of the package *(my call)*.

### 2.10 Tests

* One wrong-kind test file per module group, modelled on `tests/wilson/test_wrong_kinds.py`
  (parametrised per slot kind, `pytest.raises(TypeError, match=...)`):
  `tests/test_wrong_kinds_domains.py`, `tests/test_wrong_kinds_maps.py`,
  `tests/test_wrong_kinds_covariance.py`, `tests/test_wrong_kinds_misc.py` (data, translate,
  spectra, variance, linalg, testing), and the new `wilson` sites added to
  `tests/wilson/test_wrong_kinds.py`. Each pins one accepted NumPy integer and one accepted
  NumPy float per slot kind.
* `tests/test_dom_cov_multiply.py`: one `ValueError` case per silent shape mode (§2.8) for
  `multiply`, and for `factor(nu)`'s `solve`, `whiten` and `quad` (forms `"solve"`,
  `"whiten"`, `"elementwise"`); a mixed-size `ChannelStack` refused in `multiply`; an
  `(n, k)` batch that still works.
* The JAX rebuild carve-out: `jax.tree_util.tree_map(lambda x: None, m)` and
  `tree_map(lambda x: object(), m)` still work on every map class.
* Known to change: `tests/test_dom_domains.py:145-148` (§2.3);
  `tests/wilson/test_wrong_kinds.py:66` (§2.4); `tests/test_dom_evolutionary_cov.py:101`
  and `tests/test_dom_variance_generic.py:77` (§2.2); `tests/wilson/test_window_threading.py`
  `SCANNED` (§2.2); the `Meyer d must be an int` and `must be an int;` messages (grep).

### 2.11 Order inside the section

1. `_checks.py`, its `SCANNED` entry, and the seven integer-check copies.
2. The silent conversions: `int()`, `float()`, `operator.index`, the flags of §2.5, `fmin`,
   channel names; the string options of §2.5 (`units`, `impl`, `chol_impl`, `layout`,
   `builder`, `assembly`, `on_fail`, `cells`, `backend`, `form`); `Compose`, `Identity`,
   `GenericPSD`.
3. `AttributeError` → `TypeError` at the domain, map, spectrum, covariance and testing
   entry points.
4. Index arrays and value arrays (§2.3), and the "never complex" checks (§3.5).
5. Shapes (§2.8) and the late checks (§2.9).
6. The message format (§2.2), then the four documents (spec 2026-10-31 §5,
   `README.md:165-167`, spec 2026-11-04 line 73, `test_wrong_kinds.py:1-2`) are re-read and
   corrected where they still overstate.

## 3. `DenseOperator`

### 3.1 `rtol`

One tolerance keeps both jobs. The docstring says: `rtol` drops directions with an
eigenvalue at or below `rtol · max|λ|`, and forgives negative eigenvalues down to
`-rtol · max|λ|`. New guards:

* `rtol >= 1` raises `ValueError` (it can never trigger the PSD check, since
  `max|λ| >= |λ_min|`).
* When the PSD check fails, `rtol` was given, and the smallest eigenvalue is within the
  default round-off tolerance (`λ_min >= -n · eps · max|λ|`, with `eps` by the input dtype
  as in §3.3), the message says that the eigenvalue is within round-off of zero and that the
  default `rtol=None` accepts it. Otherwise it is the plain "not positive semi-definite"
  message.
* The default stays NumPy's `matrix_rank` tolerance, `n · eps · max|λ|` *(my call)*, not
  scipy.stats' `1e6 · eps`.

### 3.2 Fail-open fixes

* `tol = top * (n * eps)` (no overflow to inf).
* Symmetrise as `0.5 * C + 0.5 * C.T` (no overflow).
* Non-finite eigenvalues from `eigh` raise `ValueError`. The check runs on `w` right after
  `eigh`, before `top` and `tol` are computed (otherwise `top = inf` makes `tol = inf` and
  the matrix is accepted with `dof = 0` again).
* An all-zero matrix raises `ValueError` *(my call)*. With the guards above it is the only
  way left to keep no direction, and its `solve`, `quadratic_form` and `log_determinant`
  would all be 0.

### 3.3 What is stored

`op.rtol` reads back the relative tolerance actually used: `n · eps32` for a float32 input
with `rtol=None`, `n · eps64` for other inputs with `rtol=None`, or the given `rtol`. So
`dataclasses.replace(op)` and `DenseOperator(op.matrix, op.domain, op.channel_names,
rtol=op.rtol)` reproduce the operator (same `matrix`, same kept eigenpairs; verified in
review). `op.rtol` is never `None` after construction; the docstring and the field
annotation say so. Known to change: any test reading `op.rtol is None` (grep).

### 3.4 The symmetry check

`max|C - C^T| <= f · max|C|`, as now, with `f = 1e-12` for float64 and other non-float32
inputs and `f = 1e-12 · eps32 / eps64` (about 5.4e-4) for float32 input, so an
unsymmetrised float32 product `B @ D @ B.T` is accepted (verified in review for `N` = 32 to
128 and `D` spanning 0 to 8 decades) while a float64 asymmetry of 1e-11 is still refused.
The docstring (`operators.py:53-54`) gives both factors, and the refusal message
(`:98-99`) prints the factor actually applied.

**Changed after design approval.** Part 1 proposed checking each entry against the
geometric mean of its two diagonal entries, as Enchilada does for its per-pixel channel
matrices, so that a loud coordinate cannot hide asymmetry in a quiet one. A probe refuted
it for domainia's dense matrices. For an exactly symmetric stationary `C` translated to WDM
(`operator_matrix(translate_operator(DenseOperator(C, time, ("A",)), wdm))`, and `W C W^T`
directly; grids 8×16 and 16×32), round-off alone gives a per-entry asymmetry of 1.7e-12 to
8.4e-12 once the PSD spans 6 decades, 5e-9 to 6e-8 at 10 decades and 3e-5 to 2.6e-4 at 14
decades, against a global asymmetry of 9e-17 to 4.9e-16 in every case. A per-entry rule at
1e-12 would refuse such translated covariances; LISA noise spans more than 6 decades. The
global check stays.

### 3.5 Error types ("never complex" checks)

Under §2.1, an array that can never be complex in that place raises `TypeError` for a
complex dtype. Site by site:

| site | today | decision |
|---|---|---|
| `operators.py:87-88` `DenseOperator` `matrix` | `TypeError` | stays |
| `operators.py:230-231` `CovarianceOperator` `nu` | `TypeError` | stays |
| `translate.py:351-352` `from_coordinates` `R` (and its docstring, `:338-341`) | `ValueError` | `TypeError` |
| `wilson/grid.py:314` `sanitize_psd` | `ValueError` | `TypeError` |
| `wilson/stationary.py:253` `_reject_complex_psd` | `ValueError` | `TypeError` |
| `wilson/transform.py:707`, `:771` `inverse`/`inverse_many` `wmn` | `ValueError` | `TypeError` |
| `wilson/transform.py:1199` `inverse_bands` `w` | `ValueError` | `TypeError` |
| `translate.py:135-136` complex values on a time or WDM domain | `ValueError` | stays: complex values are valid on a `FourierDomain`, so it is the domain that makes them wrong (pinned by `tests/test_translate.py:193-194`) |
| `wilson/windows.py:86`, `:210` complex window samples or ramp output | `WindowError` | stays, recorded as an exception to §2.1: they check what a window returns, and `WindowError` is the inadmissible-window error (`README.md:165`) |

Known to change: `tests/test_translate_coordinates.py:168-169`,
`tests/wilson/test_noise_stationary.py:226-238` (code and docstring),
`tests/wilson/test_grid_bands.py:300` (`match="w must be real"`), and the `wmn`-complex tests
the grep finds.

### 3.6 Tests

In `tests/test_operators_dense.py`:

* the PSD boundary with the default tolerance, on a diagonal matrix (`eigh` returns its
  entries exactly) with `max|λ| = 1` and `tol` computed by the same expression as the code:
  `λ_min = -1.0 · tol` accepted, `-1.01 · tol` refused; `+1.0 · tol` dropped, `+1.01 · tol`
  kept;
* `rtol=0` on a `B B^T` with several zero eigenvalues (the file's `_psd(n, n - 7)`, so
  round-off makes `λ_min` negative) refused with the round-off message (not rank `n-1`,
  which is accepted in about half of seeds); `-I` with `rtol=0` refused without the
  round-off wording;
* `rtol=1e-2` accepting `λ_min = -1e-3 · max|λ|` (documented forgiveness);
* `rtol=1` and `rtol=5` refused;
* the overflow cases: `-1.2e307 · I` at `n=16` refused as not PSD (the `tol` overflow);
  `diag(1e308)` at `n=16` accepted with `degrees_of_freedom == 16` and
  `matrix == 1e308 · I` (the symmetrisation overflow); `np.full((4, 4), 5e307)` on a
  4-sample one-channel domain refused with the non-finite-eigenvalue `ValueError` (its
  largest eigenvalue, 2e308, overflows in `eigh`);
* the zero matrix refused;
* `dataclasses.replace(op)` on a float32 rank-deficient operator gives the same `matrix` and
  `degrees_of_freedom`;
* an unsymmetrised float32 `B @ D @ B.T` accepted; a float64 asymmetry of 1e-11 still
  refused, with the message printing `1e-12`.

## 4. Packaging

The distribution stays `domainia` (user decision; the name is reworked once the package is
complete). The PyPI name belongs to an unrelated tool, so nothing may publish a dependency on
`domainia` resolved through PyPI until then (see "Parked for the Enchilada spec").

* `pyproject.toml`, `[tool.hatch.build.targets.sdist]`: replace `exclude` with
  `only-include = ["src", "tests"]`. hatchling still adds `README.md`, `LICENSE`,
  `pyproject.toml`, `.gitignore` and `PKG-INFO` (verified in a scratch copy with a planted
  `.superpowers/` and a stray file at the root: the file set equals a clean checkout's).
  `docs/` is then out without a rule of its own. Untracked files inside `src/` or `tests/`
  are still packed; the release check builds from a clean checkout.
* `.gitignore`: add `.superpowers/`.
* `tests/test_package.py:32-36`: check the `only-include` list (and that it names no `docs`
  path) instead of the `exclude` key; its docstring's "hatchling is not installed in the
  test env" stays true for `wdmrel`.
* Metadata *(my call)*: add the classifier `Typing :: Typed`; drop `jaxlib>=0.9` from the
  `jax` extra (`jax` 0.9 requires its matching `jaxlib` unconditionally); keep the version
  in two places (`pyproject.toml` and `__init__.py`), which `test_package.py:27-29` already
  checks; no project URLs while the repo is private.
* Recorded for the release, not done now:
  * The release check, as run on 2026-11-08: from a clean checkout, build the sdist and the
    wheel twice (identical files) with hatchling in a scratch environment, `twine check
    --strict` both, install the wheel with `--no-deps` into a scratch target and run the
    suite against it (printing `domainia.__file__`).
  * Python versions: only 3.12.4 has run the full suite; 3.13.7 passed a partial NumPy-lane
    run (191 passed, 11 skipped). `requires-python = ">=3.12"` stays.
  * Before any public release, decide whether the copies of the private `wdm` code under
    `tests/wilson/oracle/` may ship in a public sdist.

## 5. Test gaps

Tests only, except the `test_window_threading.py` item, which also changes `variance.py`,
`covariance/evolutionary.py` and `domains.py`.

* `tests/test_dom_domains.py:27`, `:62`, `:185`: add `match=` to the three unmatched
  `pytest.raises(TypeError)`.
* `tests/test_translate_coordinates.py:176-179`: pin `to_coordinates`' role word (a partial
  `WDMDomain` and a `wilson.Grid` passed as the domain), next to the `from_coordinates` pair.
* `tests/test_dom_cov_multiply.py:84`: add the padded `BlockTridiagonalCovariance` and
  `EvolutionaryCovariance` layouts to the never-dense test.
* `tests/test_dom_restored_coverage.py:63-66`: `jax.jit` of `WDMToTime.adjoint`; and a
  `jax.jit` of an `after()` map's adjoint (only `apply` is jitted today, at
  `tests/test_dom_maps_fourier_wdm.py:253-268`).
* `tests/test_dom_windows.py:497-499`: `MidpointVariance(psd, cells).on(JB)` has no leaves,
  rebuilds with equal cells, and gives the NumPy values.
* *(my call)* `tests/wilson/test_window_threading.py:45-66`: add `_cells_and_window` to
  `CONSUMERS`; pass `window=` by keyword at `variance.py:68` and
  `covariance/evolutionary.py:132`; give `MidpointVariance` (`variance.py:100`) a cells-only
  helper in `domains.py` (`WDMDomain` → `.cells`, `CellSet` passes, else the existing
  `TypeError` text). No behaviour changes.
* Left alone: `from .conftest import _oracle_paths` and `test_transform_shim.py` importing
  from `test_single_precision.py`. Both work under both pytest import modes.

## 6. Text

* **1.x tags.** Strip the 25 leading 1.x plan tags (24 module docstrings and one function
  docstring, `G5:` at `tests/wilson/test_evolutionary_kernels.py:119`; 24 files) and reword
  the four sentences that lean on a tag: `tests/test_dom_cov_api.py:1` ("section D-C
  gate"), `tests/test_dom_ev_ported.py:1` ("in C8 (part-1 plan note 7)"),
  `tests/wilson/test_noise_stationary.py:106` ("public since C4") and
  `tests/test_dom_db_imports.py:1` ("the section's public names", which leans on its
  leading tag). `tests/wilson/vendored/` is out of scope, as in the Nov 7 text sweep (its
  `S2` comment stays). Nothing reads the tags, and no document in the repo or its history
  defines them.
* **`dense_lambda_generic`** (`testing.py:9-11`, `:133-134`): docstring only. Λ is in the
  units of `U`: `Λ(U) = Λ(C) / κ²`, which is `N / (2 dt)` times the Wilson-unit Λ only for
  rfft-unit bins. Passing `FourierToWDM.C` gives Wilson units only for a map built directly
  over the window, not for a `restrict=True` or `after()` map. The module docstring's "its
  WDM form" is reworded to match the README's units rule. A test in
  `tests/test_dom_testing.py` checks `dense_lambda_generic(W.C, …)` against `push_rows` for
  a directly built `FourierToWDM` `W`.
* *(my call)* `wilson.evolutionary.eq45_separable_kernels` (`evolutionary.py:716-722`,
  the same unit-norm-`U` convention) gets the same one-line units note.
* **Toy-model ports.** Trim the two redundant repeats, `linalg.py:420` and
  `maps/fourier.py:308`. The "ported order" warnings stay: they guard bit-identical
  results.

## 7. Record

* Append a dated "Corrections (2026-11-08)" list to the Nov 7 plan's "Execution errata":
  * "22 files" (Known open points) is 24 files and 25 lines.
  * "The integer-kind check is written out five times in `domainia.wilson`" is six times in
    `wilson` plus `domains._int`.
  * "No `jax.jit` test of … an `after()` map" holds only for the adjoint; `apply` is jitted
    at `tests/test_dom_maps_fourier_wdm.py:253-268`.
  * The `DenseOperator` `rtol` question was raised after the final review and is settled by
    this spec (§3).
* Spec 2026-11-04 §1.7: add a one-line note that the text sweep (Task 15) deliberately
  dropped §1.7's request to cite the follow-up spec, because the zero-hit gate overrides it.
* Task 8's `solve` that does not reproduce bit for bit under pytest: the committed test
  compares `solve` to 1e-12 and cannot fail on a last-bit difference. The plan runs a
  scratch copy (outside the repo) of
  `tests/test_operators_covariance.py::test_a_reversed_channel_dict_meets_the_same_blocks`
  that compares `solve` with `np.testing.assert_array_equal`, 30 times under pytest with and
  without `VECLIB_MAXIMUM_THREADS=1`, and records both failure counts in its errata. The
  committed test keeps its 1e-12 tolerance.

## Gates for the whole change

* Full suite and NumPy lane (`WDM_TESTS_HIDE_JAX=1`) pass with no new warnings.
* `tests/wilson/meyer_cases.py --check` reproduces the baseline (1504 arrays).
* float64 bit identity, RESCALED families included: before the first change, a scratch
  script (outside the repo) records `meyer_cases`' `compute()` output at 779550d; after the
  last change, it compares the new tree's output with `np.array_equal` on all 1504 arrays, in
  the same environment on the same machine: 0 differ. (`--check` compares the RESCALED
  families — `FourierToWDM` dense/fft/adjoint, `forward_bands`, `inverse_bands` — only to
  1e-13, and §1.2 and §1.4 touch exactly those paths.)
* `git status --short` clean apart from the plan's own files.
* A built sdist (hatchling in a scratch environment; never built in the repo) contains no
  path outside `src/`, `tests/`, `README.md`, `LICENSE`, `pyproject.toml`, `.gitignore` and
  `PKG-INFO`, also with a stray untracked folder planted at the root of a scratch copy of the
  tree.

## Order

1. §2 (wrong kinds), in the order of §2.11. §3's checks use §2's helpers, and §1's
   real-taking maps use §2's dtype helper.
2. §3 (`DenseOperator`).
3. §1 (precision).
4. §4 (packaging).
5. §5 (test gaps).
6. §6 (text) and §7 (record), last.

## Parked for the Enchilada spec

* Enchilada's core CI lane pins `numpy==1.26.4` (`ci.yml:96`), while domainia needs
  `numpy>=2.1`. Spec 2026-10-31 §4's plan ("`transform_channels` becomes a call to
  `domainia.translate.transform`") cannot hold for the time ↔ frequency routes in that lane;
  domainia must be an optional, lazily imported dependency.
* Where Enchilada gets domainia from: a pinned commit, as it gets `wdm` today (deploy key,
  `--no-deps`, outside its `pyproject`). Never a published `Requires-Dist` on `domainia`,
  which PyPI resolves to the other tool; how that squares with spec 2026-10-31 §4's
  "optional `wdm` extra" is the Enchilada spec's decision.
* Whether a Wheel or `BlockState` may hold a generic domainia `Operator`
  (`block_state.py:59-65` and `wheel.py:87-93` are `isinstance` gates today).
* `translate_operator` returns the native operator itself on an identity translation
  (`translate.py:514-515`); the adapter must copy it.
* Enchilada raises `ValueError` for wrong-kind integers (`tests/test_domains.py:21-27`),
  domainia `TypeError`: the adapter validates first or translates the exception.
* Enchilada's float32 time → frequency keeps complex64 on NumPy ≥ 2 (complex128 at the core
  lane's 1.26.4 pin); whether to widen it.

## Non-goals

* No rename of the distribution, no publication, no CI.
* No change to the wilson kernels' single-precision rule beyond the output guard (§1.4).
* No change to `DenseOperator`'s default tolerance or to a split of `rtol`.
* No change to Enchilada.
