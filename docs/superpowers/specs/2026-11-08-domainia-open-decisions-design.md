# domainia 2.0 follow-ups II: the open decisions

Date: 2026-11-08. Status: approved design (three parts approved 2026-11-08); awaiting the
user's review of the written spec. `main` at 779550d, pushed to origin. This spec pushes
nothing.

**Test pin lists.** Each section names the tests it is known to change. The lists are a
starting point, not a census: the plan greps for every pattern a section changes (the
moved error messages, `pytest.raises(ValueError` next to a kind check, `.rtol`, `dtype=`,
`exclude`) and updates every hit.

**Sources.** The facts below come from a read-only survey of `main` at 779550d
(2026-11-08): one agent per area, each probing the code, and a completeness check that
re-ran the key claims. Its JSON is kept outside the repo. `file:line` references are to
779550d.

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
  (`maps/base.py:9-10`). For complex64 input:
  * `FourierToWDM(impl="fft")` returns float32; `impl="dense"`, `restrict=True` and
    `after()` maps return float64 (`fourier.py:322`, `z @ self.C.T` with complex128 `C`).
  * Every `FourierToWDM.adjoint`, `FourierZeroPad`, `FourierCrop` and their `Compose`
    return complex128/float64.
  * `TimeToWDM`, `WDMToTime`, `TimeToFourier.apply`, the crop, pad, restrict and select
    maps follow the input.
  * Three sites break even the result-type pattern: `BinSelect.adjoint`
    (`base.py:450`, `dtype=complex`) and `TimeToFourier.adjoint` (`time.py:208`) promote;
    `TimeToWDM.adjoint` and `WDMToTime.adjoint` go to float64 when `wilson.Grid` is given
    `dt=np.float64(...)` (`time.py:90`, `:146`), because `Grid` keeps the raw `dt` type.
  * NumPy and JAX (x64 on, eager or jit) agree row for row. With x64 off, every map that
    resolves a backend raises `RuntimeError` (`wilson/backend.py:525-540`).
  * `translate` and `operators` are float64 by contract (`tests/test_translate.py:130-140`).
* **Float32 fft overflow.** With `dt=1e4` and complex64 bins of amplitude 1e36,
  `FourierToWDM(impl="fft").apply` returns NaN with only a `RuntimeWarning`, although the
  true cells (max 1.7e33) fit float32. The overflow is at `wilson/transform.py:1155`
  (`phi.astype(rdtype) * z[..., ell]`); `max|phi|` scales with `dt` (400 at `dt=1e4`).
  The kernels check their input for finiteness, not their output.
* **Wrong kinds.** The rule is promised package-wide: spec 2026-10-31 §5 (lines 295-299),
  `README.md:165-167`; spec 2026-11-04 line 73 and `tests/wilson/test_wrong_kinds.py:1-2`
  say the rest of domainia already follows it. The probes show:
  * about 24 constructors and functions (about 30 argument slots) raise a bare
    `AttributeError` on a wrong-kind domain or object;
  * about 26 scalar sites convert silently (`int()` truncation, bare `float()` parsing
    strings, `operator.index` taking `True`, truthy flags);
  * `Compose`, `Identity` and `GenericPSD` accept anything;
  * `CellSet` index arrays and `TabulatedSpectrum.at` report wrong kinds as `ValueError`;
  * `EvolutionaryCovariance` checks `P` and `t_max` only at its first `factor()`;
  * no existing test pins any of this leniency, except the two integral-float index-array
    cases kept below (§2.3).
* **The integer check is written out seven times:** `wilson/grid.py:111` (`_require_int`),
  `:156` (`validate_grid`), `:363` (`n_samples`), `filter.py:70`, `noise.py:372`,
  `windows.py:176` (`Meyer.d`, its own message), and `domains.py:53` (`domains._int`, its
  own message). The import direction is core → `wilson` only.
* **`DenseOperator`** (`operators.py:48-154`):
  * one tolerance: refuse if `w[0] < -tol`, keep `w > tol`; `tol = top * n * eps`
    (`eps` float32's for a float32 matrix, float64's otherwise) or `rtol * top`;
  * an `rtol` of 1 or more accepts `-I` with `dof = 0`; an `rtol` below round-off refuses
    every rank-deficient PSD matrix as "not positive semi-definite";
  * `top * n * eps` overflows to inf for `top > DBL_MAX / n`: `-1.2e307 * I` is accepted
    with `dof = 0`;
  * `0.5 * (C + C.T)` overflows for entries above about 8.99e307 after `validate_finite`
    passed; `eigh` returns NaN eigenvalues, both comparisons are False, and the matrix is
    accepted with `dof = 0`;
  * the float32 `eps` is not stored: `dataclasses.replace(op)` on a float32-built
    rank-deficient operator raises "not positive semi-definite";
  * the symmetry check is `max|C - C^T| <= 1e-12 * max|C|` whatever the dtype, so an
    unsymmetrised float32 product `B @ D @ B.T` (asymmetry 5.2e-8) is refused;
  * scipy.stats `_PSD` shares one tolerance between its PSD check and its cut in the same
    way.
* **Packaging.** hatchling 1.27 builds the sdist and wheel cleanly and reproducibly (the
  wheel from the tree and the wheel from the sdist are byte-identical); `twine check
  --strict` passes on both; the suite run against the installed wheel passes (1768 + 1
  probe, 3 oracle skips). The sdist uses a deny list (`exclude = ["docs/superpowers"]`),
  so any untracked folder `.gitignore` does not cover is packed (`.superpowers/` was).
  The PyPI name `domainia` belongs to an unrelated tool (2.2.0, Python ≥ 3.13).
* **Enchilada** (`../enchilada` at 8a55ddb) does not import domainia. It has no JAX, always
  converts to float64 before its WDM backend, and keeps complex64 on time → frequency
  (`src/enchilada/_transforms.py:66-76`). domainia already matches it on the WDM layout,
  Wilson units and the Meyer transform (≤ 7.8e-16 against its pinned `wdm` 0.2.0).

## 1. Precision of the maps

### 1.1 The rule

A map that computes returns float64 or complex128, whatever precision it is given
(integers included). A map that only selects, slices or zero-fills returns its input's
dtype. In one sentence for `maps/base.py`'s module docstring:

> Maps that compute (`TimeToWDM`, `WDMToTime`, `TimeToFourier`, `FourierToWDM`,
> `FourierZeroPad`, `FourierCrop`, push) return float64/complex128 whatever the input's
> precision; maps that only select, slice or zero-fill (`Identity`, `CropTime`,
> `ZeroPadTime`, `Restrict`, `BinSelect`), and their adjoints, return the input's dtype.
> `Compose` returns what its parts return.

### 1.2 Sites

* `TimeToWDM.apply`/`adjoint`, `WDMToTime.apply`/`adjoint`, `TimeToFourier.apply`/
  `apply_complex`/`adjoint` (`maps/time.py:69-230`): convert the input to float64 (complex
  input to `apply_complex`: complex128) at entry, through the backend
  (`be.xp.asarray(x, dtype=...)`). Under JAX x64 is already required.
* `TimeToFourier.adjoint` (`time.py:208`): the spectrum is allocated complex128 as now; with
  the converted input this is now by the rule, not by accident.
* `FourierToWDM(impl="fft")` (`fourier.py:325-327`): convert `z` to complex128 before
  `forward_bands`. The dense path, `restrict`, `after()` and every adjoint already return
  double through `C`; no change beyond the docstring.
* `FourierZeroPad`, `FourierCrop`: already double through their complex128 constants; no
  change.
* `BinSelect.adjoint` (`base.py:450`): allocate in the input's complex counterpart
  (`be.xp.result_type(y.dtype, np.complex64)`), so complex64 stays complex64.
* The `Grid.dt` leak at `time.py:90` and `:146` goes away with the conversion; `wilson.Grid`
  is not changed.
* Docstrings: `maps/base.py:9-10` (the rule), `FourierToWDM` (`fourier.py:254-257`: both
  impls return float64; the single-precision fft route is gone), the module docstring of
  `wilson/transform.py` ("Where it reaches, and where it stops": the maps compute in double).

### 1.3 What does not change

* float64 results are bit for bit what they are now (converting float64 to float64 is a
  no-op). The Meyer baseline (`tests/wilson/meyer_cases.py --check`) is unchanged.
* The wilson kernels keep single precision when called directly (spec 2026-11-04 §3).
* `PushedCovariance` stays double.

### 1.4 The single-precision kernel guard *(my call)*

A single-precision run of a `domainia.wilson` transform entry point (the ten in spec
2026-11-04 §3's table, `forward_time` … `inverse_bands`, `_many` forms included) that
produces a non-finite value from finite input raises `ValueError` naming the entry point
and suggesting float64 input. The check runs only on concrete arrays (skipped for JAX
tracers, as `_check_finite` is, `transform.py:193-200`), and only when the working dtype is
single, so float64 runs gain no extra pass. The variance builders are not covered.

### 1.5 Safeguard shared with §2

`FourierDomain` keeps converting `dt` and `sigma0` with `float()` after its new kind check
(§2.4). If it kept the raw type, an `np.float64` `dt` would make `rfft_scale` a NumPy scalar
again. Under §1.2 the fft path converts to complex128 anyway, but `rfft_scale` stays a Python
float so that no other weak-type promotion depends on it.

### 1.6 Tests

* A new `tests/test_dom_map_dtypes.py`: one table over every public map method (forward,
  adjoint, `apply_complex`, `after()` maps, `Compose`, `restrict`) × input dtype (float32,
  float64, complex64, complex128 and an integer array where the map takes reals) → output
  dtype, on NumPy and on JAX with x64 on.
* A large-`dt` complex64 case: `FourierToWDM(impl="fft")` returns finite float64 cells.
* The kernel guard: `forward_bands` with complex64 bins of amplitude 1e36 at `dt=1e4` raises
  `ValueError`; the same call with complex128 succeeds.
* Known to change: the `FourierToWDM` complex64 expectation, if any test pins float32 from
  the fft path (grep `complex64` in `tests/test_dom_maps_fourier_wdm.py` and
  `tests/wilson/test_single_precision.py`).

## 2. Wrong kinds raise `TypeError` across the package

### 2.1 The rule

`TypeError` when an argument's type, or an array's dtype, can never be valid in that place.
`ValueError` when the type is right but the value is wrong: range, shape, finiteness,
integrality, or a string that is not one of the choices.

### 2.2 Helpers *(my call)*

A new private module `src/domainia/wilson/_checks.py` holds every kind and value check. It
imports nothing from `domainia` outside itself, so both `wilson` and the core can use it.

* The seven copies of the integer check (Context) are replaced by calls into it.
  `domains._int` becomes a thin wrapper that adds its range check.
* `validate_dt` and `validate_finite` keep their public names in `wilson/grid.py`,
  re-exported from `_checks`.
* One message format, `wilson`'s: `"{name} must be an integer, got {type}"`, and the same
  shape for the other kinds (`"{name} must be a real number, got {type}"`, `"{name} must be
  a bool, got {type}"`, `"{name} must be a {Expected}, got {type}"`). `Meyer.d` and
  `domains._int` lose their own wording; no test pins either.

### 2.3 Integers and index arrays

* Integers: Python `int` and NumPy integers are accepted. `bool`, `np.bool_`, any float
  (2.0 included), `str`, `None` (where `None` is not a documented default) and arrays raise
  `TypeError`. This includes the four
  `operator.index` sites that take `True` today (`CropTime` `start`/`n`, `time.py:243-244`;
  `ZeroPadTime` `n_extra`, `:273`; `FourierCrop` `start`, `fourier.py:153`), `pad_maps`'s `N`
  and `crop_maps`'s `start` (`fourier.py:54`, `:70`), and the `int()` sites:
  `DiagonalCovariance` `n` and `channel` (`covariance/diagonal.py:40-41`),
  `DenseCovariance` `n` (`dense.py:40`), `ChannelStack` `params_per_channel` (`stack.py:65`),
  `BlockTridiagonalCovariance` sizes (`blocktri.py:147`), `BinSelect`'s `(k0, n)`
  (`base.py:434`), `CellSet.bands`' band list (`domains.py:305`), `BandLayout.from_sizes`
  (`linalg.py:109`), `cblock_factor`'s `b` (`linalg.py:444`), `pcg`'s `maxit`
  (`linalg.py:507`, `:542`), `empirical_variance`'s `n_realizations` (`noise.py:444-447`).
* Index arrays (`CellSet` `m`/`n`, `TabulatedSpectrum.at` bins, `LinearMap.rows` `idx`):
  integer dtypes are accepted; integral float arrays such as `[2.0]` stay accepted (pinned
  by `tests/test_dom_domains.py:149` and `tests/test_dom_spectra.py:24`). Non-integral
  floats and NaN raise `ValueError`. bool, str, object and complex arrays raise `TypeError`.
  Known to change: `tests/test_dom_domains.py:145-148` (`(True,)` and the other wrong-dtype
  cases move from `ValueError` to `TypeError`; the non-integral and NaN cases stay
  `ValueError`).

### 2.4 Real scalars

`dt`, `t0`, `sigma0`, `fmin`, `rtol`, `tol`, `t_max`, `pcg`'s `true_rtol`: accepted if
`isinstance(x, numbers.Real)` and not a `bool`/`np.bool_` (so `int`, `float`, NumPy real
scalars and `Fraction` pass). `str`, complex, `Decimal`, `None` (where `None` is not a
documented default) and arrays of any shape raise `TypeError` *(my call: `validate_dt`
accepts `Decimal`, 0-d and size-1 arrays today, the last with a NumPy
`DeprecationWarning`)*. Range and finiteness stay `ValueError`. Accepted values are stored
as Python `float` (§1.5). Sites: `domains._positive_float` (`domains.py:60-64`), `t0`
(`domains.py:78`), `TimeToFourier`'s `fmin` (`time.py:153`, `:177`; stored raw today, so
`fmin="2"` gives a different `built_for` key), `eq45_plan`'s `t_max`
(`wilson/evolutionary.py:481`), `validate_dt` itself.

### 2.5 Flags and string options

* Flags (`FourierToWDM` `restrict`, `fourier.py:264`; `DenseCovariance` and
  `EvolutionaryCovariance` `symmetrize`, `dense.py:21`, `:42`): `bool` or `np.bool_` only,
  stored as `bool`; anything else (`"no"`, `1`, `0.5`, `None`) raises `TypeError`. The check
  runs before `FourierToWDM` forms its `built_for` key.
* String options (`units`, `impl`, `chol_impl`, `layout`, `builder`, `assembly`, `on_fail`):
  a non-string raises `TypeError`; an unknown string raises `ValueError` listing the
  choices (as today).

### 2.6 Domains, maps and other objects

* Every entry point that takes a domain, a `CellSet`, a map or a covariance checks
  `isinstance` against the concrete class and raises `TypeError` naming the expected kind,
  instead of the bare `AttributeError` at about 24 entry points today: the ten map
  constructors and `FourierToWDM.after` (`src`/`dst`/`inner`), `Restrict`, `BinSelect`,
  the `CellSet` class methods taking `grid` or `sub` (`domains.py:283`, `:297-326`, `:349`),
  `WDMDomain.full` (`:385`), `TabulatedSpectrum` `domain` and `.on` (`spectra.py:36`, `:53`),
  `ChannelStack`'s `covs`, and `domainia.testing`'s `dense_W`, `wdm_congruence` and
  `eq54_covariance` arguments.
* `Compose` requires two `LinearMap`s; `Identity` requires a domain; `GenericPSD` requires a
  callable.
* Channel names (`ChannelAxis`, the data classes' `channels`, `data.py:26-36`): a bare
  string raises `TypeError` *(my call: `ChannelAxis("AE")` splits it into `("A", "E")`
  today)*, and so does a NumPy scalar (accepted today as a one-name axis).

### 2.7 JAX rebuilds

`jax.tree_util` rebuilds a map by calling its constructor, with placeholder values (`None`,
`object()`) for the arrays the map carries (`wilson/backend.py:454`,
`register_dataclass`). So the checks apply only to static fields (counts, flags, options,
domains, `built_for`) and to map arguments, never to the arrays a map carries. A `Compose`
built from non-map parts (a JAX prefix-tree pattern with `is_leaf`) is refused; no test or
caller builds one (a spy over the 122 `Compose`-using tests saw only map pairs).

### 2.8 The shape of `b` in `multiply` and `solve`

`Covariance.multiply` (`covariance/base.py:339-348`) and every factor's `solve`
(`DiagFactor`, the dense factor, `BlockTriCovFactor`, `StackFactor`) check `b`'s leading
axes against the covariance's expected shape — `(n,)`, or `(len(covs), n_c)` for a
`ChannelStack` (a `_b_shape()` hook) — and raise `ValueError` on a mismatch. This fixes
three silent wrong answers: a `(1,)` `b` broadcast by `DiagonalCovariance`, an `(n+1,)` `b`
truncated by `BlockTridiagonalCovariance`, a third row dropped by a two-channel
`ChannelStack`. `(n, k)` column batches stay supported (only the leading axes are
compared); spec 2026-11-04 §2.5's contract gains them. Shapes are static under `jit`, so
the check also runs there.

### 2.9 Late and missing checks

* `EvolutionaryCovariance` checks `P` and `t_max` at construction, not at the first
  `factor()`.
* `to_coordinates` raises `ValueError` when the `sqrt(2)` scaling of a real or imaginary
  part overflows (`|X_k| > 1.27e308`; `translate.py:270-282`), instead of returning inf
  *(my call)*.
* `domainia.testing` follows the rule like the rest of the package *(my call)*.

### 2.10 Tests

* One wrong-kind test file per module group, modelled on `tests/wilson/test_wrong_kinds.py`
  (parametrised `pytest.raises(TypeError, match=...)` over `True`, `2.0`, `2.7`, `"2"`,
  `None`, a wrong domain): `tests/test_wrong_kinds_domains.py`,
  `tests/test_wrong_kinds_maps.py`, `tests/test_wrong_kinds_covariance.py`,
  `tests/test_wrong_kinds_misc.py` (data, translate, spectra, variance, linalg, testing).
  Each also pins one accepted NumPy integer and one accepted NumPy float.
* `tests/test_dom_cov_multiply.py`: one `ValueError` case per silent shape mode (§2.8) for
  `multiply` and for `factor(nu).solve`, plus an `(n, k)` batch that still works.
* The JAX rebuild carve-out: `jax.tree_util.tree_map(lambda x: None, m)` and
  `tree_map(lambda x: object(), m)` still work on every map class.
* Known to change: `tests/test_dom_domains.py:145-148` (§2.3); the `Meyer d must be an int`
  and `must be an int;` messages (grep).

### 2.11 Order inside the section

1. `_checks.py` and the seven integer-check copies.
2. The silent conversions: `int()`, `float()`, `operator.index`, flags, `fmin`, channel
   names; `Compose`, `Identity`, `GenericPSD`.
3. `AttributeError` → `TypeError` at the domain, map, spectrum and covariance entry points.
4. Array dtypes (§2.3) and `DenseOperator`'s neighbours (§3.5).
5. `multiply`/`solve` shapes (§2.8) and the late checks (§2.9).
6. The four documents (spec 2026-10-31 §5, `README.md:165-167`, spec 2026-11-04 line 73,
   `test_wrong_kinds.py:1-2`) are re-read and corrected where they still overstate.

## 3. `DenseOperator`

### 3.1 `rtol`

One tolerance keeps both jobs. The docstring says: `rtol` drops directions with an
eigenvalue at or below `rtol · max|λ|`, and forgives negative eigenvalues down to
`-rtol · max|λ|`. New guards:

* `rtol >= 1` raises `ValueError` (it can never trigger the PSD check, since
  `max|λ| >= |λ_min|`).
* When the PSD check fails, `rtol` was given, and `rtol · max|λ|` is below the default
  round-off tolerance (`n · eps · max|λ|`), the message says that the eigenvalue is within
  round-off of zero and that the default `rtol=None` accepts it, instead of only "not
  positive semi-definite".
* The default stays NumPy's `matrix_rank` tolerance, `n · eps · max|λ|` *(my call)*, not
  scipy.stats' `1e6 · eps`.

### 3.2 Fail-open fixes

* `tol = top * (n * eps)` (no overflow to inf).
* Symmetrise as `0.5 * C + 0.5 * C.T` (no overflow).
* Non-finite eigenvalues from `eigh` raise `ValueError`.
* An all-zero matrix raises `ValueError` *(my call)*. With the guards above it is the only
  way left to keep no direction, and its `solve`, `quadratic_form` and `log_determinant`
  would all be 0.

### 3.3 What is stored

`op.rtol` reads back the relative tolerance actually used: `n · eps32` for a float32 input
with `rtol=None`, `n · eps64` for other inputs with `rtol=None`, or the given `rtol`. So
`dataclasses.replace(op)` and `DenseOperator(op.matrix, op.domain, op.channel_names,
rtol=op.rtol)` reproduce the operator (same `matrix`, same kept eigenpairs). `op.rtol` is
never `None` after construction; the docstring and the field annotation say so. Known to
change: any test reading `op.rtol is None` (grep).

### 3.4 The symmetry check

`max|C - C^T| <= f · max|C|`, as now, with `f = 1e-12` for float64 and other non-float32
inputs and `f = 1e-12 · eps32 / eps64` (about 5.4e-4) for float32 input, so an
unsymmetrised float32 product `B @ D @ B.T` is accepted.

**Changed after design approval.** Part 1 proposed checking each entry against the
geometric mean of its two diagonal entries, as Enchilada does for its per-pixel channel
matrices, so that a loud coordinate cannot hide asymmetry in a quiet one. A probe refuted
it for domainia's dense matrices: for an exactly symmetric stationary `C` translated to WDM
(`operator_matrix(translate_operator(DenseOperator(C, time, ("A",)), wdm))`, and `W C W^T`
directly), round-off alone gives per-entry asymmetry of 2e-12 to 8e-12 once the PSD spans
6 decades (grids 8×16 and 16×32), 3e-8 to 6e-8 at 10 decades and 1.7e-4 at 14 decades,
against a global asymmetry of 1e-16 to 4e-16 in every case. A per-entry rule at 1e-12
would refuse every such translated covariance; LISA noise spans more than 6 decades. The
global check stays.

### 3.5 Error types

Under §2.1 a complex `matrix` (`operators.py:87-88`) and a complex `CovarianceOperator.nu`
(`:229-231`) stay `TypeError`. `from_coordinates`' complex `R` (`translate.py:349-352`)
moves from `ValueError` to `TypeError`, and so does every other "never complex" array check
the grep finds (§2.11 step 4).

### 3.6 Tests

In `tests/test_operators_dense.py`:

* the PSD boundary with the default tolerance: `λ_min = -1.0 · tol` accepted, `-1.01 · tol`
  refused; `+1.0 · tol` dropped, `+1.01 · tol` kept;
* `rtol=0` on a rank-deficient `B B^T` refused with the round-off message;
* `rtol=1e-2` accepting `λ_min = -1e-3 · max|λ|` (documented forgiveness);
* `rtol=1` and `rtol=5` refused;
* both overflow cases (`-1.2e307 · I` at `n=16`; `diag(1e308)`) refused;
* the zero matrix refused;
* `dataclasses.replace(op)` on a float32 rank-deficient operator gives the same `matrix` and
  `degrees_of_freedom`;
* an unsymmetrised float32 `B @ D @ B.T` accepted; a float64 asymmetry of 1e-11 still
  refused.

## 4. Packaging

The distribution stays `domainia` (user decision; the name is reworked once the package is
complete). The PyPI name belongs to an unrelated tool, so nothing may publish a dependency on
`domainia` resolved through PyPI until then (see "Parked for the Enchilada spec").

* `pyproject.toml`, `[tool.hatch.build.targets.sdist]`: replace `exclude` with
  `only-include = ["src", "tests"]`. hatchling still adds `README.md`, `LICENSE`,
  `pyproject.toml`, `.gitignore` and `PKG-INFO` (verified in a scratch copy with a planted
  `.superpowers/` and a stray file: the file set equals a clean checkout's). `docs/` is then
  out without a rule of its own.
* `.gitignore`: add `.superpowers/`.
* `tests/test_package.py:32-36`: check the `only-include` list (and that it names no `docs`
  path) instead of the `exclude` key.
* Metadata *(my call)*: add the classifier `Typing :: Typed`; drop `jaxlib>=0.9` from the
  `jax` extra (`jax` pins its own `jaxlib`); keep the version in two places
  (`pyproject.toml` and `__init__.py`), which `test_package.py` already checks; no project
  URLs while the repo is private.
* Recorded for the release, not done now:
  * The release check, as run on 2026-11-08: build the sdist and the wheel twice (identical
    files), `twine check --strict` both, install the wheel with `--no-deps` into a scratch
    target and run the suite against it (printing `domainia.__file__`).
  * Python versions: only 3.12.4 has run the full suite; 3.13.7 passed a partial NumPy-lane
    run (191 passed, 11 skipped). `requires-python = ">=3.12"` stays.
  * Before any public release, decide whether the copies of the private `wdm` code under
    `tests/wilson/oracle/` may ship in a public sdist.

## 5. Test gaps

Tests only, except the last item.

* `tests/test_dom_domains.py:27`, `:62`, `:185`: add `match=` to the three unmatched
  `pytest.raises(TypeError)`.
* `tests/test_translate_coordinates.py:176-179`: pin `to_coordinates`' role word (a partial
  `WDMDomain` and a `wilson.Grid` passed as the domain), next to the `from_coordinates` pair.
* `tests/test_dom_cov_multiply.py:84`: add the padded `BlockTridiagonalCovariance` and
  `EvolutionaryCovariance` layouts to the never-dense test.
* `tests/test_dom_restored_coverage.py:63-66`: `jax.jit` of `WDMToTime.adjoint`; and a
  `jax.jit` of an `after()` map's adjoint (only `apply` is jitted today, at
  `tests/test_dom_maps_fourier_wdm.py:253-270`).
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
  the four sentences that lean on a tag mid-sentence. Nothing reads the tags, and no
  document in the repo or its history defines them.
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
    at `tests/test_dom_maps_fourier_wdm.py:253-270`.
  * The `DenseOperator` `rtol` question was raised after the final review and is settled by
    this spec (§3).
* Spec 2026-11-04 §1.7: add a one-line note that the text sweep (Task 15) deliberately
  dropped §1.7's request to cite the follow-up spec, because the zero-hit gate overrides it.
* Task 8's `solve` that does not reproduce bit for bit under pytest: the plan runs
  `tests/test_operators_covariance.py::test_a_reversed_channel_dict_meets_the_same_blocks`
  30 times with and without `VECLIB_MAXIMUM_THREADS=1` and records the failure counts in its
  errata. The 1e-12 tolerance stays either way.

## Gates for the whole change

* Full suite and NumPy lane (`WDM_TESTS_HIDE_JAX=1`) pass with no new warnings.
* `tests/wilson/meyer_cases.py --check` reproduces the baseline (1504 arrays): float64
  results unchanged bit for bit.
* `git status --short` clean apart from the plan's own files.
* A built sdist (scratch directory, never the repo) contains no path outside
  `src/`, `tests/`, `README.md`, `LICENSE`, `pyproject.toml`, `.gitignore` and `PKG-INFO`,
  also with a stray untracked folder planted in a scratch copy of the tree.

## Order

1. §2 (wrong kinds), in the order of §2.11. §3's checks use §2's helpers.
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
  which PyPI resolves to the other tool.
* Whether a Wheel or `BlockState` may hold a generic domainia `Operator`
  (`block_state.py:59-65` and `wheel.py:87-93` are `isinstance` gates today).
* `translate_operator` returns the native operator itself on an identity translation
  (`translate.py:514-515`); the adapter must copy it.
* Enchilada raises `ValueError` for wrong-kind integers (`tests/test_domains.py:21-27`),
  domainia `TypeError`: the adapter validates first or translates the exception.
* Enchilada's float32 time → frequency keeps complex64; whether to widen it.

## Non-goals

* No rename of the distribution, no publication, no CI.
* No change to the wilson kernels' single-precision rule beyond the output guard (§1.4).
* No change to `DenseOperator`'s default tolerance or to a split of `rtol`.
* No change to Enchilada.
