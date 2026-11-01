# domainia 2.0: a general time / frequency / time-frequency library

Date: 2026-10-31. Status: approved design, awaiting spec review.

## Goal

Turn this copy of domainia (the WDM Paper II release's middle layer) into a general
library that moves data and covariance between the time, Fourier and Wilson
time-frequency domains. The Wilson bases live in the subpackage `domainia.wilson`. The
WDM transform is the Wilson basis with the Meyer window. Other admissible windows give
other Wilson bases. domainia becomes the translation layer behind
`~/Documents/lisa/enchilada`.

## Context (state at baseline commit d991ac4)

* `src/domainia` (1.1.0) refuses to import unless `wdm.__release__ == "wdm-paper-1"`.
  Its pytest `pythonpath` resolves `../wdm/src`, which is the unrelated `lisa/wdm` 0.2.0.
  Seven test files import `scripts`, `golden/` or `data/` from the old release root, so
  the suite does not run standalone.
* `wilson/` is the `wdm-paper` 1.1.0 distribution (import name `wdm`). It passes 407
  tests standalone (17 skipped for missing release data). `forward_many` maps stacked
  rffts `(C, N/2+1)` to a dense `(C, Nf+1, Nt)` grid with inactive edge entries at
  round-off. That matches enchilada's `WDMGrid`. ‖w‖² = dt·‖x‖². No JAX on import.
* The Meyer window enters every consumer as one sampled array,
  `filter.phi_centered(Nf, Nt, dt)`: 13 call sites in `transform.py`, `stationary.py`,
  `noise.py`, `domainia/variance.py` and `domainia/maps/fourier.py`. No consumer relies
  on an analytic Meyer property.
* enchilada's `_transforms._load_wdm` needs an external `wdm` with
  `forward_many`/`inverse_many` and Python ≥3.13. `lisa/wdm` 0.2.0 no longer defines
  them in `transform.py`.

## Decisions

| Question | Decision |
|---|---|
| Package identity | General library. Paper-only code is deleted, not moved. |
| Seed for `domainia.wilson` | The bundled wdm-paper code (`wilson/src/wdm`), keeping its short names (`Nf`, `Nt`, `dt`). |
| Windows | Compact support only (inside \|f\| < ΔF). Meyer is parametrised by `d` only. A user window must meet the same admissibility conditions. |
| enchilada boundary | domainia owns the array transforms and lazy covariance change of basis. enchilada keeps `L1Data`, `DataCovariance`, `BlockState` and becomes a thin adapter. |
| Approach | Restructure in place. `wdm` becomes `domainia.wilson`, `window=` is threaded through the kernels, and `domainia.translate` is new. |
| Model-specific tier | Deleted (see the cut below). |
| translate API shape | Takes and returns `{channel: array}` dicts. |
| enchilada changes | A separate follow-on spec and plan in the enchilada repo. |
| Version | `domainia` 2.0.0. |

Defaults: Python ≥3.12 (enchilada's floor; the wdmrel env is 3.12.4), numpy ≥2.1,
scipy ≥1.14.1. JAX is an optional extra and `import domainia` never imports it. The
frequency domain defaults to enchilada's `dt·rfft` units.

## 1. Layout and the cut

One hatchling distribution, `domainia`. `wilson/pyproject.toml` and the separate
`wdm-paper` distribution are removed.

```
src/domainia/
  __init__.py          no release guard, no sentinels; never imports JAX
  wilson/              ← wilson/src/wdm, every `wdm` import rewritten to `domainia.wilson`
    __init__.py        public API as today, plus Window / Meyer / RampWindow / SampledWindow / MEYER
    backend.py         NumPy/JAX shim (unchanged)
    grid.py            Grid(Nf, Nt, dt), layout, active masks, packed cells (unchanged)
    windows.py         NEW (section 2)
    filter.py          nu_d, phi_centered, phi_fft_order kept as Meyer helpers
    transform.py, matrix.py, stationary.py, noise.py, evolutionary.py,
    inner_product.py, special.py
  domains.py, data.py  TimeDomain / FourierDomain / CellSet / WDMDomain / ChannelAxis
  maps/                base, time, fourier
  covariance/          base, diagonal, dense, blocktri, stack, evolutionary (was eq45)
  spectra.py           slimmed to GenericPSD and TabulatedSpectrum
  linalg.py, variance.py, testing.py
  translate.py         NEW (section 3)
tests/
  wilson/              the wilson tests, with vendored/ and oracle/
  test_*.py            domainia tests
```

**Deleted, paper-only:** `literal/` (congruence, eq45_direct, masked, pad, time),
`assets.py`, `wilson/src/wdm/paper.py`, the `wdm-paper-1` import guard,
`__release__` / `__wdm_api__`.

**Deleted, model-specific:**
* `covariance/{exact_time, vandermonde, exact_fourier, hermitian, pushed, edges}.py`
* `nuisance.py`, `sampling.py`, `likelihood.py`, `density.py`

**Slimmed, with the model-specific parts deleted:**
* `spectra.py`: `LogLinearPSD`, `SeparablePSD` and `Envelope` go. `GenericPSD` (any
  broadcasting `S(f, t, ν)`) and `TabulatedSpectrum` stay, because `variance.py` and
  the evolutionary covariance need a spectrum type.
* `covariance/eq45.py` becomes `covariance/evolutionary.py` with class
  `EvolutionaryCovariance`. It keeps the `interp` and `direct` builders (any callable
  PSD). The `bessel`, `gram` and `separable` builders go: they accept only the deleted
  PSD classes.
* `variance.py`: the `LogLinearPSD` and `SeparablePSD` branches go. The `GenericPSD`
  branches of `BandAveragedVariance` and `MidpointVariance` stay.
* `covariance/__init__.py` and `domainia/__init__.py` export only what survives.

`domainia.wilson` keeps every stateless kernel, including the
`eq45_coeffs_bessel` / `eq45_blocks_gram` / `eq45_separable_*` functions in
`evolutionary.py`. They are self-contained, have their own tests, and take arrays rather
than the deleted classes.

**Tests.** Each test file is classified by what it exercises. The implementation plan
records the class of every file, and its first step re-derives the class from the
file's imports and assertions.

* **(a) Only deleted code: delete.** This covers `wilson/tests/test_paper.py`,
  `wilson/tests/test_package_identity.py` (replaced by a 2.0 identity test),
  `wilson/tests/test_wd_rounding.py` (release data), `tests/test_dom_package.py`
  (sentinels; replaced), `test_dom_assets.py`, the five `test_dom_literal_*.py` and
  `dom_literal_cases.py`.
  It also covers the tests whose subject is a deleted module:
  `test_dom_{cov_vandermonde, density, exact_fourier_constants, exact_time, hermitian,
  likelihood, nuisance, pushed_rows, push_exact, sampling, schur_edge,
  schur_scaled_batching, whittle, whittle_apply_data, zero_padded}.py`, and those whose
  subject is a deleted builder (`test_dom_eq45_separable.py`). Files whose subject the
  imports leave ambiguous (`test_dom_pad_mask`, `test_dom_pad_reduce`,
  `test_dom_projected_stack`, `test_dom_consistency`) are read in the plan and placed
  in (a) or (b).
* **(b) Surviving code with deleted fixtures: rewrite the fixtures, keep the
  assertions.** Fixtures built from `LogLinearPSD`, `SeparablePSD`, `nuisance`,
  `dom_toys` or the release goldens are replaced by `GenericPSD` or seeded synthetic
  inputs. `dom_db_inputs.py` (imports `scripts.golden`) and `domtest_inputs.py` become
  one synthetic fixture module, which feeds `test_dom_maps_*`, `test_dom_spectra`,
  `test_dom_variance` and `test_dom_db_imports`. `test_dom_eq45`, `test_dom_cov_api`,
  `test_dom_cov_base`, `test_dom_keyed_jit`, `test_dom_factor_constants` and
  `test_dom_g6_linalg` are in (b) as well. An assertion that only compared against a
  paper golden number is dropped; its general property (symmetry, factorisation round
  trip, dense-oracle agreement) is asserted against `domainia.testing` instead.
* **(c) Clean: keep, with imports rewritten.** The remaining wilson tests (with
  `vendored/` and `oracle/`) and the domainia tests of domains, data, linalg,
  blocktri, testing and maps base.

## 2. Windows (`domainia/wilson/windows.py`)

The fast slab spans exactly `Nt` rfft bins: f ∈ [−ΔF, ΔF) with ΔF = 1/(2·Nf·dt). A
window φ̂ is **admissible** when:

1. φ̂ is real and even;
2. φ̂(f) = 0 for |f| ≥ ΔF. This keeps the slab transform exact and every band
   covariance tridiagonal (|m − m'| ≤ 1);
3. it forms a partition of unity on the lattice: φ̂(f)² + φ̂(f − ΔF)² = 1/ΔF for
   f ∈ [0, ΔF]. This is the Hz normalization `phi_centered` uses, and it makes the atoms
   orthonormal (W Wᵀ = dt·I on the active cells).

```python
class Window(Protocol):              # frozen, hashable: static under jax.jit
    name: str
    def sample(self, Nf: int, Nt: int, dt: float) -> NDArray[np.float64]: ...
        # centered (Nt,) array in phi_centered's ordering and normalization

@dataclass(frozen=True)
class Meyer:                 # flat to ΔF/4, cos(π/2 · ν_d) rolloff to 3ΔF/4
    d: int = 4               # ν_d = I_x(d, d); d ≥ 1
@dataclass(frozen=True)
class RampWindow:            # Meyer's geometry with a user ramp
    ramp: Callable[[ndarray], ndarray]   # ν on [0, 1], ν(0)=0, ν(1)=1, ν(x)+ν(1−x)=1
    name: str
@dataclass(frozen=True)
class SampledWindow:         # any φ(u), u = f/ΔF, in the Hz normalization
    fn: Callable[[ndarray], ndarray]
    name: str

MEYER = Meyer()
class WindowError(ValueError): ...
```

* `Meyer(d).sample(Nf, Nt, dt)` returns `filter.phi_centered(Nf, Nt, dt, d)`
  bit for bit.
* `sample` validates each result once per `(window, Nf, Nt, dt)` and caches it in a
  bounded LRU. It checks evenness, support (exactly 0 at the f = −ΔF bin) and the
  partition of unity to 1e-12 relative. A failure raises `WindowError` naming the
  condition, the window and the worst bin. The returned array is read-only.
* `RampWindow` and `SampledWindow` compare by `(name, fn identity)`. Two windows built
  from the same function and name are equal.
* Threading: every function that calls `phi_centered(Nf, Nt, dt)` today takes a
  keyword `window: Window = MEYER` and calls `window.sample(Nf, Nt, dt)` instead. That
  covers the transforms and their `_many` / `_time` variants, `base_atom_spectra`,
  `atom_spectrum`, `band_atoms`, `forward_bands`, `matrix.build_W` and friends, the
  stationary covariance builders, the noise variance grids, `variance.py` and
  `maps/fourier.py`. Callers that build on these (`maps.TimeToWDM`, `FourierToWDM`,
  `EvolutionaryCovariance`, inner products) take the window from their `WDMDomain` or
  pass it through.
* `WDMDomain` gains `window: Window = MEYER`, which takes part in equality and hashing.
  Maps and covariances built for one window raise `ValueError` on a domain with another.

## 3. `domainia.translate`

NumPy only. It takes and returns `{channel: ndarray}` dicts with insertion order
preserved.

**Domains.** The existing classes plus two constructors:

```python
TimeDomain(N, dt, t0=0.0)                         # real (N,) per channel
FourierDomain.full(N, dt)                         # k0=0, n=N//2+1, units="dt_rfft"; complex per channel
WDMDomain.full(Grid(Nf, Nt, dt), window=MEYER)    # all active cells, scale 1; dense (Nf+1, Nt) per channel
```

`translate` handles only these full-record domains, with WDM in the dense layout.
Partial cell sets in packed band order stay with `domainia.maps`.

**Arrays.**

```python
def transform(channel_data: Mapping[str, ndarray], source, target) -> dict[str, ndarray]
```

* Covers all 9 source/target pairs. Same domain returns copies. WDM → WDM across a
  different `(Nf, Nt)` or window goes through the spectrum.
* Inputs are refused (`ValueError`) for an empty dict, mismatched channel shapes, the
  wrong shape for `source`, non-finite values, `N` disagreeing between source and
  target, a non-zero inactive WDM entry, or a non-real DC/Nyquist bin beyond 1e-12 of
  the channel's spectral norm. A non-dict or non-domain argument raises `TypeError`.
* Time → WDM promotes to float64. Fourier input is normalised in at least complex128
  before dividing by `dt`, as enchilada does today.
* Outputs are fresh float64 / complex128 arrays. Inactive WDM entries are set to exactly
  0 and DC/Nyquist made exactly real. A non-finite result raises `ValueError`.

**Covariance as an operator.**

```python
class Operator(Protocol):
    domain: TimeDomain | FourierDomain | WDMDomain
    channel_names: tuple[str, ...]
    degrees_of_freedom: int
    def apply(self, values: dict) -> dict: ...
    def solve(self, values: dict) -> dict: ...
    def project(self, values: dict) -> dict: ...
    def quadratic_form(self, values: dict) -> float: ...
    def log_determinant(self) -> float: ...

def log_coordinate_scale(domain) -> float
    # time 0; WDM ½·log dt (any admissible window); frequency log dt + ½·log N
def translate_operator(op: Operator, target) -> Operator
```

`translate_operator` returns a `TranslatedOperator` that holds one native operator.
Translating a `TranslatedOperator` again re-targets that native operator rather than
nesting wrappers. A target equal to the native domain returns the native operator
itself. With Δ = `log_coordinate_scale(target) − log_coordinate_scale(native)`:

* `apply(x)` = to_target(native.apply(to_native(x))) · e^{2Δ}
* `solve(x)` = to_target(native.solve(to_native(x))) · e^{−2Δ}
* `project(x)` = to_target(native.project(to_native(x)))
* `quadratic_form(x)` = native.quadratic_form(to_native(x))
* `log_determinant()` = native.log_determinant() + 2 · degrees_of_freedom · Δ

`to_native` and `to_target` are `transform`. The real-coordinate convention (√2·Re/Im
interiors, real endpoints, per-point masks) stays inside the native operator.
`TranslatedOperator` stores only the native operator and the target domain, never a
dense matrix.

## 4. enchilada adapter (follow-on, separate spec and plan in the enchilada repo)

* `_transforms.transform_channels` becomes a call to `domainia.translate.transform`.
  `_load_wdm`, its Python ≥3.13 gate and the `forward_many` probe are deleted.
* `WDMGrid` gains `window` (default Meyer, part of equality), and `transform(...)`
  accepts `window=` on first conversion into WDM.
* `TranslatedCovariance` keeps its public class and validation, and delegates its math
  to `translate_operator`. `DataCovariance` gains a `domain` property.
  `_log_coordinate_scale` is deleted.
* domainia is an optional `wdm` extra, so enchilada's core stays NumPy-only. Tests
  compare the new outputs against enchilada's current outputs on fixed inputs.

## 5. Packaging, errors, tests

**Packaging.** `pyproject.toml`: `name = "domainia"`, `version = "2.0.0"`,
`requires-python = ">=3.12"`, dependencies `numpy>=2.1,<3`, `scipy>=1.14.1,<2`, extras
`jax = ["jax>=0.9", "jaxlib>=0.9"]` and `test = ["pytest>=8"]`. Hatch packages
`src/domainia`. pytest config: `testpaths = ["tests"]`, `pythonpath = ["src"]`,
`--doctest-modules` over `src/domainia/wilson`, and the `jax` and `slow` markers. The
README is rewritten for the general library.

**Errors.** Fail closed throughout.
* `WindowError` (a `ValueError`) for an inadmissible window.
* `ValueError` for a shape, finiteness, inactive-entry or endpoint violation, or for a
  domain or window mismatch.
* `TypeError` for a wrong kind of object.

**Bit-identity baseline (first task, before any edit).** With the untouched code, record
the outputs of every public function that consumes the window into
`tests/wilson/data/meyer_baseline.npz`. Use grids `(Nf, Nt) ∈ {(8, 16), (16, 32),
(32, 8)}`, two values of `dt`, and seeded inputs. After the refactor, the default window
must reproduce every array exactly (`np.array_equal`).

**New tests.**
* Windows:
  * `Meyer(d)` for d = 1…8 passes admissibility;
  * `W Wᵀ = dt·I` on the active cells and exact time round trips (≤1e-12);
  * `RampWindow` with a valid ramp passes;
  * windows that break evenness, support or the partition of unity each raise
    `WindowError` naming the right condition;
  * a mismatched window raises `ValueError`.
* translate:
  * round trips for all 9 pairs with Meyer and one non-default window;
  * Parseval scales against `log_coordinate_scale`;
  * operator invariants: `quadratic_form` agrees across domains, the log-determinant
    Jacobian, `solve(apply(x)) == project(x)`, chains collapse, a native target
    returns the native operator;
  * every input-refusal rule.
  * The operator tests use a minimal per-point test operator in `domainia.testing`, so
    they do not depend on enchilada.
* Import: `import domainia` and `import domainia.wilson` leave `jax` out of
  `sys.modules`. The 2.0 identity test checks `__version__` and the public `__all__`.

**Verification command.**
`PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu ~/miniconda3/envs/wdmrel/bin/python -P -m pytest -q`
from the repo root must pass with no collection errors. The JAX-marked tests run in the
same env (jax 0.9.0.1).

## Non-goals

* Windows wider than |f| < ΔF (Gaussian / Hermite / IOTA seeds from compact_WDM and
  arXiv:2609.05601). They would need a multi-slab transform and banded covariances. That
  is a later project.
* `transform` over partial or packed cell sets. Those stay with `domainia.maps`.
* Rewriting enchilada in this plan (section 4 is its own follow-on).
* Preserving the paper release's reproducibility. It stays frozen in its original repo.
