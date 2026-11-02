# domainia

Move data and covariance between the time, Fourier and time-frequency domains.

* `domainia.wilson` holds the Wilson-basis kernels: forward and inverse transforms, the
  stationary and evolutionary noise models, inner products and dense oracles. With the
  default Meyer window (`MEYER`) the Wilson basis is the WDM transform. Every kernel takes
  `window=`, so the same code runs any *admissible* window: real, even, zero for
  |f| ≥ ΔF, and a partition of unity on the lattice. Such a window gives an orthonormal
  basis and block-tridiagonal stationary covariances.
  `Meyer(d)`, `RampWindow(ramp, name)` and `SampledWindow(fn, name)` are validated
  when sampled, and an inadmissible one raises `WindowError`.
* `domainia.translate` is the translation layer. `transform` moves `{channel: array}`
  dicts between the full-record domains, and `translate_operator` puts a covariance
  operator native to one domain into another lazily: no dense matrix, and the
  coordinate Jacobian is in `log_determinant`.
* `domainia.domains`, `domainia.data`, `domainia.maps`, `domainia.covariance`,
  `domainia.variance` and `domainia.spectra` cover partial domains (WDM cell sets, Fourier
  bands): maps between them, block-tridiagonal / dense / diagonal / evolutionary covariances
  and cell variances.

## Quick start

```python
import numpy as np
from domainia import FourierDomain, Grid, Meyer, TimeDomain, WDMDomain, transform

Nf, Nt, dt = 16, 64, 0.5
N = Nf * Nt
time = TimeDomain(N, dt)
freq = FourierDomain.full(N, dt)                        # dt * numpy.fft.rfft(x)
wdm = WDMDomain.full(Grid(Nf, Nt, dt))                  # (Nf + 1, Nt) cells, Meyer window
wdm8 = WDMDomain.full(Grid(Nf, Nt, dt), window=Meyer(8))

x = {"A": np.random.default_rng(0).standard_normal(N)}
w = transform(x, time, wdm)                             # {"A": (Nf + 1, Nt) float64}
X = transform(w, wdm, freq)                             # {"A": (N // 2 + 1,) complex128}
w8 = transform(w, wdm, wdm8)                            # another window, through the spectrum
assert np.allclose(transform(w8, wdm8, time)["A"], x["A"])
```

Coordinates scale as `exp(log_coordinate_scale(domain))` relative to the time
samples: 1 for time, `sqrt(dt)` for WDM (any window), `dt * sqrt(N)` for Fourier (in the
real coordinates `Re X_0`, `sqrt(2) Re/Im X_k`, `Re X_{N/2}`).

## Operators

An `Operator` lives on one full-record domain and acts on `{channel: array}` dicts:
`apply`, `solve`, `project`, `quadratic_form`, `log_determinant`, plus `domain`,
`channel_names` and `degrees_of_freedom`. `translate_operator(op, target)` returns a
`TranslatedOperator` that keeps only `op` and `target`. Translating again re-targets
`op`, and translating to `op.domain` returns `op` itself.

## Errors

Everything fails closed. `WindowError` (a `ValueError`) means an inadmissible window.
`ValueError` covers shape, finiteness, inactive-cell, endpoint, domain or window
mismatches. `TypeError` means the wrong kind of object.

## Install and test

```bash
pip install -e ".[test]"          # add ",jax" for the JAX backend
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu python -P -m pytest -q
```

`import domainia` never imports JAX. The JAX backend is reached only through
`domainia.wilson.backend`.
