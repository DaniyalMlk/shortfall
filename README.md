# shortfall

A portfolio risk engine, in pure Python with no dependencies.

Risk numbers are easy to produce and hard to trust. The arithmetic is
undergraduate; what decides whether the answer means anything is the estimator
underneath it, the conventions nobody wrote down, and whether the code says so
when it is working from too little data. This library is built around those
three, and the numerics are checked against closed forms and published results
rather than against themselves.

`ROADMAP.md` says what is built and what is not. Phases 1 to 3 — return series,
covariance estimation and its diagnostics, parametric value at risk and expected
shortfall, and historical and filtered historical simulation — are done.

## Using it

```python
from shortfall import Panel, ledoit_wolf, DAILY_TRADING

panel = Panel.from_prices({"AAPL": aapl_closes, "MSFT": msft_closes})
estimate = ledoit_wolf(panel)

estimate.intensity              # how far it was pulled towards the target
estimate.diagnostics.condition_number
estimate.diagnostics.numerically_singular
panel["AAPL"].annualised_volatility(DAILY_TRADING)
```

```python
from shortfall import Distribution, portfolio_risk

risk = portfolio_risk(
    [0.6, 0.4], estimate.matrix,
    confidence=0.99, distribution=Distribution.STUDENT_T, degrees=5,
)
risk.value_at_risk        # a positive loss: 0.023 is a 2.3% loss
risk.expected_shortfall   # never smaller than the value at risk
risk.quantile             # the same number, signed, for the other convention
risk.scaled_to(10)        # ten periods, under the square-root-of-time rule
```

The same numbers taken from the sample instead of from an assumed shape, with an
interval saying how much sample there was:

```python
from shortfall import bootstrap_interval, filtered_historical_risk, historical_risk

historical_risk(returns, confidence=0.99).effective_sample   # 2.5, not 250
bootstrap_interval(returns, confidence=0.99).relative_width  # how wide that makes it
filtered_historical_risk(returns).scaling                    # today against the window
```

Dates that do not line up are handled by an inner join:

```python
panel = Panel.aligned({
    "AAPL": {"2026-01-02": 0.011, "2026-01-03": -0.004},
    "MSFT": {"2026-01-02": 0.008, "2026-01-06": 0.002},
})   # keeps 2026-01-02 only
```

## Design

### Shrinkage, and saying how much of it happened

The sample covariance fits `n(n+1)/2` parameters from `nT` numbers. When `T` is
not large relative to `n` its eigenvalues spread out — the largest biased up, the
smallest biased down, past zero once `T < n`, where the matrix is singular
outright.

That is not a uniform loss of accuracy, and that is what makes it dangerous. An
optimiser hunting for low-variance directions finds exactly the eigenvectors
whose variance was most understated, and loads up on them. The error concentrates
precisely where the portfolio is about to.

Shrinkage pulls the sample towards a structured target with far fewer parameters,
trading a little bias for a large reduction in variance. Ledoit and Wolf's
contribution is that the optimal amount can be *estimated* rather than tuned,
which turns a judgement call into a number. The target here is constant
correlation — every pairwise correlation replaced by the average of them,
variances untouched — which suits equities, where the dominant structure really
is that everything moves together by roughly the same amount.

The intensity is reported rather than hidden. An intensity of 0.05 says the
sample was informative; one of 0.9 says the answer is mostly the prior and the
data contributed little. A caller who cannot see which of those happened cannot
tell a risk estimate from an assumption, so `Shrunk` carries the intensity, the
unclamped optimum, the sample, the target, and the diagnostics on the result.

### Risk numbers carry their own conventions

Three things go wrong with a value at risk, none visible in the number.

**Sign.** It is a loss, and a loss is a negative return, so whether an
implementation returns `0.023` or `-0.023` for the same portfolio is a coin
flip. Here it is a positive loss, and the signed return quantile is on the
result as well, so nothing has to be inferred.

**Which tail.** `confidence=0.99` means the 1% tail. The two are complements,
and an implementation that takes one while documenting the other is wrong by an
amount that grows as the tail thins.

**Normal tails.** A Gaussian understates the tail of essentially every return
series. The Student-t is offered with the scaling that a raw `t` needs — its
variance is `v/(v-2)`, not 1, so an unscaled one is 22% too wide at five degrees
of freedom, which is more than the heavy tail it was reached for is worth.

Cornish-Fisher has a sharper failure than either. Outside a bounded region of
the skewness-kurtosis plane its corrected quantile stops increasing with the
probability, which means it is not a quantile function and nothing read off it
is a quantile of anything. That is checked and refused — over the tail the
number is actually read from, rather than symmetrically around zero, because the
cubic coefficient is `-S^2/18` and a symmetric check therefore refuses every
pure-skewness correction there is. It would be the wrong question, not a
stricter one.

Its expected shortfall is a closed form rather than a quadrature. Written as a
cubic in `z`, the tail integral against the normal density is exactly the four
tail moments of the normal, each of which is exact — and a quadrature over a
tail reaching to minus infinity is precisely where numerical integration is
least reliable.

### Expected shortfall is subadditive and value at risk is not

There is a worked counterexample in the tests, stated as exact frequencies
rather than simulated. Two independent bonds, each half the book, each
defaulting with probability 4%. At 95% confidence:

| | each half alone | combined | sum of parts |
| --- | --- | --- | --- |
| value at risk | −0.005 (a *gain*) | **0.495** | −0.010 |
| expected shortfall | 0.399 | **0.511** | 0.798 |

Alone, each half's 4% default chance sits inside the 5% tail, so its value at
risk is a gain. Combined, the 7.84% chance that at least one defaults reaches
into the tail, and value at risk becomes a loss of nearly half the book —
diversifying made the measured risk worse. Expected shortfall on the same
positions is comfortably below the sum of its parts.

That is not a curiosity. A risk limit written in value at risk can be satisfied
by splitting a book into pieces that each sit just inside it while the whole sits
well outside, and it is the concrete reason the Basel framework moved to
expected shortfall.

The test also earned its place by finding a bug. It reported expected shortfall
violating a property expected shortfall provably satisfies — which is a statement
about the estimator, not about the data. The estimator had been averaging every
observation at or below the quantile, which is the obvious implementation and is
wrong whenever the quantile ties with a value many observations share. On the
counterexample, where most periods are identical, it averaged nearly the whole
sample and turned a tail mean of −0.399 into −0.0152.

The tail is now the exact sample estimator, which weights the partial
observation: at 99% over 250 returns the tail is 2.5 observations, two counting
fully and the third counting half.

### Historical estimates say how little data they had

A 99% value at risk over 250 daily returns is computed from two and a half
observations. The point estimate cannot say so, so `effective_sample` does, and
`bootstrap_interval` puts a number on what it costs.

Filtered historical simulation divides each past return by the volatility
estimated at the time it happened and multiplies by today's, keeping the
empirical shape while making the scale current. The volatility estimate uses
returns strictly *before* each point: one that included the current return would
divide it by a volatility that knew about it, flattening the standardised series
and making the filtered tail far too thin.

The identity that holds the construction together is that a constant volatility
reduces the filtered estimate to the plain one bit for bit, since the two
rescalings cancel. It is a test, because anything else means the filter is being
applied and removed in different units and nothing else would reveal it.

### Conventions are arguments, not assumptions

Two decisions are invisible in the output, which is exactly why they are made
explicitly here.

**Simple against logarithmic returns.** They differ by about half the variance,
and they aggregate in opposite directions: log returns add across time, simple
returns add across a portfolio. A series carries which one it is, and combining a
panel into a portfolio converts to simple first — because a weighted sum of log
returns is not the log return of the weighted portfolio, and for concentrated
weights the gap is not small.

**Periods per year.** The 252 that appears in most code is the US equity trading
calendar. Using it on weekly data overstates volatility by a factor of 2.2, which
does not look wrong enough to notice. There is no default anywhere in this
library; the argument is required and there are named constants so the assumption
is legible at the call site.

Alignment is an inner join. Filling a missing return with zero says the asset did
not move, which lowers its estimated volatility and drags every correlation
involving it towards zero — a lie with a direction, which is the worst kind.

### Diagnostics that answer the question actually being asked

A covariance matrix estimated from fewer observations than assets is singular in
exact arithmetic and arrives in floating point with a smallest eigenvalue around
`1e-18`. Its condition number is about `1e17`, not infinity, so code testing
`isinf` concludes the matrix is fine. `Diagnostics.numerically_singular` tests
the ratio of smallest to largest eigenvalue instead, which is the question the
caller meant.

`observations_per_parameter` is the other half of that: `nT` numbers against
`n(n+1)/2` parameters. Below about 2 the sample covariance is not worth using
unshrunk, and below 1 it is rank-deficient by construction.

### A Jacobi eigensolver, and a convergence criterion that was wrong

The eigensolver is cyclic Jacobi. It is slower than a tridiagonal reduction with
QL implicit shifts, and for this purpose it is better: it computes small
eigenvalues to high *relative* accuracy — exactly where a short-sample covariance
needs it — and every step is an exact rotation, so the eigenvectors cannot drift
out of orthogonality.

The first version stopped when the off-diagonal Frobenius mass fell below `1e-15`
of the total. Frobenius masses are sums of *squares*, so that permitted
off-diagonal entries of `3e-8`. The eigenvalues were converged, because the
diagonal had settled; the eigenvectors were good to eight digits. Reconstructing
the matrix from its eigenpairs did not notice, and neither did the trace or the
determinant. Only checking `Av` against `wv` directly did, which is now the test
the criterion is set by.

## Development

```bash
pip install -e ".[dev]"
pytest          # 355 tests
mypy --strict
ruff check .
```

Continuous integration runs the suite on Python 3.10 through 3.13, type-checks,
lints, and installs the built wheel into a clean environment to confirm it can
produce an estimate that satisfies an identity — a wheel that imports but cannot
compute is not a working library.

Numerical work is checked against a closed form or a published result where one
exists, and against identities that hold whatever the input where one does not: a
2x2 matrix has exact eigenvalues, a matrix built as `Q diag(w) Q^T` has known
ones, and the eigenvalues must reproduce the trace and the determinant, which are
computed without touching the solver.

Every expected shortfall closed form is integrated a second time by a route that
shares none of its algebra, and they agree to 1e-10 or better. The quadrature
substitutes `x = q - tan(theta)` to map the infinite tail onto a finite interval;
truncating it at a large negative number instead loses real mass at three degrees
of freedom, in the second decimal.

Where there is no closed form at all — the shrinkage intensity — the tests check
what the theory predicts the estimator will *do*: shrink hard when the target
describes the data, shrink little when it cannot, and shrink more when the same
structure is observed for a tenth as long. Three predictions, wrong in three
different directions if the derivation is wrong.

## Licence

MIT.
