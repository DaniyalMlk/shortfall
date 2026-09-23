# shortfall

A portfolio risk engine, in pure Python with no dependencies.

Risk numbers are easy to produce and hard to trust. The arithmetic is
undergraduate; what decides whether the answer means anything is the estimator
underneath it, the conventions nobody wrote down, and whether the code says so
when it is working from too little data. This library is built around those
three, and the numerics are checked against closed forms and published results
rather than against themselves.

`ROADMAP.md` says what is built. All seven phases are done: return series and
covariance estimation with its diagnostics, parametric and historical value at
risk and expected shortfall, Euler risk contributions and risk parity, factor
models and risk attribution, drawdown and the path statistics, and a command
line over all of it.

## Installing

```bash
pip install shortfall
```

Python 3.10 or newer. The library imports only the standard library, so there is
nothing else to resolve and nothing to compile.

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

Where the risk comes from, and what it is about:

```python
from shortfall import (
    attribute_risk, fit_factor_model, risk_parity, volatility_contributions,
)

allocation = volatility_contributions(weights, estimate.matrix, names=panel.names)
allocation.component       # sums to the portfolio volatility, exactly
allocation.percentage      # shares; negative for a position that hedges
allocation.identity_error  # how far from exact, for a matrix that was repaired

risk_parity(estimate.matrix).weights          # equal risk contributions
risk_parity(estimate.matrix).budget_error     # evidence, not a promise

model = fit_factor_model(panel, factors)
model.worst_residual_correlation()   # whether the diagonal assumption holds
attribute_risk(weights, model).factor_share
```

And what the path did, which none of the above can see:

```python
from shortfall import maximum_drawdown, sortino, ulcer_index

worst = maximum_drawdown(returns, index=dates)
worst.depth, worst.peak_label, worst.trough_label
worst.recovery_periods      # None when it never recovered, not zero
worst.recovery_return       # a 50% fall needs a 100% gain
ulcer_index(returns)        # the whole path, not its two extreme points
sortino(returns, 252, full_sample=True)   # Sortino's denominator, stated
```

Dates that do not line up are handled by an inner join:

```python
panel = Panel.aligned({
    "AAPL": {"2026-01-02": 0.011, "2026-01-03": -0.004},
    "MSFT": {"2026-01-02": 0.008, "2026-01-06": 0.002},
})   # keeps 2026-01-02 only
```

## From the command line

```bash
shortfall risk          returns.csv --weights 0.4,0.25,0.15,0.2 --shrink
shortfall contributions returns.csv --weights 0.4,0.25,0.15,0.2
shortfall parity        returns.csv --shrink
shortfall drawdown      returns.csv --periods 252
shortfall factors       returns.csv --factors factors.csv
shortfall --json risk   returns.csv          # for anything downstream
```

The file has a header row of column names and one row per period; a leading
non-numeric column is taken as dates. `examples/returns.csv` is a bundled
four-asset series to try it on.

Two input guards earn their place. Values at or below −100% are refused as not
being returns at all. And a column whose *median* absolute value exceeds one is
refused as prices, or as returns written as percentages — before that check, a
price series read as returns produced a one-day value at risk of 164%, and a
number like that is only caught because it is absurd, which is not a guarantee.
The median rather than the maximum, because one return above 100% is rare but
real and refusing a file over it would be wrong, while half the periods moving
that far is not a return series under any circumstances.

## A worked example

`examples/basel_equivalence.py` reproduces the published figure behind the Basel
move from 99% value at risk to 97.5% expected shortfall. Under a normal
distribution the two are 2.3263 and 2.3378 sigma — within half a per cent — so
the switch changes what the measure *is* without changing what it demands.

It then shows the part that matters more, which is that the equivalence is a
property of the normal distribution rather than of returns:

| tail | VaR 99% | ES 97.5% | gap |
| --- | --- | --- | --- |
| normal | 2.3263 | 2.3378 | 0.49% |
| t(8) | 2.5084 | 2.5720 | 2.54% |
| t(5) | 2.6065 | 2.7278 | 4.66% |
| t(4) | 2.6495 | 2.8239 | 6.58% |
| t(3) | 2.6216 | 2.9096 | 10.99% |

Expected shortfall pulls away because it is an average over the tail and so sees
the tail thicken, while a quantile only sees it move. By three degrees of
freedom the gap is twenty-two times the normal one, and the bundled portfolio's
own sample shows the same widening arriving from data rather than assumption.
The script exits non-zero if a figure fails to reproduce, and runs in continuous
integration, so the numerics cannot quietly move a number that has been in print
since 2016.

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

**The moments it needs come from the series.** `ReturnSeries.skewness()` and
`ReturnSeries.excess_kurtosis()` estimate them, defaulting to the bias-corrected
forms for the same reason `variance` defaults to `ddof=1` — and the default
matters more here than for a variance. On normal data the uncorrected excess
kurtosis has expectation exactly `-6/(n+1)`, so a short window of perfectly
well-behaved returns reports thin tails, which for a risk number is being wrong in
the comfortable direction. (`-6/n` is the figure usually quoted and it is not
right; at twenty observations the two are four standard errors apart over forty
thousand replications.)

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

### Splitting a risk number is only meaningful under a rule

Volatility, value at risk and expected shortfall are all positively homogeneous
of degree one in the weights, so Euler's theorem says each equals the sum of the
weights times its own partial derivatives. The pieces therefore add to the total
*exactly* — not approximately, and not after a normalisation applied at the end.
That identity is the whole justification for calling them contributions, and it
is what the tests assert.

The requirement is real and not a formality. It is why variance is not allocated
here: variance is homogeneous of degree two, so its Euler sum is twice the
variance, and a "variance contribution" that halves itself to make the total
work is not a derivative of anything.

Every location-scale measure in this library is `-mean + k * volatility` for a
constant `k` depending only on the distribution and the confidence level. Rather
than reimplementing `k` for the normal, Student-t and Cornish-Fisher cases —
three more chances to get a tail convention wrong — the allocation asks the
estimator itself for the risk of a zero-mean unit-volatility position. If the
total is ever wrong, the allocation is wrong identically, which is the failure
mode worth having.

Sample value at risk is deliberately *not* allocated. Its Euler derivative
conditions on the portfolio sitting exactly at its quantile, which no finite
sample observes; estimating it needs a kernel whose bandwidth changes the
answer, and a number depending on an undisclosed smoothing choice is worse than
no number. Sample expected shortfall has no such problem — its derivative is
minus the asset's average return on the days the portfolio lost most, which
anybody can check — and it is allocated, reusing the same fractional tail
weighting as the estimator so the components sum to it exactly.

Negative contributions are returned rather than clamped, because a hedge
genuinely removes risk. The measures that treat shares as a distribution refuse
instead of computing an entropy over negative numbers.

### Risk parity as a convex problem

The naive approach throws the nonlinear system `w_i (Sw)_i = b_i` at a solver.
It is not convex, it has spurious solutions with negative weights, and it fails
on exactly the ill-conditioned matrices that make the answer interesting.

Minimising `(1/2) x'Sx - sum b_i log(x_i)` over positive `x` instead is strictly
convex for a positive definite covariance, has the risk budget condition as its
stationary point, and keeps every weight positive through the barrier rather
than through a projection. Each coordinate update is a closed-form quadratic
root, so there is no step size and nothing to tune — taken on whichever branch
is numerically stable for the sign of the cross term, which is negative exactly
when an asset hedges.

Two closed forms pin it. For two assets the cross terms cancel out of the
equal-contribution condition, so risk parity is inverse volatility **whatever
the correlation is** — a solver that depends on the correlation there is wrong,
and testing at one correlation would not show it. Under constant correlation the
same holds at any number of assets. Beyond those it is checked against the
defining condition directly, and against the possibility that the implementation
simply returns inverse volatility always, which would satisfy both closed forms.

Convergence evidence is reported, and it separates two questions that get
confused: whether the iteration stopped moving, and whether the risk shares
reached their budgets. Running out of sweeps raises rather than returning
quietly, because weights that are nearly risk parity look exactly like weights
that are.

### Counting bets, and which count you were handed

The effective number of bets can be measured over positions or over principal
components, and the two answer different questions. Ten evenly weighted
positions in one sector are ten bets under the first and close to one under the
second, because the components are uncorrelated by construction while the
positions are not. The principal measure is also defined for a portfolio with
shorts, where the position measure is not.

The two counts on the same allocation are Rényi entropies of order one and two,
and Rényi entropy does not increase in its order — so `exp(entropy)` is never
below `1 / sum(p^2)`, with equality only when the portfolio is perfectly
balanced. Quoting "effective number of bets" without saying which is a free
choice between a larger number and a smaller one for the same portfolio, always
available in the flattering direction.

### Regressions are not solved with the normal equations

Forming `X'X` squares the condition number. Factor returns are correlated with
each other by construction, so a design that is merely awkward at `1e8` becomes
numerically singular at `1e16`, and the Cholesky factorisation already in this
library returns digits that are all noise. Householder QR works on the design
directly and its accuracy tracks the condition number rather than its square.

The test measures this rather than asserting it: on the Läuchli design it
confirms the Gram matrix really does round to exactly singular — `1 + eps^2` is
`1` in double precision — so the normal equations have no answer at all, while
QR recovers the coefficients to full precision. The cost is a constant factor of
about two on problems whose size is the number of factors.

Rank deficiency is refused rather than resolved arbitrarily. A design with
dependent columns admits a family of coefficient vectors with identical fitted
values, and whichever one a solver lands on is an artefact of rounding. That
check caught a genuine mistake in this library's own test fixtures, where a
noise generator seeded like the factor generator reproduced the factor columns
exactly.

### A factor model reports on its own assumption

`B F B' + D` is cheap on a large universe because `D` is diagonal — the
residuals are assumed uncorrelated across assets. That is also the assumption
that fails quietly. If a sector the factors do not span moves together, its
residuals are correlated, and the model reports diversification the portfolio
does not have.

So the fit surfaces the largest residual correlation it left behind, signed,
rather than only its R-squared. Tested by injecting a common residual shock the
factors cannot see: the R-squared values stay unremarkable and the diagnostic
goes to 0.8.

The attribution keeps two quantities apart that are routinely mixed. A
*standalone volatility* is what the risk would be from that source alone; these
are the intuitive numbers and they do **not** add, because squares add — 12% and
5% make 13%, not 17%. A *contribution* is that source's share of the total under
the Euler allocation; these add exactly, and none of them is a volatility
anybody would recognise on its own. Reporting one under the other's name is the
ordinary way an attribution stops reconciling, and nothing in the numbers tells
the reader which kind they were handed.

### Drawdown is the statistic that notices the order

Shuffle the sample and the volatility, the value at risk and the expected
shortfall are unchanged. Drawdown is not, and it is the risk an investor
actually experiences: nobody redeems because the ninety-ninth percentile of the
return distribution is unattractive, and everybody redeems after being down
thirty per cent for two years.

It is computed on the compounded wealth curve. A cumulative sum of simple
returns is the common shortcut and gives a smaller number in the same direction
every time, so it understates exactly the statistic it is quoted to make vivid.

The wealth curve has one more point than there are returns, because `n` returns
move a portfolio between `n + 1` valuations — and the first matters, since a
portfolio that falls in its first period is in drawdown from the start and a
curve beginning after the first return cannot see it.

An unrecovered drawdown reports its recovery as `None`, not as zero and not as
the distance to the end of the sample. Both of those are numbers, and averaging
them in with the recoveries that did happen is how a strategy that has never
recovered from anything comes to report a short average recovery.

Episodes are separated by requiring a return to the old peak before a new one
begins; without that rule the deepest drawdown and the second deepest are
usually the same fall measured from two nearby peaks. That also makes "longest
time under water" meaningful, and it is frequently not the deepest episode — a
fund that falls 30% and recovers in a quarter has had a bad quarter, while one
that falls 12% and takes five years has had a bad decade, and only the second
loses its investors.

**The Sortino denominator is an argument.** Sortino divides the squared
shortfalls by the count of *all* observations; the widespread variant divides by
the downside count only. They are not close and they are quoted under the same
name: the ratio between them is exactly the square root of the hit rate, so at
one downside period in ten the two Sortino ratios differ by a factor of 3.16.
The library defaults to Sortino's own and the other is available by asking.

Rolling windows are not padded back to the original length. A rolling statistic
does not exist before its window is full, and filling the gap is how a backtest
comes to use information it did not have. Rolling drawdown also restarts the
peak in each window rather than slicing the full series, so a late window does
not report the distance below a peak it never saw.

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
pytest          # 708 tests
mypy --strict
ruff check .
```

Continuous integration runs the suite on Python 3.10 through 3.13, type-checks,
lints, runs the worked example and the command line over the bundled data, and
installs the wheel *and* the sdist into separate clean environments to confirm each
can produce an estimate that satisfies an identity — a distribution that imports
but cannot compute is not a working library, and the two artefacts are built by
different code paths, so a file missing from one can be present in the other.

### Releasing

The version lives in `pyproject.toml` and is mirrored by `shortfall.__version__`;
a test asserts the two agree, and the release refuses to run if the tag disagrees
with either.

```bash
git tag v0.1.0
git push origin v0.1.0
```

The tag builds both artefacts, has `twine` read the metadata the way the index
will, installs each into a clean environment and runs an estimate out of it, and
then publishes through the index's trusted-publishing flow — so there is no API
token in this repository, in the workflow, or in the repository's secrets.
Registering the publisher is a one-time step done on the index, naming this
repository, `release.yml` and the `pypi` environment.

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

Gradients are checked against central finite differences of the estimator they
claim to differentiate, because a summation identity proves nothing about the
individual pieces: any vector rescaled to sum to the total satisfies it. The
regressions are checked against Anscombe's first dataset, in print since 1973,
and against the normal equations solved over `fractions.Fraction`, which has no
rounding error at all — so the difference is a *measurement* of the
floating-point error rather than an estimate of it.

One claim in this library was corrected by measurement rather than argument. The
intuitive statement that adding useless factors lowers adjusted R-squared is not
true: over forty independent samples the padded model came out marginally
*higher*. What holds is stronger — adjusted R-squared is approximately unbiased
for the population figure whether or not useless factors are present, so both
models recover 0.5504 while the padded model's raw R-squared is inflated well
past it. The test asserts that instead.

Where there is no closed form at all — the shrinkage intensity — the tests check
what the theory predicts the estimator will *do*: shrink hard when the target
describes the data, shrink little when it cannot, and shrink more when the same
structure is observed for a tenth as long. Three predictions, wrong in three
different directions if the derivation is wrong.

## Licence

MIT.
