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

> **Not on the package index yet.** The `pip install` line above is what it
> will be; until the first release lands, install from source:
>
> ```bash
> pip install "git+https://github.com/DaniyalMlk/shortfall.git"
> ```

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

```python
from shortfall import TailMethod, extreme_risk, mean_excess_curve

# Risk beyond the sample: the exceedances are fitted on their own, so the answer
# can be worse than anything that has happened. Nothing else here can do that.
far = extreme_risk(returns, confidence=0.999, tail_fraction=0.05)
far.value_at_risk          # a positive loss
far.expected_shortfall     # None when the fitted shape has no finite mean
far.observed_beyond        # how many losses were worse; often zero
far.fit.shape              # the tail index, with a standard error beside it
far.fit.shape_standard_error
far.fit.upper_endpoint     # not None only if the fitted tail ends

mean_excess_curve([-r for r in returns])   # where the threshold should come from
```

```python
from shortfall import Family, Marginal, copula_risk, fit_copula, kendall_tau

fitted = fit_copula(panel)                    # Student-t by default
fitted.degrees_of_freedom                     # 4.2 on a sample drawn at 4
fitted.likelihood_ratio                       # 510 against the Gaussian case
for pair in fitted.tail_dependence():
    pair.first, pair.second, pair.coefficient # 0.0 under a Gaussian copula

result = copula_risk(panel, weights, confidence=0.99, paths=20_000, seed=0)
result.expected_shortfall                     # 0.0427
result.gaussian_expected_shortfall            # 0.0393, same marginals, same draws
result.tail_dependence_premium                # 0.086 -- the assumption, priced
result.standard_error                         # 0.0006, and read it as a floor

copula_risk(panel, weights, marginal=Marginal.EXTREME_VALUE)   # fitted marginal tails
kendall_tau(first, second)                    # O(T log T), not O(T^2)
```

## From the command line

```bash
shortfall risk          returns.csv --weights 0.4,0.25,0.15,0.2 --shrink
shortfall contributions returns.csv --weights 0.4,0.25,0.15,0.2
shortfall parity        returns.csv --shrink
shortfall drawdown      returns.csv --periods 252
shortfall factors       returns.csv --factors factors.csv
shortfall validate      forecasts.csv --es-column es --replications 2000
shortfall volatility    returns.csv --horizon 250 --column fund
shortfall volatility    returns.csv --innovation student-t --confidence 0.995
shortfall volatility    returns.csv --paths 40000 --horizon 10 --draw bootstrap
shortfall tail          returns.csv --confidence 0.999 --curve 12
shortfall tail          returns.csv --threshold 0.02 --method probability_weighted_moments
shortfall copula        returns.csv --weights 0.4,0.25,0.15,0.2 --paths 40000
shortfall copula        returns.csv --family gaussian --marginal extreme_value
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

### The far tail, fitted to the exceedances rather than to the sample

Every other estimate here is told something about the whole distribution and
asked about the tail. Historical simulation reads the tail off the order
statistics, so at 99.9% over 2,000 returns it reports the second-worst one and
cannot return a number larger than the worst thing that has happened. The
parametric routes fit a shape to all of the data, where the bulk dominates the
likelihood — and the tail is the part that was asked about.

`shortfall.extreme` takes the third route. The Pickands-Balkema-de Haan result
says the distribution of the *amount by which* a high threshold is exceeded
converges to a generalised Pareto for a wide class of parent distributions, so a
shape fitted to only the exceedances says nothing about the body and extrapolates
past the largest observation. Two estimators are offered because they disagree
where the sample is short: Grimshaw's one-dimensional reduction of the
likelihood, and probability-weighted moments in closed form.

`examples/tail_comparison.py` measures what that buys, against the truth, on a
Student-t whose tail index is exactly the reciprocal of its degrees of freedom. It
runs in continuous integration, and the result is not the one the argument for
extreme value theory usually implies.

| 2,000 draws of t(4) | fitted tail | historical | fitted normal |
|---|---|---|---|
| 99% — bias / mean abs. error | +0.1% / 4.7% | −0.5% / 5.2% | −12.1% / 13.0% |
| 99.9% | +0.0% / 13.0% | −1.8% / 18.4% | −39.0% / 39.0% |
| 99.99% | +0.2% / 28.4% | −21.1% / 35.7% | −59.6% / 59.6% |

At 99% the fit buys almost nothing: twenty observations are still out there and
historical simulation reads them off directly, landing within half a percentage
point. What the fit removes further out is the **bias**, not the noise. At 99.99%
historical simulation can only return the worst loss in the file and is short by
a fifth, every time and in the same direction; the fit averages within a fraction
of a per cent of the truth — with a spread of 42%, so the honest claim for it is
that it stops being systematically short, not that it becomes accurate.

The normal is the more interesting failure. Short by 39% at 99.9% and 60% at
99.99%, with a spread of 3 to 4%: precise, consistent, and wrong in the same
direction every single time. An estimator that varies is telling you it is
uncertain. That one is not.

The threshold is a bias-variance choice with no right answer, and the measurement
is worth having because it contradicts the folklore. Raising the threshold from
the top fifth to the top twentieth moves the shape from 0.10 to 0.13 against a
true 0.25 — less biased — at more than double the spread. But the 99.9% quantile
it is chosen *for* moves by less than two percentage points across the whole range
from the top fifth to the top hundredth, because the fitted scale absorbs what
the shape gets wrong. The customary 5% is not better here than 20%. What does
break is the top 1%: at twenty exceedances the shape's spread is several times its
own true value and its mean goes negative, which claims the loss is bounded. That
is the regime `MINIMUM_EXCEEDANCES` refuses.

Three refusals rather than plausible numbers. A confidence below the threshold's
own exceedance probability is outside the fit, and reading the empirical quantile
instead would be a different estimator answering under this one's name. A fitted
shape at or above one has no finite mean, so there is no expected shortfall at any
confidence — the value at risk still exists, so `extreme_risk` returns `None` for
the mean rather than throwing the whole result away, while the fit itself raises
with the shape in the message. And a fit whose upper bound falls *below* the
largest exceedance is impossible: over 40,000 samples drawn with shapes between
−1.2 and 0.4, the moment estimator produced one about one time in six of the fits
it gave a negative shape, with the bound as low as 0.58 of the largest exceedance.
Maximum likelihood cannot reach it, because the likelihood is negative infinity
outside the support.

Two smaller things that were wrong first. The optimiser's stopping rule carried an
absolute term, which made it a length in the units of one over a loss: the same
returns quoted as fractions and as basis points converged differently and the
dimensionless shape moved in its eighth digit. And the search bound came from
doubling until the profile stopped improving, which on a shape of 0.5 stepped past
the maximum and stopped short of it, returning a fit 0.17 of log-likelihood below
what a grid search could find. Both are tested now — the second against a
two-dimensional sweep of the surface.

The estimate is deliberately not a `Risk`. `Risk` carries `scaled_to`, which
applies the square-root-of-time rule, and applying that to a fitted tail is
exactly the substitution the horizon simulation exists to refuse: the sum of a
horizon's heavy-tailed innovations is not a generalised Pareto variate with a
scaled parameter.

### An elliptical model cannot have assets crash together

Every other multi-asset route here reads a covariance matrix, and that is a
stronger assumption than it looks. Under a multivariate normal, the probability
that two assets are both beyond their own `q` quantile, divided by `q`, goes to
zero as `q` falls — at *any* correlation below one. A correlation of 0.9 buys a
great deal of co-movement in the body and asymptotically none in the tail. Under a
multivariate Student-t the limit is positive, which is better, but it is forced to
be the same number for every pair and the same above as below.

So a portfolio that is mildly correlated day to day and moves as one in a crash
has no representation in a covariance matrix, and that portfolio is the reason the
estimate is being computed. `copula_risk` fits the dependence to the ranks and
each marginal separately, then puts them back together by simulation.

Measured on the case it exists for — five equally weighted assets, 1,500
observations from a t copula at 4 degrees of freedom with every pairwise tau at
0.35, empirical marginals from the same sample, 20,000 paths, three samples and
three simulation seeds each — the fitted copula puts 99% expected shortfall 10.6%
above the Gaussian copula's on the identical marginals, the identical correlation
matrix and the identical normal draws: 0.0433 against 0.0391. The spread across
the nine runs is 2.6 percentage points, range 6.7% to 13.7%, which is the honest
precision of it. At 99.5% it is 13.7%, and the gap widens as the quantile falls.

**And it matters in the opposite place from where it is expected to.** Sweeping the
pairwise tau on the same construction, premium at 99% over three simulation seeds
each:

| tau | correlation | premium |
| --- | --- | --- |
| 0.05 | 0.078 | +19.7% |
| 0.15 | 0.233 | +17.6% |
| 0.30 | 0.454 | +12.0% |
| 0.50 | 0.707 | +5.9% |
| 0.70 | 0.891 | +1.5% |
| 0.90 | 0.988 | −1.0% |

The premium is largest where the correlation is *lowest*, and gone once the
correlation approaches one. The intuition that tail dependence matters most for
assets that are already correlated has it backwards. At a correlation near one the
Gaussian copula already moves everything together, so the portfolio is effectively
one asset and no copula can change that asset's own marginal tail. At a correlation
near zero the Gaussian copula promises genuine diversification in the extremes —
and that promise is what is illusory, because a t copula at four degrees of freedom
has a tail dependence coefficient of 0.0756 at a correlation of *zero*. The shared
mixing variable is what makes large moves arrive together and it does not care
about the correlation at all.

So the portfolio this is for is the one that looks diversified. A book correlated
at 0.08 is where a covariance matrix is most reassuring and most wrong.

**And the two measures disagree about the direction.** On that same book, fitted
copula against Gaussian:

| confidence | value at risk | expected shortfall |
| --- | --- | --- |
| 95% | −4.0% | +6.0% |
| 99% | +9.5% | +19.2% |
| 99.5% | +16.1% | +24.9% |
| 99.9% | +29.4% | +35.2% |

At 95% the value at risk is *lower* under the copula that has the tail dependence
in it. Not an error and not noise: tail dependence moves mass from the near tail to
the far tail and the total is one, so a quantile sitting close to the body has less
beyond it and comes in lower, while the mean of what is beyond is higher because of
where that mass went. Value at risk cannot see past itself; expected shortfall
integrates the whole tail.

Read only the value at risk at 95% and the conclusion is that modelling tail
dependence made the portfolio safer. That is the clearest case here for why the
measure matters as much as the estimate, and it is why both figures travel together
in the result.

The mechanism is starker in the copula alone. The fraction of draws with all five
assets below their own 5% point is 0.42% under the fitted t copula against 0.14%
under the Gaussian one; below their own 1% point, 0.057% against 0.005%.
Independence would give 3.1e-7 and 1e-10. A factor of 3 becomes a factor of 11
one quantile deeper, because one of the two limits is zero.

**The negative result is the one that makes the positive one worth anything.** Fed
1,500 observations from a genuine Gaussian copula, the same procedure fits 92
degrees of freedom — at the upper bound, which is how it says "no tail dependence
found" — and reports a premium of 0.2% over the same nine runs, spread 0.2
percentage points, range −0.1% to +0.5%.

On `examples/returns.csv`, which was not built to make this point, the fit lands at
28.5 degrees of freedom with a likelihood ratio of 5.2 against the Gaussian case.
That is weak evidence, and the premium behaves like it: +0.4% averaged over six
simulation seeds with a spread of 1.3 percentage points, so it is not
distinguishable from zero. The equity, credit and utilities pairs get tail
dependence coefficients of 0.014 to 0.023; the gold pairs, correlated at −0.19,
get zero. That is about what four series of 1,260 daily returns can say, and the
estimate saying so is the point.

The pairing of the two figures is worth what it is worth and no more. Sharing the
normal draws roughly halves the spread of their difference — 2.4 percentage points
against 4.6 over six seeds on the synthetic sample, 1.3 against 2.8 on the bundled
one. It cannot do better, because the chi-square mixing draw is not shared and is
the entire difference between the two copulas.

### The dependence parameter comes from the ranks, not from a correlation

Pearson correlation is the wrong statistic to hand a copula on two counts. It is
not invariant to the marginals, which is exactly what a copula abstracts away:
square one series and take logs of another and the dependence is unchanged by any
reasonable definition, because every pair keeps its ordering, while the
correlation moves. And it is not robust in the direction that matters — one joint
8-sigma point added to 300 independent observations moves the sample correlation
by 0.175 on average and Kendall's tau by 0.0066, which is the derived bound of
`2(1 + |tau|)/(n + 1)`.

Kendall's tau it is, then, and the inversion `rho = sin(pi tau / 2)` holds for
every elliptical copula at any degrees of freedom — which is why the correlation
matrix can be estimated without first deciding what the tail looks like. The
inversion is not a rescaling: a tau of 0.5 is a correlation of 0.707 and a tau of
0.1 is 0.156, so reading a rank correlation as a linear one understates dependence
everywhere.

Counting concordant pairs directly is quadratic. `kendall_tau` sorts and counts
inversions with a Fenwick tree instead: 6.7ms against 246ms at 2,000 observations,
and the factor grows. The quadratic definition is kept and exported, because an
`O(T log T)` count with a tie correction is exactly the kind of code that is wrong
in a way no property test catches, and it is checked against the definition on
samples built to be full of ties.

Applying a non-linear inversion pair by pair does not have to land on a valid
correlation matrix, and on real data it routinely does not. The projection onto
the nearest positive semi-definite matrix is reported rather than performed
quietly: it moves the smallest eigenvalues most, so the pairs it changes are the
ones in the near-degenerate combinations, which is where a simulated portfolio
would otherwise have concentrated.

### The degrees of freedom are fitted, and the search is checked

That one parameter is what carries joint tail arrival, so defaulting it would be
choosing the answer. It comes from a profile likelihood on the copula density with
the correlation matrix held at its rank estimate — consistent, not efficient, and
deliberately so: estimating both at once would let a misspecified tail move the
correlations.

A profile search that settles below the maximum is the failure this code is most
likely to have and it does not announce itself, so there is a test comparing the
search's likelihood against a quarter-step grid over the whole range. The
quantile function is memoised inside each likelihood evaluation, which is exact
rather than approximate: the ranks of every column are a permutation of `1..T`, so
a panel presents `T` distinct probabilities and not `T*d`. That plus a shorter
search took a fit on 800 observations of four assets from 40s to 6.6s.

A maximum at the upper bound is returned as the bound rather than raised as a
failure. "Not measurably heavy" is the correct answer to how heavy the joint tail
is, often enough that it needs to be sayable.

### Marginal tails are a separate switch from the dependence

With empirical marginals no simulated draw for a single asset can exceed that
asset's worst observation, so every bit of the extrapolation lives in the
dependence. `Marginal.EXTREME_VALUE` splices a fitted generalised Pareto onto each
loss tail and puts it back — separately, so the two assumptions can be turned on
one at a time.

The splice point has to be the fit's *realised* exceedance fraction and not the
tail fraction it was asked for. A threshold at the 5% point of 800 losses is
exceeded by 39 of them, not 40, so the fit describes the worst 4.875% and refuses
anything shallower — it knows it was told nothing about the body. Splicing at the
nominal 5% sends a thin band of probabilities to a tail that declines to answer
for them, and because those probabilities only come up sometimes, it is a failure
that appears on one seed in five.

### Scoring a model against what happened

Every estimator in this library produces a forecast. `shortfall.backtest` is
the part that asks, once the period has passed, whether the forecast was worth
anything.

```python
from shortfall import normal_risk, validate

estimate = normal_risk(mean=0.0, volatility=0.012, confidence=0.99)
result = validate(
    realised_returns,
    [estimate.value_at_risk] * len(realised_returns),
    confidence=0.99,
    expected_shortfall=[estimate.expected_shortfall] * len(realised_returns),
)
result.rejected_at(0.05)      # ('independence', 'conditional coverage')
result.traffic_light.zone     # Zone.YELLOW
```

A value at risk only directly claims a *frequency*, and that claim splits into
two that fail for different reasons and need different fixes.

**Coverage** is whether there are about the right number of breaches. Kupiec's
likelihood ratio tests it, and a model with the wrong volatility fails it.

**Independence** is whether the breaches are spread out. This is the one worth
having. A model that assumes constant volatility can get the count exactly
right over a year and put every breach in the same fortnight, and a count
cannot tell the difference. Christoffersen's Markov test can, and the simulated
power here is 0.90 against a clustered process whose unconditional rate is the
correct 1% — where the count alone rejects 0.17 of the time.

That 0.17 is not nothing, and it is not a bug in either test: clustering makes
the breach count overdispersed relative to the binomial Kupiec's null assumes,
so some samples land far enough from 1% to be rejected on the count. The point
is the gap, and that independence names the actual defect.

The supervisory traffic-light zones are derived from the binomial rather than
transcribed. The published table is a table of counts for 250 observations at
99%; the rule underneath it is about cumulative probability, and deriving it
reproduces those counts exactly while also answering for a sample that is not
250 days long — three breaches is comfortably green over 250 days and yellow
over 125. The capital add-on goes the other way. Those values are tabulated
with no formula behind them, so they are returned for the setup they are
published for and `None` anywhere else, rather than extrapolated into a number
that would look official and mean nothing.

Expected shortfall is harder, and not because of effort. Value at risk is
*elicitable* — there is a scoring function minimised in expectation by the true
quantile, which is exactly what makes a breach count a test. Expected shortfall
is not, so there is no equivalent statistic. The two Acerbi-Székely statistics
compare realised tail losses against the forecast tail mean directly; both are
zero under a correct model and negative when the tail is understated. Neither
has a closed-form null, so one is simulated from the same predictive
distribution the forecasts came from.

Two measured facts about those statistics, because both are easy to guess wrong.
Test 1 reads **-0.245** when the true volatility is *double* the forecast — it
is a ratio of tail means, and a normal's conditional tail mean grows slowly
once the threshold is already inside the distribution, so a small number there
is a large error. And an *overstated* tail can barely be tested: a forecast
twice too wide produces zero breaches in four thousand observations, and a
statistic read off zero breaches is not evidence about anything.

The sign convention is the thing to get right before any of this. Returns are
signed and forecasts are positive losses, the same convention `Risk` uses, so a
breach is `observed < -forecast`. A forecast column of negative numbers is
refused rather than scored, because it is the sign convention being crossed and
not a model predicting gains.

### A volatility process, because the independence test asks for one

When `validate` rejects on independence, the answer is not to rescale
anything. It is that the model has no notion of volatility changing and needs
one. `shortfall.volatility` is that notion.

```python
from shortfall import fit_garch, validate

fitted = fit_garch(returns)
fitted.persistence        # 0.988
fitted.half_life          # 59 periods — a shock is half gone in three months
result = validate(
    returns,
    [2.326 * v for v in fitted.volatilities],
    confidence=0.99,
)
```

What was here already is `ewma_volatility` at a decay of 0.94. That is a
filter, not a model, and the gap shows in three places: the decay is assumed
rather than estimated, a shock never decays towards anything, and — because
there is no long-run level — the forecast is *flat at every horizon*. GARCH
adds the term that fixes all three:

    variance[t] = omega + alpha * residual[t-1]^2 + beta * variance[t-1]

EWMA is the boundary case `omega = 0`, `alpha + beta = 1`: a process with
infinite unconditional variance, which is exactly why it cannot mean-revert.

**Does it actually fix what it is for?** Mostly. Over twenty independent
regime-switching samples of 2000 observations:

| | breaches (20 expected) | independence rejected |
|---|---|---|
| constant forecast | ~51 | **17 / 20** |
| GARCH forecast | ~28 | **1 / 20** |

The clustering is essentially gone. The *count* is halved rather than fixed,
because a Gaussian GARCH still understates the tail of a series whose
standardised residuals are fat. Saying it was fixed would have been the easy
claim and it is not what the numbers say.

### The shape drawn at that volatility

Halving the breach excess left the other half on the table, and the other half
is the innovation distribution. The variance process says how the *scale* moves;
it says nothing about the shape drawn at that scale, and with a Gaussian
likelihood the 99% point of a standardised residual is 2.326 whatever the
residuals look like.

```python
from shortfall import Innovation, fat_tail_test, fit_garch

verdict = fat_tail_test(returns)
verdict.fat                    # True
verdict.degrees_of_freedom     # 4.1

fitted = fit_garch(returns, innovation=Innovation.STUDENT_T)
fitted.risk(confidence=0.99)   # Risk(value_at_risk=..., expected_shortfall=...)
```

The degrees of freedom are estimated **in the same likelihood** as the variance
parameters rather than fitted to the residuals afterwards: a normal likelihood
over-weights the largest residuals of a fat-tailed series enough to pull `alpha`
up, so a two-stage fit gets the tail right and the dynamics wrong. The density
used is the Student-t **standardised to unit variance**, so the shape parameter
cannot quietly rescale the variance the recursion is carrying — an unscaled `t`
would be 22% out at five degrees of freedom and the recursion would absorb it.

Same twenty regime-switching samples, 99% forecasts, 20 breaches expected:

| | breaches | excess over nominal | independence rejected |
|---|---|---|---|
| constant forecast | ~51 | 31 | **17 / 20** |
| GARCH, normal innovations | 28.2 | 8.2 | **1 / 20** |
| GARCH, estimated tail | 22.45 | 2.45 | **1 / 20** |

About 70% of what the variance model left behind, and the clustering verdict
does not move — the quantile changed, not the dynamics. The residue is not noise
either: a GARCH fitted to a regime-switching series does not have identically
distributed standardised residuals, because the process is not a GARCH.

**The other half of the claim.** On data that never had a fat tail the two agree
to within half a breach in twenty, 19.75 against 19.25, because the estimate goes
to the cap where the density is the normal's to four decimals. Without that
number, "multiply every forecast by 1.1" would have produced the table above.

**Whether to fit it at all is a test, not a preference.** The models are nested,
so the likelihood ratio answers it, and `--innovation auto` runs it. The p-value
is conservative for a structural reason: the null puts the inverse degrees of
freedom at zero, on the boundary of the parameter space, so the asymptotic null
is a half-and-half mixture of chi-square with zero and one degrees of freedom.
Read against chi-square with one it reports about twice the true probability.
Over 200 samples with Gaussian innovations a nominal 5% test rejected 5 times,
which is the 2.5% that halving predicts.

**A large estimate is not a fat tail.** Above 200 degrees of freedom the density
is within a percent of the normal everywhere that matters, so the likelihood is
flat and whatever the optimiser stops at is noise. `degrees_identified` is false
there and the number is withheld rather than reported, because "our fitted tail
index is 640" is a claim about a series that is simply Gaussian.

The expected shortfall gap is always the larger of the two. At five degrees of
freedom the 99% value at risk is 12.0% above the normal figure and the expected
shortfall 29.4% above; at 99.5% they are 21.3% and 40.6%. A desk that checks its
value at risk against breach counts and never looks at the expected shortfall has
been measuring the smaller of the two errors.

### Filtering, and which filter

`filtered_historical_risk` divides each past return by the volatility estimated at
the time and multiplies by a current one, keeping the empirical *shape* while the
*scale* becomes current. The filter used to be an exponential weighting at a fixed
decay; it can now be anything, which in practice means a fitted model:

```python
fitted = fit_garch(returns)
filtered_historical_risk(
    returns,
    volatilities=fitted.volatilities,
    current=fitted.next_variance(returns[-1]) ** 0.5,   # a forecast, not yesterday
)
```

**Measured, and it does not say what this argument is usually made with.** Six
regime-switching series, a 500-observation window, 4,200 one-step 99% forecasts
scored walk-forward:

| | breach rate (1% nominal) |
|---|---|
| no filter | 1.29% |
| exponential weighting, decay 0.94 | **1.05%** |
| fitted GARCH | 0.76% |

Filtering beats not filtering. Between the two filters the exponential weighting
lands closest to nominal and the model is conservative — so the reasons to prefer
the model are the ones it has anyway: an estimated rather than assumed decay, a
long-run level to revert to, and a forecast at horizons past one step, which an
exponential weighting cannot give at all.

The independence test rejected none of the eighteen runs, which is not evidence
that nothing clustered: 700 observations at 99% is seven breaches, and the test has
nothing to work with at that count.

**One trap found while measuring this.** `Garch.next_variance` uses the *last*
conditional variance of the fitted series, so it is the forecast for the day after
the sample and nothing else. Calling it with an earlier day's return inside a
walk-forward loop mixes that day's surprise with the end of the sample's level, and
it produces a plausible number: it moved the breach rate above from 0.76% to 1.74%
and looked like a finding about the model. Index `volatilities` instead — element
`t` is already the forecast made from returns strictly before `t`.

### A horizon is simulated, not scaled

`Garch.risk` covers one period and refuses more, because over `h` periods the sum
of the innovations is not a member of the family they came from — for the
Student-t it is not a Student-t at all, and even under normal innovations it is a
variance mixture rather than a normal. What the model gives at a horizon is the
*variance*; turning that into a quantile needs the distribution of the sum.

```python
from shortfall import Innovations, fit_garch, horizon_risk

fitted = fit_garch(returns)
ten_day = horizon_risk(fitted, returns, steps=10, paths=40_000)
ten_day.value_at_risk            # with ten_day.standard_error beside it
ten_day.quantile_against_square_root_of_time
horizon_risk(fitted, returns, steps=10, innovations=Innovations.PARAMETRIC)
```

Running the recursion forward gets three things the analytic route cannot. The
**variance path is stochastic** rather than its own expectation, so a large draw
early raises the variance for every remaining step. **Mean reversion** is in it,
because it is in the recursion. And the **shape** of the innovations can be the
empirical one: resampling the model's own standardised residuals assumes no tail
shape at all, which is filtered historical simulation with a filter that
forecasts — the EWMA one in `historical.py` has no long-run level and so no
forecast.

**The two square-root-of-time comparisons disagree, and the sign is not a
constant.** That is the whole argument for simulating. Over five samples at ten
steps and 30,000 paths:

| innovations | quantile ÷ scaled one-step quantile | above one |
|---|---|---|
| normal | mean **1.074**, range 1.011–1.150 | 10 of 10 |
| fitted `t`, ~4.5 degrees of freedom | mean **0.986**, range 0.930–1.025 | 4 of 10 |

Two effects pull against each other. The stochastic variance path makes the total
leptokurtic even when each innovation is normal, pushing the ratio above one — the
ratio of value at risk to volatility goes from 2.334 at one step, the normal's
2.326 as it must, to 2.480 at ten. And aggregation pulls the total towards
normality while the one-step quantile keeps the whole of the innovation's own
tail, pushing it back down.

Note the third column, which is the stronger version of the point. Under normal
innovations the direction is reliable; under a fat tail it straddles one and the
sign depends on the sample. So a caller cannot pick a multiplier that is even
*consistently* wrong, which is a worse position to be in than a known bias.

The same effect makes the innovation shape matter much less over a horizon than
over a day: resampling the residuals instead of drawing normals raises the
expected shortfall by 22% at one step and by 7% at ten.

**The answer is an estimate and says so.** Every result carries a Monte Carlo
standard error, from the spread across twenty batches, with no density at the
quantile to estimate. Read it as a lower bound: over 30 independent runs at ten
steps it came to 0.89 of the observed spread at 20,000 paths — inside the 13%
precision of a 30-run standard deviation — and to 0.73 at 2,000 paths, where a
batch of a hundred paths is estimating a 99% quantile from its own single worst
path.

**Square-root-of-time is not a rounding correction.** Starting at four times
the long-run variance with a persistence of 0.975, a one-year horizon
volatility is **39% below** what scaling today's volatility by the square root
of time gives. Ten days is 4% below. In a calm market the error runs the other
way, and that is the expensive direction: it understates risk while positions
are going on rather than coming off.

**A finding about the test, not the model.** Over thirty samples, a constant
forecast is rejected on independence 29 times against a regime-switching
series and only 13 times against a GARCH(1,1) at realistic parameters — even
though volatility plainly clusters in both. A passing independence test is not
evidence that volatility is constant; it says the breaches did not arrive
together, which is weaker than it looks.

Two implementation decisions worth the words. The constraints go in a
**parameter transform** rather than a penalty — `omega` is an exponential, the
persistence a capped logistic, and a second logistic splits it between `alpha`
and `beta` — so the boundary is at infinity in the free coordinates and no
step can reach it. A penalty lets the optimiser evaluate a persistence of 1.4
and reflect back, and on a persistent series the optimum sits close enough to
the boundary that it never settles. And Nelder-Mead stops on the **size of the
simplex** as well as the spread of its values, because on this surface the
vertices can agree to twelve digits while still far apart in the persistence
direction — the parameters wrong and the likelihood saying they are right.

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

## Volatility from the whole bar

Everything above reads a period as one number, the close-to-close return. A bar
usually carries three more, and the two extremes say a great deal: a day that
closes where it opened after travelling four per cent is not a quiet day, and
the close-to-close estimator records it as one.

```bash
shortfall bars --bars ohlc.csv --ticks 200
```

```
estimator                volatility  vs close-to-close                                        assumes
-----------------------  ----------  -----------------  ---------------------------------------------
close-to-close              17.156%              1.000                nothing; uses one price in four
parkinson                   16.442%              0.958                     no drift, no overnight gap
garman-klass                16.192%              0.944                     no drift, no overnight gap
rogers-satchell             16.144%              0.941                               no overnight gap
garman-klass-yang-zhang     16.190%              0.944                                       no drift
yang-zhang                  16.274%              0.949  nothing; uses all four and the previous close
```

The disagreement is the output. Two estimates of the same quantity that differ
by a third are saying something about the data, and the `assumes` column says
what.

### The efficiency, measured rather than quoted

150 independent samples of 60 bars, each bar built from 256 observations of a
geometric Brownian motion, no drift and no gap — which is to say, data that
satisfies every assumption every estimator makes:

| estimator | close-to-close samples it is worth |
|---|---|
| close-to-close | 1.00 |
| Parkinson | 5.02 |
| Rogers-Satchell | 6.42 |
| Yang-Zhang | 6.78 |
| Garman-Klass-Yang-Zhang | 7.52 |
| Garman-Klass | 7.74 |

Close to the figures the original papers derive, which is the point of
measuring: on data that meets the assumptions, they hold.

### The number the papers do not quote

Every one of those efficiencies is derived for the *continuous* high and low.
Real extremes come from finitely many trades and sit inside the true ones, so
every range estimator is biased **down**. Measured, as a fraction of the true
volatility:

| ticks a bar | close-to-close | Parkinson | predicted |
|---|---|---|---|
| 16 | 0.991 | 0.847 | 0.818 |
| 64 | 1.004 | 0.920 | 0.909 |
| 256 | 1.000 | 0.960 | 0.954 |

Close-to-close is unbiased at every tick count, because it reads only
endpoints and those are observed exactly however few trades happened in
between. That is the other half of the trade: the range estimators buy a
five- to eightfold variance reduction and pay for it with a systematic error
that more data does not remove.

`tick_bias_factor` predicts it. Getting the prediction right needed a step a
first attempt skipped. The discrete extreme's expected shortfall is
`0.5826 sigma / sqrt(m)` at each end — the same overshoot constant,
`-zeta(1/2)/sqrt(2 pi)`, that appears in the continuity correction for a
discretely monitored barrier. Turning it into a *factor* means dividing by the
expected range of the continuous motion, `2 sqrt(2/pi) sigma`, not by the
volatility. Without that normalisation the correction is out by a factor of
1.6: it predicts a 26% shortfall at sixteen ticks where the measured one is
15%.

### Which assumption is false in your data

A trend, in multiples of the per-bar volatility. Shown as the inflation over
the same estimator's no-drift reading, so the tick bias above is divided out:

| drift / volatility | Parkinson | Garman-Klass | Rogers-Satchell | Yang-Zhang |
|---|---|---|---|---|
| 0.5 | 1.043 | 1.012 | 0.992 | 0.994 |
| 1.0 | 1.174 | 1.059 | 0.986 | 0.989 |
| 2.0 | 1.594 | 1.226 | 0.965 | 0.971 |

The inflation is quadratic in the ratio, which decides when to care. An equity
at 16% annual volatility and a 12% expected return has a daily ratio near
0.05, where the effect is four parts in ten thousand and nobody should think
about it. A five-minute bar in a directional hour, or a monthly bar in a
trending year, reaches one and Parkinson reads a sixth more volatility than
there is.

An overnight gap carrying half the total variance:

| estimator | reads |
|---|---|
| close-to-close | 1.005 |
| Parkinson | 0.681 |
| Garman-Klass | 0.668 |
| Rogers-Satchell | 0.667 |
| Garman-Klass-Yang-Zhang | 0.975 |
| Yang-Zhang | 0.978 |

The three intraday estimators measure everything from the open, so a jump that
moves the open, high, low and close together is invisible to them. They read
`1/sqrt(2)` of the truth, times the tick bias they already carry — which is
exactly what the numbers are, and the suite asserts that decomposition rather
than just the totals.

### Non-negativity is proved, not clamped

Garman-Klass subtracts `(2 log 2 - 1) c^2` from `0.5 r^2` and looks as though
it could go below zero on a bar that closed at its own extreme. It cannot: the
high is at least the larger of the open and the close and the low at most the
smaller, so `r >= |c|` by construction and `0.5 r^2` dominates `0.386 c^2`
term by term. Rogers-Satchell is a sum of two products of same-signed factors,
and Yang-Zhang is a positive combination of two sample variances and a
Rogers-Satchell. A clamp here would have been unreachable code standing in for
a proof, so there is a proof and a search over four thousand bars built to sit
on their own extremes, looking for the counterexample.

## Which of two adequate models is better

`validate` asks whether one model is adequate. Ranking two that both pass is a
different question, and it needs a scoring function that is minimised —
uniquely — at the truth:

```python
from shortfall import compare, fz0_loss

first = fz0_loss(returns, var_a, es_a, confidence=0.975)
second = fz0_loss(returns, var_b, es_b, confidence=0.975)
compare(first, second).better        # 1 or 2, or None for a tie
```

The value at risk has such a score of its own — the pinball loss, whose
expected value differentiates to `Φ(v/σ) − c` and so vanishes exactly at the
quantile. **Expected shortfall does not.** It is not elicitable: no `S(e, l)`
is minimised at the shortfall for every distribution. The pair jointly is, and
`fz0_loss` is the zero-homogeneous Fissler-Ziegel function.

### The score anybody builds first is wrong by about a third

The obvious construction is the pinball loss for the quantile plus a squared
error on the breaches for the shortfall, and *given the quantile* it does
elicit the conditional tail mean — which is why it looks right. Jointly it does
not. Its expected value differentiates in the quantile, at the true pair, to
`−φ(z)(VaR − ES)²/α`: strictly negative whenever the shortfall differs from the
quantile, which is always.

So its optimum sits above the truth, and by the squared gap between the two —
a measure of how thick the tail is. Measured: **48.8% above the true value at
risk and 34.5% above the true shortfall at 95% confidence**, 46.0% and 34.9% at
97.5%, 44.3% and 35.7% at 99%.

And the two rank the same pair in opposite orders. A truthful forecaster
against one shading up to the obvious score's own optimum: Fissler-Ziegel
prefers the truthful one, 0.84921 against 1.06369, and the obvious score
prefers the shaded one, 0.07849 against 0.17513. Either comparison looks
decisive.

### What the robust variance is actually worth

`compare` uses a Newey-West standard error and reports the naive one beside it,
because the gap turns out to be small. Measured over four thousand observations
against a log-volatility process: **0.99** at zero persistence — below one,
which is the Bartlett estimator's own finite-sample noise — and **1.10** and
**1.06** at persistences of 0.95 and 0.995. The largest effect on a t statistic
is −3.53 becoming −3.20. The bandwidth earns its place because its direction is
not knowable in advance, not because it is large.

```bash
shortfall score forecasts.csv --es-columns es_a es_b --confidence 0.975
```

## The measure that is both coherent and elicitable

The section above settles for scoring value at risk and expected shortfall
jointly, because neither can be both scored and trusted on its own. There is a
measure that can: among law-invariant risk measures the **expectiles** are the
only ones that are coherent and elicitable at once. The `tau`-expectile is the
point where the expected overshoot and the expected undershoot stand in the
ratio `(1 - tau) / tau`, and the score that elicits it is a weighted squared
error — `|tau - 1{l <= e}| (l - e)^2` — whose first-order condition is that
definition. At `tau = 1/2` it is the mean.

Building them is short. The question worth answering is whether they can be
used, and three measurements say what it costs.

**An expectile level is not transferable between distributions.** The `tau`
reproducing a 97.5% expected shortfall is 0.998603 under a normal and 0.997335
under a standardised Student-t with five degrees of freedom. That gap of 0.0013
looks like nothing and is not: carrying the normal's level over to the `t`
overstates the true expected shortfall by **16.5%**, by 7.9% at eight degrees
of freedom and by 2.4% at twenty. A confidence level is fixed once by a rule
and means the same thing on every book. An expectile level has to be
recalibrated per distribution, and the recalibration reintroduces the
assumption the measure was supposed to avoid. The `expectile` command prints
this translation and its cost, because the number is the point.

**Coherence stops exactly at a half.** The subadditivity gap is exactly zero at
`tau = 1/2` — the half-expectile is the mean and the mean is additive — positive
above it and negative below, growing with the distance: `-0.006` at 0.49,
`-0.214` at 0.2, `-0.418` at 0.05 on one dependence structure, and `-2.37` at
0.02 over a search of two hundred of them.

**Comonotonic additivity is the property given away.** Expected shortfall has
it, so two positions that move together get no diversification credit. An
expectile gives some anyway: on perfectly dependent data the sum of the
expectiles exceeds the expectile of the sum by 0.40% at `tau = 0.9` and 0.69%
at 0.99.

The sample estimator is exact rather than iterated. The sample score is
piecewise quadratic with a break at every order statistic, so its derivative is
piecewise linear and the root on each piece is a division. Checking it against
an independent minimisation of the same score showed which side of that
comparison is imprecise: a quadratic minimum means two forecasts a distance `d`
apart differ in score by `O(d^2)`, so the search tops out near `1e-08` in the
argument while the closed form's residual in the defining condition is `1e-17`.

```bash
shortfall expectile returns.csv --level 0.99 --confidence 0.975
```

## Weighting the whole tail instead of averaging a slice of it

Expected shortfall averages the worst `p` of the distribution, which weights
every loss inside that slice equally and every loss outside it at nothing. Both
halves of that are choices nobody made deliberately: a loss at the 0.1% level
and one at the 2.4% level enter a 2.5% expected shortfall with the same weight,
and a loss at 2.6% does not enter at all.

`shortfall.spectral` replaces the slice with a weight function.

```
rho(X) = integral over u in [0, 1] of phi(u) q_u(X) du
```

`phi` is non-negative and integrates to one, and the measure is **coherent
exactly when `phi` is non-increasing** — the weight has to fall as the outcome
improves. Expected shortfall is the single-step case, which is the whole of why
it is coherent; value at risk is the point-mass limit, which is the whole of
why it is not.

```bash
shortfall spectrum returns.csv --confidence 0.975 --deep 0.005
```

### Four identities, none of them checked against a number this module made

The **shortfall spectrum reproduces `sample_expected_shortfall` to 3.5e-18**.
That existing estimator averages the worst `n p` observations with a partial
one at the edge, by hand; this one integrates a step spectrum against the
empirical quantile. They are unrelated derivations of the same sum.

A **discrete mixture of expected shortfalls equals the step spectrum's
measure, to 6.3e-17** — Kusuoka's representation in the one form that can be
checked against arithmetic rather than against a quadrature.

The **Wang transform has a closed form under a normal**: `mean - shift *
volatility`, exactly, for every shift. Nothing from this module appears in that
statement, so it is what validates the quadrature and the block masses
together.

And **comonotonic additivity holds to rounding**. Every distortion measure is
additive on positions that move together, so two perfectly dependent books get
no diversification credit — which is what a capital rule wants at the top of
the dependence range. On the same comonotonic data the spectra report a gap of
at most **8.6e-15 relative** and an expectile reports **4.3e-04**: eleven
orders of magnitude apart. That is the property expectiles buy elicitability
with, and it is not given up here.

### Two measures agreeing on the headline can disagree by a factor of two

On a Student-t with five degrees of freedom scaled to a 1% daily volatility,
the 97.5% expected shortfall is 2.714%. Matching every spectrum to that same
charge — so a reader given only the number could not tell them apart — the
share of it contributed by the worst 0.5% of outcomes is:

| spectrum | charge | from the worst 0.5% |
| --- | --- | --- |
| expected shortfall | 2.717% | 29.2% |
| proportional hazards | 2.717% | 37.8% |
| exponential | 2.717% | 38.0% |
| Wang transform | 2.717% | 59.7% |

The Wang transform carries **twice as much of an identical headline figure** in
the part of the tail a sample has least to say about. That is the argument for
stating a spectrum rather than a confidence level.

### Coherence is shown, not cited

The proportional-hazards family is increasing below an exponent of one, so it is
not coherent there, and the branch is implemented rather than refused. Over two
hundred ordinary paired samples — Gaussian and cubed-Gaussian marginals,
correlations across the whole range — it violates subadditivity on **200 of
200** at every exponent below one, by up to 97% of the measure itself. At an
exponent of exactly one the gap is **exactly 0.0**, because the measure is the
mean and the mean is additive, and above one there are no violations. A reader
can be told that coherence needs a non-increasing spectrum, or shown what
happens without one on data that was not built to break it.

### The quadrature was wrong twice, the same way twice

The integral under a normal is a composite Gauss-Legendre rule, and the
shortfall comparison above found both failures.

A Gauss rule integrates a polynomial exactly and a jump not at all. The
shortfall spectrum steps from `1/p` to zero at `u = p`, and with that step
straddling a panel the 97.5% figure came out **1.7e-03 relative** from the
closed form and the 99% one **5.3e-03** — twenty basis points of risk on a two
and a half per cent number. Every spectrum now reports its own discontinuities
and they are forced onto panel edges.

Then the endpoint grading, halving inwards from 0.25 thirty times, left the
innermost edge at 2.3e-10 and cost **2.0e-07 relative** on a 99.5% shortfall.
That residual is the truncated sliver times `1/p`, so the error *grew as the
tail got thinner* — the wrong way round for a tail measure. Sixty halvings
instead, each one more panel, and every spectrum with a closed form agrees with
it to **6.7e-16**. The shortfall spectrum is deliberately not special-cased in
`normal_spectral`, because that comparison is the only check on the quadrature
and the other spectra inherit it.

## The tail of a sum, from the structure rather than the history

Every estimator above reads a distribution that is assumed, resampled, or
fitted at the extreme. `shortfall.saddlepoint` uses the one piece of structure
a portfolio actually has: the loss is a **sum**, whose cumulant generating
function is the sum of the obligors' and is closed form even where the density
is not. Tilting that to put the mean on the loss level of interest moves the
expansion point with the question instead of leaving it at the centre.

```bash
shortfall portfolio book.csv --top 5
```

```
100 obligors, mean loss 35.8427, volatility 20.6827, maximum 948.0000
lattice span 1, so the exact distribution is a convolution and the errors below are measured

loss         exact   saddlepoint        err        normal        err
----  ------------  ------------  ---------  ------------  ---------
36    4.453795e-01  4.454120e-01  +7.31e-05  4.969663e-01  +1.16e-01
54    1.792587e-01  1.793490e-01  +5.04e-04  1.899995e-01  +5.99e-02
72    5.334832e-02  5.337284e-02  +4.60e-04  4.021541e-02  -2.46e-01
108   2.205403e-03  2.206045e-03  +2.91e-04  2.426159e-04  -8.90e-01
143   4.426077e-05  4.426917e-05  +1.90e-04  1.103398e-07  -9.98e-01
```

The exact column is the point. Whenever the exposures share a common unit the
loss distribution is a finite convolution, so the approximation's error is
*measured* against something containing nothing of it — no tilt, no expansion,
no normal distribution function. Across exceedance levels from 1e-02 to 1e-06
the saddlepoint is high by 3.3e-04 falling to 1.1e-04, while a normal with the
same first two moments is low by 69% rising to 99.99%. The saddlepoint's error
shrinks into the tail; the normal's grows to everything.

**And the normal approximation is not a deep-tail caveat.** It first errs by
more than ten per cent at an exceedance of 0.46 — essentially at the mean,
because the portfolio is skewed and a normal is not. The saddlepoint is worst
in the *body*, at 5.9e-03 where the exceedance is 0.975.

### Three things that are not the formula

**The lattice correction is not a refinement.** Integer exposures leave the
loss with no density at all, and the continuous form reads 4.7% to 7.2% high —
*growing* into the tail, which is the shape of failure the method exists to
avoid. Correcting it is worth a factor of 150 at a one-in-a-hundred loss and
700 at a one-in-a-million one.

Detecting the span needed a decision. **Every double is a dyadic rational**, so
"is this lattice-valued" asked exactly has the answer "always" and returns a
useless span near 1e-16 for an exposure of pi. The question needs a resolution,
so the denominator is capped: 0.25 and 0.1 come back as 0.05 exactly, pi comes
back as no lattice at all.

**The shortfall comes from an exact decomposition, not a second expansion.**
`E[L 1{L>x}] = sum_i e_i p_i P(L^(i) > x - e_i)` — conditioning on one obligor
defaulting replaces its loss by a constant and leaves the rest independent of
it. So the only approximation is the same tail probability as the total, the
per-obligor contributions a desk asks for fall out for free, and the result is
an order *more* accurate than the probability it divides by. It also hands over
two exact identities: the contributions at a zero level sum to the mean to
1e-14, and at any level they sum to the numerator of the conditional mean.

**The singularity at the mean is removable and the branch cannot be on a
zero.** `w` is a difference of two nearly equal quantities under a square root,
so the generic form loses digits to cancellation long before the saddlepoint
reaches zero, while the limiting form's error shrinks linearly in the tilt. The
gap between them runs 8.2e-02, 8.0e-03, 8.0e-04, 8.0e-05, 7.6e-06 as the tilt
goes 1e-02 to 1e-06, then **rises** to 2.8e-03, 1.14 and 96.0 by 1e-10. The
floor sits at the last clean decade.

### It is not a distribution, and what that means had to be measured

On a hundred names the approximation stays inside `[0, 1]` to within 6.1e-18
and is non-monotone at nine of nine hundred lattice points, all at tail
probabilities below 4e-16 — round-off, not a failure, and saying so needs the
measurement. Two names at a 400-to-1 exposure ratio *is* a failure: the raw
value runs from -3.15 to 4.15, the sequence is not monotone, and the worst
relative error is 300%. There is no asymptotic regime with two summands, so the
result carries a flag rather than only a number.

### The sign the exact reference caught

Lugannani and Rice state the approximation for the *distribution* function, so
the correction enters the tail with a minus. With it the wrong way round the
answer is still plausible and still monotone, and reads 14% high at the mean
rising to 65% in the tail — an error growing as the tail thins, which is exactly
the shape of failure the saddlepoint exists to avoid. It cannot be told from a
method that simply does not work, except against something exact.

## The range, when no dependence is named

Every estimator above answers with a number because it has been told how the
positions move together. Name nothing, keep only the marginal loss
distributions, and the answer becomes an interval — a computable one, with both
ends attained by couplings that are constructed rather than assumed.

```python
from shortfall import dependence_bounds, exponential_loss, normal_loss, pareto_loss

book = [
    normal_loss(0.0, 0.02, label="equity"),
    exponential_loss(0.015, label="credit"),
    pareto_loss(2.5, 0.004, label="operational"),
]
bounds = dependence_bounds(book, confidence=0.99)

print(f"{bounds.best.upper:.4%} to {bounds.worst.lower:.4%}")   # 5.0543% to 17.1541%
print(f"comonotonic {bounds.comonotonic:.4%}")                  # comonotonic 14.0843%
print(f"a factor of {bounds.ratio:.2f}")                        # a factor of 3.39
```

Three marginals, one level, and value at risk is free to be anywhere across a
factor of 3.39. Note where the comonotonic coupling falls: **not at the top**.
Value at risk is not subadditive, so lining every loss up to be large together
is not the worst thing the dependence can do — a coupling that makes the tail
sum flat rather than extreme pushes the quantile higher still, and here it is
1.22 times the comonotonic figure.

The command line puts the covariance-based number inside the range it is one
point of:

```console
$ shortfall bounds examples/returns.csv --confidence 0.99
4 positions over 1260 periods, 99.0% confidence, 512 tail cells

coupling               value at risk  against comonotonic
---------------------  -------------  -------------------
best case (attained)          0.043%                0.014
fitted covariance             1.725%                0.680
comonotonic                   2.537%                1.000
worst case (attained)         2.897%                1.143
```

### Two of the four bounds are proofs and two are arrangements

The bound above the worst case is an inequality and nothing else:
`VaR <= ES` under any coupling, and expected shortfall is subadditive, so the
worst case is at most the sum of the marginals' own expected shortfalls.
Symmetrically, the sum of the lower tail means lies below the best case. Both
are exact closed forms for the marginals built here, which is why each marginal
carries its tail means rather than having them integrated off its quantile
function.

The other two ends come from the rearrangement algorithm: discretise each tail
onto equiprobable cells, then repeatedly replace one column with the
arrangement that runs against the sum of the others. The minimum row sum is the
value at risk of an explicit coupling, so it is attained — provided the grid
understates each marginal, which the left-endpoint one does. The right-endpoint
grid is reported but is **not** a bound, because the algorithm returns an
arrangement and not the discrete optimum, and saying so is cheaper than being
wrong about it.

### The oracle, and the divisibility problem it exposed

The uniform distribution is completely mixable, so uniform marginals have the
closed forms `d (1 + alpha) / 2` and `d alpha / 2` for the two cases. At 512
cells the corrected estimate reproduces the first to between 2.2e-16 and
1.1e-14 at two, four, eight and sixteen positions — every one a divisor of 512
— and to 2.9e-05 at three and five, which is one grid step and the best any
arrangement of 512 points into three equal row sums can do. Whether the tail
can be flattened at all is partly a question about arithmetic, which is not
something the published error bounds mention.

### Most of the gap is the grid, not the dependence

The mean row sum survives any rearrangement, and it is exactly the
left-endpoint Riemann sum of the tail mean the proof uses exactly. So the
distance from the attained bound to the proved one splits into quadrature plus
non-mixability with no residue. They are not the same size: on eight
exponential marginals the gap is 0.0667, of which **0.0631 is the grid and
0.0036 the dependence**. Reading the raw ratio of the two bounds would put 95%
of it down to the wrong cause.

### What the superadditivity is worth

On Pareto marginals with tail index 2 at the 99% level the worst case runs
1.476 times the comonotonic coupling at two positions, 1.791 at four, 1.927 at
eight, 1.985 at sixteen and 1.9997 at thirty-two, approaching the limit
`theta / (theta - 1) = 2`. Sweep the index at eight positions and it reaches
2.712 against a limit of 3 at index 1.5, 1.927 against 2, 1.488 against 1.5 and
1.249 against 1.25: the lighter the tail, the sooner the limit arrives.
Exponential marginals cap at 1.2171, which is their own ratio of expected
shortfall to value at risk at that level.

### Which bracket exists is not symmetric

The worst case's reported grid needs the quantile at one, so it exists only for
a loss bounded above. The best case's needs the quantile at zero, so it exists
for a loss bounded below. A uniform loss has both, a Pareto or exponential loss
only the second, a normal loss neither — and the inequalities hold in every
case, which is why they are what the interval is bracketed by.

## A view imposed by reweighting, not by filtering

Every estimator above reads a sample and takes it as given, which leaves no way
to ask what the risk would be *if* something were true. The usual answer is to
keep the scenarios that match the story and drop the rest, and that does not
produce a stressed scenario set: it produces a third of a sample, with the
dependence between assets re-estimated on whatever is left.

`shortfall.entropy` reweights instead. Among all distributions on the existing
scenarios that satisfy the view, it takes the one closest to the original in
relative entropy. Nothing is discarded, the view holds exactly, and everything
the view did not mention moves only as far as the sample's own dependence
implies.

```bash
shortfall stress returns.csv --on equity --mean -0.01
```

```
1000 scenarios, view on equity: equity mean

                            before    after
--------------------------  ------  -------
mean                        0.028%  -0.394%
value at risk (99.0%)       1.152%   1.411%
expected shortfall (99.0%)  1.284%   1.477%

The view cost 0.421059 nats and left 656.4 effective scenarios of 1000 (65.6%).
Nothing was discarded: every scenario still carries weight, so the dependence
in the sample is intact.

equity moved -1.014%. Regressing the portfolio on it gives a slope of 0.4172,
so shifting the whole loss distribution by hand would raise the expected
shortfall by 0.423%.
It actually moved 0.193%, which is 0.46 of that. The reweighting concentrates
on scenarios where equity was extreme, and the portfolio's worst scenarios are
only partly those, so the shortcut gets the mean right and overstates the tail.
```

Those last three lines are the reason the module exists, and they are also the
honest case against it.

### For a mean, a regression would have done

A view on the mean of one series moves the mean of another by that series'
least-squares slope on the first — not approximately, and not only to first
order. On a thousand scenarios of two correlated series the sample slope is
0.652444, and the slopes implied by views of 0.1, 0.25, 0.5 and 1.0 standard
deviations are 0.652141, 0.651922, 0.652120 and 0.653246: agreement to between
0.05% and 0.12% over a tenfold range of view strength. Nothing linear was
assumed anywhere. The exponential tilt reproduces the projection.

That is worth stating plainly rather than burying, because it says the
machinery is not inventing a relationship — and it says that if the mean of
something else is all anybody wants, this is an expensive way to get a
regression coefficient.

### The tail does not follow, and it misses in both directions

The same views move the second series' expected shortfall by far less than a
parallel shift of its distribution would. At a view of half a standard
deviation the mean moves by the regression amount, 0.3261, while the 99%
expected shortfall moves by 0.1263 — **0.39** of it — and the 95% by 0.81. The
attenuation deepens both further into the tail and as the view strengthens: at
a full standard deviation, 0.35 and 0.63. So the shortcut overstates a 99%
expected shortfall by between two and three times.

And the error does not have a fixed sign, which is worse than it being large. A
view on a *mean* spreads its weight across the whole sample. A view on the
*probability of a tail event* puts its weight where the losses already are, and
moves the tail further than the shift predicts. On one three-asset file, with
one portfolio, the mean view delivers **0.46** of the predicted move and the
tail-probability view **1.79** — so the sentence the command prints under its
table is chosen on the sign, because a fixed one would be wrong half the time.
There is no correction factor to apply to the shortcut, only the reweighting.

### The dual, and a line search that had to be told what it was looking for

The primal problem has one unknown per scenario and a handful of constraints.
Its dual has one unknown per view, its gradient *is* the view residual, and its
Hessian is the posterior covariance of the view functions — so the Hessian is
positive semidefinite by construction, Newton applies, and the stopping test is
a statement about what the caller asked for. Every solve in the tests converges
in five to eight steps. The optimal dual value is minus the relative entropy,
which is computed both ways and checked: a sign dropped in the objective would
otherwise still converge, smoothly, to the wrong point.

The line search took three attempts. Accepting a step on `candidate <= value`
never terminates, because at the optimum the objective stops changing and
zero-progress steps are accepted until the iteration budget runs out — a
converged solve reported as a failure to converge. Requiring strict decrease
terminates and stops too early: the dual value of a small-probability view is
itself around 1e-04 and goes flat to the last bit of a double while the residual
is still 1e-09. Accepting a step that improves the objective *or the residual*
took the same solves to a worst residual of 1.4e-16 and the relative entropy
from 2.3e-08 relative to 5.1e-13.

### The closed form is the only oracle, and the blend is optimal exactly once

A view on a probability has an answer in arithmetic: scale the weights inside
the set, scale them outside, and nothing else can change, because relative
entropy on a partition is minimised cell by cell. Over twenty combinations of
threshold and target the solver matches it to 1.4e-16 on the weights. Views on
a mean have no closed form, so every accuracy assertion in the tests is on the
probability case and the mean views are checked for properties instead.

The same fact settles a question about partial confidence. Taking
`(1 - c) p + c q` is what confidence in a view usually means, and for a
probability view it is *also* the entropy-minimising distribution for the view
value it produces — exactly, at every confidence, measured at 0.0% excess —
because blending two per-cell scalings leaves each cell proportional. For a
mean view it is not, since a blend of two exponential tilts is not an
exponential tilt, and the cost sits where nobody would look for it: at **low**
confidence in a **strong** view. On a view moving the mean a full standard
deviation the blend carries 31.8% more relative entropy than re-solving at the
value it actually produced when `c = 0.25`, 13.8% at 0.5 and 4.1% at 0.75.

### Four failures, four messages

A target outside the range the scenarios span, a view that is constant, two
views that are linearly dependent, and two views each reachable and jointly
impossible are different faults with different fixes. The first two are caught
before the solve starts, by name.

The last two both arrive as a singular Hessian, and telling them apart took a
test. A Hessian singular *at the prior* means the view functions are dependent
over the scenarios and no tilt separates them. One that goes singular later
means the tilt has already driven the weights onto a face of the simplex, where
functions independent over the whole set no longer are — which is what happens
on the way out of the feasible region. The residual distinguishes them, and the
first version of this module reported two disjoint sets each asked for a
probability of 0.6 as linearly dependent views.

## Development

```bash
pip install -e ".[dev]"
pytest          # 1650 tests
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
