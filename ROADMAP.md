# Roadmap

Phases are ordered but not dated. An item is checked when the code is written,
exercised end to end, and covered by tests that run.

Numerical work is checked against a closed form or a published worked example
wherever one exists, and against a slower reference implementation where one
does not. A test that only agrees with the code it tests is not evidence.

## Phase 1 — Return series and covariance

- [x] Return series: construction, alignment across assets, simple and log returns
- [x] Annualisation with an explicit periods-per-year, never an assumed 252
- [x] Sample covariance and correlation, with the unbiased and maximum-likelihood forms
- [x] Ledoit-Wolf shrinkage towards a constant-correlation target
- [x] Shrinkage intensity reported, not hidden, with the target it shrank towards
- [x] Nearest positive semi-definite repair for a matrix that is not one
- [x] Condition number and eigenvalue diagnostics on any estimated matrix

## Phase 2 — Parametric risk

- [x] Normal value at risk and expected shortfall in closed form
- [x] Student-t value at risk and expected shortfall, with the tail scaling right
- [x] Portfolio variance from weights and a covariance matrix
- [x] Cornish-Fisher expansion, refusing the parameters where it is not monotone
- [x] Sign and tail conventions stated once and enforced everywhere

## Phase 3 — Historical and simulated risk

- [x] Empirical quantiles with the interpolation method named
- [x] Historical value at risk and expected shortfall over a return series
- [x] Bootstrap confidence intervals for both
- [x] Filtered historical simulation over a volatility model
- [x] Coherence checks: expected shortfall is subadditive where value at risk is not

## Phase 4 — Risk contributions

- [x] Marginal and component contributions under the Euler allocation
- [x] Contributions sum to the total risk, as an identity the tests assert
- [x] Risk parity weights, with the convergence evidence reported
- [x] Diversification ratio and effective number of bets

## Phase 5 — Factor models

- [x] Factor exposures by regression, with residual diagnostics
- [x] Specific risk and the factor-implied covariance
- [x] Risk attributed between factor and specific components
- [x] Attribution reconciles to the total, as an identity the tests assert

## Phase 6 — Path statistics

- [x] Drawdown series, maximum drawdown, and the dates it ran between
- [x] Underwater duration and time to recovery
- [x] Calmar, Sortino and the ulcer index, each with its denominator stated
- [x] Rolling windows over any of the above

## Phase 7 — Interface, documentation and continuous integration

- [x] Command line entry point over a returns file
- [x] Worked example reproducing a published figure end to end
- [x] README covering the conventions and the design decisions
- [x] Continuous integration across supported Python versions with types and lint

## Phase 8 — Distribution

- [x] `py.typed` inside the package, so the annotations reach anyone who installs it
- [x] Distribution metadata an index can present: authors, keywords, classifiers,
      project URLs, and the licence as an SPDX expression carrying the LICENSE text
      into the artefact, which the legacy licence table did not
- [x] `shortfall --version`, asserted against both the installed metadata and the
      version declared in `pyproject.toml`
- [x] A release driven by a version tag, publishing with the index's trusted
      publishing flow, so no upload credential exists in the repository — and
      refusing to publish when the tag and the declared version disagree
- [x] The sdist and the wheel each installed into a clean environment and made to
      produce an estimate, on pull requests as well as on a tag
- [ ] A first release on the index, which waits on the publisher being registered
      there for this project

## Phase 9 — Gaps found by using it

- [x] Skewness and excess kurtosis on a return series, which the Cornish-Fisher
      estimator requires and nothing here could produce
- [x] A tail-observation count that is not inflated by `count * probability` failing
      to be an exact integer in binary, which overstated the data behind a figure

## Phase 10 — Scoring a model against what happened

Every estimator here produces a forecast, and until this phase nothing could
ask whether a forecast had been any good.

- [x] Chi-square and binomial tail probabilities, to the accuracy standard the
      rest of `distributions` is held to, with the survival functions evaluated
      on their own branch rather than as one minus a distribution function
- [x] Exception indicators over a one-step-ahead forecast series, with the
      loss-sign convention refused rather than guessed at
- [x] Kupiec's unconditional coverage test on the breach count
- [x] Christoffersen's independence test on whether breaches cluster, which a
      count cannot see and which is how a constant-volatility model fails while
      getting the count right
- [x] The conditional coverage test that combines the two
- [x] Supervisory traffic-light zones derived from the binomial, reproducing the
      published 250-day table exactly and also answering for a sample that is
      not 250 days long; the tabulated capital add-on returned only for the
      setup it is published for
- [x] The Acerbi-Szekely statistics for expected shortfall, which is not
      elicitable and so has no breach-count equivalent, with a simulated null
- [x] Size and power of each test established by simulation against a model
      that is correct and a model whose defect was put there on purpose
- [x] A `validate` command over a file of returns and forecasts

Three things the tests record because they are easy to assume and wrong.

Kupiec rejects a clustered process about one time in six even when its
unconditional rate is exactly right, because clustering makes the breach count
overdispersed relative to the binomial its null assumes. The claim worth making
is the gap — independence catches it five times as often — not that the count
is uninformative.

The Acerbi-Szekely test 1 lands at -0.245 when the true volatility is double
the forecast. It is a ratio of tail means, and a normal's conditional tail mean
grows slowly once the threshold is already inside the distribution, so the
statistic is far smaller than the error it is detecting.

An overstated tail can barely be tested at all. A forecast twice too wide
produces zero breaches in four thousand observations, and a statistic read off
zero breaches is not evidence about a tail mean.

## Phase 11 — A volatility process

Phase 10 left the library able to detect a model whose breaches cluster and
unable to offer anything to fix it. `ewma_volatility` at a fixed decay is a
filter rather than a model: the decay is assumed, a shock decays towards
nothing, and the forecast is flat at every horizon.

- [x] GARCH(1,1) fitted by maximum likelihood
- [x] Constraints carried by a parameter transform rather than a penalty, so no
      step can reach the inadmissible region
- [x] Nelder-Mead stopping on the size of the simplex as well as the spread of
      its values, because this surface has a flat ridge the second test alone
      stops early on
- [x] Variance targeting as an option, for the short samples where omega is the
      product of a level and a nearly unidentified factor
- [x] Persistence, long-run variance and the half-life of a shock
- [x] Multi-step forecasts through the mean-reverting recursion, and the
      aggregate horizon variance square-root-of-time approximates
- [x] Standardised residuals, so the fit can be doubted
- [x] A `volatility` command
- [x] Parameters recovered from simulated data with known ones, and the
      optimiser checked against a function whose minimum is known in closed form

The measured results, none of which were assumed beforehand.

Recovery: (2e-6, 0.08, 0.90) comes back as (2.008e-6, 0.0787, 0.8998) from 4000
observations. From 1000 the persistence is materially low, because the
likelihood is flat along that direction and four years of daily data has not
seen enough slow decay to pin it down.

Square-root-of-time: from four times the long-run variance at a persistence of
0.975, the one-year horizon is 39% below the scaled figure and ten days is 4%
below. In a calm market it runs the other way, which is the direction that
costs money.

The loop closes, but only mostly. Over twenty regime-switching samples the
constant forecast is rejected on independence 17 times and the GARCH forecast
once; the breach count goes from about 51 to about 28 against a nominal 20.
Halved rather than fixed — a Gaussian GARCH still understates the tail of a
series whose standardised residuals are fat.

And a finding about phase 10 rather than this one: over thirty samples, a
constant forecast is caught 29 times on a regime-switching series and 13 times
on a GARCH at realistic parameters. The independence test is much weaker
against smooth volatility than against regime switches, so passing it is not
evidence that volatility is constant.

## Phase 12 — The innovation distribution

The last phase halved the breach excess and left the other half on the table.
The variance process was the part that was missing; the *shape* drawn at that
variance was still normal, so the 99% point of a standardised residual was read
off as 2.326 whatever the residuals looked like.

- [x] The Student-t density standardised to unit variance, so the shape
      parameter cannot rescale the variance the recursion is carrying
- [x] The degrees of freedom estimated inside the same likelihood as the
      variance parameters, not fitted to the residuals afterwards
- [x] Conditional value at risk and expected shortfall from the fitted model,
      in the innovation distribution rather than in a normal quantile applied
      to its volatility
- [x] A likelihood ratio test of whether the fat tail is there, since the two
      models are nested
- [x] An unidentified estimate reported as unidentified rather than as a number
- [x] `--innovation auto`, which tests before it assumes

Measured, over the same twenty regime-switching samples as phase 11.

The 99% breach count falls from 28.2 per two thousand observations to 22.45
against a nominal 20 — the excess over nominal from 8.2 to 2.45, so about 70%
of what the variance model left behind. The independence verdict is unchanged
at one rejection in twenty, which is what should happen: the quantile moved and
the clustering stayed fixed.

The residue is not noise. A GARCH fitted to a regime-switching series does not
have identically distributed standardised residuals, because the process is not
a GARCH; one tail index for the whole sample is closer than a normal's and still
an approximation.

On data that never had a fat tail the two agree to within half a breach in
twenty — 19.75 against 19.25 — because the estimate goes to the cap where the
density is the normal's to four decimal places. That is the half of the claim
that stops "widen every forecast by 10%" from passing for the same result.

The test is conservative and the reason is structural: the null puts the inverse
degrees of freedom at zero, which is the boundary of the parameter space, so the
asymptotic null is a half-and-half mixture of chi-square with zero and one
degrees of freedom rather than chi-square with one. Read against chi-square with
one it reports about twice the true probability. Measured over 200 samples with
Gaussian innovations, a nominal 5% test rejected 5 times.

The cap has one visible consequence worth stating. The normal is the limit of
the Student-t family and the cap stops short of it, so on a thin-tailed series
the best admissible t is fractionally worse than the normal and the statistic
comes out slightly negative — about 7e-5 nats per observation. The p-value
clamps it at zero.

## Phase 13 — A horizon that is simulated rather than scaled

One-step risk is a closed form and horizon risk is not. `Garch.risk` refused a
horizon for the right reason — the sum of `h` innovations is not a member of the
family they came from — and that refusal left a caller with an aggregate variance
and no way to turn it into a quantile except the assumption just declined.

- [x] Horizon value at risk and expected shortfall by running the recursion
      forward and taking the quantile of the accumulated return
- [x] Innovations drawn from the fitted family, or resampled from the model's own
      standardised residuals, which assumes no tail shape
- [x] A Monte Carlo standard error, reported rather than left to be guessed at
- [x] Validated against every closed form that exists: one step under normal and
      under Student-t innovations, and the horizon volatility against the
      analytic aggregation at three horizons
- [x] `--paths` on the command line, off by default
- [x] The direction of the horizon effect measured, in both signs

Measured, and the headline is that the sign is not a constant.

Over ten samples at ten steps and 30,000 paths, the horizon value at risk against
the square-root-of-time figure came to a mean of 1.074 under normal innovations
(range 1.011 to 1.150, above one on all ten) and 0.986 under a fitted tail near
four and a half degrees of freedom (range 0.930 to 1.025, above one on four of
ten).

Two effects pull against each other: the stochastic variance path makes the
accumulated return leptokurtic — the ratio of value at risk to volatility goes
from 2.334 at one step, which is the normal's 2.326 as it must be, to 2.480 at ten
— while aggregation pulls the total towards normality when the innovation itself
is fat.

The counts are the sharper finding and they replace a first draft that quoted the
fat case's mean as though the sign were reliable. It is not: under normal
innovations the direction holds every time and under a fat tail it straddles one.
A caller cannot choose a multiplier that is even consistently wrong, which is a
worse position than a known bias. The volatility comparison does not change sign
between the two cases either, so nothing applied to a scaled volatility reproduces
either column.

The same aggregation makes the innovation's shape matter much less over a horizon
than over a day. Resampling the standardised residuals instead of drawing normals
raises the expected shortfall by 22% at one step and by 7% at ten. A one-step tail
adjustment does not scale to a horizon whatever it is multiplied by.

The error bar is the part that needed the most care and is reported as a lower
bound. It is the spread across twenty batches, which needs no density estimate at
the quantile, and over 30 independent runs at ten steps it came to 0.89 of the
observed spread of the reported figure at 20,000 paths — inside the 13% precision
of a 30-run standard deviation — and 0.73 of it at 2,000 paths, where a batch of a
hundred paths estimates a 99% quantile from its own single worst path.

The bootstrap floor is 250 residuals rather than the fit's own 100. A bootstrap
cannot draw past the worst residual it holds, so at a hundred residuals a 99%
figure sits on the boundary of the data instead of inside it. Below that the
parametric draw is the honest route: it extrapolates, and says so.

## Phase 14 — The one-step filter, and what measuring it showed

Phase 13 closed the horizon half of the filtering gap and left the one-step half
open: `filtered_historical_risk` still rescaled by an exponential weighting at an
assumed decay, with no long-run level and no forecast to rescale *to*.

- [x] The filter supplied by the caller, one value per return and aligned the same
      way, so a fitted model's series goes straight in
- [x] The level rescaled to stated separately, so it can be a forecast rather than
      the filter's last value
- [x] The identity a constant filter must satisfy kept, and tested at three
      different constants
- [x] The two filters scored against each other walk-forward rather than argued
      about

Measured over six regime-switching series, a 500-observation window and 4,200
one-step 99% forecasts: no filter breached 1.29% of the time, an exponential
weighting 1.05%, and a fitted GARCH 0.76%, against a nominal 1%.

That is not the result the argument for model filtering is usually made with.
Filtering beats not filtering, and between the two filters the exponential
weighting lands closest to nominal while the model is conservative. The reasons to
prefer the model are the ones it has anyway — an estimated decay, a long-run level,
and a forecast at horizons past one step, which an exponential weighting cannot
give at all — and the README says so rather than reporting the comparison as a win.

The independence test rejected none of the eighteen runs, which is not evidence
that nothing clustered: seven breaches per run is not enough for it to see
anything. That is the same weakness phase 11 measured from the other direction.

One trap found while measuring, now in `next_variance`'s own docstring. It uses the
*last* conditional variance of the fitted series, so it forecasts the day after
the sample and nothing else. Calling it with an earlier day's return in a
walk-forward loop mixes that day's surprise with the end of the sample's level and
returns a plausible number: it moved the breach rate from 0.76% to 1.74% and looked
like a finding about the model rather than about the call.

## Phase 15 — The far tail, where the sample has nothing to say

Historical simulation cannot return a loss larger than the worst one observed, and
the parametric routes fit a shape to the whole sample, where the body dominates
the likelihood. At 99.9% on a few years of daily data both answer a question about
the tail with information about the middle.

- [x] A generalised Pareto fitted to the exceedances over a threshold, by
      Grimshaw's one-dimensional reduction of the likelihood
- [x] The same fit by probability-weighted moments, in closed form, because the
      two disagree where the sample is short
- [x] Asymptotic standard errors where they exist, and absent rather than
      misleading where they do not
- [x] Value at risk, expected shortfall and return levels from the fit, with the
      exponential limit forms across the removable singularity at shape zero
- [x] Threshold diagnostics: the mean excess curve and the Hill shape as a
      function of the order statistics it uses
- [x] A refusal for a confidence inside the body, naming the lowest legal one
- [x] A refusal for a shape with no finite mean, keeping the value at risk
- [x] A refusal for a fit whose upper bound is below a loss it was fitted to
- [x] A command line report with the historical figure printed beside it, and the
      count of observations beyond the answer
- [x] The comparison against the truth measured in a worked example that runs in
      continuous integration

The measurement is the part worth keeping, because it does not say what the
argument for extreme value theory implies. On 2,000 draws of a Student-t on four
degrees of freedom, whose tail index is exactly 0.25:

At 99% the fit and historical simulation land within half a percentage point of
each other — 4.7% against 5.2% mean absolute error — because twenty observations
are still out there to be read off. What the fit removes as the question moves
past the data is the bias, not the noise. At 99.99% historical simulation returns
the worst loss in the file and is short by 21.1% on average, in the same direction
every time; the fit averages +0.2%. Its spread is 41.6%, so the claim worth making
is that it stops being systematically short, not that it becomes accurate.

The fitted normal is short by 39.0% at 99.9% and 59.6% at 99.99%, with a spread of
3 to 4%. Precise, consistent and wrong in one direction, which is worse than
noisy: a varying estimator is telling you it is uncertain.

The threshold trade contradicts the usual advice. Raising the threshold from the
top fifth to the top twentieth cuts the shape's bias — 0.10 to 0.13 against a true
0.25 — at double the spread, while the 99.9% quantile it is chosen for moves by
under two percentage points anywhere from the top fifth to the top hundredth,
because the fitted scale absorbs what the shape gets wrong. The default of 5% is
defensible and not better than 20% for the number it is picked for. The top 1% is
where it breaks: twenty exceedances give a shape whose spread is several times its
own true value and whose mean is negative.

Two implementation errors are recorded because both produced plausible output. The
optimiser's stopping rule had an absolute term, so it was a length in the units of
one over a loss and the dimensionless shape depended on whether returns were
quoted as fractions or as basis points. And the search bound came from doubling
until the profile stopped improving, which stepped past the maximum on a shape of
0.5 and settled 0.17 of log-likelihood below what a grid search could find.

One finding about a second opinion. The Hill estimator reads high on generalised
Pareto data — 0.55 against a true 0.35 on the worst 400 of 4,000 — because it
assumes a tail Pareto about the origin and a generalised Pareto is shifted by
`scale / shape`, which at that threshold is 2.86 against a 90th percentile of 3.5.
So a Hill curve disagreeing with the fit is a statement about the threshold, not
evidence that either is broken.

## Phase 16 — Dependence that is not elliptical

Every multi-asset estimate so far read a covariance matrix. Under a normal that
forces the probability of two assets being in their own tails together to zero at
any correlation below one; under a multivariate t it forces one number for every
pair, symmetric between the tails. A portfolio that is mildly correlated day to
day and moves as one in a crash therefore had no representation here, and it is the
portfolio the estimate exists for.

- [x] Kendall's tau-b counted with a Fenwick tree rather than over pairs, with the
      tie correction, checked against the quadratic definition on samples built to
      be full of ties
- [x] Spearman's rho, and both elliptical inversions, with the projection onto the
      nearest valid correlation matrix reported rather than performed quietly
- [x] Gaussian and Student-t copula log-likelihoods, with the marginal densities
      dividing out
- [x] Degrees of freedom by profile likelihood, checked against a grid over the
      whole range, with a maximum at the upper bound returned as the bound
- [x] The coefficient of tail dependence in closed form, cross-checked against the
      elementary antiderivative at five degrees of freedom
- [x] Marginals empirical or spliced with a fitted generalised Pareto tail, as a
      separate switch from the dependence
- [x] Simulated portfolio risk with the Gaussian-copula figure from the same normal
      draws beside it, and a batched Monte Carlo standard error
- [x] A `copula` command, and the whole payload asserted to survive a strict JSON
      encoder

Measured on the case it exists for: five equally weighted assets, 1,500
observations from a t copula at 4 degrees of freedom with every pairwise tau at
0.35, empirical marginals from the same sample, 20,000 paths, over three samples
and three simulation seeds each. The fitted copula puts 99% expected shortfall
10.6% above the Gaussian copula's on the identical marginals, correlation matrix
and normal draws — 0.0433 against 0.0391 — with a spread of 2.6 percentage points
across the nine runs and a range of 6.7% to 13.7%. At 99.5% it is 13.7%. Fitted
degrees of freedom 4.2, range 4.0 to 4.5.

A first attempt at this figure averaged three samples with one simulation seed
each, which conflated sample variation with simulation variation and reported
8.6% with a range that did not contain the properly averaged answer. The spread is
quoted here because the point estimate on its own was misleading.

Sweeping the pairwise tau on the same construction turned up the finding worth
having, because it reverses the obvious expectation. Premium at 99%, three
simulation seeds each: +19.7% at a tau of 0.05, +17.6% at 0.15, +12.0% at 0.30,
+5.9% at 0.50, +1.5% at 0.70 and −1.0% at 0.90. It is largest where the
correlation is *lowest*.

The reason: at a correlation near one the Gaussian copula already moves everything
together, so the portfolio is one asset and no copula changes that asset's own
marginal tail. At a correlation near zero the Gaussian copula promises real
diversification in the extremes, and that is the promise that is false — the t
copula's tail dependence coefficient at a correlation of zero and four degrees of
freedom is 0.0756, because the shared mixing variable does not consult the
correlation. The portfolio this method is for is the one that looks diversified.

The mechanism is starker in the copula alone. All five assets below their own 5%
point: 0.42% of draws under the fitted copula against 0.14% under the Gaussian one.
Below their own 1% point: 0.057% against 0.005%. Independence gives 3.1e-7 and
1e-10. A factor of three becomes a factor of eleven one quantile deeper, because
one of the two limits is zero.

**The negative result matters more than the positive one.** On 1,500 observations
from a genuine Gaussian copula the same procedure fits 92 degrees of freedom and
reports a premium of 0.2% over the same nine runs, spread 0.2 percentage points,
range −0.1% to +0.5%. On `examples/returns.csv` it lands at 28.5 degrees of
freedom with a likelihood ratio of 5.2 — weak evidence — and the premium behaves
accordingly: +0.4% over six simulation seeds with a spread of 1.3 percentage
points, not distinguishable from zero.

The paired draws are worth about a factor of two and not more. Sharing the normals
halves the spread of the difference — 2.4 percentage points against 4.6 on the
synthetic sample, 1.3 against 2.8 on the bundled one — and cannot do better,
because the chi-square mixing draw is not shared and is the whole difference
between the two copulas.

Two things were measured because they were about to be asserted instead. One joint
8-sigma point added to 300 independent observations moves a Pearson correlation by
0.175 and Kendall's tau by 0.0066; the bound on the second is
`2(1 + |tau|)/(n + 1)`, which is 0.0066 and *not* the `2/n` a first draft claimed —
appending a point changes the denominator as well as the numerator. And the closed
form for tail dependence at four degrees of freedom and a correlation of 0.5 is
0.25317, not the 0.2546 written from memory; it is now checked against the
elementary antiderivative of the t density at five degrees of freedom, which shares
no code with the series the library evaluates.

One defect, and it would have been intermittent. The generalised Pareto splice
point has to be the fit's realised exceedance fraction, not the tail fraction asked
for: a threshold at the 5% point of 800 losses is exceeded by 39 of them, so the
fit describes the worst 4.875% and refuses anything shallower. Splicing at the
nominal 5% routed a thin band of probabilities to a tail that declines to answer
for them, which surfaced on one seed in five.

The optimisation worth recording is exact rather than approximate. The quantile
function is the expensive part of a copula likelihood, and the ranks of every
column are a permutation of `1..T` — so a panel of `T` observations and `d` assets
presents `T` distinct probabilities, not `T*d`. Memoising within the evaluation
took one pass on 800 observations of four assets from 835ms to 201ms, and the whole
fit from 40s to 6.6s, landing at a likelihood no lower than a quarter-step grid
search over the same range.
