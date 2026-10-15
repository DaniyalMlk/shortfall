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

The two measures then disagree about the sign, which is the sharper finding. On
that same book, fitted against Gaussian: value at risk −4.0% and expected
shortfall +6.0% at 95%; +9.5% and +19.2% at 99%; +16.1% and +24.9% at 99.5%;
+29.4% and +35.2% at 99.9%. At 95% the value at risk is *lower* under the copula
that has the tail dependence in it, because tail dependence moves mass from the
near tail to the far tail and the total is one — a quantile close to the body has
less beyond it, while the mean of what is beyond is larger. A reader taking the
95% value at risk alone would conclude the assumption made the portfolio safer.
Both figures travel together in the result for that reason.

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

## Phase 17 — The whole bar, not just the close

- [x] A validated bar type: a high below its own close or a low above its open
      is refused where it enters, not where it bites
- [x] Parkinson, Garman-Klass, Rogers-Satchell, the open-jump-adjusted
      Garman-Klass and Yang-Zhang, each with the assumption it rests on stated
- [x] Efficiency against close-to-close measured on the caller's own sample
      size rather than quoted from the literature
- [x] The discretisation bias of the observed range measured, and predicted by
      a closed form that is checked against the measurement
- [x] Drift and gap sensitivity measured for every estimator, so the choice
      between them is made on what is false in the data rather than on which
      number is largest
- [x] Non-negativity proved rather than clamped
- [x] A command-line entry point reading OHLC rows and reporting the
      disagreement between the estimators as the diagnostic it is

## Phase 18 — Which of two adequate models is better

- [x] The pinball loss, strictly consistent for the value at risk alone
- [x] A strictly consistent joint score for the pair, since the shortfall is
      not elicitable by itself
- [x] Consistency demonstrated exactly rather than simulated
- [x] The score anybody builds first, shown not to be consistent, with the size
      of its bias and the first-order condition that explains it
- [x] A Diebold-Mariano comparison with an autocorrelation-robust variance, and
      a measurement of what that variance is actually worth
- [x] A command-line entry point scoring two forecast series against the same
      returns

`backtest` asks whether one model is adequate: the right number of breaches,
spread out, with tails of the right size. That is a hypothesis test and it
cannot rank two models that both pass. Ranking them needs a *scoring function*,
and a scoring function is worth nothing unless it is strictly consistent — the
expected score minimised, uniquely, at the truth. Otherwise a forecaster who
optimises it is optimising towards something else.

**The value at risk has one; the shortfall does not have one of its own.** The
pinball loss is strictly consistent for a quantile — for a normal law its
expected value differentiates to `Phi(v/sigma) - c`, zero exactly at the
quantile. Expected shortfall is not elicitable at all, so the pair has to be
scored jointly, and the zero-homogeneous Fissler-Ziegel function is the usual
choice. Its two first-order conditions reduce to the definitions of the two
quantities, and minimising its exact expected value on a refining grid recovers
the true pair to 2e-08 at three confidence levels.

**Zero homogeneity is a weaker statement than it sounds, and the test had it
wrong first.** The score is not invariant under rescaling: multiplying
everything by `k` moves it by exactly `log k`, from the `log e` term. The shift
does not depend on the forecast, so every *difference* between two models is
untouched and the ranking cannot move — which is what makes the member
scale-free, and is not what the first draft of the test asserted. It failed by
precisely `log 1000`.

**The score anybody builds first is wrong by about a third.** The obvious
construction is the pinball loss for the quantile plus a squared error on the
breaches for the shortfall, and *given the quantile* it does elicit the
conditional tail mean, which is why it looks right. Jointly it does not. Its
expected value differentiates in the quantile, at the true pair, to

    -phi(z) (VaR - ES)^2 / alpha

which is strictly negative whenever the shortfall differs from the quantile —
that is, always — so the optimum sits above the truth by the squared gap
between them, which is a measure of how thick the tail is. Measured: **48.8%
above the true value at risk and 34.5% above the true shortfall at 95%
confidence**, 46.0% and 34.9% at 97.5%, 44.3% and 35.7% at 99%. Not a
subtlety; a third of the number.

**And the two scores rank the same pair in opposite orders.** A truthful
forecaster at 97.5% against one shading both numbers up to the obvious score's
own optimum: under Fissler-Ziegel the truthful one wins, 0.84921 against
1.06369; under the obvious score the shaded one wins, 0.07849 against 0.17513.
A desk choosing its model with the obvious score picks the wrong one, and the
comparison looks decisive either way round.

**The robust variance is real and modest, which is worth saying because the
first draft asserted otherwise.** "Score differences autocorrelate under
volatility clustering, so a naive standard error understates the sampling
error" is a plausible sentence and was written before it was checked. Measured
over four thousand observations comparing two exponentially weighted models
against a log-volatility process: at zero persistence the robust standard error
is **0.99** times the naive one — *below* one, which is the Bartlett estimator's
own finite-sample noise rather than a correction in the wrong direction — and
at persistences of 0.95 and 0.995 it is **1.10** and **1.06**. The largest
effect on a t statistic across those runs is -3.53 becoming -3.20. The
bandwidth earns its place because its direction is not knowable in advance, not
because it is large, and `compare` returns both standard errors so a reader can
see that for themselves.

## Phase 19 — The measure that is both, and what it costs to use it

- [x] The expectile, by the condition that defines it, with both of its partial
      moments carried on the result
- [x] An exact sample estimator rather than an iterated one, from the piecewise
      structure of the sample score
- [x] The asymmetric squared loss, with consistency checked against the exact
      expected score
- [x] Closed forms under a normal and a standardised Student-t
- [x] The level matching, so an expectile can be stated in units already in use
- [x] Subadditivity above a half and its failure below, found rather than cited
- [x] Comonotonic additivity, which expectiles do not have, measured
- [x] A command-line entry point that translates its own level and prices the
      translation

Phase 18 ends on a compromise. Value at risk is elicitable and not coherent,
expected shortfall is coherent and not elicitable, and the package's answer is
to score the pair jointly. There is a measure that is both: among law-invariant
risk measures the **expectiles** are the only ones that are coherent and
elicitable at once, and the score that elicits them is a weighted squared error
whose first-order condition *is* their definition.

So the interesting work is not building them, which is short. It is answering
whether they can be used, and that turns out to be an empirical question.

**A level is not transferable, and the gap is not small.** A confidence level
is fixed once by a rule and means the same thing on every book. An expectile
level does not, because the same `tau` is a different amount of conservatism on
a different tail. The `tau` reproducing a 97.5% expected shortfall is
**0.998603** under a normal and **0.997335** under a standardised Student-t with
five degrees of freedom — 0.0013 apart, which looks like nothing. Carrying the
normal's level over to the `t` overstates the true expected shortfall by
**16.5%**, and by 7.9% at eight degrees of freedom and 2.4% at twenty. The
error is worst exactly where a tail measure was supposed to help. Recalibrating
per distribution fixes it and reintroduces the distributional assumption the
exercise was meant to avoid.

**Coherence stops at a half, and the boundary is exact.** The gap
`e(X) + e(Y) - e(X + Y)` is exactly zero at `tau = 1/2`, because the
half-expectile is the mean and the mean is additive. Above a half it is
positive, below it negative, and the magnitude grows with the distance from a
half: on one dependence structure, `-0.006` at 0.49, `-0.063` at 0.4, `-0.214`
at 0.2 and `-0.418` at 0.05. A search over two hundred dependence structures
found `-2.37` at `tau = 0.02`. Below a half an expectile is not a coherent risk
measure, and the theorem's boundary is a real edge rather than a technicality.

**And comonotonic additivity is the property given up.** Expected shortfall has
it: two positions that move together offer no diversification, and it reports
none, which is what a capital rule wants at the top of the dependence range. An
expectile reports some anyway. On perfectly dependent data — one position an
increasing transform of the other — the sum of the expectiles exceeds the
expectile of the sum by 0.40% at `tau = 0.9`, 0.48% at 0.975, 0.69% at 0.99,
and exactly zero at a half. That is small enough to argue about rather than
small enough to ignore, which is the honest summary.

**The estimator is exact, and the test of it found out why that matters.** The
sample score is piecewise quadratic with a break at every order statistic, so
its derivative is piecewise linear and the root on each piece is one division;
scanning the pieces finds the one containing it. The usual implementation
bisects the same derivative instead. Checking the closed form against an
independent minimisation of the sample score showed which side of that
comparison is the imprecise one: the score is quadratic at its minimum, so two
forecasts a distance `d` apart differ in score by `O(d^2)`, and the search tops
out around `1e-08` in the argument while the closed form's residual in the
defining condition is `1e-17`.

Convergence is asserted against the sandwich formula rather than against the
error getting smaller. The estimating equation gives an asymptotic standard
error of `1.4285 / sqrt(n)` for a standard normal at `tau = 0.95`, and the root
mean squared error over independent samples matches it at 500 and 4000
observations with the bias inside three standard errors of zero. The first
version of that test reused one seed across sample sizes and read a 1.5-sigma
draw at the largest as a failure to converge — the same trap as the simulation
study in phase 15, and now written into the test's own docstring.

## Phase 20 — Weighting the whole tail, not averaging a slice of it

- [x] A spectrum as a first-class object, with its density, its block masses in
      closed form and its own discontinuities reported
- [x] The shortfall, exponential, proportional-hazards and Wang spectra, plus an
      explicit mixture of shortfall spectra
- [x] The sample estimator as a finite weighted sum with no approximation in it
- [x] The split of a charge into the far tail and the rest, so two matched
      spectra can be told apart
- [x] Subadditivity and comonotonic additivity on the data, with the
      non-coherent branch implemented and its failure measured
- [x] Closed forms under a normal, with the Wang transform exact
- [x] A command-line entry point that matches several spectra to one charge

Expected shortfall weights every loss inside its slice equally and everything
outside it at nothing. Both halves of that are choices nobody made
deliberately. A spectral measure replaces the slice with a weight function, and
coherence becomes a property of that function — non-increasing — rather than of
the construction. Expected shortfall is the single-step case, which is the whole
of why it is coherent; value at risk is the point-mass limit, which is the whole
of why it is not.

**Four identities, and none is checked against a figure this module produced.**
The shortfall spectrum reproduces `historical.sample_expected_shortfall` to
**3.5e-18** — that estimator averages the worst `n p` observations with a
partial one at the edge, by hand, and this one integrates a step spectrum
against the empirical quantile, so they are unrelated derivations of the same
sum. A discrete mixture of expected shortfalls equals the step spectrum's
measure to **6.3e-17**, which is Kusuoka's representation in the one form
checkable against arithmetic instead of against a quadrature. The Wang
transform's measure of a normal is `mean - shift * volatility` **exactly**, for
every shift, with nothing from this module in the statement. And comonotonic
additivity holds to rounding.

**That last one is the contrast with Phase 19.** Every distortion measure is
additive on positions that move together, so two perfectly dependent books get
no diversification credit, which is what a capital rule wants at the top of the
dependence range. On the same comonotonic data the spectra here report a gap of
at most **8.6e-15 relative** against the expectile's **4.3e-04** — eleven orders
of magnitude. Phase 19's answer to "can expectiles be used" was that the level
does not transfer; this is the other half of the bill.

**Two measures agreeing on the headline can disagree by a factor of two about
where it came from.** Matching each spectrum to the 97.5% expected-shortfall
charge on a `t(5)` scaled to a 1% daily volatility — so every row is the same
number a committee would see — the share contributed by the worst 0.5% of
outcomes is **29.2% under expected shortfall, 37.8% under proportional hazards,
38.0% under the exponential spectrum and 59.7% under the Wang transform**. The
Wang transform carries twice as much of an identical figure in the part of the
tail a sample has least to say about. That is the argument for stating a
spectrum rather than a confidence level.

**Coherence is shown rather than cited.** The proportional-hazards family is
increasing below an exponent of one, and the branch is implemented instead of
refused: over two hundred ordinary paired samples it violates subadditivity on
**200 of 200** at every exponent below one, by up to 97% of the measure. At an
exponent of exactly one the gap is **exactly 0.0** — the measure is the mean and
the mean is additive — and above one there are no violations at all. A reader
can be told the rule or shown the failure on data that was not built to produce
it.

**The quadrature under a normal was wrong twice, the same way twice, and the
shortfall comparison found both.** A Gauss rule integrates a polynomial exactly
and a jump not at all. The shortfall spectrum steps at `u = p`, and with the
step straddling a panel the 97.5% figure came out **1.7e-03 relative** from the
closed form and the 99% one **5.3e-03**, which is twenty basis points of risk on
a two and a half per cent number. Every spectrum therefore reports its own
discontinuities and they are forced onto panel edges — the same requirement, and
the same size of failure, as putting a payoff kink on a grid node. Then the
endpoint grading left the innermost edge at 2.3e-10 and cost **2.0e-07** on a
99.5% shortfall: the truncated sliver times `1/p`, so the error grew as the tail
thinned, which is the wrong way round for a tail measure. Sixty halvings instead
and the agreement is **6.7e-16**. The shortfall spectrum is deliberately not
special-cased in `normal_spectral`, because that comparison is the only check on
the quadrature and the other spectra inherit it.

One test defect of my own, fixed rather than loosened. The Wang estimator's
convergence was first asserted as one draw per sample size getting smaller, and
it does not reliably: the 2,000-observation draw came in at 2.3e-05 and the
32,000 one at 1.0e-04. That is one lucky sample, not a failure to converge, and
a single-draw comparison is a coin flip dressed as a convergence test. It now
asserts the root-mean-square error over twelve independent samples against the
predicted root-`n` rate. This trap has now appeared in five consecutive phases
and the rule that stops it is the same every time: independent seeds per row,
and a derived prediction rather than "it got smaller".

## Phase 21 — The tail of a sum, from the structure rather than the history

- [x] An obligor type and a portfolio of them, validated
- [x] The cumulant generating function and its first three derivatives in
      closed form, with the tilted probability written so a large tilt
      saturates instead of overflowing
- [x] The lattice span of the exposures detected by exact rational arithmetic,
      with a stated resolution
- [x] The saddlepoint solved on a bracket, using the fact that the first
      derivative is increasing
- [x] Lugannani-Rice tail probabilities, with the lattice correction
- [x] The limiting form at the mean, over a neighbourhood whose width is
      measured rather than chosen
- [x] An expected shortfall from an exact decomposition over obligors, and the
      per-name contributions it hands over for free
- [x] The exact distribution by convolution on the lattice, as the reference
- [x] The error measured against it across the tail, and against a normal
      approximation
- [x] Where the approximation leaves `[0, 1]` or loses monotonicity, measured
- [x] A command-line entry point printing the exact column beside the
      approximate one

Every tail measure before this one reads a distribution that is assumed, from
`parametric`, resampled, from `historical`, or fitted at the extreme, from
`extreme`. None of them uses the fact that a portfolio loss is a **sum**, whose
cumulant generating function is the sum of the obligors' and is closed form even
where the density is not.

Tilting the distribution to put its mean on the loss level of interest moves the
expansion point with the question instead of leaving it at the centre, which is
why the relative error stays bounded as the tail thins. On a hundred names with
default probabilities from 0.49% to 7.72% and integer exposures from 1 to 20,
measured against an exact convolution: the saddlepoint is high by 3.3e-04,
2.5e-04, 2.0e-04, 1.6e-04 and 1.1e-04 at exceedance levels of 1e-02 down to
1e-06, while a normal with the same first two moments is **low by 69%, 94%,
99.45%, 99.96% and 99.99%**. The saddlepoint's error shrinks into the tail and
the normal's grows to everything.

**And the normal approximation is not a deep-tail caveat.** It first errs by
more than ten per cent at an exceedance of **0.46** — essentially at the mean,
because the portfolio is skewed and a normal is not. The saddlepoint, by
contrast, is worst in the *body*: its largest relative error over the whole
lattice is 5.9e-03, at an exceedance of 0.975.

### Three things that are not the formula

**The lattice correction is not a refinement.** Exposures sharing a common unit
leave the loss with no density at all, and the continuous form of the formula
applied to it reads 4.7% to 7.2% high across those same levels — *growing* into
the tail, which is the shape of failure the method exists to avoid. Correcting
it is worth a factor of 150 at a one-in-a-hundred loss and 700 at a
one-in-a-million one. The correction replaces `t sqrt(K'')` by
`(2/d) sinh(td/2) sqrt(K'')` and solves at the midpoint, and the continuous form
is the limit of that as the span goes to zero rather than a separate branch.

Detecting the span needed a decision that was not obvious. **Every double is a
dyadic rational**, so "is this lattice-valued" asked exactly has the answer
"always" and returns a useless span near 1e-16 for an exposure of pi. The
question needs a resolution, so the denominator is capped at 1024 — exposures of
0.25 and 0.1 come back as 0.05 exactly, and pi comes back as no lattice at all.

**The shortfall comes from an exact decomposition, not a second expansion.**
`E[L 1{L>x}] = sum_i e_i p_i P(L^(i) > x - e_i)`, because conditioning on one
obligor defaulting replaces its contribution by a constant and leaves the rest
independent of it. So the only approximation is the same tail probability as the
total, the two cannot drift apart, the per-obligor contributions a desk asks for
fall out for free, and the result is an order *more* accurate than the
probability it divides by — within 1.2e-04 at a loss of 40 and 4.3e-05 at 120.
It also gives two exact identities to check against: the contributions at a zero
level sum to the mean to 1e-14, and at any level they sum to the numerator of the
conditional mean.

**The singularity at the mean is removable and the branch cannot be on a
zero.** `w` is a difference of two nearly equal quantities under a square root,
so the generic form loses digits to cancellation long before the saddlepoint is
exactly zero, while the limiting form's own error shrinks linearly in the tilt.
Swept: the gap between the two runs 8.2e-02, 8.0e-03, 8.0e-04, 8.0e-05, 7.6e-06
as the tilt goes 1e-02 to 1e-06 — the limit's error, falling — and then **rises**
to 4.1e-06, 2.8e-03, 1.14 and 96.0 at 1e-07 through 1e-10, which is the
cancellation taking over. The floor sits at 1e-06, the last decade in which the
generic form is still clean.

### It is not a distribution, and what that means had to be measured

Nothing in the expansion is monotone and nothing keeps it inside `[0, 1]`. On
the hundred-name portfolio it stays inside to within 6.1e-18 and is
non-monotone at nine of nine hundred lattice points, every one at a tail
probability below 4e-16 — round-off, not a failure of the method, and saying so
needs the measurement. Two names with a 400-to-1 exposure ratio *is* a failure
of the method: the raw value runs from **-3.15 to 4.15**, the sequence is not
monotone, and the worst relative error is **300%**. There is no asymptotic regime
with two summands, and the result carries a flag rather than only a number.

### One sign, and one convention

The correction enters the tail with a minus, because Lugannani and Rice state
the approximation for the *distribution* function. With it the wrong way round
the answer is still plausible and still monotone, and reads 14% high at the mean
rising to 65% in the tail — an error that grows as the tail thins, which is
exactly the shape of failure the saddlepoint exists to avoid, so it cannot be
told from a method that simply does not work. Only the exact reference
distinguishes the two.

And `distributions.binomial_sf` is `P(X >= k)` where this module's tail is
`P(L > x)`. Compared without the shift, the convolution and the survival
function disagree by 0.10 at the median of a binomial — which reads as a
catastrophic accuracy failure and is a convention. The convolution is exact to
1.1e-16 against integer arithmetic; it was never the suspect.

## Phase 22 — A view, imposed without discarding anything

- [x] A view as a linear statement about the posterior, with the mean and the
      probability of a set as named constructors
- [x] The solve in the dual, whose dimension is the number of views rather than
      the number of scenarios, and whose Hessian is the posterior covariance of
      the view functions
- [x] The identity between the optimal dual value and the relative entropy
      asserted, since the two are reached by different routes
- [x] A line search that accepts a step improving the objective *or* the
      residual, because the objective of a small-probability view goes flat
      three orders of magnitude before the residual does
- [x] The probability-view case checked against its closed form, which exists
      because relative entropy on a partition is minimised cell by cell
- [x] An already-satisfied view returning the prior's own floats rather than a
      normalised copy of them
- [x] Four failure modes separated and named: a target outside the span, a
      constant view, dependent views, and views that are individually reachable
      and jointly impossible
- [x] The relative entropy and the effective number of scenarios returned with
      every solve, because a view pushed into the tail spends the sample
- [x] The confidence blend provided, with its cost against re-solving measured
      rather than assumed to be zero
- [x] Weighted quantile and expected shortfall, the quantile uninterpolated and
      the shortfall reporting the mass it actually averaged over
- [x] A command-line entry point that puts the reweighted risk beside the
      unstressed one and beside the shortcut it replaces

The gap this closes is not an estimator. Every other module here reads a sample
and reports what is in it; none of them can answer a conditional question, and
the thing people do instead — keep the scenarios that fit the story — throws
away the dependence that made the sample worth using in the first place.
Reweighting keeps every scenario and satisfies the view exactly, which is a
strictly better trade whenever the sample is the thing being trusted.

### What it is worth, stated honestly

For a mean, nothing: the move it induces in an unmentioned series is that
series' least-squares slope on the view function, to within 0.12% over a
tenfold range of view strength. A regression would have done.

For a tail, everything, and not by a factor that can be guessed. A mean view
moves a 99% expected shortfall by 0.35 to 0.45 of what shifting the whole
distribution by the regression amount predicts; a tail-probability view on the
same portfolio moves it by 1.79 of that. The error changes sign with the kind
of view, so the shortcut has no correction and the reweighting is the only way
to the number.

### The line search was wrong twice before it was right

Not in a way that returned a wrong answer: in a way that returned a worse one
and said nothing. A non-strict descent test accepts zero-progress steps at the
optimum until the budget runs out, and reports a solved problem as
unconverged. A strict one terminates and stops while the residual is still
1e-09, because the dual value of a small-probability view is itself 1e-04 and
reaches the last bit of a double first. The stopping criterion belongs on the
quantity the caller asked about, and once the residual is allowed to justify a
step the same solves reach 1.4e-16.

## Phase 23 — The range value at risk can take with no dependence named

Every module above answers with a number because it has been told what the
dependence is — a covariance matrix, a fitted copula, a panel that lived it.
The answer is then conditional on that, and the usual way of saying so is a
footnote. This phase computes the interval instead: given the marginal loss
distributions and nothing else, how far apart can value at risk be?

- [x] The classical sandwich, both halves of it proofs rather than numerics —
      the sum of the marginals' expected shortfalls above the worst case, the
      sum of their lower tail means below the best case
- [x] Marginals that carry their tail means exactly, so neither proof inherits
      a quadrature error, with uniform, exponential, Pareto and normal built in
- [x] The rearrangement algorithm, with the objective and the stopping rule on
      the quantity the caller asked about rather than on the columns settling
- [x] Both discretisations run, so the grid's contribution is reported rather
      than absorbed, and the grid that does not exist returned as nothing
- [x] The gap split exactly into a quadrature part and a mixing part, using the
      invariance of the mean row sum under rearrangement
- [x] The superadditivity of value at risk measured across position count and
      tail weight, against the limits the marginals imply
- [x] A command-line route that prints the covariance-based figure inside the
      interval it is one point of
- [ ] A first release on the package index

### The oracle is complete mixability, and it exposed a divisibility problem

The uniform distribution is completely mixable, so for uniform marginals the
worst case is exactly `d (1 + alpha) / 2` and the best case exactly
`d alpha / 2`. Reproducing those is the only way to know the rearrangement is
finding the arrangement rather than returning a plausible number — and doing it
showed something the error bounds do not mention. At 512 cells the corrected
estimate recovers the closed form to between 2.2e-16 and 1.1e-14 at two, four,
eight and sixteen positions, every one a divisor of 512, and to 2.9e-05 at
three and five, which is one grid step and the best any arrangement of 512
points into three equal row sums can achieve. The attained bound itself sits
one to two cells below the closed form at every count, falling as one over the
cell count: five halvings each give a ratio of 2.00.

### Most of the gap between the two kinds of bound is arithmetic

Reading the ratio of the attained bound to the proved one as a statement about
couplings is wrong, and by a lot. The mean row sum cannot be moved by a
rearrangement and is precisely the left-endpoint Riemann sum of the same tail
mean the proof uses exactly, which splits the gap into quadrature plus
non-mixability with no residue. On eight exponential marginals the whole gap is
0.0667, of which 0.0631 is the grid and 0.0036 the dependence: 95% of it
attributable to the wrong cause by the naive reading.

### What the superadditivity is worth

On Pareto marginals with tail index 2 at the 99% level the worst case runs
1.476 times the comonotonic coupling at two positions, 1.791 at four, 1.927 at
eight, 1.985 at sixteen and 1.9997 at thirty-two, against the limit
`theta / (theta - 1) = 2`. Sweeping the index at eight positions gives 2.712
against a limit of 3 at index 1.5, then 1.927 against 2, 1.488 against 1.5 and
1.249 against 1.25 — the lighter the tail, the sooner the limit arrives.
Exponential marginals cap at 1.2171, their own ratio of expected shortfall to
value at risk. A book's value at risk can nearly double with no marginal
changing.

### Which bracket exists is not symmetric

The worst case's overstating grid needs the quantile at one, so it exists only
for a loss bounded above; the best case's understating grid needs the quantile
at zero, so it exists for a loss bounded below. A uniform loss has both, a
Pareto or exponential loss only the second, a normal loss neither. The
inequalities hold in every case, which is why they are what the output is
bracketed by and the grids are only reported.

## Phase 24 — Which directions of a covariance estimate are noise

- [x] The Marchenko-Pastur band and density, checked against quadrature
- [x] Eigenvalue clipping on the correlation matrix, preserving the trace and
      the diagonal exactly
- [x] The detection threshold, and the statement that below it a factor leaves
      no trace at all
- [x] The upward bias of a detectable eigenvalue, and its inverse
- [x] The limiting eigenvector overlap, which is zero below the threshold
- [x] An out-of-sample comparison against the sample and shrunk estimators,
      with equal weighting as the control
- [x] A `clean` subcommand

### The band is not a rule of thumb

For `T` observations of `N` independent unit-variance series the eigenvalues of
the sample correlation matrix do not cluster near one. With `q = N / T` they
fill `[(1 - sqrt(q))**2, (1 + sqrt(q))**2]` with the Marchenko-Pastur density,
which integrates to one and has mean one — both checked here by Simpson at four
ratios rather than asserted. At `q = 0.5` the band runs from 0.086 to 2.914, so
an eigenvalue of two and a half in a sixty-asset estimate from a hundred and
twenty observations is not a factor carrying two and a half times its share of
the variance. It is what pure noise produces, routinely.

Cleaning replaces every eigenvalue inside the band with the average of that
group. The average rather than zero, because the *total* variance was measured
well even though its distribution across directions was not: the trace is
preserved to rounding, and so is the diagonal of the covariance, because the
rebuilt matrix is rescaled by the square roots of its own diagonal rather than
having ones written into it — which also keeps it positive semidefinite, as
overwriting would not.

Working on the correlation matrix rather than the covariance is not tidiness.
The law is stated for unit-variance series, so applying it to a covariance means
fitting a scale to the bulk; the correlation matrix has that scale fixed at one.

### A factor can be too small to be in the estimate at all

Plant one factor of strength `s`, so the truth's largest eigenvalue is `1 + s`.
Above `sqrt(q)` the sample's largest converges to `(1 + s)(1 + q / s)`: outside
the band, detectable, and biased *upward* — at `q = 0.25` and `s = 0.8` the truth
is 1.8 and the sample reads 2.3625, which is 31% high, and `implied_spike`
inverts that map to ten figures.

Below `sqrt(q)` the sample's largest converges to the band edge **regardless of
s**, and the squared overlap of the sample eigenvector with the true one
converges to zero. Simulated at `q = 0.5`: a factor adding 40% to one
direction's variance leaves a largest eigenvalue inside the band, in the same
place a panel with no factor at all puts its largest. The factor is not weakly
estimated. It is absent, and nothing recovers it from this much data.

That is the same lesson as the cap that reprices to 2.4e-15 at seven different
correlations, in a new place: an estimator cannot report a parameter its data
does not depend on, and the useful output is the threshold rather than a number.

### More assets can reveal a factor that fewer assets hid

Which runs against the instinct that more assets means more parameters and a
worse estimate. The same three factors with the same loadings, over the same
hundred and twenty periods: at 24 assets the weakest one's eigenvalue is 1.965
against a band reaching 2.094, and the estimate has two factors in it; at 30
assets the eigenvalue is 2.395 against a band reaching 2.250, and it has three.
A factor's correlation-matrix eigenvalue grows with the number of assets loading
on it and the band grows only like `sqrt(q)`, so widening a panel can bring a
factor into view. Both directions are in the tests, because the band widening is
the part that makes the result surprising.

### Two of the three obvious tests cannot tell these estimators apart

Forty assets, eighty training observations, the true variance of each
estimator's minimum-variance portfolio in units of 1e-4:

| truth | sample | shrunk | clipped | equal weight | truth |
|---|---|---|---|---|---|
| isotropic | 0.0538 | 0.0257 | 0.0256 | 0.0250 | 0.0250 |
| one common factor, equal loadings | 1.4045 | 0.7605 | 0.8043 | 0.6650 | 0.6650 |
| three factors, unequal loadings | 0.0604 | 0.0371 | 0.0291 | 0.0433 | 0.0267 |

The fourth column is the control and it condemns the first two rows. When the
truth treats the assets alike the equal-weight portfolio is *exactly* optimal —
it matches the truth column to nine figures — so no covariance estimate can beat
it and the comparison measures nothing. Shrinkage winning the second row by 5%
is a fact about the panel, whose truth is the shrinkage target.

Only the third row distinguishes the assets, and there cleaning closes 92.9% of
the gap between the sample estimate and the truth, shrinkage 69.3% and equal
weighting 50.7%. The sample estimate is worse than equal weighting on every row.

So neither cleaning nor shrinking dominates, `compare_estimators` returns a
table rather than a recommendation, and the thing carried out of the phase is
about benchmarks: the first two panels anyone would reach for are the two on
which the question cannot be asked.

## Phase 25 — A book that is not linear in its factors

- [x] The exposure whitened and diagonalised, so a quadratic form in correlated
      factors becomes a sum of independent pieces
- [x] The first four cumulants in closed form, with the measurement that every
      term of the third and fourth carries a curvature
- [x] The support, which is a hard floor for a long-gamma book and a ceiling for
      a short-gamma one
- [x] The distribution function by saddlepoint, with its accuracy measured
      against two exact cases
- [x] Both tails computed in their own right, by negating the form rather than
      complementing the other one
- [x] Expected shortfall as an integral of the distribution function, with the
      unbounded case's truncation refused if it does not converge
- [x] The delta-only comparison through the same code, so there is one
      implementation rather than two
- [x] A `curvature` subcommand, and both directions of the error in the worked
      example

### What the phase is for

Everything before this treats the portfolio as a vector of weights. A book of
options is not: to second order its profit is `a + delta' x + 0.5 x' Gamma x`,
and the quadratic term is not a correction to be bolted onto a linear answer. It
is where every asymmetry in the distribution comes from. Each term of the third
and fourth cumulants carries an eigenvalue of the whitened gamma, so a linear
book has no skewness and no excess kurtosis however its factors are distributed.
There is no "skewness of the deltas".

### The number a delta-only model cannot have

Completing the square in each independent direction gives
`inf L = a - 0.5 sum beta^2 / lambda` when every eigenvalue is positive. A
long-gamma book has a floor — not a confidence interval, a floor. On a
three-factor book whose gamma is paid for by its theta, so the expected profit is
zero:

| confidence | exact VaR | exact ES | delta-normal |
| --- | --- | --- | --- |
| 90% | 3.80% | 4.18% | 5.68% |
| 95% | 4.16% | 4.39% | 6.09% |
| 99% | 4.53% | 4.60% | 6.85% |
| 99.9% | 4.67% | 4.69% | 7.71% |

against a floor of 4.708%. The exact numbers converge on it; the delta-normal
number passes through it at a confidence of **65.9%** and never comes back, so at
99% it is quoting 1.46 times a loss the book cannot sustain. Four hundred
thousand simulated paths came within 2e-05 of the floor without crossing it, so
the bound is tight as well as true.

Flip the sign of the gamma and the same arithmetic runs the other way. The book
now receives the theta, a delta-normal model credits it, drops the gamma it was
paid for taking, and reports a **negative** value at risk at 95% — no loss at
all — against a true loss of 7.62%, and 14.04% at 99% against a quoted -1.65%.
The
mechanism is identical in both cases: the model charges or credits the theta and
ignores what the theta bought. Only one of the two errors is in the safe
direction.

On that book the curvature also carries 92.2% of the variance. A delta-only model
is not making a small error about the shape; it is missing most of the risk.

### The expansion is asymptotic in the factor count, and that is measurable

Equal eigenvalues with no linear term is a scaled chi-square, which gives an
exact oracle. The worst absolute error of the saddlepoint distribution function
over the body:

| independent directions | worst error |
| --- | --- |
| 1 | 1.4e-02 |
| 2 | 2.8e-03 |
| 5 | 2.6e-04 |
| 10 | 5.7e-05 |
| 30 | 1.2e-05 |

So the method is not uniformly accurate, and a book dominated by one direction of
gamma sits at the top of that table rather than the bottom. The same thing shows
up a second way: the lower and upper tail expansions are independent, so their
sum is one only to the expansion's accuracy, and on the strongly skewed
three-factor book above the worst gap is 1.09e-02 — which places it near the
one-factor end, not the three-factor end.

### The logarithm that destroyed the answer near the mean

The generating function has `-0.5 log(1 - lambda t)` in it. Written that way the
distribution function was wrong by a per cent within a thousandth of a standard
deviation of the mean. The argument is near one for a small saddlepoint, `1.0 - x`
carries an absolute error of an ulp of one while the answer is of size `x`, and
the quantity that error lands in is `t K'(t) - K(t)` — itself a cancellation of
size `t^2`. With `log1p(-lambda t)` the same point holds to 6e-05 at a millionth
of a standard deviation.

### The branch that read exactly 1.0 at the mean

At the mean both of Lugannani-Rice's correction terms vanish and their difference
does not, so the difference has to be taken as a limit. The obvious test is
whether the level equals the mean, and it is the wrong test: a level one rounding
away has a saddlepoint of 1e-16, which is not zero, and the direct form then
divides two quantities that agree in every bit they have. The distribution
function read exactly `1.0` at the mean of a one-dimensional form and the clamp
on `[0, 1]` turned that into a plausible number rather than a visible one — the
first version of the accuracy sweep reported errors of 0.32, 0.37 and 0.44 at
one, two and ten factors and 2.6e-04 at five, which is non-monotone and was the
only reason it was caught at all.

The test is now on the dimensionless `t sqrt(k2)`, with the threshold at the
balance of the two errors: the discontinuity across it is 4e-06, two orders
inside the expansion's own error.

## Phase 26 — The worst a plausible set of factor moves can do

- [x] A chi-square quantile, so an ellipsoid is named by the probability it
      holds rather than by a bare number of standard deviations
- [x] The whitening transform exposed from behind `standard_form`, so a point
      in the independent coordinates can be mapped back to a factor move
- [x] The exact maximum loss over the ellipsoid through the secular equation,
      including the hard case
- [x] The closed form for a linear book, and the statement that it is a
      volatility multiple
- [x] The worst-case factor move itself, in each factor's own standard
      deviations
- [x] The one-factor-at-a-time stress test, for comparison, scaled onto the
      same ellipsoid
- [x] A `worst-case` subcommand and a worked example wired into CI

### A different question from a quantile

Everything else here is a quantile or a tail average of a distribution. A risk
committee also asks what the worst a plausible set of factor moves can do, and
what those moves are. The plausible set is the Mahalanobis ellipsoid
`x' Sigma^-1 x <= radius^2` — the set the covariance itself defines, so that no
direction is privileged by the shape of the set rather than by the book. In the
coordinates `whitening` produces that set is the Euclidean ball, and the problem
is the trust-region subproblem with a diagonal Hessian: a scalar equation, not
an optimiser.

### The shift goes on the largest curvature, and the first attempt did not

The stationarity condition is `(m - b_i) u_i = g_i` with
`m >= max(0, max(b))`, where `b` is the loss curvature. The first
implementation shifted by the *smallest* instead. It produced plausible
numbers, satisfied its own validation, and lost to a three-thousand-point
random search on the sphere by a factor of four on the first instance tried.
Nothing short of an independent reference would have caught it: the output of
the wrong version is a loss, is positive, grows with the radius, and sits on
the boundary.

### The hard case is a book shape, not a corner

If the direction of fastest loss growth carries no linear exposure then `g_i`
is zero there, the term drops out of the secular sum, and the equation has no
root: the left side at the floor is already below `radius^2`. The answer is to
pin the multiplier at the extreme curvature, take the solution in the other
coordinates, and spend the remaining norm along the degenerate direction.

That is exactly a delta-hedged short-gamma position, which makes it the
ordinary case for an option book rather than an edge case. A two-factor book
with gamma `diag(-120, -20)` and no delta loses 22.10 at the 99% radius, with
the whole move in the first factor. And there the one-at-a-time test is
*exact*, because the factors are uncorrelated and the worst direction is an
axis — the one shape where the naive test is right, which is worth knowing.

### For a linear book the whole exercise is a volatility multiple

`radius * sqrt(delta' Sigma delta)`, attained in the direction `-Sigma delta`.
Which is parametric value at risk with a different constant in front. So for a
book with no curvature a worst-case stress test and a VaR are the same
calculation and the only live question is which multiplier to use. Saying so is
more use than letting a reader discover it.

### One factor at a time understates, by a factor that grows with the book

Both searches over the *same* ellipsoid, so the only difference is that one of
them searches all of it. The median ratio of the joint worst case to the worst
single-factor loss, over four hundred random books at each size:

| factors | median | q1 | q3 | minimum | same deltas, no gamma |
|---|---|---|---|---|---|
| 2 | 1.36 | 1.10 | 2.25 | 1.00 | 1.15 |
| 3 | 2.04 | 1.46 | 3.52 | 1.01 | 1.38 |
| 5 | 3.76 | 2.59 | 6.76 | 1.16 | 1.92 |
| 8 | 6.78 | 4.56 | 9.72 | 1.31 | 2.66 |

Never below one, which it cannot be: the joint search contains every
single-factor move it is compared against. The last column is the same deltas
with the curvature removed, so roughly half the gap is the correlation and half
is the gamma.

### And the sum of the single-factor losses is not a bound either

That sum is the other number such a report carries, and the one treated as
conservative. The joint worst case exceeds it on 85.4% of random three-factor
books, 97.6% at five factors and 100% at eight, by a median factor of 1.93,
3.02 and 4.59. Adding up independent stresses produces a number no single point
in the factor space attains; it is not an upper bound and not a lower one.

### The usual version of the test is not even inside the set

"Move each factor by three of its own standard deviations" places `e_i` at
Mahalanobis length `sigma_i sqrt((Sigma^-1)_ii)`, which is at least one and
equals one only when that factor is uncorrelated with every other. Over random
correlated covariances that happens never — outside in 100 of 100 draws at
every factor count tried. So the common stress test is neither a subset of the
plausible set nor a bound on anything in it, and it still comes out smaller.

### And "three sigma" is not one stress test

The radius has to be named by a probability because the phrase does not
distinguish the sets it refers to. A ball of radius 3 holds 99.73% of the
probability over one factor, 97.07% over three, 46.79% over ten and 1.9e-11
over fifty. The 99% radius runs the other way: 2.58, 3.37, 4.82, 8.73.

## Phase 27 — A book with two states, and the risk that is not an average

- [x] A validated finite normal mixture, with its moments in closed form from
      the normal's own rather than by quadrature
- [x] A quantile bracketed by the components' own and solved by a safeguarded
      Newton step, because a bimodal density defeats Newton alone
- [x] Exact value at risk and expected shortfall: only the quantile is solved,
      and the tail mean is closed form given it
- [x] The posterior probability that a tail loss came from each regime
- [x] The weighted average of the regimes' own expected shortfalls, computed so
      the gap to the right answer can be measured rather than described
- [x] Exact convolution of independent mixtures, and the failure of value at
      risk to be subadditive stated from it rather than simulated
- [x] Expectation maximisation with the variance floor the unbounded likelihood
      requires, and the degeneracy reported
- [x] The horizon law under a two-state Markov chain, as an occupation-time
      mixture with `h + 1` components rather than `2 ** h` paths
- [x] A `regime` subcommand

### What the phase is for

Every distribution in Phase 3 has one mode. A risk committee's description of
its own book has two: a calm state and a stressed one. A finite mixture is the
smallest model that says that, and it is also the model whose arithmetic is
most reliably done wrong, because the word "mixture" invites averaging and
almost nothing here averages.

The running example is a stressed regime happening 5% of the time at four
times the volatility. Variance 1.75, excess kurtosis 10.47.

### The expected shortfall of a mixture is not the mixture of expected shortfalls

`sum(w_i * ES_i)` answers a different question — the loss if a regime were
drawn first and *then* a bad day inside it. The tail is not drawn that way.

| confidence | VaR | ES | averaged | error | stressed share of the tail |
|---|---|---|---|---|---|
| 90% | 1.366 | 2.244 | 2.018 | -10.1% | 18.3% |
| 95% | 1.806 | 2.926 | 2.372 | -18.9% | 32.6% |
| 97.5% | 2.279 | 3.844 | 2.688 | -30.1% | 56.9% |
| 99% | 3.445 | 5.607 | 3.065 | -45.3% | 97.3% |
| 99.9% | 8.215 | 9.684 | 3.872 | -60.0% | 100.0% |

One-signed, and growing with the level, so it is worst where the number is
used. The last column is why: a 5% regime is responsible for 97.3% of the
losses past the 99th percentile. The error is exactly zero when the two
components are identical, which is the only case in which the averaging
question and the real one coincide.

### Fat where it is measured, thin in the limit

The ratio of expected shortfall to value at risk is scale free for a zero-mean
normal, so the mixture's excess over it is the entire fat-tail signal: 42.1%
at 99%, 3.5% at `1 - 1e-4`, 1.2% at `1 - 1e-6`, 0.2% at `1 - 1e-12`. A mixture
of normals has every moment and is asymptotically a normal tail, so an
argument from its limiting behaviour is about a region no capital rule refers
to.

### The subadditivity failure lives in a window of width half the tail

Two independent positions losing 100 with probability 0.9%: at 99% each has a
value at risk of 0.031 and the pair has 99.998. Bisecting the stressed weight
from both sides, the violation holds on **(0.004943, 0.010000)** and nowhere
else. The upper edge is exactly `1 - c`, above which each position's own
quantile is already in the stressed region. The lower edge is where the pair's
chance of at least one stress reaches the same threshold, `1 - sqrt(c) =
0.0050126`, displaced by the components having width.

### The square-root rule, and the two conditions behind the table

Conditional on the regime sequence a horizon sum is normal and depends on the
sequence only through the occupation counts, so the law has `h + 1` components.
Against square-root scaling, under a chain staying calm 98% and stressed 90%:

| periods | 99% | 90% |
|---|---|---|
| 5 | 0.958 | 1.062 |
| 10 | 0.924 | 1.121 |
| 60 | 0.797 | 1.349 |

The direction depends on the confidence level, crossing at **97.41%** at five
periods, 97.32% at ten and 97.00% at sixty. In the tail the rule overstates,
because aggregation diversifies the regime itself: excess kurtosis falls from
7.65 at one period to 1.74 at sixty.

Both conditions behind that are load-bearing. The chain is started at its own
stationary distribution; started elsewhere the per-period risk drifts towards
it and the ratio reports the starting point — 1.47 against 1.09 on the bundled
returns, which is how the error was found. And the regimes differ only in
volatility; give the stressed one a mean and the drift accumulates linearly
against a quantile scaling as a square root, taking the ten-period ratio from
0.924 at a mean of zero to 1.043 at 0.5 and 1.361 at 2.0, crossing one at
**0.319**. "The square-root rule overstates" is a statement about a
volatility-only model.

### The likelihood has no maximum

Park a component on an observation and shrink its variance: the density
diverges. On forty standard normal draws the log-likelihood climbs -55.48,
-54.87, -50.92, -41.71, -32.50 as that component's deviation goes 1 to 1e-12.

An implementation without a floor does not fail loudly. Nine components on
those forty points walks into the spike, and the giveaway is that the answer
depends on the floor: -17.80 at 1e-8 and -3.99 at 1e-14, same data, same code.
Two components on the same points gives -50.44 with a smallest deviation of
0.177 and is not flagged. Where the fit is identified it works — 20,000 draws
recover a 99% expected shortfall within 1.1%, 200 draws within 7.1% — and the
starts are sample quantiles, so the same data gives the same answer twice.
