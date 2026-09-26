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
