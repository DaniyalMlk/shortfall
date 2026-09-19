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
