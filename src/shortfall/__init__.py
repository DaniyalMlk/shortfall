"""shortfall — a portfolio risk engine, in pure Python with no dependencies."""

from __future__ import annotations

from .covariance import (
    Diagnostics,
    Shrunk,
    average_correlation,
    blend,
    constant_correlation_target,
    correlation,
    diagnose,
    ledoit_wolf,
    portfolio_variance,
    sample_covariance,
)
from .linalg import (
    NotPositiveDefinite,
    NotSquare,
    NotSymmetric,
    cholesky,
    condition_number,
    eigenvalues,
    eigh,
    is_positive_semidefinite,
    nearest_psd,
    quadratic_form,
)
from .series import (
    ANNUAL,
    DAILY_CALENDAR,
    DAILY_TRADING,
    MONTHLY,
    QUARTERLY,
    WEEKLY,
    Convention,
    Misaligned,
    Panel,
    ReturnSeries,
    TooShort,
)

__version__ = "0.1.0"

__all__ = [
    "ANNUAL",
    "DAILY_CALENDAR",
    "DAILY_TRADING",
    "MONTHLY",
    "QUARTERLY",
    "WEEKLY",
    "Convention",
    "Diagnostics",
    "Misaligned",
    "NotPositiveDefinite",
    "NotSquare",
    "NotSymmetric",
    "Panel",
    "ReturnSeries",
    "Shrunk",
    "TooShort",
    "__version__",
    "average_correlation",
    "blend",
    "cholesky",
    "condition_number",
    "constant_correlation_target",
    "correlation",
    "diagnose",
    "eigenvalues",
    "eigh",
    "is_positive_semidefinite",
    "ledoit_wolf",
    "nearest_psd",
    "portfolio_variance",
    "quadratic_form",
    "sample_covariance",
]
