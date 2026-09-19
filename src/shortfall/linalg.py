"""Symmetric matrix routines, in pure Python.

Everything here takes a symmetric matrix and nothing else. That is not a
limitation worth apologising for: every matrix in this library is a covariance
or a correlation, both symmetric by construction, and a routine that only has to
handle the symmetric case can be both simpler and more accurate than a general
one.

The eigensolver is the cyclic Jacobi method. It is not the fastest way to
diagonalise a matrix — a tridiagonal reduction followed by QL implicit shifts
is — but for the symmetric case it has two properties that matter more here than
speed. It computes small eigenvalues to high *relative* accuracy, which is
exactly where a covariance matrix estimated from a short sample needs accuracy;
and it produces an orthogonal basis by construction, because every step is an
exact rotation, so the eigenvectors cannot drift out of orthogonality however
many sweeps it takes.

The matrices this library builds are as big as the asset universe, so tens to a
few hundred on a side. Jacobi is entirely comfortable there.
"""

from __future__ import annotations

import math
from typing import Final

Matrix = list[list[float]]
Vector = list[float]

#: Sweeps before the solver gives up. Cyclic Jacobi converges quadratically once
#: the off-diagonal mass is small, and a symmetric matrix of any size seen here
#: is done in well under ten. Fifty is a guard against a pathological input, not
#: a working limit.
MAX_SWEEPS: Final = 50

#: Relative size, in magnitude, that a remaining off-diagonal entry may have.
#: Near the double-precision epsilon, which is as tight as the arithmetic allows.
RESIDUAL: Final = 1e-14

#: The same tolerance expressed in the units the sweep test is written in.
#: Frobenius masses are sums of *squares*, so the threshold on them is the
#: square of the threshold on magnitudes. Getting this wrong is easy and quiet:
#: a threshold of 1e-15 on the squared mass reads as tight and actually permits
#: off-diagonal entries of 3e-8, which leaves the eigenvalues accurate — the
#: diagonal has converged — while the eigenvectors are only good to eight
#: digits, and a reconstruction test does not notice.
CONVERGED: Final = RESIDUAL * RESIDUAL


class NotSymmetric(ValueError):
    """The matrix given is not symmetric, and every routine here assumes it is."""


class NotSquare(ValueError):
    """The matrix given is not square."""


class NotPositiveDefinite(ValueError):
    """The matrix is not positive definite, and the operation requires it to be.

    Distinct from *not positive semi-definite*: a covariance matrix estimated
    from fewer observations than assets is singular and perfectly legitimate,
    and is only a problem for the operations that need to invert it.
    """


def dimension(matrix: Matrix) -> int:
    """Return the side length of a square matrix, or raise."""
    rows = len(matrix)
    if rows == 0:
        raise NotSquare("an empty matrix has no dimension")
    for index, row in enumerate(matrix):
        if len(row) != rows:
            raise NotSquare(
                f"matrix is {rows} rows but row {index} has {len(row)} entries"
            )
    return rows


def check_symmetric(matrix: Matrix, *, tolerance: float = 1e-12) -> int:
    """Return the dimension of ``matrix``, raising unless it is symmetric.

    The comparison is relative to the largest entry in the *whole* matrix, so
    the verdict does not change when the matrix is rescaled. A daily covariance
    has entries around 1e-4 and the same matrix annualised has entries around
    1e-2; they are the same matrix and must get the same answer, which an
    absolute tolerance — or a relative one with a floor under it — would not
    give them.
    """
    size = dimension(matrix)
    scale = max(abs(matrix[i][j]) for i in range(size) for j in range(size))
    if scale == 0.0:
        return size
    for i in range(size):
        for j in range(i + 1, size):
            upper, lower = matrix[i][j], matrix[j][i]
            if abs(upper - lower) > tolerance * scale:
                raise NotSymmetric(
                    f"matrix[{i}][{j}] is {upper!r} but matrix[{j}][{i}] is {lower!r}"
                )
    return size


def symmetrise(matrix: Matrix) -> Matrix:
    """Return ``(A + A^T) / 2``.

    Used where a matrix should be symmetric and is only not so by accumulated
    rounding. It is deliberately *not* applied silently inside the estimators:
    a matrix that is asymmetric by more than rounding is a bug upstream, and
    quietly averaging it away would hide it.
    """
    size = dimension(matrix)
    return [
        [(matrix[i][j] + matrix[j][i]) / 2.0 for j in range(size)] for i in range(size)
    ]


def identity(size: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def eigh(matrix: Matrix, *, max_sweeps: int = MAX_SWEEPS) -> tuple[Vector, Matrix]:
    """Diagonalise a symmetric matrix.

    Returns ``(values, vectors)`` with the eigenvalues in ascending order and
    ``vectors[i]`` the unit eigenvector for ``values[i]`` — vectors as *rows*,
    so ``A = sum(w * outer(v, v))`` over the pairs. Rows rather than columns
    because every consumer here iterates over eigenpairs, and a row is the
    natural unit of that iteration in nested lists.

    The method is cyclic Jacobi: repeatedly annihilate the largest off-diagonal
    entries with exact plane rotations until the off-diagonal mass is negligible.
    """
    size = check_symmetric(matrix)
    work = [row[:] for row in matrix]
    basis = identity(size)

    total = sum(work[i][j] ** 2 for i in range(size) for j in range(size))
    if total == 0.0:
        return [0.0] * size, basis

    for _ in range(max_sweeps):
        off = sum(work[i][j] ** 2 for i in range(size) for j in range(i + 1, size))
        if off <= CONVERGED * total:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if work[p][q] == 0.0:
                    continue
                _rotate(work, basis, p, q, size)

    pairs = sorted(
        ((work[i][i], basis[i]) for i in range(size)), key=lambda pair: pair[0]
    )
    return [value for value, _ in pairs], [vector for _, vector in pairs]


def _rotate(work: Matrix, basis: Matrix, p: int, q: int, size: int) -> None:
    """Apply the Jacobi rotation that zeroes ``work[p][q]``.

    ``t`` is the tangent of the rotation angle, and it is computed from the
    stable root of ``t^2 + 2*theta*t - 1 = 0`` — the one with the smaller
    magnitude. The other root is mathematically just as valid and numerically
    much worse: it rotates by nearly a right angle every step, so the iteration
    reorders the diagonal instead of converging on it.
    """
    theta = (work[q][q] - work[p][p]) / (2.0 * work[p][q])
    if theta >= 0.0:
        tangent = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
    else:
        tangent = -1.0 / (-theta + math.sqrt(1.0 + theta * theta))
    cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
    sine = tangent * cosine

    for k in range(size):
        a_kp = work[k][p]
        a_kq = work[k][q]
        work[k][p] = cosine * a_kp - sine * a_kq
        work[k][q] = sine * a_kp + cosine * a_kq
    for k in range(size):
        a_pk = work[p][k]
        a_qk = work[q][k]
        work[p][k] = cosine * a_pk - sine * a_qk
        work[q][k] = sine * a_pk + cosine * a_qk
    # Exactly zero rather than nearly: the rotation was chosen to annihilate
    # this entry, and leaving a rounding residue there would make the sweep
    # termination test depend on noise.
    work[p][q] = work[q][p] = 0.0

    for k in range(size):
        v_pk = basis[p][k]
        v_qk = basis[q][k]
        basis[p][k] = cosine * v_pk - sine * v_qk
        basis[q][k] = sine * v_pk + cosine * v_qk


def eigenvalues(matrix: Matrix) -> Vector:
    """Eigenvalues in ascending order."""
    values, _ = eigh(matrix)
    return values


def reconstruct(values: Vector, vectors: Matrix) -> Matrix:
    """Rebuild ``sum(w * outer(v, v))`` from eigenpairs."""
    size = len(values)
    out = [[0.0] * size for _ in range(size)]
    for value, vector in zip(values, vectors, strict=True):
        for i in range(size):
            if vector[i] == 0.0:
                continue
            scaled = value * vector[i]
            for j in range(size):
                out[i][j] += scaled * vector[j]
    return symmetrise(out)


def condition_number(matrix: Matrix) -> float:
    """Ratio of largest to smallest eigenvalue by magnitude.

    Infinite when an eigenvalue is exactly zero, which is the honest answer
    rather than a large finite one: a singular covariance matrix has directions
    of zero estimated variance, and that is not merely awkward.

    In floating point an exact zero is rare. A covariance matrix estimated from
    fewer observations than assets is rank-deficient in exact arithmetic and
    comes back here with a condition number around 1e17 instead of infinity, so
    a caller testing for singularity should use a relative threshold —
    :attr:`~shortfall.covariance.Diagnostics.numerically_singular` does — rather
    than testing for infinity and concluding the matrix is fine.
    """
    values = [abs(value) for value in eigenvalues(matrix)]
    smallest = min(values)
    if smallest == 0.0:
        return math.inf
    return max(values) / smallest


def is_positive_semidefinite(matrix: Matrix, *, tolerance: float | None = None) -> bool:
    """Whether every eigenvalue is non-negative, up to rounding.

    The default tolerance scales with the largest eigenvalue, because "zero" in
    a matrix whose entries are 1e-4 is a different number from "zero" in one
    whose entries are 1e2, and a fixed threshold is wrong for one of them.
    """
    values = eigenvalues(matrix)
    limit = max(abs(value) for value in values)
    threshold = -(limit * 1e-12) if tolerance is None else -abs(tolerance)
    return all(value >= threshold for value in values)


def nearest_psd(matrix: Matrix, *, floor: float = 0.0) -> Matrix:
    """Clip negative eigenvalues to ``floor`` and rebuild.

    This is the Frobenius-nearest positive semi-definite matrix to a symmetric
    input, which is a real result rather than a heuristic: for symmetric ``A``
    the nearest one is obtained by replacing each eigenvalue with ``max(w, 0)``.

    The diagonal is *not* restored afterwards. Doing so is common, and it would
    make the variances match the input at the cost of the result no longer being
    the nearest matrix, no longer necessarily positive semi-definite, and no
    longer the thing this function claims to return. Callers who want their
    variances back should rescale to a correlation matrix, repair that, and
    rescale — which is a different operation and says so.
    """
    values, vectors = eigh(matrix)
    return reconstruct([max(value, floor) for value in values], vectors)


def cholesky(matrix: Matrix) -> Matrix:
    """Lower-triangular ``L`` with ``L @ L.T == matrix``.

    Raises :class:`NotPositiveDefinite` rather than returning a matrix full of
    ``nan``. A factorisation that silently produced ``nan`` would propagate
    through a simulation and surface a long way from the cause.
    """
    size = check_symmetric(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0.0:
                    raise NotPositiveDefinite(
                        f"leading minor {i + 1} has pivot {total!r}; the matrix is not "
                        "positive definite. If it came from a sample with fewer "
                        "observations than assets it is singular by construction — "
                        "shrink it, or repair it towards the nearest one."
                    )
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def quadratic_form(weights: Vector, matrix: Matrix) -> float:
    """``w^T A w``, the portfolio variance when ``A`` is a covariance matrix.

    Accumulated as the diagonal terms plus twice the upper triangle, which is
    both half the multiplications and the form that cannot disagree with itself
    across the diagonal when the matrix is symmetric only to rounding.
    """
    size = check_symmetric(matrix)
    if len(weights) != size:
        raise ValueError(
            f"{len(weights)} weights against a {size}x{size} matrix; they must agree"
        )
    total = sum(weights[i] * weights[i] * matrix[i][i] for i in range(size))
    for i in range(size):
        if weights[i] == 0.0:
            continue
        for j in range(i + 1, size):
            total += 2.0 * weights[i] * weights[j] * matrix[i][j]
    return total


def matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    """``A @ v``."""
    size = dimension(matrix)
    if len(vector) != size:
        raise ValueError(f"{len(vector)} entries against a {size}x{size} matrix")
    return [sum(row[j] * vector[j] for j in range(size)) for row in matrix]


# -- least squares -----------------------------------------------------------


class RankDeficient(ValueError):
    """The design matrix has linearly dependent columns.

    Which means the regression has no unique answer: some combination of the
    columns is exactly zero, so a coefficient vector can be shifted along it
    without changing a single fitted value. Any particular answer a solver
    returns for such a design is an artefact of how it rounds.
    """


def solve_upper(upper: Matrix, target: Vector) -> Vector:
    """Solve ``R x = b`` for upper-triangular ``R`` by back substitution."""
    size = len(target)
    out = [0.0] * size
    for i in reversed(range(size)):
        total = target[i] - math.fsum(upper[i][j] * out[j] for j in range(i + 1, size))
        if upper[i][i] == 0.0:
            raise RankDeficient(f"the triangular factor has a zero at position {i}")
        out[i] = total / upper[i][i]
    return out


def least_squares(design: Matrix, target: Vector) -> tuple[Vector, Matrix]:
    """Minimise ``||design @ x - target||`` by Householder QR.

    Returns the coefficients and the upper-triangular factor ``R``, which is
    what a caller needs for standard errors: ``(X'X)^-1`` is ``R^-1 R^-T``,
    obtainable from ``R`` alone without ever forming ``X'X``.

    **Why not the normal equations.** Solving ``X'X b = X'y`` is three lines
    with the Cholesky factorisation already in this module, and it is the wrong
    method here. Forming ``X'X`` squares the condition number of ``X``, so a
    design that is merely awkward at a condition number of 1e8 — perfectly
    ordinary for factor returns, which are correlated with each other by
    construction and often nearly collinear — becomes numerically singular at
    1e16 and the factorisation either fails or returns digits that are all
    noise. Householder QR works on ``X`` directly and its accuracy depends on
    the condition number itself rather than its square.

    The cost is a constant factor of about two in arithmetic, on problems whose
    size is the number of factors. That is not a trade worth thinking about.
    """
    rows = len(design)
    if rows == 0:
        raise ValueError("a least squares problem needs at least one observation")
    columns = len(design[0])
    for index, row in enumerate(design):
        if len(row) != columns:
            raise ValueError(
                f"design row {index} has {len(row)} entries against {columns} in the "
                "first; the design matrix must be rectangular"
            )
    if len(target) != rows:
        raise ValueError(
            f"{len(target)} targets against {rows} design rows; they must agree"
        )
    if rows < columns:
        raise RankDeficient(
            f"{rows} observations cannot determine {columns} coefficients; the design "
            "has more unknowns than equations and infinitely many exact solutions"
        )

    work = [row[:] for row in design]
    rhs = list(target)
    # The scale the pivots are judged against. Taken once from the original
    # matrix, because the working copy shrinks as columns are eliminated and a
    # tolerance recomputed from it would drift with the elimination.
    scale = max(
        (abs(work[i][j]) for i in range(rows) for j in range(columns)), default=0.0
    )
    if scale == 0.0:
        raise RankDeficient("the design matrix is entirely zero")
    floor = scale * math.sqrt(rows) * RESIDUAL

    for k in range(columns):
        norm = math.sqrt(math.fsum(work[i][k] ** 2 for i in range(k, rows)))
        if norm <= floor:
            raise RankDeficient(
                f"column {k} of the design is a linear combination of the columns "
                f"before it: what remains of it after eliminating them has norm "
                f"{norm:.3e} against a scale of {scale:.3e}. The regression has no "
                "unique solution; drop a factor or combine the dependent ones."
            )
        # Reflect away from the leading entry rather than towards it. Choosing
        # the other sign subtracts two nearly equal numbers when the column
        # already points along the axis, which is the one case the reflection
        # was not needed for and the one case it destroys.
        alpha = -norm if work[k][k] >= 0.0 else norm
        vector = [work[i][k] for i in range(k, rows)]
        vector[0] -= alpha
        squared = math.fsum(value * value for value in vector)
        if squared > 0.0:
            for j in range(k, columns):
                dot = math.fsum(
                    vector[i - k] * work[i][j] for i in range(k, rows)
                )
                factor = 2.0 * dot / squared
                for i in range(k, rows):
                    work[i][j] -= factor * vector[i - k]
            dot = math.fsum(vector[i - k] * rhs[i] for i in range(k, rows))
            factor = 2.0 * dot / squared
            for i in range(k, rows):
                rhs[i] -= factor * vector[i - k]
        work[k][k] = alpha
        for i in range(k + 1, rows):
            work[i][k] = 0.0

    upper = [[work[i][j] for j in range(columns)] for i in range(columns)]
    return solve_upper(upper, rhs[:columns]), upper


def upper_inverse_gram(upper: Matrix) -> Matrix:
    """``(R' R)^-1`` from the triangular factor, which is ``(X'X)^-1``.

    The diagonal of this scaled by the residual variance gives the squared
    standard errors of the coefficients. Computed by inverting ``R`` — itself a
    triangular solve per column — rather than by forming and inverting ``X'X``,
    for the reason given in :func:`least_squares`.
    """
    size = len(upper)
    inverse = [[0.0] * size for _ in range(size)]
    for column in range(size):
        unit = [1.0 if i == column else 0.0 for i in range(size)]
        solved = solve_upper(upper, unit)
        for i in range(size):
            inverse[i][column] = solved[i]
    # (R'R)^-1 = R^-1 R^-T
    return symmetrise(
        [
            [
                math.fsum(inverse[i][k] * inverse[j][k] for k in range(size))
                for j in range(size)
            ]
            for i in range(size)
        ]
    )


__all__ = [
    "MAX_SWEEPS",
    "RESIDUAL",
    "Matrix",
    "NotPositiveDefinite",
    "NotSquare",
    "NotSymmetric",
    "RankDeficient",
    "Vector",
    "check_symmetric",
    "cholesky",
    "condition_number",
    "dimension",
    "eigenvalues",
    "eigh",
    "identity",
    "is_positive_semidefinite",
    "least_squares",
    "matrix_vector",
    "nearest_psd",
    "quadratic_form",
    "reconstruct",
    "solve_upper",
    "symmetrise",
    "upper_inverse_gram",
]
