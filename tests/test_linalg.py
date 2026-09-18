"""The symmetric eigensolver and the matrix routines built on it.

Checked against closed forms where one exists — a 2x2 symmetric matrix has
exact eigenvalues, and a matrix built as ``Q diag(w) Q^T`` from a known
orthogonal ``Q`` has known ones — and otherwise against identities that hold
whatever the input: reconstruction, orthonormality, and the trace and
determinant, which are the sum and product of the eigenvalues.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.linalg import (
    Matrix,
    NotPositiveDefinite,
    NotSquare,
    NotSymmetric,
    check_symmetric,
    cholesky,
    condition_number,
    eigenvalues,
    eigh,
    identity,
    is_positive_semidefinite,
    matrix_vector,
    nearest_psd,
    quadratic_form,
    reconstruct,
    symmetrise,
)


def random_symmetric(size: int, seed: int) -> Matrix:
    rng = random.Random(seed)
    out = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            value = rng.gauss(0.0, 1.0)
            out[i][j] = value
            out[j][i] = value
    return out


def rotation(size: int, seed: int) -> Matrix:
    """An orthogonal matrix, by Gram-Schmidt on random vectors.

    Built independently of anything under test, so a matrix assembled from it
    has eigenvalues this file chose rather than eigenvalues the solver found.
    """
    rng = random.Random(seed)
    basis: Matrix = []
    for _ in range(size):
        vector = [rng.gauss(0.0, 1.0) for _ in range(size)]
        for done in basis:
            overlap = sum(a * b for a, b in zip(vector, done, strict=True))
            vector = [a - overlap * b for a, b in zip(vector, done, strict=True)]
        norm = math.sqrt(sum(a * a for a in vector))
        basis.append([a / norm for a in vector])
    return basis


def assemble(values: list[float], basis: Matrix) -> Matrix:
    return reconstruct(values, basis)


# -- shape and symmetry ----------------------------------------------------


def test_a_ragged_matrix_is_refused() -> None:
    with pytest.raises(NotSquare, match="row 1"):
        check_symmetric([[1.0, 2.0], [2.0]])


def test_an_empty_matrix_is_refused() -> None:
    with pytest.raises(NotSquare):
        check_symmetric([])


def test_an_asymmetric_matrix_is_refused() -> None:
    with pytest.raises(NotSymmetric, match=r"\[0\]\[1\]"):
        check_symmetric([[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("factor", [1e-8, 1.0, 1e6])
def test_the_symmetry_verdict_does_not_change_when_the_matrix_is_rescaled(
    factor: float,
) -> None:
    # A daily covariance and the same matrix annualised are the same matrix and
    # must get the same answer. A relative tolerance with a floor under it —
    # which is the easy way to write this — turns absolute below the floor and
    # calls a nine-percent asymmetry in a matrix of 1e-8 entries symmetric.
    asymmetric = [[1.0 * factor, 1.0 * factor], [1.1 * factor, 1.0 * factor]]
    with pytest.raises(NotSymmetric):
        check_symmetric(asymmetric)

    rounded = [[1.0 * factor, 1.0 * factor], [factor + 1e-15 * factor, 1.0 * factor]]
    assert check_symmetric(rounded) == 2


def test_an_all_zero_matrix_is_symmetric() -> None:
    assert check_symmetric([[0.0, 0.0], [0.0, 0.0]]) == 2


def test_symmetrise_averages_the_two_halves() -> None:
    assert symmetrise([[1.0, 2.0], [4.0, 1.0]]) == [[1.0, 3.0], [3.0, 1.0]]


# -- the eigensolver against closed forms ----------------------------------


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ([[4.0, 1.0], [1.0, 2.0]], (3.0 - math.sqrt(2.0), 3.0 + math.sqrt(2.0))),
        ([[2.0, 0.0], [0.0, 5.0]], (2.0, 5.0)),
        ([[1.0, 1.0], [1.0, 1.0]], (0.0, 2.0)),
        ([[0.0, 3.0], [3.0, 0.0]], (-3.0, 3.0)),
    ],
)
def test_a_two_by_two_matches_the_quadratic_formula(
    matrix: Matrix, expected: tuple[float, float]
) -> None:
    # (a + d)/2 +/- sqrt(((a - d)/2)^2 + b^2), exactly.
    assert eigenvalues(matrix) == pytest.approx(list(expected), abs=1e-14)


def test_a_matrix_built_from_known_eigenvalues_returns_them() -> None:
    wanted = [-2.5, 0.25, 1.0, 3.75, 9.0]
    basis = rotation(5, seed=17)
    assert eigenvalues(assemble(wanted, basis)) == pytest.approx(wanted, abs=1e-12)


def test_a_repeated_eigenvalue_is_returned_with_its_multiplicity() -> None:
    # Degenerate spectra are where an eigensolver is most likely to misbehave,
    # because the eigenvectors are not unique.
    wanted = [2.0, 2.0, 2.0, 7.0]
    got = eigenvalues(assemble(wanted, rotation(4, seed=3)))
    assert got == pytest.approx(wanted, abs=1e-12)


def test_an_identity_matrix_is_already_diagonal() -> None:
    values, vectors = eigh(identity(4))
    assert values == pytest.approx([1.0] * 4)
    assert vectors == identity(4)


def test_a_zero_matrix_has_zero_eigenvalues() -> None:
    values, vectors = eigh([[0.0, 0.0], [0.0, 0.0]])
    assert values == [0.0, 0.0]
    assert vectors == identity(2)


@pytest.mark.parametrize("size", [2, 3, 5, 8, 12])
def test_the_decomposition_reconstructs_its_input(size: int) -> None:
    matrix = random_symmetric(size, seed=size * 31)
    values, vectors = eigh(matrix)
    rebuilt = reconstruct(values, vectors)
    scale = max(abs(matrix[i][j]) for i in range(size) for j in range(size))
    for i in range(size):
        for j in range(size):
            assert rebuilt[i][j] == pytest.approx(matrix[i][j], abs=1e-12 * scale)


@pytest.mark.parametrize("size", [2, 3, 5, 8, 12])
def test_the_eigenvectors_are_orthonormal(size: int) -> None:
    _, vectors = eigh(random_symmetric(size, seed=size * 7 + 1))
    for a in range(size):
        for b in range(size):
            overlap = sum(vectors[a][k] * vectors[b][k] for k in range(size))
            assert overlap == pytest.approx(1.0 if a == b else 0.0, abs=1e-12)


@pytest.mark.parametrize("size", [2, 3, 6])
def test_the_eigenvalues_reproduce_the_trace(size: int) -> None:
    # The trace is the sum of the eigenvalues, and it is computed here without
    # touching the solver, so agreement is evidence rather than tautology.
    matrix = random_symmetric(size, seed=size * 101)
    trace = sum(matrix[i][i] for i in range(size))
    assert sum(eigenvalues(matrix)) == pytest.approx(trace, abs=1e-12)


def test_the_eigenvalues_reproduce_the_determinant() -> None:
    matrix = [[4.0, 1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 2.0, 5.0]]
    # Expanded by hand along the first row: 4(15 - 4) - 1(5 - 0) + 0 = 39.
    product = 1.0
    for value in eigenvalues(matrix):
        product *= value
    assert product == pytest.approx(39.0, abs=1e-11)


def test_the_eigenvalues_come_back_in_ascending_order() -> None:
    values = eigenvalues(random_symmetric(9, seed=44))
    assert values == sorted(values)


@pytest.mark.parametrize("size", [2, 3, 4, 6, 9, 14])
def test_an_eigenvector_satisfies_its_own_equation(size: int) -> None:
    # The tightest check here, and the one the convergence criterion is really
    # about. The eigenvalues converge well before the eigenvectors do, so a
    # reconstruction test passes while Av and wv still disagree in the eighth
    # digit; only this notices.
    matrix = random_symmetric(size, seed=2024 + size)
    values, vectors = eigh(matrix)
    for value, vector in zip(values, vectors, strict=True):
        left = matrix_vector(matrix, vector)
        right = [value * entry for entry in vector]
        assert left == pytest.approx(right, abs=1e-11)


# -- conditioning and definiteness ------------------------------------------


def test_a_singular_matrix_has_an_infinite_condition_number() -> None:
    assert math.isinf(condition_number([[1.0, 1.0], [1.0, 1.0]]))


def test_the_condition_number_of_a_known_matrix() -> None:
    lower, upper = 3.0 - math.sqrt(2.0), 3.0 + math.sqrt(2.0)
    assert condition_number([[4.0, 1.0], [1.0, 2.0]]) == pytest.approx(upper / lower)


def test_definiteness_is_judged_relative_to_the_largest_eigenvalue() -> None:
    # The same matrix scaled by 1e-8 is exactly as definite as before, so a
    # fixed absolute threshold would give two different answers for it.
    matrix = [[1.0, 0.999999], [0.999999, 1.0]]
    assert is_positive_semidefinite(matrix)
    tiny = [[value * 1e-8 for value in row] for row in matrix]
    assert is_positive_semidefinite(tiny)


def test_an_indefinite_matrix_is_reported_as_such() -> None:
    assert not is_positive_semidefinite([[1.0, 2.0], [2.0, 1.0]])


def test_repair_clips_the_negative_eigenvalues_and_leaves_the_rest() -> None:
    wanted = [-1.0, -0.25, 0.5, 4.0]
    basis = rotation(4, seed=9)
    repaired = nearest_psd(assemble(wanted, basis))
    assert eigenvalues(repaired) == pytest.approx([0.0, 0.0, 0.5, 4.0], abs=1e-12)


def test_repair_leaves_an_already_definite_matrix_alone() -> None:
    matrix = assemble([0.5, 1.0, 2.0], rotation(3, seed=5))
    repaired = nearest_psd(matrix)
    for i in range(3):
        for j in range(3):
            assert repaired[i][j] == pytest.approx(matrix[i][j], abs=1e-12)


def test_repair_can_floor_above_zero() -> None:
    repaired = nearest_psd(assemble([-1.0, 0.0, 3.0], rotation(3, seed=6)), floor=0.1)
    assert min(eigenvalues(repaired)) == pytest.approx(0.1, abs=1e-12)


# -- factorisation and quadratic forms --------------------------------------


def test_cholesky_factors_a_known_matrix() -> None:
    lower = cholesky([[4.0, 2.0], [2.0, 3.0]])
    assert lower[0] == pytest.approx([2.0, 0.0])
    assert lower[1] == pytest.approx([1.0, math.sqrt(2.0)])


@pytest.mark.parametrize("size", [2, 4, 7])
def test_the_cholesky_factor_multiplies_back(size: int) -> None:
    matrix = assemble([0.5 + i for i in range(size)], rotation(size, seed=size + 60))
    lower = cholesky(matrix)
    for i in range(size):
        for j in range(size):
            product = sum(lower[i][k] * lower[j][k] for k in range(size))
            assert product == pytest.approx(matrix[i][j], abs=1e-11)


def test_cholesky_refuses_a_singular_matrix_rather_than_returning_nans() -> None:
    # A silent matrix of nan would propagate through a simulation and surface a
    # very long way from here.
    with pytest.raises(NotPositiveDefinite, match="positive definite"):
        cholesky([[1.0, 1.0], [1.0, 1.0]])


def test_cholesky_names_the_leading_minor_that_failed() -> None:
    with pytest.raises(NotPositiveDefinite, match="minor 2"):
        cholesky([[1.0, 2.0], [2.0, 1.0]])


def test_a_quadratic_form_matches_the_expansion() -> None:
    matrix = [[4.0, 2.0], [2.0, 3.0]]
    weights = [0.6, -0.4]
    expected = (
        weights[0] ** 2 * 4.0 + 2 * weights[0] * weights[1] * 2.0 + weights[1] ** 2 * 3.0
    )
    assert quadratic_form(weights, matrix) == pytest.approx(expected, abs=1e-15)


def test_a_quadratic_form_of_a_definite_matrix_is_positive() -> None:
    matrix = assemble([0.2, 1.0, 3.0], rotation(3, seed=15))
    assert quadratic_form([0.3, 0.3, 0.4], matrix) > 0.0


def test_weights_must_match_the_matrix() -> None:
    with pytest.raises(ValueError, match="must agree"):
        quadratic_form([1.0], [[1.0, 0.0], [0.0, 1.0]])
