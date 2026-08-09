"""VAR data generation."""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

import numpy as np


# Domains

NUM_REGIMES = 2
VAR_TARGET_SPECTRAL_RADIUS = 0.75


# Functions

def make_causal_adjacency_graphs(
    d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two causal adjacency matrices."""
    A1 = np.zeros((d, d), dtype=float)
    A2 = np.zeros((d, d), dtype=float)

    for i in range(d):
        A1[i, i] = 1.0
        A1[i, (i + 2) % d] = 1.0

        A2[i, i] = 1.0
        A2[i, (i + 3) % d] = 1.0

    return A1, A2


def make_coefficient_graphs(
    d: int,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Sample signed coefficient matrices masked by the causal graphs."""
    A1, A2 = make_causal_adjacency_graphs(d)

    W1 = (
        rng.uniform(alpha, beta, size=(d, d))
        * rng.choice([-1.0, 1.0], size=(d, d))
    )
    W2 = (
        rng.uniform(alpha, beta, size=(d, d))
        * rng.choice([-1.0, 1.0], size=(d, d))
    )

    B1 = A1 * W1
    B2 = A2 * W2

    return [B1, B2]


def companion_matrix_from_var_coefficients(
    coefficient_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    """Construct the companion matrix of a VAR(p) process."""
    p = len(coefficient_matrices)
    if p <= 0:
        raise ValueError("Need at least one lag matrix.")

    d = coefficient_matrices[0].shape[0]

    for B in coefficient_matrices:
        if B.shape != (d, d):
            raise ValueError(
                f"All coefficient matrices must have shape {(d, d)}. "
                f"Got {B.shape}."
            )

    C = np.zeros((d * p, d * p), dtype=float)
    C[:d, : d * p] = np.concatenate(coefficient_matrices, axis=1)

    if p > 1:
        C[d:, :-d] = np.eye(d * (p - 1))

    return C


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the spectral radius of a matrix."""
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals)))


def var_companion_spectral_radius(
    coefficient_matrices: Sequence[np.ndarray],
) -> float:
    """Return the spectral radius of a VAR(p) companion matrix."""
    C = companion_matrix_from_var_coefficients(coefficient_matrices)
    return spectral_radius(C)


def scale_var_coefficients_to_stability(
    coefficient_matrices: Sequence[np.ndarray],
    target_radius: float = VAR_TARGET_SPECTRAL_RADIUS,
    max_iter: int = 100,
) -> List[np.ndarray]:
    """Scale VAR coefficients to the target companion spectral radius."""
    if target_radius <= 0:
        raise ValueError(
            f"target_radius must be positive. Got {target_radius}."
        )

    coeffs = [
        np.asarray(B, dtype=float).copy()
        for B in coefficient_matrices
    ]

    rho = var_companion_spectral_radius(coeffs)
    if rho == 0 or rho <= target_radius:
        return coeffs

    scale = 1.0

    for _ in range(max_iter):
        rho = var_companion_spectral_radius(
            [scale * B for B in coeffs]
        )
        if rho <= target_radius:
            return [scale * B for B in coeffs]

        scale *= 0.9 * target_radius / rho

    warnings.warn(
        "Could not scale VAR coefficients to target spectral radius "
        "within max_iter. Returning the latest scaled coefficients.",
        RuntimeWarning,
    )
    return [scale * B for B in coeffs]


def make_var_lag_coefficient_graphs(
    d: int,
    p: int,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    target_radius: float = VAR_TARGET_SPECTRAL_RADIUS,
) -> List[List[np.ndarray]]:
    """Generate two regimes of stable VAR(p) coefficient matrices."""
    if p <= 0:
        raise ValueError(
            f"VAR lag order p must be positive. Got p={p}."
        )

    A1, A2 = make_causal_adjacency_graphs(d)
    adjacency_by_regime = [A1, A2]

    coefficient_graphs: List[List[np.ndarray]] = []

    for regime in range(NUM_REGIMES):
        lag_matrices = []

        for lag_index in range(p):
            lag_decay = 1.0 / (lag_index + 1)

            W = rng.uniform(alpha, beta, size=(d, d))
            signs = rng.choice([-1.0, 1.0], size=(d, d))

            B_lag = (
                lag_decay
                * adjacency_by_regime[regime]
                * W
                * signs
            )
            lag_matrices.append(B_lag)

        lag_matrices = scale_var_coefficients_to_stability(
            lag_matrices,
            target_radius=target_radius,
        )

        coefficient_graphs.append(lag_matrices)

    return coefficient_graphs


def controlled_nonlinear_var_step(
    B: np.ndarray,
    z_prev: np.ndarray,
    control: float,
    noise_std_dev: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take one controlled nonlinear VAR(1) step."""
    d = len(z_prev)
    noise = rng.normal(
        loc=0.0,
        scale=noise_std_dev,
        size=d,
    )
    return control * np.tanh(B @ z_prev) + noise


def validate_change_points(
    num_timepoints: int,
    num_regimes: int,
    change_points: Sequence[int],
) -> None:
    """Validate the changepoints of a piecewise process."""
    if len(change_points) != num_regimes - 1:
        raise ValueError(
            "Need len(change_points) == num_regimes - 1. "
            f"Got {len(change_points)} and "
            f"num_regimes={num_regimes}."
        )

    if sorted(change_points) != list(change_points):
        raise ValueError(
            "change_points must be sorted increasingly. "
            f"Got {change_points}."
        )

    if any(
        cp <= 0 or cp >= num_timepoints
        for cp in change_points
    ):
        raise ValueError(
            "Each changepoint must satisfy 0 < cp < T. "
            f"Got {change_points} for T={num_timepoints}."
        )


def controlled_nonlinear_var(
    num_variables: int,
    num_timepoints: int,
    num_regimes: int,
    change_points: Sequence[int],
    coefficient_graphs: Sequence[np.ndarray],
    control: float,
    noise_std_dev: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a piecewise controlled nonlinear VAR(1) process."""
    validate_change_points(
        num_timepoints,
        num_regimes,
        change_points,
    )

    if len(coefficient_graphs) != num_regimes:
        raise ValueError(
            "Need one coefficient graph per regime. "
            f"Got {len(coefficient_graphs)} graphs for "
            f"{num_regimes} regimes."
        )

    boundaries = list(change_points) + [num_timepoints]
    z_prev = np.zeros(num_variables)
    time_series = np.zeros(
        (num_timepoints, num_variables)
    )

    regime = 0
    for t in range(num_timepoints):
        while (
            regime < num_regimes - 1
            and t >= boundaries[regime]
        ):
            regime += 1

        z_prev = controlled_nonlinear_var_step(
            B=coefficient_graphs[regime],
            z_prev=z_prev,
            control=control,
            noise_std_dev=noise_std_dev,
            rng=rng,
        )
        time_series[t] = z_prev

    return time_series


def lagged_linear_signal(
    coefficient_matrices: Sequence[np.ndarray],
    history: Sequence[np.ndarray],
) -> np.ndarray:
    """Compute the linear signal of a VAR(p) process."""
    p = len(coefficient_matrices)

    if len(history) != p:
        raise ValueError(
            f"history length must match VAR order p={p}. "
            f"Got len(history)={len(history)}."
        )

    d = coefficient_matrices[0].shape[0]
    signal = np.zeros(d)

    for B_lag, x_lag in zip(
        coefficient_matrices,
        history,
    ):
        signal += B_lag @ x_lag

    return signal


def controlled_nonlinear_var_p_step(
    coefficient_matrices: Sequence[np.ndarray],
    history: Sequence[np.ndarray],
    control: float,
    noise_std_dev: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take one controlled nonlinear VAR(p) step."""
    d = coefficient_matrices[0].shape[0]
    signal = lagged_linear_signal(
        coefficient_matrices,
        history,
    )
    noise = rng.normal(
        loc=0.0,
        scale=noise_std_dev,
        size=d,
    )
    return control * np.tanh(signal) + noise


def linear_var_step(
    coefficient_matrices: Sequence[np.ndarray],
    history: Sequence[np.ndarray],
    noise_std_dev: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take one linear VAR(p) step."""
    p = len(coefficient_matrices)

    if len(history) != p:
        raise ValueError(
            f"history length must match VAR order p={p}. "
            f"Got len(history)={len(history)}."
        )

    d = coefficient_matrices[0].shape[0]
    signal = np.zeros(d)

    for B_lag, x_lag in zip(
        coefficient_matrices,
        history,
    ):
        signal += B_lag @ x_lag

    noise = rng.normal(
        loc=0.0,
        scale=noise_std_dev,
        size=d,
    )
    return signal + noise


def piecewise_linear_var(
    num_variables: int,
    num_timepoints: int,
    num_regimes: int,
    change_points: Sequence[int],
    coefficient_graphs: Sequence[Sequence[np.ndarray]],
    noise_std_dev: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a piecewise linear VAR(p) process."""
    validate_change_points(
        num_timepoints,
        num_regimes,
        change_points,
    )

    if len(coefficient_graphs) != num_regimes:
        raise ValueError(
            "Need one coefficient-matrix list per regime. "
            f"Got {len(coefficient_graphs)} for "
            f"{num_regimes} regimes."
        )

    p = len(coefficient_graphs[0])
    if p <= 0:
        raise ValueError("VAR order p must be positive.")

    for regime_coeffs in coefficient_graphs:
        if len(regime_coeffs) != p:
            raise ValueError(
                "All regimes must have the same VAR order p."
            )

    boundaries = list(change_points) + [num_timepoints]
    time_series = np.zeros(
        (num_timepoints, num_variables)
    )
    history = [
        np.zeros(num_variables)
        for _ in range(p)
    ]

    regime = 0
    for t in range(num_timepoints):
        while (
            regime < num_regimes - 1
            and t >= boundaries[regime]
        ):
            regime += 1

        x_t = linear_var_step(
            coefficient_matrices=coefficient_graphs[regime],
            history=history,
            noise_std_dev=noise_std_dev,
            rng=rng,
        )

        time_series[t] = x_t
        history = [x_t] + history[:-1]

    return time_series


def piecewise_controlled_nonlinear_var_p(
    num_variables: int,
    num_timepoints: int,
    num_regimes: int,
    change_points: Sequence[int],
    coefficient_graphs: Sequence[Sequence[np.ndarray]],
    control: float,
    noise_std_dev: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a piecewise controlled nonlinear VAR(p) process."""
    validate_change_points(
        num_timepoints,
        num_regimes,
        change_points,
    )

    if len(coefficient_graphs) != num_regimes:
        raise ValueError(
            "Need one coefficient-matrix list per regime. "
            f"Got {len(coefficient_graphs)} for "
            f"{num_regimes} regimes."
        )

    p = len(coefficient_graphs[0])
    if p <= 0:
        raise ValueError("VAR order p must be positive.")

    for regime_coeffs in coefficient_graphs:
        if len(regime_coeffs) != p:
            raise ValueError(
                "All regimes must have the same lag order p."
            )

    boundaries = list(change_points) + [num_timepoints]
    time_series = np.zeros(
        (num_timepoints, num_variables)
    )
    history = [
        np.zeros(num_variables)
        for _ in range(p)
    ]

    regime = 0
    for t in range(num_timepoints):
        while (
            regime < num_regimes - 1
            and t >= boundaries[regime]
        ):
            regime += 1

        x_t = controlled_nonlinear_var_p_step(
            coefficient_matrices=coefficient_graphs[regime],
            history=history,
            control=control,
            noise_std_dev=noise_std_dev,
            rng=rng,
        )

        time_series[t] = x_t
        history = [x_t] + history[:-1]

    return time_series


def empirical_quantile_match(
    time_series: np.ndarray,
    num_regimes: int,
    change_points: Sequence[int],
) -> np.ndarray:
    """Match later-regime marginals to the first regime."""
    X = np.asarray(
        time_series,
        dtype=float,
    ).copy()
    T, d = X.shape

    validate_change_points(
        T,
        num_regimes,
        change_points,
    )

    boundaries = [0] + list(change_points) + [T]

    for j in range(d):
        z = X[:, j].copy()

        ref_start = boundaries[0]
        ref_end = boundaries[1]
        ref_sorted = np.sort(z[ref_start:ref_end])
        n_ref = ref_end - ref_start

        if n_ref <= 1:
            continue

        for r in range(1, num_regimes):
            start = boundaries[r]
            end = boundaries[r + 1]
            n = end - start

            if n <= 0:
                continue

            if n == 1:
                z[start:end] = ref_sorted[n_ref // 2]
                continue

            segment = z[start:end].copy()
            sorted_indices = np.argsort(
                segment,
                kind="mergesort",
            )
            source_positions = np.floor(
                np.linspace(0, n_ref - 1, n)
            ).astype(int)

            matched_segment = segment.copy()
            matched_segment[sorted_indices] = (
                ref_sorted[source_positions]
            )
            z[start:end] = matched_segment

        X[:, j] = z

    return X