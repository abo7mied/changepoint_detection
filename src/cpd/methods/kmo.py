"""KMO VAR-score baseline."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def make_flat_var_design_for_kmo(
    X,
    L,
    include_intercept=False,
):
    """Construct the flat VAR design used by the KMO baseline."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must have shape (T, d). Got {X.shape}."
        )

    if L <= 0:
        raise ValueError(
            f"L must be positive. Got L={L}."
        )

    if len(X) <= L:
        raise ValueError(
            f"Need len(X) > L. "
            f"Got len(X)={len(X)}, L={L}."
        )

    Phi_rows = []
    Y_rows = []
    target_times = []

    for t in range(L, len(X)):
        row = X[t - L:t].reshape(-1)

        if include_intercept:
            row = np.concatenate([
                [1.0],
                row,
            ])

        Phi_rows.append(row)
        Y_rows.append(X[t])
        target_times.append(t)

    return (
        np.asarray(Phi_rows),
        np.asarray(Y_rows),
        np.asarray(target_times, dtype=int),
    )


def fit_global_var_for_kmo(
    X,
    L,
    include_intercept=False,
    ridge=1e-8,
):
    """Fit the global VAR model used by the KMO baseline."""
    Phi, Y, target_times = (
        make_flat_var_design_for_kmo(
            X=X,
            L=L,
            include_intercept=include_intercept,
        )
    )

    q = Phi.shape[1]
    A = Phi.T @ Phi + ridge * np.eye(q)
    B = np.linalg.solve(A, Phi.T @ Y)
    E = Y - Phi @ B

    return Phi, Y, target_times, B, E


def compute_kmo_varscore_curve(
    series,
    L,
    candidates,
    h_choice="diag_inv_cov",
    include_intercept=False,
    ridge=1e-8,
):
    """Compute the KMO VAR-score curve."""
    X = np.asarray(series, dtype=float)

    if len(candidates) == 0:
        raise ValueError("candidates is empty.")

    Phi, Y, target_times, _, E = (
        fit_global_var_for_kmo(
            X=X,
            L=L,
            include_intercept=include_intercept,
            ridge=ridge,
        )
    )

    n, q = Phi.shape
    d = Y.shape[1]

    if n < 3:
        raise ValueError(
            "Need at least 3 supervised rows for KMO. "
            f"Got n={n}."
        )

    Xi = np.einsum(
        "tq,td->tqd",
        Phi,
        E,
    ).reshape(n, q * d)

    Xi = Xi - Xi.mean(
        axis=0,
        keepdims=True,
    )

    C = np.cumsum(Xi, axis=0)
    total = C[-1]

    if h_choice == "identity":
        Hmat = None
        variances = None

    elif h_choice == "diag_inv_cov":
        variances = Xi.var(axis=0) + ridge
        Hmat = None

    elif h_choice == "pinv_cov":
        Sigma = np.cov(
            Xi,
            rowvar=False,
        )

        Hmat = np.linalg.pinv(
            Sigma
            + ridge * np.eye(Sigma.shape[0])
        )

        variances = None

    else:
        raise ValueError(
            "Unknown KMO H choice. Valid choices are "
            "'identity', 'diag_inv_cov', and 'pinv_cov'. "
            f"Got {h_choice!r}."
        )

    scores = {}
    best_tau = None
    best_score = -np.inf

    for tau in candidates:
        tau = int(tau)

        k = int(
            np.searchsorted(
                target_times,
                tau,
                side="left",
            )
        )

        if k <= 0 or k >= n:
            continue

        Z = C[k - 1] - (k / n) * total

        if h_choice == "identity":
            score = float(
                np.sum(Z**2)
            )

        elif h_choice == "diag_inv_cov":
            score = float(
                np.sum((Z**2) / variances)
            )

        else:
            score = float(
                Z @ Hmat @ Z
            )

        if not np.isfinite(score):
            score = -np.inf

        scores[tau] = score

        if score > best_score:
            best_score = score
            best_tau = tau

    if best_tau is None:
        raise RuntimeError(
            "KMO baseline produced no finite score "
            "on the candidate set."
        )

    return best_tau, scores


def estimate_kmo_varscore_amoc(
    series,
    L,
    candidates,
    competitor,
    include_intercept=False,
    ridge=1e-8,
):
    """Run the KMO baseline from a competitor configuration."""
    prefix = "kmo_"

    if not competitor.score_direction.startswith(prefix):
        raise ValueError(
            "KMO competitor score_direction must start "
            f"with {prefix!r}. "
            f"Got {competitor.score_direction!r}."
        )

    h_choice = competitor.score_direction[
        len(prefix):
    ]

    return compute_kmo_varscore_curve(
        series=series,
        L=L,
        candidates=candidates,
        h_choice=h_choice,
        include_intercept=include_intercept,
        ridge=ridge,
    )