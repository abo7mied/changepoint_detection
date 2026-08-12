"""Log-normalized cross-regime changepoint detection."""

import math

import numpy as np

from cpd.models import (
    construct_supervised_prediction_pairs,
    make_model,
)


def train_validation_split(
    examples,
    labels,
    validation_fraction=0.3,
    min_train_size=5,
    min_validation_size=5,
):
    """Chronologically split prediction pairs into training and validation sets."""
    n = len(examples)

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must lie in (0, 1). "
            f"Got {validation_fraction}."
        )

    if n < min_train_size + min_validation_size:
        raise ValueError(
            f"Need at least {min_train_size + min_validation_size} "
            f"prediction pairs. Got {n}."
        )

    validation_size = max(
        min_validation_size,
        int(round(validation_fraction * n)),
    )
    train_size = n - validation_size

    if train_size < min_train_size:
        raise ValueError(
            f"Training split must contain at least "
            f"{min_train_size} pairs. Got {train_size}."
        )

    return (
        examples[:train_size],
        labels[:train_size],
        examples[train_size:],
        labels[train_size:],
    )


def compute_log_normalized_score(
    left_on_right_error,
    left_on_left_error,
    right_on_left_error,
    right_on_right_error,
    score_direction="forward_backward",
    epsilon=1e-10,
    use_epsilon=True,
):
    """Compute the forward or forward-backward score."""
    if use_epsilon and epsilon <= 0:
        raise ValueError(
            f"epsilon must be positive. Got {epsilon}."
        )

    offset = epsilon if use_epsilon else 0.0

    forward_score = math.log(
        (left_on_right_error + offset)
        / (left_on_left_error + offset)
    )

    if score_direction == "forward_only":
        return float(forward_score)

    if score_direction == "forward_backward":
        backward_score = math.log(
            (right_on_left_error + offset)
            / (right_on_right_error + offset)
        )

        return float(
            forward_score + backward_score
        )

    raise ValueError(
        "score_direction must be either "
        "'forward_only' or 'forward_backward'. "
        f"Got {score_direction!r}."
    )


def estimate_causal_cpd_log_normalized_cross_regime(
    series,
    window_size,
    L,
    candidates,
    competitor,
    device='cpu',
    use_epsilon=True,
    reverse_right_to_left=True,
    split_train_validation=False,
    validation_fraction=0.3,
    min_train_size=5,
    min_validation_size=5,
    verbose=False,
):
    """Estimate the changepoint using cross-regime transfer."""
    X = np.asarray(series, dtype=float)
    h = int(window_size)

    if h <= L:
        raise ValueError(
            "window_size must be larger than L. "
            f"Got h={h}, L={L}."
        )

    if len(candidates) == 0:
        raise ValueError("candidates is empty.")

    scores = {}
    best_score = -np.inf
    best_tau = None

    for i, tau in enumerate(candidates):
        tau = int(tau)

        window_left = X[tau - h:tau]
        window_right = X[tau:tau + h]

        if (
            len(window_left) != h
            or len(window_right) != h
        ):
            raise ValueError(
                f"Candidate tau={tau} does not have "
                f"full windows of size h={h}."
            )

        (
            left_examples,
            left_labels,
        ) = construct_supervised_prediction_pairs(
            window_left,
            L,
        )

        (
            right_examples,
            right_labels,
        ) = construct_supervised_prediction_pairs(
            window_right,
            L,
        )

        if split_train_validation:
            (
                left_train_examples,
                left_train_labels,
                left_test_examples,
                left_test_labels,
            ) = train_validation_split(
                left_examples,
                left_labels,
                validation_fraction,
                min_train_size,
                min_validation_size,
            )

            (
                _,
                _,
                right_test_examples,
                right_test_labels,
            ) = train_validation_split(
                right_examples,
                right_labels,
                validation_fraction,
                min_train_size,
                min_validation_size,
            )

        else:
            left_train_examples = left_examples
            left_train_labels = left_labels
            left_test_examples = left_examples
            left_test_labels = left_labels
            right_test_examples = right_examples
            right_test_labels = right_labels

        f_left = make_model(
            competitor.model_choice,
            L,
            device,
        )

        f_left.fit(
            left_train_examples,
            left_train_labels,
        )

        _, left_on_right_error = f_left.test(
            right_test_examples,
            right_test_labels,
        )

        _, left_on_left_error = f_left.test(
            left_test_examples,
            left_test_labels,
        )

        if reverse_right_to_left:
            backward_right = window_right[::-1]
            backward_left = window_left[::-1]
        else:
            backward_right = window_right
            backward_left = window_left

        (
            backward_right_examples,
            backward_right_labels,
        ) = construct_supervised_prediction_pairs(
            backward_right,
            L,
        )

        (
            backward_left_examples,
            backward_left_labels,
        ) = construct_supervised_prediction_pairs(
            backward_left,
            L,
        )

        if split_train_validation:
            (
                right_train_examples,
                right_train_labels,
                right_test_examples,
                right_test_labels,
            ) = train_validation_split(
                backward_right_examples,
                backward_right_labels,
                validation_fraction,
                min_train_size,
                min_validation_size,
            )

            (
                _,
                _,
                left_test_examples,
                left_test_labels,
            ) = train_validation_split(
                backward_left_examples,
                backward_left_labels,
                validation_fraction,
                min_train_size,
                min_validation_size,
            )

        else:
            right_train_examples = backward_right_examples
            right_train_labels = backward_right_labels
            right_test_examples = backward_right_examples
            right_test_labels = backward_right_labels
            left_test_examples = backward_left_examples
            left_test_labels = backward_left_labels

        f_right = make_model(
            competitor.model_choice,
            L,
            device,
        )

        f_right.fit(
            right_train_examples,
            right_train_labels,
        )

        _, right_on_left_error = f_right.test(
            left_test_examples,
            left_test_labels,
        )

        _, right_on_right_error = f_right.test(
            right_test_examples,
            right_test_labels,
        )

        score = compute_log_normalized_score(
            left_on_right_error=left_on_right_error,
            left_on_left_error=left_on_left_error,
            right_on_left_error=right_on_left_error,
            right_on_right_error=right_on_right_error,
            score_direction=competitor.score_direction,
            epsilon=competitor.epsilon,
            use_epsilon=use_epsilon,
        )

        if not np.isfinite(score):
            score = -np.inf

        scores[tau] = float(score)

        if score > best_score:
            best_score = score
            best_tau = tau

        if verbose:
            print(
                f"    {competitor.label}: processed "
                f"{i + 1}/{len(candidates)} candidates",
                end="\r",
            )

    if verbose:
        print(" " * 100, end="\r")

    if best_tau is None:
        raise RuntimeError(
            "Could not estimate changepoint: "
            "no finite score was produced."
        )

    return best_tau, scores