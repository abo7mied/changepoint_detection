"""Experimental setup and competitor construction."""

import itertools
import warnings

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np


# Classes

@dataclass(frozen=True)
class CompetitorConfig:
    """A fully specified CPD competitor."""

    model_choice: str
    score_direction: str
    epsilon: float
    model_lag_order: int
    window_size: int

    @property
    def forward_only(self):
        if self.score_direction == "forward_only":
            return True

        if self.score_direction == "forward_backward":
            return False

        raise ValueError(
            "score_direction must be either "
            "'forward_only' or 'forward_backward'. "
            f"Got {self.score_direction!r}."
        )

    @property
    def label(self):
        if self.model_choice == "KMOVARScore":
            base = (
                f"{self.model_choice} | "
                f"{self.score_direction}"
            )
        else:
            base = (
                f"{self.model_choice} | "
                f"{self.score_direction} | "
                f"eps={self.epsilon:g}"
            )

        return (
            f"{base} | "
            f"L={self.model_lag_order} | "
            f"h={self.window_size}"
        )


# Functions

def _coerce_positive_integer(value, name):
    """Validate and return a positive integer."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} cannot be boolean."
        )

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer. "
            f"Got {value!r}."
        ) from exc

    if (
        not np.isfinite(numeric)
        or not numeric.is_integer()
        or numeric <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer. "
            f"Got {value!r}."
        )

    return int(numeric)


def _sensitivity_value_label(value):
    """Return a stable label for a sensitivity value."""
    if callable(value):
        return getattr(
            value,
            "__name__",
            repr(value),
        )

    if isinstance(value, float):
        return f"{value:g}"

    return str(value)


def _normalize_grid_choices(values, name):
    """Convert one value or a sequence into a nonempty list."""
    if (
        callable(values)
        or isinstance(values, (str, bytes))
        or np.isscalar(values)
    ):
        choices = [values]
    else:
        choices = list(values)

    if not choices:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return choices


def resolve_window_size(
    T,
    L,
    window_size_spec,
    auto_shrink_window_if_needed=False,
):
    """Resolve a fixed or series-dependent window size."""
    if callable(window_size_spec):
        raw_h = window_size_spec(T)
    else:
        raw_h = window_size_spec

    if isinstance(raw_h, (bool, np.bool_)):
        raise ValueError(
            "Window size cannot be boolean."
        )

    try:
        raw_h_float = float(raw_h)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Window-size choice must resolve to an integer. "
            f"Got {raw_h!r}."
        ) from exc

    if (
        not np.isfinite(raw_h_float)
        or not raw_h_float.is_integer()
    ):
        raise ValueError(
            "Window-size choice must resolve to a "
            f"finite integer. Got {raw_h!r}."
        )

    h = int(raw_h_float)
    max_h = (T // 2) - 1

    if h > max_h:
        if auto_shrink_window_if_needed:
            warnings.warn(
                f"Configured window h={h} is too large "
                f"for T={T}. Shrinking to h={max_h}.",
                RuntimeWarning,
            )
            h = max_h
        else:
            raise ValueError(
                f"Configured window h={h} is too large "
                f"for T={T}. Need h <= {max_h}."
            )

    if h <= 0:
        raise ValueError(
            "Resolved window size must be positive. "
            f"Got h={h}."
        )

    if h <= L:
        raise ValueError(
            f"Resolved window size h={h} "
            f"must be larger than L={L}."
        )

    return h


def iter_competitor_configs(
    T,
    model_choices,
    score_directions,
    epsilon_values,
    kmo_h_choices,
    model_lag_orders,
    window_sizes,
    auto_shrink_window_if_needed=False,
    model_lag_order=None,
    window_size=None,
):
    """Yield competitors for one scan configuration."""
    if (
        (model_lag_order is None)
        != (window_size is None)
    ):
        raise ValueError(
            "model_lag_order and window_size must "
            "either both be supplied or both be omitted."
        )

    if model_lag_order is None:
        lag_window_pairs = []

        for lag_order_raw, window_spec in itertools.product(
            _normalize_grid_choices(
                model_lag_orders,
                "MODEL_LAG_ORDERS",
            ),
            _normalize_grid_choices(
                window_sizes,
                "WINDOW_SIZES",
            ),
        ):
            lag_order = _coerce_positive_integer(
                lag_order_raw,
                "Fitted model lag order",
            )

            h = resolve_window_size(
                T,
                lag_order,
                window_spec,
                auto_shrink_window_if_needed,
            )

            lag_window_pairs.append(
                (lag_order, h)
            )

    else:
        lag_order = _coerce_positive_integer(
            model_lag_order,
            "Fitted model lag order",
        )

        h = resolve_window_size(
            T,
            lag_order,
            window_size,
            auto_shrink_window_if_needed,
        )

        lag_window_pairs = [
            (lag_order, h)
        ]

    seen = set()

    for lag_order, h in lag_window_pairs:
        for model_choice_raw in model_choices:
            model_choice = str(model_choice_raw)

            if model_choice == "KMOVARScore":
                score_epsilon_pairs = [
                    (
                        f"kmo_{h_choice}",
                        float("nan"),
                    )
                    for h_choice in kmo_h_choices
                ]

            else:
                score_epsilon_pairs = [
                    (
                        str(score_direction),
                        float(epsilon),
                    )
                    for score_direction, epsilon
                    in itertools.product(
                        score_directions,
                        epsilon_values,
                    )
                ]

            for (
                score_direction,
                epsilon,
            ) in score_epsilon_pairs:
                if np.isnan(epsilon):
                    epsilon_key = None
                else:
                    epsilon_key = epsilon

                key = (
                    model_choice,
                    score_direction,
                    epsilon_key,
                    lag_order,
                    h,
                )

                if key in seen:
                    continue

                seen.add(key)

                yield CompetitorConfig(
                    model_choice=model_choice,
                    score_direction=score_direction,
                    epsilon=epsilon,
                    model_lag_order=lag_order,
                    window_size=h,
                )


def make_candidates(T, h, step):
    """Return candidates with complete left and right windows."""
    if step <= 0:
        raise ValueError(
            "CANDIDATE_STEP must be positive. "
            f"Got {step}."
        )

    candidates = list(
        range(
            h,
            T - h + 1,
            step,
        )
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No valid candidates for "
            f"T={T}, h={h}, step={step}."
        )

    return candidates


def make_full_experiment_replicate_seeds(
    num_replicates,
    base_seed,
):
    """Construct reproducible replicate seeds."""
    if num_replicates <= 0:
        raise ValueError(
            "FULL_EXPERIMENT_NUM_REPLICATES must "
            f"be positive. Got {num_replicates}."
        )

    seed_sequence = np.random.SeedSequence(
        base_seed
    )

    children = seed_sequence.spawn(
        num_replicates
    )

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in children
    ]