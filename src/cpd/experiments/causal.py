"""Causal experiment configuration and dataset generation."""

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from cpd.data.var import (
    empirical_quantile_match,
    make_var_lag_coefficient_graphs,
    piecewise_controlled_nonlinear_var_p,
    piecewise_linear_var,
)
from cpd.design import (
    _coerce_positive_integer,
    _normalize_grid_choices,
    _sensitivity_value_label,
    resolve_window_size,
)


@dataclass(frozen=True)
class CausalExperimentParameters:
    """Parameters controlling the causal experiment design."""

    dataset_lengths: Sequence[Any]
    noise_std_devs: Sequence[Any]
    changepoint_percentiles: Sequence[Any]
    empirical_quantile_match_options: Sequence[Any]
    data_generating_processes: Sequence[Any]
    dataset_var_lag_orders: Sequence[Any]
    dimension_sizes: Sequence[Any]
    model_lag_orders: Sequence[Any]
    window_sizes: Sequence[Any]

    default_dataset_length: int
    default_noise_std_dev: float
    default_changepoint_percentile: float
    default_empirical_quantile_match: bool
    default_data_generating_process: str
    default_dataset_var_lag_order: int
    default_dimension: int
    default_model_lag_order: int
    default_window_size: Any

    experiment_design_mode: str
    include_default_setup: bool
    auto_shrink_window_if_needed: bool

    num_regimes: int
    control: float
    alpha: float
    beta: float
    var_target_spectral_radius: float
    base_seed: int


@dataclass(frozen=True)
class DatasetConfig:
    """One complete causal dataset and scan configuration."""

    T: int
    sigma: float
    changepoint_percentile: float
    empirical_quantile_match: bool
    data_generating_process: str
    data_lag_order: int
    dimension: int
    model_lag_order: int
    window_size: int
    sensitivity_factor: str = "default"
    sensitivity_value: str = "default"
    seed: int = 42

    @property
    def tau_star(self):
        tau = int(round(self.changepoint_percentile * self.T))
        return int(np.clip(tau, 1, self.T - 1))

    @property
    def label(self):
        qmatch = "matched" if self.empirical_quantile_match else "raw"

        if self.data_generating_process == "controlled_nonlinear_var":
            dgp_label = f"tanhVAR({self.data_lag_order})"
        elif self.data_generating_process == "linear_var":
            dgp_label = f"VAR({self.data_lag_order})"
        else:
            dgp_label = self.data_generating_process

        return (
            f"factor={self.sensitivity_factor}, "
            f"value={self.sensitivity_value}, "
            f"{dgp_label}, d={self.dimension}, "
            f"T={self.T}, sigma={self.sigma:g}, "
            f"cp_pct={self.changepoint_percentile:.4f}, "
            f"tau*={self.tau_star}, {qmatch}, "
            f"fit_L={self.model_lag_order}, h={self.window_size}"
        )


def _make_dataset_config(
    values,
    sensitivity_factor,
    sensitivity_value,
    parameters,
):
    T = _coerce_positive_integer(values["T"], "Dataset length T")
    dimension = _coerce_positive_integer(values["dimension"], "Dimension")
    data_lag_order = _coerce_positive_integer(
        values["data_lag_order"],
        "Dataset VAR lag order",
    )
    model_lag_order = _coerce_positive_integer(
        values["model_lag_order"],
        "Fitted model lag order",
    )

    sigma = float(values["sigma"])
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(
            f"Noise standard deviation must be finite and nonnegative. Got {sigma}."
        )

    changepoint_percentile = float(values["changepoint_percentile"])
    if not 0 < changepoint_percentile < 1:
        raise ValueError(
            "Changepoint percentile must satisfy 0 < p < 1. "
            f"Got {changepoint_percentile}."
        )

    data_generating_process = str(values["data_generating_process"])
    valid_processes = {
        "controlled_nonlinear_var",
        "linear_var",
    }

    if data_generating_process not in valid_processes:
        raise ValueError(
            f"Unknown data-generating process {data_generating_process!r}. "
            f"Valid choices are {sorted(valid_processes)}."
        )

    window_size = resolve_window_size(
        T=T,
        L=model_lag_order,
        window_size_spec=values["window_size_spec"],
        auto_shrink_window_if_needed=(
            parameters.auto_shrink_window_if_needed
        ),
    )

    return DatasetConfig(
        T=T,
        sigma=sigma,
        changepoint_percentile=changepoint_percentile,
        empirical_quantile_match=bool(
            values["empirical_quantile_match"]
        ),
        data_generating_process=data_generating_process,
        data_lag_order=data_lag_order,
        dimension=dimension,
        model_lag_order=model_lag_order,
        window_size=window_size,
        sensitivity_factor=str(sensitivity_factor),
        sensitivity_value=_sensitivity_value_label(sensitivity_value),
        seed=parameters.base_seed,
    )


def iter_dataset_configs(
    parameters: CausalExperimentParameters,
) -> Iterable[DatasetConfig]:
    mode = str(parameters.experiment_design_mode).strip().lower()

    if mode not in {"one_factor_at_a_time", "cartesian"}:
        raise ValueError(
            "experiment_design_mode must be "
            "'one_factor_at_a_time' or 'cartesian'. "
            f"Got {parameters.experiment_design_mode!r}."
        )

    defaults = {
        "T": parameters.default_dataset_length,
        "sigma": parameters.default_noise_std_dev,
        "changepoint_percentile": (
            parameters.default_changepoint_percentile
        ),
        "empirical_quantile_match": (
            parameters.default_empirical_quantile_match
        ),
        "data_generating_process": (
            parameters.default_data_generating_process
        ),
        "data_lag_order": parameters.default_dataset_var_lag_order,
        "dimension": parameters.default_dimension,
        "model_lag_order": parameters.default_model_lag_order,
        "window_size_spec": parameters.default_window_size,
    }

    setup_specs = []

    if mode == "cartesian":
        choices = itertools.product(
            _normalize_grid_choices(
                parameters.dataset_lengths,
                "dataset_lengths",
            ),
            _normalize_grid_choices(
                parameters.noise_std_devs,
                "noise_std_devs",
            ),
            _normalize_grid_choices(
                parameters.changepoint_percentiles,
                "changepoint_percentiles",
            ),
            _normalize_grid_choices(
                parameters.empirical_quantile_match_options,
                "empirical_quantile_match_options",
            ),
            _normalize_grid_choices(
                parameters.data_generating_processes,
                "data_generating_processes",
            ),
            _normalize_grid_choices(
                parameters.dataset_var_lag_orders,
                "dataset_var_lag_orders",
            ),
            _normalize_grid_choices(
                parameters.dimension_sizes,
                "dimension_sizes",
            ),
            _normalize_grid_choices(
                parameters.model_lag_orders,
                "model_lag_orders",
            ),
            _normalize_grid_choices(
                parameters.window_sizes,
                "window_sizes",
            ),
        )

        for values in choices:
            (
                T,
                sigma,
                changepoint_percentile,
                empirical_quantile_match_value,
                data_generating_process,
                data_lag_order,
                dimension,
                model_lag_order,
                window_size_spec,
            ) = values

            setup_specs.append(
                (
                    "cartesian",
                    "combined",
                    {
                        "T": T,
                        "sigma": sigma,
                        "changepoint_percentile": changepoint_percentile,
                        "empirical_quantile_match": (
                            empirical_quantile_match_value
                        ),
                        "data_generating_process": (
                            data_generating_process
                        ),
                        "data_lag_order": data_lag_order,
                        "dimension": dimension,
                        "model_lag_order": model_lag_order,
                        "window_size_spec": window_size_spec,
                    },
                )
            )
    else:
        if parameters.include_default_setup:
            setup_specs.append(
                ("default", "all_defaults", defaults.copy())
            )

        factor_definitions = [
            ("dataset_length", "T", parameters.dataset_lengths),
            ("noise_std_dev", "sigma", parameters.noise_std_devs),
            (
                "changepoint_percentile",
                "changepoint_percentile",
                parameters.changepoint_percentiles,
            ),
            (
                "empirical_quantile_match",
                "empirical_quantile_match",
                parameters.empirical_quantile_match_options,
            ),
            (
                "data_generating_process",
                "data_generating_process",
                parameters.data_generating_processes,
            ),
            (
                "dataset_var_lag_order",
                "data_lag_order",
                parameters.dataset_var_lag_orders,
            ),
            ("dimension", "dimension", parameters.dimension_sizes),
            (
                "model_lag_order",
                "model_lag_order",
                parameters.model_lag_orders,
            ),
            ("window_size", "window_size_spec", parameters.window_sizes),
        ]

        for factor_name, field_name, choices in factor_definitions:
            for choice in _normalize_grid_choices(choices, factor_name):
                values = defaults.copy()
                values[field_name] = choice
                setup_specs.append((factor_name, choice, values))

    seen = set()
    yielded = 0

    for factor_name, factor_value, values in setup_specs:
        config = _make_dataset_config(
            values=values,
            sensitivity_factor=factor_name,
            sensitivity_value=factor_value,
            parameters=parameters,
        )

        key = (
            config.T,
            config.sigma,
            config.changepoint_percentile,
            config.empirical_quantile_match,
            config.data_generating_process,
            config.data_lag_order,
            config.dimension,
            config.model_lag_order,
            config.window_size,
        )

        if key in seen:
            continue

        seen.add(key)
        yielded += 1
        yield config

    if yielded == 0:
        raise ValueError("The experiment design produced no unique setups.")


def generate_dataset(
    config: DatasetConfig,
    parameters: CausalExperimentParameters,
):
    rng = np.random.default_rng(config.seed)

    coefficient_graphs = make_var_lag_coefficient_graphs(
        d=config.dimension,
        p=config.data_lag_order,
        alpha=parameters.alpha,
        beta=parameters.beta,
        rng=rng,
        target_radius=parameters.var_target_spectral_radius,
    )

    if config.data_generating_process == "controlled_nonlinear_var":
        X_raw = piecewise_controlled_nonlinear_var_p(
            num_variables=config.dimension,
            num_timepoints=config.T,
            num_regimes=parameters.num_regimes,
            change_points=[config.tau_star],
            coefficient_graphs=coefficient_graphs,
            control=parameters.control,
            noise_std_dev=config.sigma,
            rng=rng,
        )
    elif config.data_generating_process == "linear_var":
        X_raw = piecewise_linear_var(
            num_variables=config.dimension,
            num_timepoints=config.T,
            num_regimes=parameters.num_regimes,
            change_points=[config.tau_star],
            coefficient_graphs=coefficient_graphs,
            noise_std_dev=config.sigma,
            rng=rng,
        )
    else:
        raise ValueError(
            "Unknown data_generating_process. Valid options are "
            "'controlled_nonlinear_var' and 'linear_var'. "
            f"Got {config.data_generating_process!r}."
        )

    if config.empirical_quantile_match:
        X_used = empirical_quantile_match(
            time_series=X_raw,
            num_regimes=parameters.num_regimes,
            change_points=[config.tau_star],
        )
    else:
        X_used = X_raw

    return X_used, X_raw, coefficient_graphs