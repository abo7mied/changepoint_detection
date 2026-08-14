"""Spectral experiment configuration and dataset generation."""

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from cpd.data.spectral import (
    controlled_spectral_mixture,
    validate_spectral_coefficients,
)
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


# --- 1: Coefficient Design ---

@dataclass(frozen=True)
class SpectralExperimentParameters:
    """Parameters controlling the spectral experiment design."""

    dataset_lengths: Sequence[Any]
    noise_scales: Sequence[Any]
    changepoint_percentiles: Sequence[Any]
    empirical_quantile_match_options: Sequence[Any]
    data_generating_processes: Sequence[Any]
    dataset_var_lag_orders: Sequence[Any]
    model_lag_orders: Sequence[Any]
    window_sizes: Sequence[Any]
    dimension_sizes: Sequence[Any]

    spectral_noise_coefficients: Sequence[float]
    spectral_transfer_masses: Sequence[float]
    spectral_num_affected_variables: Sequence[int]
    spectral_transfer_directions: Sequence[str]
    spectral_concentration_profiles: Mapping[str, Mapping[str, Sequence[float]]]

    default_dataset_length: int
    default_noise_scale: float
    default_changepoint_percentile: float
    default_empirical_quantile_match: bool
    default_data_generating_process: str
    default_dataset_var_lag_order: int
    default_model_lag_order: int
    default_window_size: Any
    default_dimension: int

    default_spectral_noise_coefficient: float
    default_spectral_transfer_mass: float
    default_spectral_num_affected_variables: int
    default_spectral_transfer_direction: str
    default_spectral_concentration_profile: str

    low_bands: Sequence[str]
    high_bands: Sequence[str]
    spectral_variable_order: Sequence[int]

    experiment_design_mode: str
    include_default_setup: bool
    auto_shrink_window_if_needed: bool

    num_regimes: int
    sampling_rate: float
    spectral_ar_damping: float
    spectral_filter_order: int
    spectral_burn_in: int

    control: float
    alpha: float
    beta: float
    var_target_spectral_radius: float
    base_seed: int


@dataclass(frozen=True)
class SpectralCoefficientExperiment:
    """One before-versus-after spectral coefficient experiment."""

    name: str
    description: str
    dimension: int
    coefficients: np.ndarray = field(
        repr=False,
        compare=False,
    )
    sensitivity_factor: str = "custom"
    sensitivity_value: str = "custom"


@dataclass(frozen=True)
class DatasetConfig:
    """One complete spectral dataset and scan configuration."""

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
    spectral_experiment_name: Optional[str] = None
    spectral_regime_coefficients: Optional[np.ndarray] = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def tau_star(self):
        tau = int(round(
            self.changepoint_percentile * self.T
        ))
        return int(np.clip(tau, 1, self.T - 1))

    @property
    def label(self):
        qmatch = (
            "matched"
            if self.empirical_quantile_match
            else "raw"
        )

        if (
            self.data_generating_process
            == "controlled_nonlinear_var"
        ):
            dgp_label = (
                f"tanhVAR({self.data_lag_order})"
            )
            noise_label = f"sigma={self.sigma:g}"
        elif self.data_generating_process == "linear_var":
            dgp_label = f"VAR({self.data_lag_order})"
            noise_label = f"sigma={self.sigma:g}"
        elif (
            self.data_generating_process
            == "spectral_mixture"
        ):
            name = (
                self.spectral_experiment_name
                or "unspecified"
            )
            dgp_label = f"spectral_mixture[{name}]"
            noise_label = (
                f"noise_scale={self.sigma:g}"
            )
        else:
            dgp_label = (
                self.data_generating_process
            )
            noise_label = f"sigma={self.sigma:g}"

        return (
            f"factor={self.sensitivity_factor}, "
            f"value={self.sensitivity_value}, "
            f"{dgp_label}, d={self.dimension}, "
            f"T={self.T}, {noise_label}, "
            f"cp_pct={self.changepoint_percentile:.4f}, "
            f"tau*={self.tau_star}, {qmatch}, "
            f"fit_L={self.model_lag_order}, "
            f"h={self.window_size}"
        )   


def make_spectral_coefficients_from_bundles(
    regime_bundle_specs,
    dimension,
    parameters,
):
    band_names = (
        tuple(parameters.low_bands)
        + tuple(parameters.high_bands)
    )
    band_index = {
        name: index
        for index, name in enumerate(band_names)
    }

    if len(regime_bundle_specs) != parameters.num_regimes:
        raise ValueError(
            f"Expected {parameters.num_regimes} regimes, "
            f"got {len(regime_bundle_specs)}."
        )

    coefficients = np.zeros(
        (
            parameters.num_regimes,
            dimension,
            6,
        ),
        dtype=float,
    )

    for regime, variable_specs in enumerate(
        regime_bundle_specs
    ):
        if len(variable_specs) != dimension:
            raise ValueError(
                f"Regime {regime} must contain "
                f"{dimension} variable specifications."
            )

        for variable, spec in enumerate(
            variable_specs
        ):
            if (
                "bundles" not in spec
                or "noise" not in spec
            ):
                raise ValueError(
                    f"Invalid specification for "
                    f"regime {regime}, "
                    f"variable {variable}."
                )

            noise = float(spec["noise"])

            if not np.isfinite(noise) or noise < 0:
                raise ValueError(
                    "Noise must be finite and nonnegative."
                )

            coefficients[
                regime,
                variable,
                5,
            ] = noise

            used_bands = set()
            total_bundle_weight = 0.0

            for bands, total_weight in spec["bundles"]:
                bands = list(bands)
                total_weight = float(total_weight)

                if not bands:
                    raise ValueError(
                        "A spectral bundle cannot be empty."
                    )

                if len(bands) != len(set(bands)):
                    raise ValueError(
                        "A spectral bundle contains "
                        "a repeated band."
                    )

                unknown_bands = [
                    band
                    for band in bands
                    if band not in band_index
                ]

                if unknown_bands:
                    raise ValueError(
                        "Unknown spectral bands: "
                        f"{unknown_bands}."
                    )

                overlap = used_bands.intersection(
                    bands
                )

                if overlap:
                    raise ValueError(
                        f"Bands {sorted(overlap)} "
                        "appear in multiple bundles."
                    )

                if (
                    not np.isfinite(total_weight)
                    or total_weight < 0
                ):
                    raise ValueError(
                        "Bundle weights must be finite "
                        "and nonnegative."
                    )

                used_bands.update(bands)

                per_band_weight = (
                    total_weight / len(bands)
                )

                for band in bands:
                    coefficients[
                        regime,
                        variable,
                        band_index[band],
                    ] = per_band_weight

                total_bundle_weight += total_weight

            total = total_bundle_weight + noise

            if not np.isclose(
                total,
                1.0,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "Bundle weights plus noise must "
                    f"sum to 1. Got {total:g}."
                )

    return coefficients


def affected_variables_from_count(
    num_affected_variables,
    dimension,
    parameters,
):
    count = int(num_affected_variables)

    if not 1 <= count <= dimension:
        raise ValueError(
            "num_affected_variables must lie in "
            f"[1, {dimension}]. Got {count}."
        )

    variable_order = tuple(
        parameters.spectral_variable_order[:dimension]
    )

    if tuple(sorted(variable_order)) != tuple(
        range(dimension)
    ):
        raise ValueError(
            "The first dimension entries of "
            "spectral_variable_order must be a "
            f"permutation of 0,...,{dimension - 1}."
        )

    return tuple(variable_order[:count])


def make_band_bundles_from_group_masses(
    low_mass,
    high_mass,
    concentration,
    parameters,
):
    profiles = parameters.spectral_concentration_profiles

    if concentration not in profiles:
        raise ValueError(
            f"Unknown concentration profile {concentration!r}."
        )

    profile = profiles[concentration]
    bundles = []

    for band, proportion in zip(
        parameters.low_bands,
        profile["low"],
    ):
        bundles.append(
            ([band], float(low_mass) * float(proportion))
        )

    for band, proportion in zip(
        parameters.high_bands,
        profile["high"],
    ):
        bundles.append(
            ([band], float(high_mass) * float(proportion))
        )

    return bundles


def make_spectral_transfer_bundle_specs(
    noise,
    transfer_mass,
    concentration,
    num_affected_variables,
    direction,
    dimension,
    parameters,
):
    noise = float(noise)
    transfer_mass = float(transfer_mass)
    signal_mass = 1.0 - noise

    if not 0 <= noise < 1:
        raise ValueError(f"noise must lie in [0, 1). Got {noise}.")

    if not 0 <= transfer_mass <= signal_mass:
        raise ValueError(
            f"transfer_mass must lie in [0, {signal_mass:g}]. "
            f"Got {transfer_mass:g}."
        )

    affected_variables = set(
        affected_variables_from_count(
            num_affected_variables,
            dimension,
            parameters,
        )
    )

    midpoint = signal_mass / 2
    half_transfer = transfer_mass / 2

    if direction == "low_to_high":
        low_before = midpoint + half_transfer
        low_after = midpoint - half_transfer
    elif direction == "high_to_low":
        low_before = midpoint - half_transfer
        low_after = midpoint + half_transfer
    else:
        raise ValueError(
            "direction must be 'low_to_high' or 'high_to_low'. "
            f"Got {direction!r}."
        )

    high_before = signal_mass - low_before
    high_after = signal_mass - low_after

    before_bundles = make_band_bundles_from_group_masses(
        low_mass=low_before,
        high_mass=high_before,
        concentration=concentration,
        parameters=parameters,
    )
    after_bundles = make_band_bundles_from_group_masses(
        low_mass=low_after,
        high_mass=high_after,
        concentration=concentration,
        parameters=parameters,
    )

    regime_1_specs = []
    regime_2_specs = []

    for variable in range(dimension):
        regime_1_specs.append(
            {
                "bundles": list(before_bundles),
                "noise": noise,
            }
        )
        regime_2_specs.append(
            {
                "bundles": list(
                    after_bundles
                    if variable in affected_variables
                    else before_bundles
                ),
                "noise": noise,
            }
        )

    return [regime_1_specs, regime_2_specs]


def _make_spectral_experiment(
    noise,
    transfer_mass,
    concentration,
    num_affected,
    direction,
    sensitivity_factor,
    sensitivity_value,
    dimension,
    parameters,
):
    affected_variables = affected_variables_from_count(
        num_affected,
        dimension,
        parameters,
    )

    prefix = (
        "default"
        if sensitivity_factor == "default"
        else f"vary_{sensitivity_factor}"
    )

    name = (
        f"{prefix}"
        f"__noise_{float(noise):.2f}"
        f"__m_{float(transfer_mass):.2f}"
        f"__{concentration}"
        f"__k_{int(num_affected)}"
        f"__{direction}"
    )

    description = (
        f"sensitivity factor={sensitivity_factor}, "
        f"value={sensitivity_value}; "
        f"{direction.replace('_', ' ')}, "
        f"noise={float(noise):.2f}, "
        f"total transferred mass={float(transfer_mass):.2f}, "
        f"concentration={concentration}, "
        f"affected variables={affected_variables} "
        f"({len(affected_variables)}/{dimension}), "
        f"dimension={dimension}."
    )

    regime_bundle_specs = (
        make_spectral_transfer_bundle_specs(
            noise=noise,
            transfer_mass=transfer_mass,
            concentration=concentration,
            num_affected_variables=num_affected,
            direction=direction,
            dimension=dimension,
            parameters=parameters,
        )
    )

    coefficients = (
        make_spectral_coefficients_from_bundles(
            regime_bundle_specs,
            dimension,
            parameters,
        )
    )

    return SpectralCoefficientExperiment(
        name=name,
        description=description,
        dimension=dimension,
        coefficients=coefficients,
        sensitivity_factor=str(sensitivity_factor),
        sensitivity_value=str(sensitivity_value),
    )


def default_spectral_experiment(
    parameters,
    dimension=None,
):
    if dimension is None:
        dimension = parameters.default_dimension

    dimension = _coerce_positive_integer(
        dimension,
        "Dimension",
    )

    return _make_spectral_experiment(
        noise=(
            parameters
            .default_spectral_noise_coefficient
        ),
        transfer_mass=(
            parameters
            .default_spectral_transfer_mass
        ),
        concentration=(
            parameters
            .default_spectral_concentration_profile
        ),
        num_affected=(
            parameters
            .default_spectral_num_affected_variables
        ),
        direction=(
            parameters
            .default_spectral_transfer_direction
        ),
        sensitivity_factor="default",
        sensitivity_value="all_defaults",
        dimension=dimension,
        parameters=parameters,
    )


def build_spectral_coefficient_experiments(
    parameters,
    dimension=None,
):
    if dimension is None:
        dimension = parameters.default_dimension

    dimension = _coerce_positive_integer(
        dimension,
        "Dimension",
    )

    defaults = {
        "noise": parameters.default_spectral_noise_coefficient,
        "transfer_mass": parameters.default_spectral_transfer_mass,
        "concentration": (
            parameters.default_spectral_concentration_profile
        ),
        "num_affected": (
            parameters.default_spectral_num_affected_variables
        ),
        "direction": parameters.default_spectral_transfer_direction,
    }

    mode = str(parameters.experiment_design_mode).strip().lower()

    if mode == "cartesian":
        specs = [
            (
                "cartesian",
                "combined",
                {
                    "noise": noise,
                    "transfer_mass": transfer_mass,
                    "concentration": concentration,
                    "num_affected": num_affected,
                    "direction": direction,
                },
            )
            for (
                noise,
                transfer_mass,
                concentration,
                num_affected,
                direction,
            ) in itertools.product(
                parameters.spectral_noise_coefficients,
                parameters.spectral_transfer_masses,
                parameters.spectral_concentration_profiles.keys(),
                parameters.spectral_num_affected_variables,
                parameters.spectral_transfer_directions,
            )
        ]
    elif mode == "one_factor_at_a_time":
        specs = []

        if parameters.include_default_setup:
            specs.append(
                ("default", "all_defaults", defaults.copy())
            )

        factor_definitions = [
            (
                "spectral_noise_coefficient",
                "noise",
                parameters.spectral_noise_coefficients,
            ),
            (
                "spectral_transfer_mass",
                "transfer_mass",
                parameters.spectral_transfer_masses,
            ),
            (
                "spectral_concentration_profile",
                "concentration",
                parameters.spectral_concentration_profiles.keys(),
            ),
            (
                "spectral_num_affected_variables",
                "num_affected",
                parameters.spectral_num_affected_variables,
            ),
            (
                "spectral_transfer_direction",
                "direction",
                parameters.spectral_transfer_directions,
            ),
        ]

        for factor_name, field_name, choices in factor_definitions:
            for choice in choices:
                values = defaults.copy()
                values[field_name] = choice
                specs.append((factor_name, choice, values))
    else:
        raise ValueError(
            "experiment_design_mode must be "
            "'one_factor_at_a_time' or 'cartesian'. "
            f"Got {parameters.experiment_design_mode!r}."
        )

    experiments = []
    seen = set()

    for factor_name, factor_value, values in specs:
        key = (
            float(values["noise"]),
            float(values["transfer_mass"]),
            str(values["concentration"]),
            int(values["num_affected"]),
            str(values["direction"]),
        )

        if key in seen:
            continue

        seen.add(key)

        experiments.append(
            _make_spectral_experiment(
                noise=values["noise"],
                transfer_mass=values["transfer_mass"],
                concentration=values["concentration"],
                num_affected=values["num_affected"],
                direction=values["direction"],
                sensitivity_factor=factor_name,
                sensitivity_value=factor_value,
                dimension=dimension,
                parameters=parameters,
            )
        )

    if not experiments:
        raise ValueError(
            "The spectral design produced no unique experiments."
        )

    return experiments


# --- 2: Dataset Enumeration and Generation ---


def _make_dataset_config(
    values,
    sensitivity_factor,
    sensitivity_value,
    spectral_experiment,
    parameters,
):
    dimension = _coerce_positive_integer(
        values["dimension"],
        "Dimension",
    )
    T = _coerce_positive_integer(values["T"], "Dataset length T")
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
            f"Noise scale must be finite and nonnegative. Got {sigma}."
        )

    changepoint_percentile = float(values["changepoint_percentile"])
    if not 0 < changepoint_percentile < 1:
        raise ValueError(
            "Changepoint percentile must satisfy 0 < p < 1. "
            f"Got {changepoint_percentile}."
        )

    data_generating_process = str(values["data_generating_process"])
    valid_processes = {
        "spectral_mixture",
        "controlled_nonlinear_var",
        "linear_var",
    }

    if data_generating_process not in valid_processes:
        raise ValueError(
            f"Unknown data-generating process {data_generating_process!r}. "
            f"Valid choices are {sorted(valid_processes)}."
        )

    quantile_match = bool(values["empirical_quantile_match"])

    if data_generating_process == "spectral_mixture":
        if quantile_match:
            raise ValueError(
                "Empirical quantile matching cannot be enabled "
                "for spectral_mixture."
            )
        if spectral_experiment is None:
            raise ValueError(
                "A spectral coefficient experiment is required "
                "for spectral_mixture."
            )
    else:
        spectral_experiment = None

    window_size = resolve_window_size(
        T=T,
        L=model_lag_order,
        window_size_spec=values["window_size_spec"],
        auto_shrink_window_if_needed=(
            parameters.auto_shrink_window_if_needed
        ),
    )

    if spectral_experiment is not None:
        expected_shape = (
            parameters.num_regimes,
            dimension,
            6,
        )

        if spectral_experiment.coefficients.shape != expected_shape:
            raise ValueError(
                "Spectral coefficients must have shape "
                f"{expected_shape}. Got "
                f"{spectral_experiment.coefficients.shape}."
            )

    return DatasetConfig(
        T=T,
        sigma=sigma,
        changepoint_percentile=changepoint_percentile,
        empirical_quantile_match=quantile_match,
        data_generating_process=data_generating_process,
        data_lag_order=data_lag_order,
        dimension=dimension,
        model_lag_order=model_lag_order,
        window_size=window_size,
        sensitivity_factor=str(sensitivity_factor),
        sensitivity_value=_sensitivity_value_label(
            sensitivity_value
        ),
        seed=parameters.base_seed,
        spectral_experiment_name=(
            None
            if spectral_experiment is None
            else spectral_experiment.name
        ),
        spectral_regime_coefficients=(
            None
            if spectral_experiment is None
            else np.asarray(
                spectral_experiment.coefficients,
                dtype=float,
            ).copy()
        ),
    )


def iter_dataset_configs(
    parameters: SpectralExperimentParameters,
) -> Iterable[DatasetConfig]:
    mode = str(parameters.experiment_design_mode).strip().lower()

    if mode not in {"one_factor_at_a_time", "cartesian"}:
        raise ValueError(
            "experiment_design_mode must be "
            "'one_factor_at_a_time' or 'cartesian'."
        )

    default_experiment = default_spectral_experiment(
        parameters,
        dimension=parameters.default_dimension,
    )
    defaults = {
        "T": parameters.default_dataset_length,
        "sigma": parameters.default_noise_scale,
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
        "model_lag_order": parameters.default_model_lag_order,
        "window_size_spec": parameters.default_window_size,
        "dimension": parameters.default_dimension,
    }

    setup_specs = []

    if mode == "cartesian":
        outer_choices = itertools.product(
            _normalize_grid_choices(
                parameters.dataset_lengths,
                "dataset_lengths",
            ),
            _normalize_grid_choices(
                parameters.noise_scales,
                "noise_scales",
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

        for outer_values in outer_choices:
            (
                T,
                sigma,
                changepoint_percentile,
                quantile_match,
                data_generating_process,
                data_lag_order,
                dimension,
                model_lag_order,
                window_size_spec,
            ) = outer_values

            values = {
                "T": T,
                "sigma": sigma,
                "changepoint_percentile": changepoint_percentile,
                "empirical_quantile_match": quantile_match,
                "data_generating_process": data_generating_process,
                "data_lag_order": data_lag_order,
                "dimension": dimension,
                "model_lag_order": model_lag_order,
                "window_size_spec": window_size_spec,
            }

            experiments = (
                build_spectral_coefficient_experiments(
                    parameters,
                    dimension=dimension,
                )
                if str(data_generating_process)
                == "spectral_mixture"
                else [None]
            )

            for experiment in experiments:
                setup_specs.append(
                    (
                        "cartesian",
                        "combined",
                        values.copy(),
                        experiment,
                    )
                )
    else:
        if parameters.include_default_setup:
            setup_specs.append(
                (
                    "default",
                    "all_defaults",
                    defaults.copy(),
                    default_experiment,
                )
            )

        outer_factors = [
            ("dataset_length", "T", parameters.dataset_lengths),
            ("noise_scale", "sigma", parameters.noise_scales),
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
            (
                "dimension",
                "dimension",
                parameters.dimension_sizes,
            ),
        ]

        for factor_name, field_name, choices in outer_factors:
            if (
                factor_name == "dataset_var_lag_order"
                and parameters.default_data_generating_process
                == "spectral_mixture"
            ):
                nondefault = [
                    choice
                    for choice in _normalize_grid_choices(
                        choices,
                        factor_name,
                    )
                    if int(choice)
                    != int(parameters.default_dataset_var_lag_order)
                ]

                if nondefault:
                    warnings.warn(
                        "Skipping dataset_var_lag_orders sensitivity "
                        "because the default DGP is spectral_mixture.",
                        RuntimeWarning,
                    )

                continue

            for choice in _normalize_grid_choices(choices, factor_name):
                values = defaults.copy()
                values[field_name] = choice

                experiment = (
                    default_spectral_experiment(
                        parameters,
                        dimension=int(values["dimension"]),
                    )
                    if str(values["data_generating_process"])
                    == "spectral_mixture"
                    else None
                )

                setup_specs.append(
                    (factor_name, choice, values, experiment)
                )

        if (
            parameters.default_data_generating_process
            == "spectral_mixture"
        ):
            spectral_experiments = (
                build_spectral_coefficient_experiments(
                    parameters,
                    dimension=parameters.default_dimension,
                )
            )

            for experiment in spectral_experiments:
                setup_specs.append(
                    (
                        experiment.sensitivity_factor,
                        experiment.sensitivity_value,
                        defaults.copy(),
                        experiment,
                    )
                )

        for choice in _normalize_grid_choices(
            parameters.model_lag_orders,
            "model_lag_orders",
        ):
            values = defaults.copy()
            values["model_lag_order"] = choice
            setup_specs.append(
                (
                    "model_lag_order",
                    choice,
                    values,
                    default_experiment,
                )
            )

        for choice in _normalize_grid_choices(
            parameters.window_sizes,
            "window_sizes",
        ):
            values = defaults.copy()
            values["window_size_spec"] = choice
            setup_specs.append(
                (
                    "window_size",
                    choice,
                    values,
                    default_experiment,
                )
            )

    seen = set()
    yielded = 0

    for factor_name, factor_value, values, experiment in setup_specs:
        config = _make_dataset_config(
            values=values,
            sensitivity_factor=factor_name,
            sensitivity_value=factor_value,
            spectral_experiment=experiment,
            parameters=parameters,
        )

        coefficient_key = None
        if config.spectral_regime_coefficients is not None:
            coefficient_key = np.asarray(
                config.spectral_regime_coefficients,
                dtype=float,
            ).tobytes()

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
            coefficient_key,
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
    parameters: SpectralExperimentParameters,
):
    rng = np.random.default_rng(config.seed)

    if config.data_generating_process == "spectral_mixture":
        if parameters.num_regimes != 2:
            raise ValueError(
                "The current AMOC experiment requires num_regimes=2."
            )

        if config.spectral_regime_coefficients is None:
            raise ValueError(
                "A spectral_mixture configuration must provide "
                "spectral_regime_coefficients."
            )

        base_coefficients = validate_spectral_coefficients(
            config.spectral_regime_coefficients,
            name=(
                "spectral coefficients for experiment "
                f"{config.spectral_experiment_name!r}"
            ),
            require_unit_total=True,
        )

        expected_shape = (
            parameters.num_regimes,
            config.dimension,
            6,
        )

        if base_coefficients.shape != expected_shape:
            raise ValueError(
                f"Expected coefficients with shape {expected_shape}; "
                f"got {base_coefficients.shape}."
            )

        coefficients = base_coefficients.copy()
        coefficients[:, :, 5] *= config.sigma

        coefficients = validate_spectral_coefficients(
            coefficients,
            name="effective spectral coefficients",
            require_unit_total=False,
        )

        X_raw, generation_details = controlled_spectral_mixture(
            num_variables=config.dimension,
            num_timepoints=config.T,
            num_regimes=parameters.num_regimes,
            change_points=[config.tau_star],
            regime_coefficients=coefficients,
            f_s=parameters.sampling_rate,
            M=parameters.spectral_ar_damping,
            filter_order=parameters.spectral_filter_order,
            burn_in=parameters.spectral_burn_in,
            seed=config.seed,
            return_details=True,
        )

        generation_details["spectral_experiment_name"] = (
            config.spectral_experiment_name
        )
        generation_details["base_regime_coefficients"] = (
            base_coefficients.copy()
        )
        generation_details["effective_regime_coefficients"] = (
            coefficients.copy()
        )
        generation_details["global_noise_scale"] = float(config.sigma)

    elif config.data_generating_process in {
        "controlled_nonlinear_var",
        "linear_var",
    }:
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
        else:
            X_raw = piecewise_linear_var(
                num_variables=config.dimension,
                num_timepoints=config.T,
                num_regimes=parameters.num_regimes,
                change_points=[config.tau_star],
                coefficient_graphs=coefficient_graphs,
                noise_std_dev=config.sigma,
                rng=rng,
            )

        generation_details = {
            "data_generating_process": (
                config.data_generating_process
            ),
            "coefficient_graphs": coefficient_graphs,
        }
    else:
        raise ValueError(
            "Unknown data_generating_process. Valid options are "
            "'spectral_mixture', 'controlled_nonlinear_var', "
            f"and 'linear_var'. Got {config.data_generating_process!r}."
        )

    if config.empirical_quantile_match:
        if config.data_generating_process == "spectral_mixture":
            raise ValueError(
                "Empirical quantile matching is disabled for "
                "spectral_mixture."
            )

        X_used = empirical_quantile_match(
            time_series=X_raw,
            num_regimes=parameters.num_regimes,
            change_points=[config.tau_star],
        )
    else:
        X_used = X_raw

    return X_used, X_raw, generation_details


def build_analysis_series(X, generation_details=None):
    del generation_details

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must have shape (T, d). Got {X.shape}.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")

    return {"mixture": X}


def find_spectral_experiment(name, experiments, parameters):
    if name is None:
        return None

    for experiment in experiments:
        if experiment.name == name:
            return experiment

    default_experiment = default_spectral_experiment(parameters)

    if default_experiment.name == name:
        return default_experiment

    return None
