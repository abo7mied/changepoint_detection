"""Run the causal predictive-mechanism experiment."""

from __future__ import annotations

from itertools import product

from cpd.design import CompetitorConfig
from cpd.experiments.causal import (
    CausalExperimentParameters,
    generate_dataset,
    iter_dataset_configs,
)
from cpd.plotting import PlotConfig
from cpd.runner import (
    RunnerConfig,
    RunnerDependencies,
    run_experiment,
    run_full_replicate_experiment,
)
from cpd.saving import SavingConfig


# --- 1: Sweeps ---

DIMENSION_SIZES = [5, 10, 20, 50, 100, 150]
DATASET_LENGTHS = [500, 1000, 2000, 4000]
NOISE_STD_DEVS = [1, 2, 4]
CHANGEPOINT_PERCENTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

EMPIRICAL_QUANTILE_MATCH_OPTIONS = [False]
DATA_GENERATING_PROCESSES = [
    "linear_var",
    "controlled_nonlinear_var",
    "sign_flip_quadratic",
]

QUADRATIC_COEFFICIENT = 3.0
DATASET_VAR_LAG_ORDERS = [1]

MODEL_LAG_ORDERS = [1, 2, 5, 10]
WINDOW_SIZES = [50, 100, 150, 250, 400]

MODEL_CHOICES = [
    "VARModel",
    "KMOVARScore",
    # "MLPModel",
    # "AEModel",
    # "RNNModel",
]

SCORE_DIRECTIONS = [
    "forward_backward",
    "forward_only",
]

EPSILON_VALUES = [1e-10]

KMO_H_CHOICES = [
    "diag_inv_cov",
    # "identity",
    # "pinv_cov",
]


# --- 2: Defaults ---

EXPERIMENT_DESIGN_MODE = "one_factor_at_a_time"

DEFAULT_DIMENSION = 10
DEFAULT_DATASET_LENGTH = 2000
DEFAULT_NOISE_STD_DEV = 1
DEFAULT_CHANGEPOINT_PERCENTILE = 0.5
DEFAULT_EMPIRICAL_QUANTILE_MATCH = False
DEFAULT_DATA_GENERATING_PROCESS = "linear_var"
DEFAULT_DATASET_VAR_LAG_ORDER = 1

DEFAULT_MODEL_LAG_ORDER = 2
DEFAULT_WINDOW_SIZE = 150

DEVICE = "cpu"

MLP_HIDDEN_SIZE = 32
MLP_LEARNING_RATE = 0.5
MLP_EPOCHS = 500
MLP_BATCH_SIZE = 8

RNN_HIDDEN_SIZE = 16
RNN_LEARNING_RATE = 0.1
RNN_EPOCHS = 200
RNN_BATCH_SIZE = 16

AE_BOTTLENECK_SIZE = 4
AE_LEARNING_RATE = 0.1
AE_EPOCHS = 200
AE_BATCH_SIZE = 16

VALIDATION_FRACTION = 0.3
MIN_TRAIN_EXAMPLES = 5
MIN_VALIDATION_EXAMPLES = 5

NUM_REGIMES = 2
CONTROL = 0.5
ALPHA = 0.2
BETA = 0.5
VAR_TARGET_SPECTRAL_RADIUS = 0.75

BASE_SEED = 42
CANDIDATE_STEP = 10

FULL_EXPERIMENT_NUM_REPLICATES = 100
FULL_EXPERIMENT_BASE_SEED = BASE_SEED


# --- 3: Toggles ---

INCLUDE_DEFAULT_SETUP = True
AUTO_SHRINK_WINDOW_IF_NEEDED = False

USE_EPSILON = True
REVERSE_RIGHT_TO_LEFT = True
SPLIT_TRAIN_VALIDATION = False

KMO_INCLUDE_INTERCEPT = False

RUN_FULL_REPLICATE_EXPERIMENT = False

VERBOSE_CANDIDATE_PROGRESS = False
PRINT_SCORES = False
PRINT_PER_DATASET_TABLES = False

SAVE_SCORE_PLOTS = True
SAVE_SCORE_VALUES = True
SAVE_SAMPLE_SERIES_PLOTS = True
SAVE_SAMPLE_SERIES_VALUES = True
SAVE_PREDICTIVE_MECHANISM_HEATMAPS = True


# --- 4: Output ---

ANALYSIS_DOMAIN_NAME = "time"

EXPERIMENT_OUTPUTS_BASE_DIR = "experiment_outputs"
FULL_EXPERIMENT_OUTPUTS_BASE_DIR = (
    "full_experiment_outputs"
)

FINAL_TABLE_CSV_FILENAME = (
    "final_percentage_error_table.csv"
)
FINAL_TABLE_MD_FILENAME = (
    "final_percentage_error_table.md"
)
FULL_EXPERIMENT_SUMMARY_FILENAME = (
    "full_experiment_summary.json"
)
SETUP_SUMMARY_FILENAME = "setup_summary.json"

PLOT_FORMAT = "png"
PLOT_DPI = 180
MAX_FILENAME_LENGTH = 180


# --- 5: Configuration Assembly ---

MODEL_PARAMETERS = {
    "MLPModel": {
        "hidden_size": MLP_HIDDEN_SIZE,
        "lr": MLP_LEARNING_RATE,
        "epochs": MLP_EPOCHS,
        "batch_size": MLP_BATCH_SIZE,
        "loss_type": "mse",
    },
    "RNNModel": {
        "hidden_size": RNN_HIDDEN_SIZE,
        "lr": RNN_LEARNING_RATE,
        "epochs": RNN_EPOCHS,
        "batch_size": RNN_BATCH_SIZE,
        "loss_type": "mse",
    },
    "AEModel": {
        "bottleneck_size": AE_BOTTLENECK_SIZE,
        "lr": AE_LEARNING_RATE,
        "epochs": AE_EPOCHS,
        "batch_size": AE_BATCH_SIZE,
        "loss_type": "mse",
    },
}

EXPERIMENT_PARAMETERS = CausalExperimentParameters(
    dataset_lengths=DATASET_LENGTHS,
    noise_std_devs=NOISE_STD_DEVS,
    changepoint_percentiles=(
        CHANGEPOINT_PERCENTILES
    ),
    empirical_quantile_match_options=(
        EMPIRICAL_QUANTILE_MATCH_OPTIONS
    ),
    data_generating_processes=(
        DATA_GENERATING_PROCESSES
    ),
    dataset_var_lag_orders=(
        DATASET_VAR_LAG_ORDERS
    ),
    dimension_sizes=DIMENSION_SIZES,
    model_lag_orders=MODEL_LAG_ORDERS,
    window_sizes=WINDOW_SIZES,

    default_dataset_length=(
        DEFAULT_DATASET_LENGTH
    ),
    default_noise_std_dev=(
        DEFAULT_NOISE_STD_DEV
    ),
    default_changepoint_percentile=(
        DEFAULT_CHANGEPOINT_PERCENTILE
    ),
    default_empirical_quantile_match=(
        DEFAULT_EMPIRICAL_QUANTILE_MATCH
    ),
    default_data_generating_process=(
        DEFAULT_DATA_GENERATING_PROCESS
    ),
    default_dataset_var_lag_order=(
        DEFAULT_DATASET_VAR_LAG_ORDER
    ),
    default_dimension=DEFAULT_DIMENSION,
    default_model_lag_order=(
        DEFAULT_MODEL_LAG_ORDER
    ),
    default_window_size=DEFAULT_WINDOW_SIZE,

    experiment_design_mode=(
        EXPERIMENT_DESIGN_MODE
    ),
    include_default_setup=INCLUDE_DEFAULT_SETUP,
    auto_shrink_window_if_needed=(
        AUTO_SHRINK_WINDOW_IF_NEEDED
    ),

    num_regimes=NUM_REGIMES,
    control=CONTROL,
    alpha=ALPHA,
    beta=BETA,
    var_target_spectral_radius=(
        VAR_TARGET_SPECTRAL_RADIUS
    ),
    base_seed=BASE_SEED,
)

PLOT_CONFIG = PlotConfig(
    plot_format=PLOT_FORMAT,
    plot_dpi=PLOT_DPI,
    max_filename_length=MAX_FILENAME_LENGTH,
    save_sample_series_plots=(
        SAVE_SAMPLE_SERIES_PLOTS
    ),
    save_score_plots=SAVE_SCORE_PLOTS,
    save_coefficient_heatmaps=False,
    save_predictive_mechanism_heatmaps=(
        SAVE_PREDICTIVE_MECHANISM_HEATMAPS
    ),
)

SAVING_CONFIG = SavingConfig(
    save_sample_series_values=(
        SAVE_SAMPLE_SERIES_VALUES
    ),
    save_score_values=SAVE_SCORE_VALUES,
    final_table_csv_filename=(
        FINAL_TABLE_CSV_FILENAME
    ),
    final_table_md_filename=(
        FINAL_TABLE_MD_FILENAME
    ),
    max_filename_length=MAX_FILENAME_LENGTH,
)

RUNNER_CONFIG = RunnerConfig(
    experiment_kind="causal",
    experiment_design_mode=(
        EXPERIMENT_DESIGN_MODE
    ),
    analysis_domain_name=ANALYSIS_DOMAIN_NAME,
    candidate_step=CANDIDATE_STEP,

    verbose_candidate_progress=(
        VERBOSE_CANDIDATE_PROGRESS
    ),
    print_scores=PRINT_SCORES,
    print_per_dataset_tables=(
        PRINT_PER_DATASET_TABLES
    ),

    experiment_outputs_base_dir=(
        EXPERIMENT_OUTPUTS_BASE_DIR
    ),
    full_experiment_outputs_base_dir=(
        FULL_EXPERIMENT_OUTPUTS_BASE_DIR
    ),
    full_experiment_summary_filename=(
        FULL_EXPERIMENT_SUMMARY_FILENAME
    ),
    setup_summary_filename=(
        SETUP_SUMMARY_FILENAME
    ),

    full_experiment_num_replicates=(
        FULL_EXPERIMENT_NUM_REPLICATES
    ),
    full_experiment_base_seed=(
        FULL_EXPERIMENT_BASE_SEED
    ),

    plot_config=PLOT_CONFIG,
    saving_config=SAVING_CONFIG,

    cross_regime_kwargs={
        "use_epsilon": USE_EPSILON,
        "reverse_right_to_left": (
            REVERSE_RIGHT_TO_LEFT
        ),
        "split_train_validation": (
            SPLIT_TRAIN_VALIDATION
        ),
        "val_frac": VALIDATION_FRACTION,
        "min_train": MIN_TRAIN_EXAMPLES,
        "min_val": MIN_VALIDATION_EXAMPLES,
        "device": DEVICE,
        "model_parameters": MODEL_PARAMETERS,
    },
    kmo_kwargs={
        "include_intercept": KMO_INCLUDE_INTERCEPT,
        "ridge": 1e-8,
    },
)


# --- 6: Functions ---

def configured_dataset_configs():
    return iter_dataset_configs(
        EXPERIMENT_PARAMETERS
    )


def configured_competitors(dataset_config):
    for model_choice in MODEL_CHOICES:
        if model_choice == "KMOVARScore":
            for h_choice in KMO_H_CHOICES:
                yield CompetitorConfig(
                    model_choice=model_choice,
                    score_direction=f"kmo_{h_choice}",
                    epsilon=EPSILON_VALUES[0],
                    model_lag_order=(
                        dataset_config.model_lag_order
                    ),
                    window_size=(
                        dataset_config.window_size
                    ),
                )

            continue

        for score_direction, epsilon in product(
            SCORE_DIRECTIONS,
            EPSILON_VALUES,
        ):
            yield CompetitorConfig(
                model_choice=model_choice,
                score_direction=score_direction,
                epsilon=epsilon,
                model_lag_order=(
                    dataset_config.model_lag_order
                ),
                window_size=(
                    dataset_config.window_size
                ),
            )


def configured_dataset_generator(dataset_config):
    return generate_dataset(
        dataset_config,
        EXPERIMENT_PARAMETERS,
    )


DEPENDENCIES = RunnerDependencies(
    iter_dataset_configs=configured_dataset_configs,
    iter_competitors=configured_competitors,
    generate_dataset=configured_dataset_generator,
)


def main():
    if RUN_FULL_REPLICATE_EXPERIMENT:
        return run_full_replicate_experiment(
            RUNNER_CONFIG,
            DEPENDENCIES,
        )

    return run_experiment(
        RUNNER_CONFIG,
        DEPENDENCIES,
    )


if __name__ == "__main__":
    main()
