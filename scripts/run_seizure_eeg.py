"""Run the neonatal seizure EEG experiment."""

from __future__ import annotations

from itertools import product
from pathlib import Path

from cpd.design import CompetitorConfig
from cpd.experiments.seizure_eeg import (
    SeizureEEGExperimentParameters,
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


# --- 1: Dataset ---

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = (
    REPO_ROOT / "dataset/seizure_eeg/raw"
)

PATIENT_NUMBERS = list(
    range(1, 80)
)

SECTION_NUMBERS = None

MIN_ANNOTATORS = 2

CHANNEL_LABELS = None


# --- 2: Model and Scan ---

MODEL_LAG_ORDER = 2
WINDOW_SIZE = 1000
CANDIDATE_STEP = 50

MODEL_CHOICES = [
    "VARModel",
    "KMOVARScore",
    "MLPModel",
    "AEModel",
]

SCORE_DIRECTIONS = [
    "forward_backward",
]

EPSILON_VALUES = [
    1e-10,
]

KMO_H_CHOICES = [
    "diag_inv_cov",
]

DEVICE = "cpu"

MLP_HIDDEN_SIZE = 32
MLP_LEARNING_RATE = 0.5
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 8

AE_BOTTLENECK_SIZE = 4
AE_LEARNING_RATE = 0.1
AE_EPOCHS = 200
AE_BATCH_SIZE = 16

VALIDATION_FRACTION = 0.3
MIN_TRAIN_EXAMPLES = 5
MIN_VALIDATION_EXAMPLES = 5

USE_EPSILON = True
REVERSE_RIGHT_TO_LEFT = False
SPLIT_TRAIN_VALIDATION = True

KMO_INCLUDE_INTERCEPT = False

BASE_SEED = 42

FULL_EXPERIMENT_MODE = "observed_datasets"


# --- 3: Toggles ---

RUN_FULL_EXPERIMENT = True

VERBOSE_CANDIDATE_PROGRESS = True
PRINT_SCORES = False
PRINT_PER_DATASET_TABLES = False

SAVE_SCORE_PLOTS = True
SAVE_SCORE_VALUES = True
SAVE_SAMPLE_SERIES_PLOTS = True
SAVE_SAMPLE_SERIES_VALUES = True


# --- 4: Output ---

ANALYSIS_DOMAIN_NAME = "time"
EXPERIMENT_DESIGN_MODE = "seizure_sections"

EXPERIMENT_OUTPUTS_BASE_DIR = (
    "seizure_eeg_experiment_outputs"
)

FULL_EXPERIMENT_OUTPUTS_BASE_DIR = (
    "seizure_eeg_full_experiment_outputs"
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

SETUP_SUMMARY_FILENAME = (
    "setup_summary.json"
)

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
    "AEModel": {
        "bottleneck_size": AE_BOTTLENECK_SIZE,
        "lr": AE_LEARNING_RATE,
        "epochs": AE_EPOCHS,
        "batch_size": AE_BATCH_SIZE,
        "loss_type": "mse",
    },
}

EXPERIMENT_PARAMETERS = (
    SeizureEEGExperimentParameters(
        raw_data_dir=RAW_DATA_DIR,
        patient_numbers=PATIENT_NUMBERS,
        section_numbers=SECTION_NUMBERS,
        min_annotators=MIN_ANNOTATORS,
        channel_labels=CHANNEL_LABELS,
        model_lag_order=MODEL_LAG_ORDER,
        window_size=WINDOW_SIZE,
        base_seed=BASE_SEED,
    )
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
    save_predictive_mechanism_heatmaps=False,
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
    analysis_domain_name=(
        ANALYSIS_DOMAIN_NAME
    ),
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
    full_experiment_num_replicates=1,
    full_experiment_base_seed=BASE_SEED,
    full_experiment_mode=FULL_EXPERIMENT_MODE,
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
        "validation_fraction": (
            VALIDATION_FRACTION
        ),
        "min_train_size": MIN_TRAIN_EXAMPLES,
        "min_validation_size": (
            MIN_VALIDATION_EXAMPLES
        ),
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


def configured_competitors(
    dataset_config,
):
    for model_choice in MODEL_CHOICES:
        if model_choice == "KMOVARScore":
            for h_choice in KMO_H_CHOICES:
                yield CompetitorConfig(
                    model_choice=model_choice,
                    score_direction=(
                        f"kmo_{h_choice}"
                    ),
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


def configured_dataset_generator(
    dataset_config,
):
    return generate_dataset(
        dataset_config,
        EXPERIMENT_PARAMETERS,
    )


DEPENDENCIES = RunnerDependencies(
    iter_dataset_configs=(
        configured_dataset_configs
    ),
    iter_competitors=configured_competitors,
    generate_dataset=(
        configured_dataset_generator
    ),
)


def main():
    if RUN_FULL_EXPERIMENT:
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
