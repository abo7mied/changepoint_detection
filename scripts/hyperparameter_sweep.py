"""Run MLP, autoencoder, or RNN hyperparameter sweeps."""

from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_causal as causal_setup
import run_spectral as spectral_setup

from cpd.design import (
    CompetitorConfig,
    make_candidates,
)
from cpd.experiments.causal import (
    DatasetConfig as CausalDatasetConfig,
    generate_dataset as generate_causal_dataset,
)
from cpd.experiments.spectral import (
    DatasetConfig as SpectralDatasetConfig,
    default_spectral_experiment,
    generate_dataset as generate_spectral_dataset,
)
from cpd.methods.cross_regime import (
    estimate_causal_cpd_log_normalized_cross_regime,
)
from cpd.runner import (
    make_full_experiment_replicate_seeds,
)
from cpd.saving import create_unique_directory


# --- 1: Sweeps ---

MLP_HIDDEN_SIZES = [8, 16, 32, 64]
MLP_LEARNING_RATES = [0.01, 0.05, 0.1, 0.5]
MLP_BATCH_SIZES = [8]
MLP_EPOCH_COUNTS = [50]

AE_BOTTLENECK_SIZES = [2, 4, 8, 16]
AE_LEARNING_RATES = [0.01, 0.05, 0.1, 0.5]
AE_BATCH_SIZES = [8]
AE_EPOCH_COUNTS = [50]

RNN_HIDDEN_SIZES = [8, 16, 32, 64]
RNN_LEARNING_RATES = [0.01, 0.05, 0.1, 0.5]
RNN_BATCH_SIZES = [16]
RNN_EPOCH_COUNTS = [50]


# --- 2: Defaults ---

MODEL_CHOICE = "MLPModel"
EXPERIMENT_KIND = "causal"
SWEEP_MODE = "plot"

NUM_REPLICATES = 20
BASE_SEED = 42

T = 10000
DIMENSION = 10
CHANGEPOINT_PERCENTILE = 0.25
SIGMA = 1.0

CAUSAL_DATA_GENERATING_PROCESS = (
    "sign_flip_quadratic",
    # "controlled_nonlinear_var"
)
DATA_LAG_ORDER = 1

MODEL_LAG_ORDER = 2
WINDOW_SIZE = 1000
CANDIDATE_STEP = 100

SCORE_DIRECTION = "forward_backward"
EPSILON = 1e-10

DEVICE = "cpu"

VALIDATION_FRACTION = 0.3
MIN_TRAIN_EXAMPLES = 5
MIN_VALIDATION_EXAMPLES = 5

OUTPUTS_BASE_DIR = "model_hyperparameter_sweep"
RAW_RESULTS_FILENAME = "raw_results.csv"
RANKING_FILENAME = "ranking.csv"
SCORE_PLOT_FILENAME = "score_curves.png"
PLOT_DPI = 180


# --- 3: Toggles ---

USE_EPSILON = True
REVERSE_RIGHT_TO_LEFT = True
SPLIT_TRAIN_VALIDATION = True

PAIR_MODEL_INITIALIZATION = True

SAVE_RAW_RESULTS = True
SAVE_RANKING = True
SAVE_SCORE_PLOT = True

PRINT_EACH_RUN = True
PRINT_RANKING = True
PRINT_BEST_CONFIGURATION = True


# --- 4: Functions ---

def get_sweep_grid():
    """Return the active model's sweep grid."""
    if MODEL_CHOICE == "MLPModel":
        grid = {
            "size_name": "hidden_size",
            "sizes": MLP_HIDDEN_SIZES,
            "learning_rates": MLP_LEARNING_RATES,
            "batch_sizes": MLP_BATCH_SIZES,
            "epoch_counts": MLP_EPOCH_COUNTS,
        }

    elif MODEL_CHOICE == "AEModel":
        grid = {
            "size_name": "bottleneck_size",
            "sizes": AE_BOTTLENECK_SIZES,
            "learning_rates": AE_LEARNING_RATES,
            "batch_sizes": AE_BATCH_SIZES,
            "epoch_counts": AE_EPOCH_COUNTS,
        }

    elif MODEL_CHOICE == "RNNModel":
        grid = {
            "size_name": "hidden_size",
            "sizes": RNN_HIDDEN_SIZES,
            "learning_rates": RNN_LEARNING_RATES,
            "batch_sizes": RNN_BATCH_SIZES,
            "epoch_counts": RNN_EPOCH_COUNTS,
        }

    else:
        raise ValueError(
            "MODEL_CHOICE must be 'MLPModel', "
            "'AEModel', or 'RNNModel'. "
            f"Got {MODEL_CHOICE!r}."
        )

    if SWEEP_MODE == "plot":
        if not grid["batch_sizes"]:
            raise ValueError(
                "batch_sizes cannot be empty."
            )

        grid["batch_sizes"] = [
            grid["batch_sizes"][0]
        ]

    return grid


def validate_configuration():
    """Validate the sweep configuration."""
    if EXPERIMENT_KIND not in {
        "causal",
        "spectral",
    }:
        raise ValueError(
            "EXPERIMENT_KIND must be "
            "'causal' or 'spectral'. "
            f"Got {EXPERIMENT_KIND!r}."
        )

    if SWEEP_MODE not in {
        "replicate",
        "plot",
    }:
        raise ValueError(
            "SWEEP_MODE must be "
            "'replicate' or 'plot'. "
            f"Got {SWEEP_MODE!r}."
        )

    if SCORE_DIRECTION not in {
        "forward_only",
        "forward_backward",
    }:
        raise ValueError(
            "SCORE_DIRECTION must be "
            "'forward_only' or "
            "'forward_backward'."
        )

    if NUM_REPLICATES <= 0:
        raise ValueError(
            "NUM_REPLICATES must be positive."
        )

    grid = get_sweep_grid()

    for name in [
        "sizes",
        "learning_rates",
        "batch_sizes",
        "epoch_counts",
    ]:
        if not grid[name]:
            raise ValueError(
                f"{name} cannot be empty."
            )


def make_dataset_config(seed):
    """Construct one dataset configuration."""
    if EXPERIMENT_KIND == "causal":
        return CausalDatasetConfig(
            T=T,
            sigma=SIGMA,
            changepoint_percentile=(
                CHANGEPOINT_PERCENTILE
            ),
            empirical_quantile_match=False,
            data_generating_process=(
                CAUSAL_DATA_GENERATING_PROCESS
            ),
            data_lag_order=DATA_LAG_ORDER,
            dimension=DIMENSION,
            model_lag_order=MODEL_LAG_ORDER,
            window_size=WINDOW_SIZE,
            sensitivity_factor="model_sweep",
            sensitivity_value=MODEL_CHOICE,
            seed=int(seed),
        )

    experiment = default_spectral_experiment(
        spectral_setup.EXPERIMENT_PARAMETERS,
        dimension=DIMENSION,
    )

    return SpectralDatasetConfig(
        T=T,
        sigma=SIGMA,
        changepoint_percentile=(
            CHANGEPOINT_PERCENTILE
        ),
        empirical_quantile_match=False,
        data_generating_process="spectral_mixture",
        data_lag_order=DATA_LAG_ORDER,
        dimension=DIMENSION,
        model_lag_order=MODEL_LAG_ORDER,
        window_size=WINDOW_SIZE,
        sensitivity_factor="model_sweep",
        sensitivity_value=MODEL_CHOICE,
        seed=int(seed),
        spectral_experiment_name=(
            experiment.name
        ),
        spectral_regime_coefficients=(
            experiment.coefficients.copy()
        ),
    )


def generate_dataset(dataset_config):
    """Generate one causal or spectral dataset."""
    if EXPERIMENT_KIND == "causal":
        X, _, _ = generate_causal_dataset(
            dataset_config,
            causal_setup.EXPERIMENT_PARAMETERS,
        )

    else:
        X, _, _ = generate_spectral_dataset(
            dataset_config,
            spectral_setup.EXPERIMENT_PARAMETERS,
        )

    return X


def get_base_model_parameters():
    """Return the selected experiment's model parameters."""
    if EXPERIMENT_KIND == "causal":
        return causal_setup.MODEL_PARAMETERS

    return spectral_setup.MODEL_PARAMETERS


def make_model_parameters(
    size,
    learning_rate,
    batch_size,
    epochs,
):
    """Construct parameters for one sweep point."""
    base_parameters = {
        model_choice: parameters.copy()
        for model_choice, parameters
        in get_base_model_parameters().items()
    }

    selected_parameters = (
        base_parameters
        .get(
            MODEL_CHOICE,
            {},
        )
        .copy()
    )

    if MODEL_CHOICE == "MLPModel":
        selected_parameters.update(
            {
                "hidden_size": int(size),
                "lr": float(learning_rate),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
            }
        )

    elif MODEL_CHOICE == "AEModel":
        selected_parameters.update(
            {
                "bottleneck_size": int(size),
                "lr": float(learning_rate),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
            }
        )

    elif MODEL_CHOICE == "RNNModel":
        selected_parameters.update(
            {
                "hidden_size": int(size),
                "lr": float(learning_rate),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
            }
        )

    else:
        raise ValueError(
            f"Unsupported model {MODEL_CHOICE!r}."
        )

    base_parameters[MODEL_CHOICE] = (
        selected_parameters
    )

    return base_parameters


def make_competitor():
    """Construct the active model competitor."""
    return CompetitorConfig(
        model_choice=MODEL_CHOICE,
        score_direction=SCORE_DIRECTION,
        epsilon=EPSILON,
        model_lag_order=MODEL_LAG_ORDER,
        window_size=WINDOW_SIZE,
    )


def set_model_seed(seed):
    """Set the model initialization seed."""
    if not PAIR_MODEL_INITIALIZATION:
        return

    np.random.seed(
        int(seed) % (2**32 - 1)
    )


def run_one_configuration(
    X,
    dataset_config,
    candidates,
    size,
    learning_rate,
    batch_size,
    epochs,
    model_seed,
):
    """Run one hyperparameter configuration."""
    set_model_seed(model_seed)

    model_parameters = make_model_parameters(
        size=size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )

    tau_hat, scores = (
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=WINDOW_SIZE,
            L=MODEL_LAG_ORDER,
            candidates=candidates,
            competitor=make_competitor(),
            device=DEVICE,
            model_parameters=model_parameters,
            use_epsilon=USE_EPSILON,
            reverse_right_to_left=(
                REVERSE_RIGHT_TO_LEFT
            ),
            split_train_validation=(
                SPLIT_TRAIN_VALIDATION
            ),
            validation_fraction=(
                VALIDATION_FRACTION
            ),
            min_train_size=(
                MIN_TRAIN_EXAMPLES
            ),
            min_validation_size=(
                MIN_VALIDATION_EXAMPLES
            ),
            verbose=False,
        )
    )

    absolute_error = abs(
        tau_hat - dataset_config.tau_star
    )

    percentage_error = (
        100.0
        * absolute_error
        / dataset_config.T
    )

    return {
        "tau_hat": int(tau_hat),
        "absolute_error": int(
            absolute_error
        ),
        "percentage_error": float(
            percentage_error
        ),
        "scores": scores,
    }


def build_ranking(raw_results):
    """Aggregate and rank all configurations."""
    group_columns = [
        "model_choice",
        "size_name",
        "size",
        "learning_rate",
        "batch_size",
        "epochs",
    ]

    ranking = (
        raw_results
        .groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            num_runs=(
                "replicate",
                "count",
            ),
            num_successful=(
                "absolute_error",
                "count",
            ),
            mean_localization_error=(
                "absolute_error",
                "mean",
            ),
            median_localization_error=(
                "absolute_error",
                "median",
            ),
            std_localization_error=(
                "absolute_error",
                "std",
            ),
            mean_percentage_error=(
                "percentage_error",
                "mean",
            ),
            median_percentage_error=(
                "percentage_error",
                "median",
            ),
        )
    )

    ranking["num_failed"] = (
        ranking["num_runs"]
        - ranking["num_successful"]
    )

    ranking = ranking.sort_values(
        by=[
            "mean_localization_error",
            "median_localization_error",
            "size",
            "learning_rate",
            "batch_size",
            "epochs",
        ],
        na_position="last",
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1),
    )

    return ranking


def run_sweep():
    """Run the configured hyperparameter sweep."""
    validate_configuration()

    grid = get_sweep_grid()

    num_replicates = (
        NUM_REPLICATES
        if SWEEP_MODE == "replicate"
        else 1
    )

    replicate_seeds = (
        make_full_experiment_replicate_seeds(
            num_replicates=num_replicates,
            base_seed=BASE_SEED,
        )
    )

    combinations = list(
        product(
            grid["sizes"],
            grid["learning_rates"],
            grid["batch_sizes"],
            grid["epoch_counts"],
        )
    )

    candidates = make_candidates(
        T=T,
        h=WINDOW_SIZE,
        step=CANDIDATE_STEP,
    )

    records = []
    score_results = {}

    for replicate_index, seed in enumerate(
        replicate_seeds,
        start=1,
    ):
        dataset_config = make_dataset_config(
            seed
        )

        X = generate_dataset(
            dataset_config
        )

        for combination_index, combination in enumerate(
            combinations,
            start=1,
        ):
            (
                size,
                learning_rate,
                batch_size,
                epochs,
            ) = combination

            if PRINT_EACH_RUN:
                print(
                    f"Replicate "
                    f"{replicate_index}/"
                    f"{num_replicates}, "
                    f"configuration "
                    f"{combination_index}/"
                    f"{len(combinations)}: "
                    f"{grid['size_name']}={size}, "
                    f"lr={learning_rate:g}, "
                    f"batch_size={batch_size}, "
                    f"epochs={epochs}"
                )

            try:
                result = run_one_configuration(
                    X=X,
                    dataset_config=dataset_config,
                    candidates=candidates,
                    size=size,
                    learning_rate=(
                        learning_rate
                    ),
                    batch_size=batch_size,
                    epochs=epochs,
                    model_seed=seed,
                )

                status = "OK"
                tau_hat = result["tau_hat"]

                absolute_error = (
                    result["absolute_error"]
                )

                percentage_error = (
                    result["percentage_error"]
                )

                if SWEEP_MODE == "plot":
                    score_results[
                        (
                            size,
                            learning_rate,
                            batch_size,
                            epochs,
                        )
                    ] = result

            except Exception as exc:
                status = (
                    f"ERROR: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(status)

                tau_hat = np.nan
                absolute_error = np.nan
                percentage_error = np.nan

            records.append(
                {
                    "replicate": replicate_index,
                    "seed": int(seed),
                    "experiment_kind": (
                        EXPERIMENT_KIND
                    ),
                    "model_choice": MODEL_CHOICE,
                    "size_name": (
                        grid["size_name"]
                    ),
                    "size": int(size),
                    "learning_rate": float(
                        learning_rate
                    ),
                    "batch_size": int(
                        batch_size
                    ),
                    "epochs": int(epochs),
                    "tau_star": int(
                        dataset_config.tau_star
                    ),
                    "tau_hat": tau_hat,
                    "absolute_error": (
                        absolute_error
                    ),
                    "percentage_error": (
                        percentage_error
                    ),
                    "status": status,
                }
            )

    raw_results = pd.DataFrame(records)
    ranking = build_ranking(raw_results)

    return (
        raw_results,
        ranking,
        score_results,
    )


def plot_score_curves(
    score_results,
    output_path,
):
    """Plot one score curve per configuration."""
    grid = get_sweep_grid()

    combinations = list(
        product(
            grid["sizes"],
            grid["learning_rates"],
            grid["batch_sizes"],
            grid["epoch_counts"],
        )
    )

    num_plots = len(combinations)
    num_columns = min(4, num_plots)
    num_rows = int(
        np.ceil(
            num_plots / num_columns
        )
    )

    tau_star = int(
        round(
            CHANGEPOINT_PERCENTILE * T
        )
    )

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        figsize=(
            4.5 * num_columns,
            3.5 * num_rows,
        ),
        sharex=True,
        squeeze=False,
    )

    for ax, combination in zip(
        axes.ravel(),
        combinations,
    ):
        (
            size,
            learning_rate,
            batch_size,
            epochs,
        ) = combination

        result = score_results.get(
            (
                size,
                learning_rate,
                batch_size,
                epochs,
            )
        )

        configuration_title = (
            f"{grid['size_name']}={size}, "
            f"lr={learning_rate:g}\n"
            f"batch={batch_size}, "
            f"epochs={epochs}"
        )

        if result is None:
            ax.set_title(
                f"{configuration_title}\n"
                "ERROR"
            )
            ax.grid(alpha=0.25)
            continue

        scores = result["scores"]

        taus = np.array(
            sorted(scores),
            dtype=int,
        )

        values = np.array(
            [
                scores[tau]
                for tau in taus
            ],
            dtype=float,
        )

        ax.plot(
            taus,
            values,
            linewidth=1.5,
        )

        ax.axvline(
            tau_star,
            linestyle="--",
            linewidth=1.5,
            label="True CP",
        )

        ax.axvline(
            result["tau_hat"],
            linestyle=":",
            linewidth=1.5,
            label="Estimated CP",
        )

        ax.set_title(
            f"{configuration_title}\n"
            f"tau_hat={result['tau_hat']}, "
            f"error={result['absolute_error']}"
        )

        ax.set_xlabel(
            "Candidate changepoint"
        )
        ax.set_ylabel("CPD score")
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[num_plots:]:
        ax.set_visible(False)

    handles = []
    labels = []

    for ax in axes.ravel()[:num_plots]:
        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if handles:
            break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
        )

    fig.suptitle(
        f"{MODEL_CHOICE} Hyperparameter Sweep\n"
        f"dataset={EXPERIMENT_KIND}, "
        f"window_size={WINDOW_SIZE}",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    fig.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_ranked_results(ranking):
    """Print the ranked configurations."""
    columns = [
        "rank",
        "size_name",
        "size",
        "learning_rate",
        "batch_size",
        "epochs",
        "num_successful",
        "num_failed",
        "mean_localization_error",
        "median_localization_error",
        "std_localization_error",
        "mean_percentage_error",
    ]

    print("\nRanked configurations:\n")

    print(
        ranking[columns].to_string(
            index=False
        )
    )


def print_best_configuration(ranking):
    """Print the best successful configuration."""
    successful = ranking[
        ranking["num_successful"] > 0
    ]

    if successful.empty:
        print(
            "\nNo configuration completed "
            "successfully."
        )
        return

    best = successful.iloc[0]
    size_name = best["size_name"]

    print("\nBest configuration:")

    print(
        f"{size_name:<15} = "
        f"{int(best['size'])}"
    )

    print(
        "learning_rate   = "
        f"{best['learning_rate']:g}"
    )

    print(
        "batch_size      = "
        f"{int(best['batch_size'])}"
    )

    print(
        "epochs          = "
        f"{int(best['epochs'])}"
    )

    print(
        "mean localization error   = "
        f"{best['mean_localization_error']:.2f}"
    )

    print(
        "median localization error = "
        f"{best['median_localization_error']:.2f}"
    )

    if np.isfinite(
        best["std_localization_error"]
    ):
        print(
            "std localization error    = "
            f"{best['std_localization_error']:.2f}"
        )


def main():
    output_name = (
        f"{EXPERIMENT_KIND}_"
        f"{MODEL_CHOICE.lower()}_"
        f"{SWEEP_MODE}_sweep"
    )

    output_dir = create_unique_directory(
        f"{OUTPUTS_BASE_DIR}/{output_name}"
    )

    (
        raw_results,
        ranking,
        score_results,
    ) = run_sweep()

    if SAVE_RAW_RESULTS:
        raw_results.to_csv(
            output_dir / RAW_RESULTS_FILENAME,
            index=False,
        )

    if SAVE_RANKING:
        ranking.to_csv(
            output_dir / RANKING_FILENAME,
            index=False,
        )

    if (
        SWEEP_MODE == "plot"
        and SAVE_SCORE_PLOT
    ):
        plot_score_curves(
            score_results=score_results,
            output_path=(
                output_dir
                / SCORE_PLOT_FILENAME
            ),
        )

    if PRINT_RANKING:
        print_ranked_results(
            ranking
        )

    if PRINT_BEST_CONFIGURATION:
        print_best_configuration(
            ranking
        )

    print(
        f"\nOutput directory: "
        f"{output_dir}"
    )

    return raw_results, ranking


if __name__ == "__main__":
    main()