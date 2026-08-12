"""Plots used by the causal and spectral experiments."""

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from cpd.data.spectral import validate_spectral_coefficients
from cpd.data.var import validate_change_points
from cpd.saving import safe_filename


@dataclass(frozen=True)
class PlotConfig:
    plot_format: str
    plot_dpi: int
    max_filename_length: int
    save_sample_series_plots: bool
    save_score_plots: bool
    save_coefficient_heatmaps: bool
    save_predictive_mechanism_heatmaps: bool


def plot_regime_time_series(
    time_series,
    change_points,
    num_regimes,
    title="Time Series with Change Point",
    xlabel="Time",
    ylabel="Value",
    figsize=(14, 6),
    linewidth=1.0,
    alpha=0.85,
    colors=None,
    show_change_lines=True,
    show_regime_labels=True,
    grid=True,
    shared_boundary_anchor=False,
):
    """Plot a multivariate time series by regime."""

    X = np.asarray(time_series, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"time_series must have shape (T, d). Got {X.shape}."
        )

    T, d = X.shape
    change_points = [int(cp) for cp in change_points]

    validate_change_points(
        num_timepoints=T,
        num_regimes=num_regimes,
        change_points=change_points,
    )

    if colors is None:
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
        ]

    regime_starts = [0] + change_points
    regime_ends = change_points + [T]

    if shared_boundary_anchor:
        time = np.arange(1, T + 1)
    else:
        time = np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)

    for regime in range(num_regimes):
        start = regime_starts[regime]
        end = regime_ends[regime]

        if shared_boundary_anchor and regime > 0:
            plot_start = start - 1
        else:
            plot_start = start

        color = colors[regime % len(colors)]

        for variable in range(d):
            ax.plot(
                time[plot_start:end],
                X[plot_start:end, variable],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=(
                    f"Regime {regime + 1}"
                    if variable == 0
                    else None
                ),
            )

    if show_change_lines:
        for index, changepoint in enumerate(change_points):
            ax.axvline(
                changepoint,
                color="black",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label="Change point" if index == 0 else None,
            )

    if show_regime_labels:
        for regime in range(num_regimes):
            start = regime_starts[regime]
            end = regime_ends[regime]

            if shared_boundary_anchor:
                plot_start = start if regime == 0 else start - 1
                midpoint = (
                    time[plot_start] + time[end - 1]
                ) / 2
            else:
                midpoint = (start + end) / 2

            ax.text(
                midpoint,
                1.01,
                f"Regime {regime + 1}",
                ha="center",
                va="bottom",
                fontsize=11,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, alpha=0.3)

    ax.legend(loc="best")
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, ax


def save_predictive_mechanism_heatmap(
    coefficient_graphs,
    dataset_config,
    num_regimes,
    plot_config,
    output_dir,
):
    """Save regime-specific VAR coefficient heatmaps."""

    if not plot_config.save_predictive_mechanism_heatmaps:
        return None

    if len(coefficient_graphs) != num_regimes:
        raise ValueError(
            f"Expected {num_regimes} predictive mechanisms. "
            f"Got {len(coefficient_graphs)}."
        )

    lag_order = len(coefficient_graphs[0])

    if lag_order <= 0:
        raise ValueError(
            "Predictive mechanism must contain at least one lag matrix."
        )

    dimension = int(dataset_config.dimension)

    matrices = np.empty(
        (
            num_regimes,
            lag_order,
            dimension,
            dimension,
        ),
        dtype=float,
    )

    for regime, regime_coefficients in enumerate(coefficient_graphs):
        if len(regime_coefficients) != lag_order:
            raise ValueError(
                "All regimes must have the same predictive lag order."
            )

        for lag_index, matrix in enumerate(regime_coefficients):
            matrix = np.asarray(matrix, dtype=float)

            if matrix.shape != (dimension, dimension):
                raise ValueError(
                    f"Matrix for regime {regime + 1}, "
                    f"lag {lag_index + 1} must have shape "
                    f"{(dimension, dimension)}. Got {matrix.shape}."
                )

            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    "Predictive mechanism contains non-finite values."
                )

            matrices[regime, lag_index] = matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_abs = float(np.max(np.abs(matrices)))

    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    fig, axes = plt.subplots(
        nrows=lag_order,
        ncols=num_regimes,
        figsize=(
            6.0 * num_regimes,
            max(4.5, 4.2 * lag_order),
        ),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    source_labels = [
        f"x{variable}"
        for variable in range(dimension)
    ]
    target_labels = [
        f"x{variable}"
        for variable in range(dimension)
    ]

    image = None

    for lag_index in range(lag_order):
        for regime in range(num_regimes):
            ax = axes[lag_index, regime]
            matrix = matrices[regime, lag_index]

            image = ax.imshow(
                matrix,
                aspect="auto",
                vmin=-max_abs,
                vmax=max_abs,
                cmap="coolwarm",
            )

            if num_regimes == 2:
                regime_title = (
                    "Before changepoint (Regime 1)"
                    if regime == 0
                    else "After changepoint (Regime 2)"
                )
            else:
                regime_title = f"Regime {regime + 1}"

            ax.set_title(
                f"{regime_title} — lag {lag_index + 1}"
            )
            ax.set_xticks(np.arange(dimension))
            ax.set_xticklabels(
                source_labels,
                rotation=35,
                ha="right",
            )
            ax.set_yticks(np.arange(dimension))
            ax.set_yticklabels(target_labels)
            ax.set_xlabel("Source variable")

            if regime == 0:
                ax.set_ylabel("Target variable")

            for row in range(dimension):
                for column in range(dimension):
                    value = float(matrix[row, column])
                    red, green, blue, _ = image.cmap(
                        image.norm(value)
                    )
                    luminance = (
                        0.299 * red
                        + 0.587 * green
                        + 0.114 * blue
                    )
                    text_color = (
                        "black"
                        if luminance > 0.55
                        else "white"
                    )

                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.88,
        pad=0.03,
    )
    colorbar.set_label("Predictive coefficient")

    if (
        dataset_config.data_generating_process
        == "controlled_nonlinear_var"
    ):
        mechanism_name = (
            f"tanhVAR({dataset_config.data_lag_order})"
        )
    elif dataset_config.data_generating_process == "linear_var":
        mechanism_name = f"VAR({dataset_config.data_lag_order})"
    else:
        mechanism_name = (
            dataset_config.data_generating_process
        )

    fig.suptitle(
        f"Predictive mechanism coefficients: {mechanism_name}\n"
        f"seed={dataset_config.seed}",
        fontsize=13,
    )

    filename = safe_filename(
        f"predictive_mechanism__{dataset_config.label}__"
        f"seed_{dataset_config.seed}.{plot_config.plot_format}",
        max_len=plot_config.max_filename_length,
    )

    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=plot_config.plot_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return str(output_path.resolve())


def save_spectral_coefficient_heatmap(
    experiment,
    band_names,
    plot_config,
    output_dir,
):
    """Save the before-and-after spectral coefficient heatmap."""

    if not plot_config.save_coefficient_heatmaps:
        return None

    coefficients = validate_spectral_coefficients(
        experiment.coefficients,
        name=f"coefficients for experiment {experiment.name!r}",
        require_unit_total=True,
    )

    if coefficients.ndim != 3:
        raise ValueError(
            "Spectral coefficients must have shape "
            "(num_regimes, dimension, 6)."
        )

    num_regimes, dimension, num_components = coefficients.shape

    if num_regimes != 2 or num_components != 6:
        raise ValueError(
            "Before-and-after coefficient heatmaps require shape "
            f"(2, dimension, 6). Got {coefficients.shape}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    column_labels = list(band_names) + ["noise"]
    row_labels = [
        f"x{variable}"
        for variable in range(dimension)
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, max(4.5, 0.65 * dimension + 2.0)),
        sharey=True,
        constrained_layout=True,
    )

    panel_titles = [
        "Before changepoint (Regime 1)",
        "After changepoint (Regime 2)",
    ]

    image = None

    for regime, ax in enumerate(axes):
        matrix = coefficients[regime]

        image = ax.imshow(
            matrix,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )

        ax.set_title(panel_titles[regime])
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_xticklabels(
            column_labels,
            rotation=35,
            ha="right",
        )
        ax.set_yticks(np.arange(dimension))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Component")

        if regime == 0:
            ax.set_ylabel("Observed variable")

        for row in range(dimension):
            for column in range(num_components):
                value = float(matrix[row, column])
                red, green, blue, _ = image.cmap(
                    image.norm(value)
                )
                luminance = (
                    0.299 * red
                    + 0.587 * green
                    + 0.114 * blue
                )
                text_color = (
                    "black"
                    if luminance > 0.55
                    else "white"
                )

                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.88,
        pad=0.03,
    )
    colorbar.set_label("Mixture weight")

    title = f"Spectral coefficients: {experiment.name}"

    if experiment.description:
        title += f"\n{experiment.description}"

    fig.suptitle(title, fontsize=13)

    filename = safe_filename(
        f"spectral_coefficients__experiment_{experiment.name}"
        f"__before_vs_after.{plot_config.plot_format}",
        max_len=plot_config.max_filename_length
    )
    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=plot_config.plot_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return str(output_path.resolve())


def export_spectral_coefficient_heatmaps(
    experiments,
    band_names,
    plot_config,
    output_dir,
):
    """Save one coefficient heatmap per spectral experiment."""

    return {
        experiment.name: save_spectral_coefficient_heatmap(
            experiment=experiment,
            band_names=band_names,
            plot_config=plot_config,
            output_dir=output_dir,
        )
        for experiment in experiments
    }


def causal_sample_series_title(dataset_config):
    """Return the causal sample-series title."""

    return (
        f"Generated causal time series\n"
        f"{dataset_config.label}"
    )


def spectral_sample_series_title(
    dataset_config,
    series_domain,
    analysis_domain_name,
):
    """Return the spectral sample-series title."""

    if series_domain == analysis_domain_name:
        domain_title = "Observed convex spectral mixture"
    else:
        domain_title = series_domain.replace("_", " ").title()

    return f"{domain_title}\n{dataset_config.label}"


def save_sample_series_plot(
    X,
    dataset_config,
    series_domain,
    num_regimes,
    plot_config,
    output_dir,
    spectral_case=False,
    analysis_domain_name="observed_mixture",
):
    """Save a generated multivariate time-series plot."""

    if not plot_config.save_sample_series_plots:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if spectral_case:
        title = spectral_sample_series_title(
            dataset_config=dataset_config,
            series_domain=series_domain,
            analysis_domain_name=analysis_domain_name,
        )
    else:
        title = causal_sample_series_title(dataset_config)

    fig, _ = plot_regime_time_series(
        time_series=X,
        change_points=[dataset_config.tau_star],
        num_regimes=num_regimes,
        title=title,
        ylabel="Value",
        shared_boundary_anchor=spectral_case,
    )

    filename = safe_filename(
        f"sample_series__domain_{series_domain}__"
        f"{dataset_config.label}.{plot_config.plot_format}",
        max_len=plot_config.max_filename_length
    )
    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=plot_config.plot_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return str(output_path.resolve())


def save_sample_series_plot_without_changepoint(
    X,
    dataset_config,
    series_domain,
    plot_config,
    output_dir,
    analysis_domain_name="observed_mixture",
):
    """Save the spectral series without changepoint markings."""

    if not plot_config.save_sample_series_plots:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if series_domain == analysis_domain_name:
        domain_title = "Observed convex spectral mixture"
    else:
        domain_title = series_domain.replace("_", " ").title()

    experiment_name = (
        dataset_config.spectral_experiment_name
        or "unspecified"
    )

    title = (
        f"{domain_title}\n"
        f"T={dataset_config.T}, "
        f"noise_scale={dataset_config.sigma:g}, "
        f"experiment={experiment_name}"
    )

    fig, _ = plot_regime_time_series(
        time_series=X,
        change_points=[],
        num_regimes=1,
        title=title,
        ylabel="Value",
        show_change_lines=False,
        show_regime_labels=False,
        shared_boundary_anchor=True,
    )

    filename = safe_filename(
        f"sample_series_unmarked__domain_{series_domain}__"
        f"{dataset_config.label}.{plot_config.plot_format}",
        max_len=plot_config.max_filename_length
    )
    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=plot_config.plot_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return str(output_path.resolve())


def save_score_curve_plot(
    scores,
    tau_star,
    tau_hat,
    dataset_config,
    competitor,
    series_domain,
    plot_config,
    output_dir,
):
    """Save one candidate score curve."""

    if not plot_config.save_score_plots:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taus = np.array(sorted(scores), dtype=int)
    values = np.array(
        [scores[tau] for tau in taus],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        taus,
        values,
        linewidth=1.8,
        label="Score",
    )
    ax.axvline(
        tau_star,
        linestyle="--",
        linewidth=1.6,
        label=f"tau* = {tau_star}",
    )
    ax.axvline(
        tau_hat,
        linestyle=":",
        linewidth=1.8,
        label=f"tau_hat = {tau_hat}",
    )

    ax.set_title(
        f"Score curve [{series_domain}]\n"
        f"{dataset_config.label}\n"
        f"{competitor.label}",
        fontsize=11,
    )
    ax.set_xlabel("Candidate changepoint tau")
    ax.set_ylabel("CPD score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    filename = safe_filename(
        f"score_curve__domain_{series_domain}__"
        f"{dataset_config.label}__"
        f"{competitor.label}.{plot_config.plot_format}",
        max_len=plot_config.max_filename_length
    )
    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=plot_config.plot_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return str(output_path.resolve())


def save_pairwise_replicate_boxplots(
    replicate_records,
    metric,
    setup_label,
    output_dir,
    plot_config,
    tau_star=None,
):
    """Save pairwise boxplots over matched successful replicates."""

    valid_metrics = {
        "pct_error",
        "abs_error",
        "tau_hat",
    }

    if metric not in valid_metrics:
        raise ValueError(
            f"Unsupported boxplot metric {metric!r}. "
            f"Valid choices are {sorted(valid_metrics)}."
        )

    if metric == "tau_hat" and tau_star is None:
        raise ValueError(
            "tau_star is required for tau_hat boxplots."
        )

    method_labels = []
    seen = set()

    for record in replicate_records:
        for result in record["raw_results"]:
            if result.competitor_label not in seen:
                seen.add(result.competitor_label)
                method_labels.append(
                    result.competitor_label
                )

    metric_dir = Path(output_dir) / metric
    metric_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for method_a, method_b in itertools.combinations(
        method_labels,
        2,
    ):
        values_a = []
        values_b = []

        for record in replicate_records:
            by_method = {
                result.competitor_label: result
                for result in record["raw_results"]
            }

            result_a = by_method.get(method_a)
            result_b = by_method.get(method_b)

            if result_a is None or result_b is None:
                continue

            if not result_a.status.startswith("OK"):
                continue

            if not result_b.status.startswith("OK"):
                continue

            value_a = getattr(result_a, metric)
            value_b = getattr(result_b, metric)

            if value_a is None or value_b is None:
                continue

            value_a = float(value_a)
            value_b = float(value_b)

            if not (
                np.isfinite(value_a)
                and np.isfinite(value_b)
            ):
                continue

            values_a.append(value_a)
            values_b.append(value_b)

        if not values_a:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.boxplot(
            [values_a, values_b],
            labels=[method_a, method_b],
            showmeans=True,
        )

        if metric == "pct_error":
            metric_title = "Percentage localization error"
            ax.set_ylabel(metric_title)
        elif metric == "abs_error":
            metric_title = "Absolute localization error"
            ax.set_ylabel(metric_title)
        else:
            metric_title = "Estimated changepoint"
            ax.set_ylabel("Estimated changepoint tau_hat")
            ax.axhline(
                float(tau_star),
                linestyle="--",
                linewidth=1.6,
                label=f"True tau = {int(tau_star)}",
            )
            ax.legend(loc="best")

        ax.set_title(
            f"{metric_title} across "
            f"{len(values_a)} paired replicates\n"
            f"{setup_label}",
            fontsize=11,
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()

        filename = safe_filename(
            f"pairwise_boxplot__{metric}__"
            f"{method_a}__vs__{method_b}__"
            f"{setup_label}.{plot_config.plot_format}",
            max_len=plot_config.max_filename_length,
        )
        output_path = metric_dir / filename

        fig.savefig(
            output_path,
            dpi=plot_config.plot_dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        saved_paths.append(str(output_path.resolve()))

    return saved_paths