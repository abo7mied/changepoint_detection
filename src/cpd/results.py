"""Experiment results, summaries, and table construction."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ExperimentResult:
    """Result from one dataset, domain, and competitor."""

    dataset_label: str
    competitor_label: str
    model_choice: str
    score_direction: str
    epsilon: float
    model_lag_order: int
    window_size: int
    tau_star: int
    tau_hat: Optional[int]
    nearest_candidate: Optional[int]
    abs_error: Optional[int]
    pct_error: float
    best_possible_abs_error: Optional[int]
    best_possible_pct_error: Optional[float]
    score_at_tau_hat: Optional[float]
    status: str
    series_domain: Optional[str] = None
    score_plot_path: Optional[str] = None
    score_values_path: Optional[str] = None


def fmt_value(value, digits=4):
    """Format one table value."""

    if value is None:
        return "-"

    if isinstance(value, float):
        if math.isinf(value):
            return "inf"

        if math.isnan(value):
            return "nan"

        return f"{value:.{digits}f}"

    return str(value)


def print_table(results):
    """Print one ranked result table."""

    include_domain = any(
        result.series_domain is not None
        for result in results
    )

    rows = []

    for rank, result in enumerate(results, start=1):
        row = {
            "rank": rank,
            "model": result.model_choice,
            "score": result.score_direction,
            "eps": f"{result.epsilon:g}",
            "L": result.model_lag_order,
            "h": result.window_size,
            "tau*": result.tau_star,
            "tau_hat": fmt_value(result.tau_hat, digits=0),
            "abs_err": fmt_value(result.abs_error, digits=0),
            "pct_err": fmt_value(result.pct_error, digits=3),
            "nearest_cand": fmt_value(
                result.nearest_candidate,
                digits=0,
            ),
            "best_pct": fmt_value(
                result.best_possible_pct_error,
                digits=3,
            ),
            "score@hat": fmt_value(
                result.score_at_tau_hat,
                digits=5,
            ),
            "plot": (
                result.score_plot_path
                if result.score_plot_path is not None
                else "-"
            ),
            "scores_csv": (
                result.score_values_path
                if result.score_values_path is not None
                else "-"
            ),
            "status": result.status,
        }

        if include_domain:
            row["domain"] = result.series_domain

        rows.append(row)

    columns = ["rank"]

    if include_domain:
        columns.append("domain")

    columns.extend(
        [
            "model",
            "score",
            "eps",
            "L",
            "h",
            "tau*",
            "tau_hat",
            "abs_err",
            "pct_err",
            "nearest_cand",
            "best_pct",
            "score@hat",
            "plot",
            "scores_csv",
            "status",
        ]
    )

    print_plain_table(columns, rows)


def final_dataset_column_label(dataset_label):
    """Return the label used for one dataset column."""

    return dataset_label


def build_final_percentage_error_table(
    all_results,
    analysis_domain_name=None,
):
    """Build the final percentage-error comparison table."""

    dataset_labels = list(all_results)
    dataset_columns = [
        final_dataset_column_label(label)
        for label in dataset_labels
    ]

    include_domain = any(
        result.series_domain is not None
        for results in all_results.values()
        for result in results
    )

    if include_domain:
        by_method = {}

        for dataset_label, results in all_results.items():
            for result in results:
                key = (
                    result.series_domain,
                    result.competitor_label,
                )
                by_method.setdefault(key, {})
                by_method[key][dataset_label] = result
    else:
        by_method = {}

        for dataset_label, results in all_results.items():
            for result in results:
                key = result.competitor_label
                by_method.setdefault(key, {})
                by_method[key][dataset_label] = result

    rows_for_sorting = []

    for method_key, result_by_dataset in by_method.items():
        if include_domain:
            series_domain, competitor_label = method_key
        else:
            series_domain = None
            competitor_label = method_key

        percentage_errors = []

        for dataset_label in dataset_labels:
            result = result_by_dataset.get(dataset_label)

            if (
                result is None
                or not result.status.startswith("OK")
            ):
                percentage_errors.append(float("inf"))
            else:
                percentage_errors.append(result.pct_error)

        finite_errors = [
            error
            for error in percentage_errors
            if np.isfinite(error)
        ]

        mean_percentage_error = (
            float(np.mean(finite_errors))
            if finite_errors
            else float("inf")
        )

        rows_for_sorting.append(
            {
                "domain": series_domain,
                "competitor": competitor_label,
                "mean_pct_err_float": mean_percentage_error,
                "result_by_dataset": result_by_dataset,
            }
        )

    if include_domain:
        preferred_domain_order = (
            []
            if analysis_domain_name is None
            else [analysis_domain_name]
        )
        domain_rank = {
            name: index
            for index, name in enumerate(
                preferred_domain_order
            )
        }

        rows_for_sorting.sort(
            key=lambda row: (
                domain_rank.get(
                    row["domain"],
                    len(domain_rank),
                ),
                not np.isfinite(
                    row["mean_pct_err_float"]
                ),
                row["mean_pct_err_float"],
                row["competitor"],
            )
        )
    else:
        rows_for_sorting.sort(
            key=lambda row: (
                not np.isfinite(
                    row["mean_pct_err_float"]
                ),
                row["mean_pct_err_float"],
                row["competitor"],
            )
        )

    columns = ["rank"]

    if include_domain:
        columns.append("domain")

    columns.extend(
        [
            "competitor",
            "mean_pct_err",
            *dataset_columns,
        ]
    )

    table_rows = []

    for rank, row_info in enumerate(
        rows_for_sorting,
        start=1,
    ):
        row = {
            "rank": str(rank),
            "competitor": row_info["competitor"],
            "mean_pct_err": fmt_value(
                row_info["mean_pct_err_float"],
                digits=3,
            ),
        }

        if include_domain:
            row["domain"] = row_info["domain"]

        for dataset_label, dataset_column in zip(
            dataset_labels,
            dataset_columns,
        ):
            result = row_info["result_by_dataset"].get(
                dataset_label
            )

            if result is None:
                row[dataset_column] = "-"
            elif not result.status.startswith("OK"):
                row[dataset_column] = "ERR"
            else:
                row[dataset_column] = fmt_value(
                    result.pct_error,
                    digits=3,
                )

        table_rows.append(row)

    return columns, table_rows


def table_rows_to_markdown(columns, rows):
    """Convert table rows to a Markdown table."""

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(
            ["---"] * len(columns)
        ) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[column])
                for column in columns
            )
            + " |"
        )

    return "\n".join(lines)


def print_plain_table(columns, rows):
    """Print a plain-text table."""

    if not rows:
        print("(no rows)")
        return

    widths = {
        column: max(
            len(column),
            *[
                len(str(row[column]))
                for row in rows
            ],
        )
        for column in columns
    }

    header = " | ".join(
        column.ljust(widths[column])
        for column in columns
    )
    separator = "-+-".join(
        "-" * widths[column]
        for column in columns
    )

    print(header)
    print(separator)

    for row in rows:
        print(
            " | ".join(
                str(row[column]).ljust(widths[column])
                for column in columns
            )
        )


def finite_or_none(value):
    """Convert non-finite numbers to None."""

    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if (
        isinstance(value, float)
        and not np.isfinite(value)
    ):
        return None

    return value


def relative_output_path(path, root_dir):
    """Represent an output path relative to its experiment root."""

    if path is None:
        return None

    resolved_path = Path(path).resolve()
    resolved_root = Path(root_dir).resolve()

    try:
        return str(
            resolved_path.relative_to(resolved_root)
        )
    except ValueError:
        return str(resolved_path)


def serialize_experiment_result(result, root_dir):
    """Return a JSON-safe experiment result."""

    serialized = {
        "dataset_label": result.dataset_label,
        "competitor_label": result.competitor_label,
        "model_choice": result.model_choice,
        "score_direction": result.score_direction,
        "epsilon": finite_or_none(result.epsilon),
        "model_lag_order": int(
            result.model_lag_order
        ),
        "window_size": int(result.window_size),
        "tau_star": int(result.tau_star),
        "tau_hat": (
            None
            if result.tau_hat is None
            else int(result.tau_hat)
        ),
        "nearest_candidate": (
            None
            if result.nearest_candidate is None
            else int(result.nearest_candidate)
        ),
        "abs_error": (
            None
            if result.abs_error is None
            else int(result.abs_error)
        ),
        "pct_error": finite_or_none(
            result.pct_error
        ),
        "best_possible_abs_error": (
            None
            if result.best_possible_abs_error is None
            else int(result.best_possible_abs_error)
        ),
        "best_possible_pct_error": finite_or_none(
            result.best_possible_pct_error
        ),
        "score_at_tau_hat": finite_or_none(
            result.score_at_tau_hat
        ),
        "status": result.status,
        "score_plot": relative_output_path(
            result.score_plot_path,
            root_dir,
        ),
        "score_values": relative_output_path(
            result.score_values_path,
            root_dir,
        ),
    }

    if result.series_domain is not None:
        serialized["series_domain"] = (
            result.series_domain
        )

    return serialized


def summarize_replicate_metric(values):
    """Summarize one replicate metric."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_variance": None,
            "sample_std": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sample_variance": (
            float(np.var(values, ddof=1))
            if values.size >= 2
            else None
        ),
        "sample_std": (
            float(np.std(values, ddof=1))
            if values.size >= 2
            else None
        ),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def group_replicate_results_by_competitor(
    replicate_records,
):
    """Group replicate results by competitor."""

    grouped = {}

    for replicate_record in replicate_records:
        for result in replicate_record["raw_results"]:
            grouped.setdefault(
                result.competitor_label,
                [],
            ).append(result)

    return grouped


def build_setup_method_summary(replicate_records):
    """Summarize and rank competitors for one setup."""

    grouped = group_replicate_results_by_competitor(
        replicate_records
    )
    rows = []

    for competitor_label, results in grouped.items():
        successful = [
            result
            for result in results
            if result.status.startswith("OK")
        ]

        percentage_errors = [
            result.pct_error
            for result in successful
        ]
        absolute_errors = [
            float(result.abs_error)
            for result in successful
            if result.abs_error is not None
        ]

        representative = results[0]

        rows.append(
            {
                "competitor_label": competitor_label,
                "model_choice": (
                    representative.model_choice
                ),
                "score_direction": (
                    representative.score_direction
                ),
                "epsilon": finite_or_none(
                    representative.epsilon
                ),
                "model_lag_order": int(
                    representative.model_lag_order
                ),
                "window_size": int(
                    representative.window_size
                ),
                "num_replicates": len(results),
                "num_successful": len(successful),
                "num_failed": (
                    len(results) - len(successful)
                ),
                "percentage_error": (
                    summarize_replicate_metric(
                        percentage_errors
                    )
                ),
                "absolute_error": (
                    summarize_replicate_metric(
                        absolute_errors
                    )
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["percentage_error"]["mean"] is None,
            (
                float("inf")
                if row["percentage_error"]["mean"] is None
                else row["percentage_error"]["mean"]
            ),
            row["competitor_label"],
        )
    )

    for rank, row in enumerate(rows, start=1):
        row["rank_by_mean_percentage_error"] = rank

    return rows


def dataset_config_summary(config):
    """Return a JSON-safe dataset configuration summary."""

    summary = {
        "label": config.label,
        "sensitivity_factor": (
            config.sensitivity_factor
        ),
        "sensitivity_value": (
            config.sensitivity_value
        ),
        "T": int(config.T),
        "sigma": float(config.sigma),
        "changepoint_percentile": float(
            config.changepoint_percentile
        ),
        "tau_star": int(config.tau_star),
        "empirical_quantile_match": bool(
            config.empirical_quantile_match
        ),
        "data_generating_process": (
            config.data_generating_process
        ),
        "data_lag_order": int(
            config.data_lag_order
        ),
        "model_lag_order": int(
            config.model_lag_order
        ),
        "window_size": int(config.window_size),
    }

    if hasattr(config, "dimension"):
        summary["dimension"] = int(config.dimension)

    if hasattr(config, "spectral_experiment_name"):
        summary["spectral_experiment_name"] = (
            config.spectral_experiment_name
        )

    return summary