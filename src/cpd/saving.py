"""Output paths and file writing."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cpd.results import (
    build_final_percentage_error_table,
    relative_output_path,
    table_rows_to_markdown,
)


@dataclass(frozen=True)
class SavingConfig:
    """Saving switches and filenames."""

    save_sample_series_values: bool
    save_score_values: bool
    final_table_csv_filename: str
    final_table_md_filename: str
    max_filename_length: int


@dataclass(frozen=True)
class ExperimentOutputPaths:
    """Output paths for one experiment execution."""

    run_dir: Path
    score_plots_dir: Path
    score_values_dir: Path
    sample_series_plots_dir: Path
    sample_series_values_dir: Path
    final_table_csv_path: Path
    final_table_md_path: Path
    predictive_mechanism_heatmaps_dir: Optional[Path] = None
    coefficient_heatmaps_dir: Optional[Path] = None

    @property
    def heatmaps_dir(self):
        if self.predictive_mechanism_heatmaps_dir is not None:
            return self.predictive_mechanism_heatmaps_dir

        return self.coefficient_heatmaps_dir


def safe_filename(text, max_len=220):
    """Create a safe collision-resistant filename."""

    raw = str(text)

    extension_match = re.search(
        r"(?i)(\.(?:eps|jpeg|jpg|pdf|pgf|png|ps|raw|rgba|"
        r"svg|svgz|tif|tiff|webp|csv|json|md))$",
        raw,
    )

    extension = (
        extension_match.group(1)
        if extension_match
        else ""
    )
    stem = (
        raw[:-len(extension)]
        if extension
        else raw
    )

    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        stem,
    ).strip("._-")

    if not cleaned:
        cleaned = "output"

    available_stem_length = max_len - len(extension)

    if available_stem_length <= 12:
        raise ValueError(
            f"max_len={max_len} is too small to preserve "
            f"extension {extension!r}."
        )

    if len(cleaned) > available_stem_length:
        digest = hashlib.sha1(
            cleaned.encode("utf-8")
        ).hexdigest()[:10]

        prefix_length = (
            available_stem_length
            - len(digest)
            - 1
        )
        prefix = (
            cleaned[:prefix_length].rstrip("._-")
            or "output"
        )
        cleaned = f"{prefix}_{digest}"

    return f"{cleaned}{extension}"


def create_unique_directory(base_dir):
    """Create a directory using the original suffix convention."""

    base_path = Path(base_dir)
    candidate = base_path
    suffix = 2

    while candidate.exists():
        candidate = base_path.with_name(
            f"{base_path.name}_{suffix}"
        )
        suffix += 1

    candidate.mkdir(
        parents=True,
        exist_ok=False,
    )

    return candidate.resolve()


def create_experiment_output_paths_at(
    run_dir,
    experiment_kind,
    saving_config,
):
    """Create the output layout at an exact path."""

    if experiment_kind not in {"causal", "spectral"}:
        raise ValueError(
            "experiment_kind must be 'causal' or 'spectral'. "
            f"Got {experiment_kind!r}."
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    score_plots_dir = run_dir / "score_plots"
    score_values_dir = run_dir / "score_values"
    sample_series_plots_dir = (
        run_dir / "sample_series_plots"
    )
    sample_series_values_dir = (
        run_dir / "sample_series_values"
    )

    standard_directories = [
        score_plots_dir,
        score_values_dir,
        sample_series_plots_dir,
        sample_series_values_dir,
    ]

    for directory in standard_directories:
        directory.mkdir(
            parents=False,
            exist_ok=False,
        )

    predictive_heatmaps_dir = None
    coefficient_heatmaps_dir = None

    if experiment_kind == "causal":
        predictive_heatmaps_dir = (
            run_dir
            / "predictive_mechanism_heatmaps"
        )
        predictive_heatmaps_dir.mkdir(
            parents=False,
            exist_ok=False,
        )
    else:
        coefficient_heatmaps_dir = (
            run_dir / "coefficient_heatmaps"
        )
        coefficient_heatmaps_dir.mkdir(
            parents=False,
            exist_ok=False,
        )

    return ExperimentOutputPaths(
        run_dir=run_dir.resolve(),
        score_plots_dir=score_plots_dir.resolve(),
        score_values_dir=score_values_dir.resolve(),
        sample_series_plots_dir=(
            sample_series_plots_dir.resolve()
        ),
        sample_series_values_dir=(
            sample_series_values_dir.resolve()
        ),
        final_table_csv_path=(
            run_dir
            / saving_config.final_table_csv_filename
        ).resolve(),
        final_table_md_path=(
            run_dir
            / saving_config.final_table_md_filename
        ).resolve(),
        predictive_mechanism_heatmaps_dir=(
            None
            if predictive_heatmaps_dir is None
            else predictive_heatmaps_dir.resolve()
        ),
        coefficient_heatmaps_dir=(
            None
            if coefficient_heatmaps_dir is None
            else coefficient_heatmaps_dir.resolve()
        ),
    )


def create_unique_experiment_output_paths(
    base_dir,
    experiment_kind,
    saving_config,
):
    """Create a collision-free experiment output layout."""

    base_path = Path(base_dir)
    candidate = base_path
    suffix = 2

    while candidate.exists():
        candidate = base_path.with_name(
            f"{base_path.name}_{suffix}"
        )
        suffix += 1

    return create_experiment_output_paths_at(
        run_dir=candidate,
        experiment_kind=experiment_kind,
        saving_config=saving_config,
    )


def save_sample_series_values(
    X,
    dataset_config,
    series_domain,
    saving_config,
    output_dir,
    sampling_rate=None,
):
    """Save one generated multivariate series as CSV."""

    if not saving_config.save_sample_series_values:
        return None

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must have shape (T, d). Got {X.shape}."
        )

    if sampling_rate is None:
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )
    else:
        sampling_rate = float(sampling_rate)

        if (
            not np.isfinite(sampling_rate)
            or sampling_rate <= 0
        ):
            raise ValueError(
                "sampling_rate must be finite and positive."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = safe_filename(
        f"sample_series__domain_{series_domain}__"
        f"{dataset_config.label}.csv",
        max_len=saving_config.max_filename_length,
    )
    output_path = output_dir / filename

    if sampling_rate is None:
        fieldnames = [
            "time_index",
            "regime",
            *[
                f"x{variable}"
                for variable in range(X.shape[1])
            ],
        ]
    else:
        fieldnames = [
            "time_index",
            "time_seconds",
            "regime",
            *[
                f"x{variable}"
                for variable in range(X.shape[1])
            ],
        ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for time_index, row in enumerate(X):
            record = {
                "time_index": time_index,
                "regime": (
                    1
                    if time_index < dataset_config.tau_star
                    else 2
                ),
            }

            if sampling_rate is not None:
                record["time_seconds"] = (
                    time_index / sampling_rate
                )

            record.update(
                {
                    f"x{variable}": float(value)
                    for variable, value in enumerate(row)
                }
            )

            writer.writerow(record)

    return str(output_path.resolve())


def save_score_values(
    scores,
    dataset_config,
    competitor,
    series_domain,
    saving_config,
    output_dir,
):
    """Save raw candidate scores as CSV."""

    if not saving_config.save_score_values:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = safe_filename(
        f"score_values__domain_{series_domain}__"
        f"{dataset_config.label}__"
        f"{competitor.label}.csv",
        max_len=saving_config.max_filename_length,
    )
    output_path = output_dir / filename

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["tau", "score"],
        )
        writer.writeheader()

        for tau in sorted(scores):
            writer.writerow(
                {
                    "tau": tau,
                    "score": scores[tau],
                }
            )

    return str(output_path.resolve())


def save_final_percentage_error_table(
    columns,
    rows,
    csv_path,
    md_path,
):
    """Save the final table as CSV and Markdown."""

    csv_path = Path(csv_path).resolve()
    md_path = Path(md_path).resolve()

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    md_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(columns),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row[column]
                    for column in columns
                }
            )

    markdown = table_rows_to_markdown(
        columns,
        rows,
    )
    md_path.write_text(
        markdown + "\n",
        encoding="utf-8",
    )

    return csv_path, md_path


def save_setup_replicate_table(
    results,
    dataset_label,
    csv_path,
    md_path,
    analysis_domain_name=None,
):
    """Save the ordinary table for one replicate."""

    columns, rows = build_final_percentage_error_table(
        {
            dataset_label: list(results),
        },
        analysis_domain_name=analysis_domain_name,
    )

    return save_final_percentage_error_table(
        columns=columns,
        rows=rows,
        csv_path=csv_path,
        md_path=md_path,
    )


def save_json(data, output_path):
    """Save strict JSON with the original formatting."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    return output_path.resolve()


def full_experiment_setup_name(
    dataset_config,
    setup_index,
):
    """Return the directory name for one setup."""

    return (
        f"setup_{setup_index:03d}__"
        f"{safe_filename(dataset_config.label, max_len=170)}"
    )


def collect_plot_manifest(
    directory,
    root_dir,
    plot_format,
):
    """List saved plots relative to the experiment root."""

    directory = Path(directory)
    root_dir = Path(root_dir).resolve()

    if not directory.exists():
        return []

    plots = sorted(
        directory.rglob(f"*.{plot_format}")
    )

    return [
        relative_output_path(
            str(path),
            root_dir,
        )
        for path in plots
    ]