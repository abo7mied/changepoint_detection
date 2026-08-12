"""Shared causal and spectral experiment runner."""

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from cpd.design import make_candidates
from cpd.methods.cross_regime import (
    estimate_causal_cpd_log_normalized_cross_regime,
)
from cpd.methods.kmo import estimate_kmo_varscore_amoc
from cpd.plotting import (
    PlotConfig,
    export_spectral_coefficient_heatmaps,
    save_pairwise_replicate_boxplots,
    save_predictive_mechanism_heatmap,
    save_sample_series_plot,
    save_sample_series_plot_without_changepoint,
    save_score_curve_plot,
    save_spectral_coefficient_heatmap,
)
from cpd.results import (
    ExperimentResult,
    build_final_percentage_error_table,
    build_setup_method_summary,
    dataset_config_summary,
    finite_or_none,
    print_plain_table,
    print_table,
    relative_output_path,
    serialize_experiment_result,
    summarize_replicate_metric,
)
from cpd.saving import (
    ExperimentOutputPaths,
    SavingConfig,
    collect_plot_manifest,
    create_experiment_output_paths_at,
    create_unique_directory,
    create_unique_experiment_output_paths,
    full_experiment_setup_name,
    save_final_percentage_error_table,
    save_json,
    save_sample_series_values,
    save_score_values,
    save_setup_replicate_table,
)


@dataclass(frozen=True)
class RunnerConfig:
    """Shared runner switches and output parameters."""

    experiment_kind: str
    experiment_design_mode: str
    analysis_domain_name: str
    candidate_step: int

    verbose_candidate_progress: bool
    print_scores: bool
    print_per_dataset_tables: bool

    experiment_outputs_base_dir: str
    full_experiment_outputs_base_dir: str
    full_experiment_summary_filename: str
    setup_summary_filename: str

    full_experiment_num_replicates: int
    full_experiment_base_seed: int

    plot_config: PlotConfig
    saving_config: SavingConfig

    sampling_rate: Optional[float] = None
    spectral_band_names: Sequence[str] = ()

    cross_regime_kwargs: Mapping[str, Any] = field(
        default_factory=dict
    )
    kmo_kwargs: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.experiment_kind not in {
            "causal",
            "spectral",
        }:
            raise ValueError(
                "experiment_kind must be 'causal' or 'spectral'. "
                f"Got {self.experiment_kind!r}."
            )


@dataclass(frozen=True)
class RunnerDependencies:
    """Experiment-specific functions used by the runner."""

    iter_dataset_configs: Callable[[], Sequence[Any]]
    iter_competitors: Callable[[Any], Sequence[Any]]
    generate_dataset: Callable[[Any], tuple]

    build_analysis_series: Optional[Callable] = None

    spectral_experiments: Sequence[Any] = ()
    get_spectral_experiment: Optional[Callable] = None


def make_full_experiment_replicate_seeds(
    num_replicates,
    base_seed,
):
    """Return the shared reproducible replicate seeds."""

    if num_replicates <= 0:
        raise ValueError(
            "full_experiment_num_replicates must be positive. "
            f"Got {num_replicates}."
        )

    seed_sequence = np.random.SeedSequence(base_seed)
    children = seed_sequence.spawn(num_replicates)

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in children
    ]


def export_sample_series(
    series_by_domain,
    dataset_config,
    runner_config,
    output_paths,
):
    """Export the sample plots and CSV values."""

    exported = {}

    for series_domain, X_domain in series_by_domain.items():
        plot_path = save_sample_series_plot(
            X=X_domain,
            dataset_config=dataset_config,
            series_domain=series_domain,
            num_regimes=2,
            plot_config=runner_config.plot_config,
            output_dir=output_paths.sample_series_plots_dir,
            spectral_case=(
                runner_config.experiment_kind
                == "spectral"
            ),
            analysis_domain_name=(
                runner_config.analysis_domain_name
            ),
        )

        unmarked_plot_path = None

        if runner_config.experiment_kind == "spectral":
            unmarked_plot_path = (
                save_sample_series_plot_without_changepoint(
                    X=X_domain,
                    dataset_config=dataset_config,
                    series_domain=series_domain,
                    plot_config=runner_config.plot_config,
                    output_dir=(
                        output_paths.sample_series_plots_dir
                    ),
                    analysis_domain_name=(
                        runner_config.analysis_domain_name
                    ),
                )
            )

        values_path = save_sample_series_values(
            X=X_domain,
            dataset_config=dataset_config,
            series_domain=series_domain,
            saving_config=runner_config.saving_config,
            output_dir=(
                output_paths.sample_series_values_dir
            ),
            sampling_rate=(
                runner_config.sampling_rate
                if runner_config.experiment_kind
                == "spectral"
                else None
            ),
        )

        exported[series_domain] = {
            "plot": plot_path,
            "values": values_path,
        }

        if runner_config.experiment_kind == "spectral":
            exported[series_domain][
                "plot_without_changepoint"
            ] = unmarked_plot_path

    return exported


def evaluate_one_competitor(
    X,
    dataset_config,
    competitor,
    series_domain,
    candidates,
    runner_config,
    output_paths,
):
    """Run one competitor and calculate localization error."""

    tau_star = dataset_config.tau_star

    nearest_candidate = min(
        candidates,
        key=lambda candidate: abs(
            candidate - tau_star
        ),
    )

    best_possible_abs_error = abs(
        nearest_candidate - tau_star
    )
    best_possible_pct_error = (
        100.0
        * best_possible_abs_error
        / dataset_config.T
    )

    result_series_domain = (
        series_domain
        if runner_config.experiment_kind == "spectral"
        else None
    )

    try:
        if competitor.model_choice == "KMOVARScore":
            kmo_kwargs = dict(
                runner_config.kmo_kwargs
            )

            tau_hat, scores = estimate_kmo_varscore_amoc(
                series=X,
                L=competitor.model_lag_order,
                candidates=candidates,
                competitor=competitor,
                **kmo_kwargs,
            )
        else:
            cross_regime_kwargs = dict(
                runner_config.cross_regime_kwargs
            )
            cross_regime_kwargs.setdefault(
                "verbose",
                runner_config.verbose_candidate_progress,
            )

            tau_hat, scores = (
                estimate_causal_cpd_log_normalized_cross_regime(
                    series=X,
                    window_size=competitor.window_size,
                    L=competitor.model_lag_order,
                    candidates=candidates,
                    competitor=competitor,
                    **cross_regime_kwargs,
                )
            )

        abs_error = abs(tau_hat - tau_star)
        pct_error = (
            100.0
            * abs_error
            / dataset_config.T
        )
        score_at_tau_hat = scores.get(
            tau_hat,
            None,
        )

        score_plot_path = save_score_curve_plot(
            scores=scores,
            tau_star=tau_star,
            tau_hat=tau_hat,
            dataset_config=dataset_config,
            competitor=competitor,
            series_domain=series_domain,
            plot_config=runner_config.plot_config,
            output_dir=output_paths.score_plots_dir,
        )

        score_values_path = save_score_values(
            scores=scores,
            dataset_config=dataset_config,
            competitor=competitor,
            series_domain=series_domain,
            saving_config=runner_config.saving_config,
            output_dir=output_paths.score_values_dir,
        )

        if runner_config.print_scores:
            if result_series_domain is None:
                print(
                    f"Scores for {competitor.label}:"
                )
            else:
                print(
                    f"Scores for [{series_domain}] "
                    f"{competitor.label}:"
                )

            for tau, score in scores.items():
                print(
                    f"  tau={tau:5d}, "
                    f"score={score: .6f}"
                )

        return ExperimentResult(
            dataset_label=dataset_config.label,
            series_domain=result_series_domain,
            competitor_label=competitor.label,
            model_choice=competitor.model_choice,
            score_direction=competitor.score_direction,
            epsilon=competitor.epsilon,
            model_lag_order=(
                competitor.model_lag_order
            ),
            window_size=competitor.window_size,
            tau_star=tau_star,
            tau_hat=tau_hat,
            nearest_candidate=nearest_candidate,
            abs_error=abs_error,
            pct_error=pct_error,
            best_possible_abs_error=(
                best_possible_abs_error
            ),
            best_possible_pct_error=(
                best_possible_pct_error
            ),
            score_at_tau_hat=score_at_tau_hat,
            status="OK",
            score_plot_path=score_plot_path,
            score_values_path=score_values_path,
        )

    except Exception as exc:
        return ExperimentResult(
            dataset_label=dataset_config.label,
            series_domain=result_series_domain,
            competitor_label=competitor.label,
            model_choice=competitor.model_choice,
            score_direction=competitor.score_direction,
            epsilon=competitor.epsilon,
            model_lag_order=(
                competitor.model_lag_order
            ),
            window_size=competitor.window_size,
            tau_star=tau_star,
            tau_hat=None,
            nearest_candidate=nearest_candidate,
            abs_error=None,
            pct_error=float("inf"),
            best_possible_abs_error=(
                best_possible_abs_error
            ),
            best_possible_pct_error=(
                best_possible_pct_error
            ),
            score_at_tau_hat=None,
            status=(
                f"ERROR: {type(exc).__name__}: {exc}"
            ),
        )


def run_one_dataset(
    dataset_config,
    runner_config,
    dependencies,
    output_paths,
):
    """Generate one dataset and run every competitor."""

    X, _, generation_details = (
        dependencies.generate_dataset(
            dataset_config
        )
    )

    if runner_config.experiment_kind == "spectral":
        if dependencies.build_analysis_series is None:
            raise ValueError(
                "The spectral runner requires "
                "build_analysis_series."
            )

        series_by_domain = (
            dependencies.build_analysis_series(
                X,
                generation_details=generation_details,
            )
        )
    else:
        series_by_domain = {
            runner_config.analysis_domain_name: X,
        }

    exported_series = export_sample_series(
        series_by_domain=series_by_domain,
        dataset_config=dataset_config,
        runner_config=runner_config,
        output_paths=output_paths,
    )

    mechanism_heatmap_path = None

    if runner_config.experiment_kind == "causal":
        mechanism_heatmap_path = (
            save_predictive_mechanism_heatmap(
                coefficient_graphs=generation_details,
                dataset_config=dataset_config,
                num_regimes=2,
                plot_config=runner_config.plot_config,
                output_dir=(
                    output_paths
                    .predictive_mechanism_heatmaps_dir
                ),
            )
        )

    competitors = list(
        dependencies.iter_competitors(
            dataset_config
        )
    )

    scan_settings = sorted(
        {
            (
                competitor.model_lag_order,
                competitor.window_size,
            )
            for competitor in competitors
        }
    )

    print("\n" + "=" * 100)
    print(f"Dataset: {dataset_config.label}")

    if runner_config.experiment_kind == "spectral":
        print(
            "Analysis domains: "
            f"{list(series_by_domain)}"
        )
        competitor_description = (
            f"competitors per domain={len(competitors)}"
        )
    else:
        competitor_description = (
            f"competitors={len(competitors)}"
        )

    print(
        f"Model/window settings={scan_settings}, "
        f"candidate step={runner_config.candidate_step}, "
        f"{competitor_description}"
    )

    for lag_order, window_size in scan_settings:
        candidates = make_candidates(
            dataset_config.T,
            window_size,
            runner_config.candidate_step,
        )

        print(
            f"  L={lag_order}, h={window_size}: "
            f"candidate range="
            f"[{candidates[0]}, {candidates[-1]}], "
            f"num candidates={len(candidates)}"
        )

        if (
            dataset_config.tau_star < candidates[0]
            or dataset_config.tau_star
            > candidates[-1]
        ):
            warnings.warn(
                f"tau*={dataset_config.tau_star} is "
                "outside the candidate range "
                f"[{candidates[0]}, "
                f"{candidates[-1]}] for "
                f"L={lag_order}, h={window_size}. "
                "That configuration cannot exactly "
                "recover this changepoint.",
                RuntimeWarning,
            )

    if (
        runner_config.plot_config.save_sample_series_plots
        or runner_config.saving_config
        .save_sample_series_values
    ):
        if runner_config.experiment_kind == "spectral":
            print(
                "Exported sample-series representations:"
            )

            for domain, paths in exported_series.items():
                print(
                    f"  {domain}: "
                    f"plot={paths['plot'] or '-'}, "
                    "plot_without_changepoint="
                    f"{paths['plot_without_changepoint'] or '-'}, "
                    f"values={paths['values'] or '-'}"
                )
        else:
            paths = exported_series[
                runner_config.analysis_domain_name
            ]

            print("Exported sample time series:")
            print(
                f"  plot={paths['plot'] or '-'}"
            )
            print(
                f"  values={paths['values'] or '-'}"
            )

    if (
        runner_config.experiment_kind == "causal"
        and runner_config.plot_config
        .save_predictive_mechanism_heatmaps
    ):
        print(
            "Exported predictive mechanism heatmap:"
        )
        print(
            f"  heatmap="
            f"{mechanism_heatmap_path or '-'}"
        )

    print("=" * 100)

    results = []

    for series_domain, X_domain in (
        series_by_domain.items()
    ):
        if runner_config.experiment_kind == "spectral":
            print(f"\n  Domain: {series_domain}")

        for competitor in competitors:
            candidates = make_candidates(
                dataset_config.T,
                competitor.window_size,
                runner_config.candidate_step,
            )

            if runner_config.experiment_kind == "spectral":
                print(
                    "    Running competitor: "
                    f"{competitor.label}"
                )
            else:
                print(
                    "  Running competitor: "
                    f"{competitor.label}"
                )

            result = evaluate_one_competitor(
                X=X_domain,
                dataset_config=dataset_config,
                competitor=competitor,
                series_domain=series_domain,
                candidates=candidates,
                runner_config=runner_config,
                output_paths=output_paths,
            )

            results.append(result)

    if runner_config.experiment_kind == "spectral":
        domain_order = {
            name: index
            for index, name in enumerate(
                series_by_domain
            )
        }

        results.sort(
            key=lambda result: (
                domain_order.get(
                    result.series_domain,
                    len(domain_order),
                ),
                not result.status.startswith("OK"),
                result.pct_error,
                result.model_choice,
                result.score_direction,
                result.epsilon,
                result.model_lag_order,
                result.window_size,
            )
        )
    else:
        results.sort(
            key=lambda result: (
                not result.status.startswith("OK"),
                result.pct_error,
                result.model_choice,
                result.score_direction,
                result.epsilon,
                result.model_lag_order,
                result.window_size,
            )
        )

    return results


def print_and_save_final_table(
    all_results,
    runner_config,
    output_paths,
):
    """Print and save the final cross-dataset table."""

    columns, rows = (
        build_final_percentage_error_table(
            all_results,
            analysis_domain_name=(
                runner_config.analysis_domain_name
            ),
        )
    )

    csv_path, md_path = (
        save_final_percentage_error_table(
            columns=columns,
            rows=rows,
            csv_path=(
                output_paths.final_table_csv_path
            ),
            md_path=output_paths.final_table_md_path,
        )
    )

    print("\n" + "=" * 100)
    print("Saved experiment outputs:")
    print(
        f"  Run directory:         "
        f"{output_paths.run_dir}"
    )
    print(f"  CSV:                   {csv_path}")
    print(f"  Markdown:              {md_path}")
    print(
        f"  Score plots:           "
        f"{output_paths.score_plots_dir}"
    )
    print(
        f"  Score values:          "
        f"{output_paths.score_values_dir}"
    )
    print(
        f"  Sample-series plots:   "
        f"{output_paths.sample_series_plots_dir}"
    )
    print(
        f"  Sample-series values:  "
        f"{output_paths.sample_series_values_dir}"
    )

    if runner_config.experiment_kind == "causal":
        print(
            "  Mechanism heatmaps:    "
            f"{output_paths.predictive_mechanism_heatmaps_dir}"
        )
    else:
        print(
            "  Coefficient heatmaps:  "
            f"{output_paths.coefficient_heatmaps_dir}"
        )

    print("\n" + "=" * 100)
    print("FINAL PERCENTAGE-ERROR TABLE")
    print(
        "Cell value = "
        "100 * |tau_hat - tau_star| / T. "
        "Lower is better."
    )
    print("=" * 100)

    print_plain_table(columns, rows)


def run_experiment(
    runner_config,
    dependencies,
):
    """Run every dataset setup once."""

    output_paths = (
        create_unique_experiment_output_paths(
            base_dir=(
                runner_config
                .experiment_outputs_base_dir
            ),
            experiment_kind=(
                runner_config.experiment_kind
            ),
            saving_config=(
                runner_config.saving_config
            ),
        )
    )

    dataset_configs = list(
        dependencies.iter_dataset_configs()
    )

    competitor_counts = [
        len(
            list(
                dependencies.iter_competitors(
                    config
                )
            )
        )
        for config in dataset_configs
    ]

    competitor_count_display = (
        competitor_counts[0]
        if (
            competitor_counts
            and len(set(competitor_counts)) == 1
        )
        else competitor_counts
    )

    if runner_config.experiment_kind == "spectral":
        heatmap_paths = (
            export_spectral_coefficient_heatmaps(
                experiments=(
                    dependencies.spectral_experiments
                ),
                band_names=(
                    runner_config.spectral_band_names
                ),
                plot_config=runner_config.plot_config,
                output_dir=(
                    output_paths
                    .coefficient_heatmaps_dir
                ),
            )
        )
    else:
        heatmap_paths = {}

    print("Synthetic CPD experiment")
    print(
        "Experiment design mode: "
        f"{runner_config.experiment_design_mode}"
    )
    print(
        f"Output directory: {output_paths.run_dir}"
    )

    if runner_config.experiment_kind == "spectral":
        print(
            "Number of spectral coefficient "
            "experiments: "
            f"{len(dependencies.spectral_experiments)}"
        )

        for experiment in (
            dependencies.spectral_experiments
        ):
            print(
                f"  - {experiment.name}: "
                f"{experiment.description}"
            )
            print(
                "    heatmap: "
                f"{heatmap_paths.get(experiment.name) or '-'}"
            )

        print(
            "Number of dataset setups after "
            "coefficient expansion: "
            f"{len(dataset_configs)}"
        )
        print(
            "Number of competitors per domain: "
            f"{competitor_count_display}"
        )
    else:
        print(
            "Number of dataset setups: "
            f"{len(dataset_configs)}"
        )
        print(
            "Number of competitors per dataset: "
            f"{competitor_count_display}"
        )

    all_results = {}

    for dataset_config in dataset_configs:
        results = run_one_dataset(
            dataset_config=dataset_config,
            runner_config=runner_config,
            dependencies=dependencies,
            output_paths=output_paths,
        )

        all_results[dataset_config.label] = results

        if runner_config.print_per_dataset_tables:
            print("\nRanked results:")
            print_table(results)

    print_and_save_final_table(
        all_results=all_results,
        runner_config=runner_config,
        output_paths=output_paths,
    )

    return all_results


def build_global_method_summary(
    results_by_competitor,
):
    """Build the full-experiment global ranking."""

    rows = []

    for competitor_label, results in (
        results_by_competitor.items()
    ):
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
                "num_results": len(results),
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


def run_full_replicate_experiment(
    runner_config,
    dependencies,
):
    """Run every setup over the shared replicate seeds."""

    base_dataset_configs = list(
        dependencies.iter_dataset_configs()
    )

    competitor_counts = [
        len(
            list(
                dependencies.iter_competitors(
                    config
                )
            )
        )
        for config in base_dataset_configs
    ]

    num_competitors_per_replicate = (
        competitor_counts[0]
        if (
            competitor_counts
            and len(set(competitor_counts)) == 1
        )
        else competitor_counts
    )

    replicate_seeds = (
        make_full_experiment_replicate_seeds(
            num_replicates=(
                runner_config
                .full_experiment_num_replicates
            ),
            base_seed=(
                runner_config
                .full_experiment_base_seed
            ),
        )
    )

    full_root = create_unique_directory(
        runner_config.full_experiment_outputs_base_dir
    )

    setup_summaries = []
    global_results_by_competitor = {}

    print(
        "Replicated full "
        f"{runner_config.experiment_kind} "
        "CPD experiment"
    )
    print(
        "Experiment design mode: "
        f"{runner_config.experiment_design_mode}"
    )
    print(f"Output directory: {full_root}")
    print(
        "Number of parameter setups: "
        f"{len(base_dataset_configs)}"
    )
    print(
        f"Replicates per setup: "
        f"{len(replicate_seeds)}"
    )
    print(
        "Competitors per replicate: "
        f"{num_competitors_per_replicate}"
    )

    for setup_index, base_config in enumerate(
        base_dataset_configs,
        start=1,
    ):
        setup_name = full_experiment_setup_name(
            base_config,
            setup_index,
        )
        setup_dir = full_root / setup_name
        setup_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        replicates_dir = setup_dir / "replicates"
        box_plots_dir = setup_dir / "box_plots"

        replicates_dir.mkdir(
            parents=False,
            exist_ok=False,
        )
        box_plots_dir.mkdir(
            parents=False,
            exist_ok=False,
        )

        coefficient_heatmap_path = None

        if runner_config.experiment_kind == "spectral":
            coefficient_heatmaps_dir = (
                setup_dir / "coefficient_heatmaps"
            )
            coefficient_heatmaps_dir.mkdir(
                parents=False,
                exist_ok=False,
            )

            if dependencies.get_spectral_experiment is None:
                raise ValueError(
                    "The spectral full runner requires "
                    "get_spectral_experiment."
                )

            spectral_experiment = (
                dependencies.get_spectral_experiment(
                    base_config
                )
            )

            if spectral_experiment is not None:
                coefficient_heatmap_path = (
                    save_spectral_coefficient_heatmap(
                        experiment=(
                            spectral_experiment
                        ),
                        band_names=(
                            runner_config
                            .spectral_band_names
                        ),
                        plot_config=(
                            runner_config.plot_config
                        ),
                        output_dir=(
                            coefficient_heatmaps_dir
                        ),
                    )
                )

        print("\n" + "=" * 100)
        print(
            f"Full-experiment setup "
            f"{setup_index}/"
            f"{len(base_dataset_configs)}"
        )
        print(base_config.label)
        print(f"Setup directory: {setup_dir}")
        print("=" * 100)

        replicate_records = []

        for replicate_index, seed in enumerate(
            replicate_seeds,
            start=1,
        ):
            replicate_dir = (
                replicates_dir
                / (
                    f"replicate_{replicate_index:03d}"
                    f"__seed_{seed}"
                )
            )

            replicate_paths = (
                create_experiment_output_paths_at(
                    run_dir=replicate_dir,
                    experiment_kind=(
                        runner_config.experiment_kind
                    ),
                    saving_config=(
                        runner_config.saving_config
                    ),
                )
            )

            replicate_config = replace(
                base_config,
                seed=int(seed),
            )

            print(
                f"\nReplicate {replicate_index}/"
                f"{len(replicate_seeds)} "
                f"for setup {setup_index}: "
                f"seed={seed}"
            )

            results = run_one_dataset(
                dataset_config=replicate_config,
                runner_config=runner_config,
                dependencies=dependencies,
                output_paths=replicate_paths,
            )

            csv_path, md_path = (
                save_setup_replicate_table(
                    results=results,
                    dataset_label=(
                        replicate_config.label
                    ),
                    csv_path=(
                        replicate_paths
                        .final_table_csv_path
                    ),
                    md_path=(
                        replicate_paths
                        .final_table_md_path
                    ),
                    analysis_domain_name=(
                        runner_config
                        .analysis_domain_name
                    ),
                )
            )

            for result in results:
                global_results_by_competitor.setdefault(
                    result.competitor_label,
                    [],
                ).append(result)

            record = {
                "replicate_index": replicate_index,
                "seed": int(seed),
                "directory": relative_output_path(
                    replicate_paths.run_dir,
                    full_root,
                ),
                "final_table_csv": relative_output_path(
                    csv_path,
                    full_root,
                ),
                "final_table_markdown": (
                    relative_output_path(
                        md_path,
                        full_root,
                    )
                ),
                "raw_results": results,
            }

            if runner_config.experiment_kind == "causal":
                record[
                    "predictive_mechanism_heatmaps"
                ] = collect_plot_manifest(
                    directory=(
                        replicate_paths
                        .predictive_mechanism_heatmaps_dir
                    ),
                    root_dir=full_root,
                    plot_format=(
                        runner_config
                        .plot_config.plot_format
                    ),
                )

            replicate_records.append(record)

        boxplot_paths = {}

        for metric in [
            "pct_error",
            "abs_error",
            "tau_hat",
        ]:
            boxplot_paths[metric] = (
                save_pairwise_replicate_boxplots(
                    replicate_records=(
                        replicate_records
                    ),
                    metric=metric,
                    setup_label=base_config.label,
                    output_dir=box_plots_dir,
                    plot_config=(
                        runner_config.plot_config
                    ),
                    tau_star=(
                        base_config.tau_star
                        if metric == "tau_hat"
                        else None
                    ),
                )
            )

        method_summary = build_setup_method_summary(
            replicate_records
        )

        serialized_replicates = []

        for record in replicate_records:
            serialized_record = {
                "replicate_index": (
                    record["replicate_index"]
                ),
                "seed": record["seed"],
                "directory": record["directory"],
                "final_table_csv": (
                    record["final_table_csv"]
                ),
                "final_table_markdown": (
                    record["final_table_markdown"]
                ),
                "results": [
                    serialize_experiment_result(
                        result,
                        full_root,
                    )
                    for result in (
                        record["raw_results"]
                    )
                ],
            }

            if runner_config.experiment_kind == "causal":
                serialized_record[
                    "predictive_mechanism_heatmaps"
                ] = record[
                    "predictive_mechanism_heatmaps"
                ]

            serialized_replicates.append(
                serialized_record
            )

        setup_summary = {
            "setup_index": setup_index,
            "setup_name": setup_name,
            "setup_directory": relative_output_path(
                setup_dir,
                full_root,
            ),
            "parameters": dataset_config_summary(
                base_config
            ),
            "num_replicates": len(replicate_seeds),
            "replicate_seeds": [
                int(seed)
                for seed in replicate_seeds
            ],
            "box_plots": {
                "percentage_error": [
                    relative_output_path(
                        path,
                        full_root,
                    )
                    for path in (
                        boxplot_paths["pct_error"]
                    )
                ],
                "absolute_error": [
                    relative_output_path(
                        path,
                        full_root,
                    )
                    for path in (
                        boxplot_paths["abs_error"]
                    )
                ],
                "estimated_tau": [
                    relative_output_path(
                        path,
                        full_root,
                    )
                    for path in (
                        boxplot_paths["tau_hat"]
                    )
                ],
            },
            "method_summary_and_ranking": (
                method_summary
            ),
            "replicates": serialized_replicates,
        }

        if runner_config.experiment_kind == "spectral":
            setup_summary["coefficient_heatmap"] = (
                relative_output_path(
                    coefficient_heatmap_path,
                    full_root,
                )
            )

        setup_summary["plot_manifest"] = (
            collect_plot_manifest(
                directory=setup_dir,
                root_dir=full_root,
                plot_format=(
                    runner_config
                    .plot_config.plot_format
                ),
            )
        )

        setup_summary_path = (
            setup_dir
            / runner_config.setup_summary_filename
        )
        setup_summary["setup_summary_json"] = (
            relative_output_path(
                setup_summary_path,
                full_root,
            )
        )

        save_json(
            setup_summary,
            setup_summary_path,
        )

        setup_summaries.append(setup_summary)

    global_method_summary = (
        build_global_method_summary(
            global_results_by_competitor
        )
    )

    full_summary = {
        "mode": "full_replicate_experiment",
        "experiment_design_mode": (
            runner_config.experiment_design_mode
        ),
        "output_directory": str(full_root),
        "num_parameter_setups": (
            len(base_dataset_configs)
        ),
        "num_replicates_per_setup": (
            len(replicate_seeds)
        ),
        "replicate_seeds": [
            int(seed)
            for seed in replicate_seeds
        ],
        "num_competitors_per_replicate": (
            num_competitors_per_replicate
        ),
        "ranking_metric": (
            "mean percentage localization error"
        ),
        "global_method_summary_and_ranking": (
            global_method_summary
        ),
        "setups": setup_summaries,
        "plot_manifest": collect_plot_manifest(
            directory=full_root,
            root_dir=full_root,
            plot_format=(
                runner_config.plot_config.plot_format
            ),
        ),
    }

    if runner_config.experiment_kind == "causal":
        full_summary["experiment_family"] = (
            "causal_predictive_mechanism"
        )

    full_summary_path = (
        full_root
        / runner_config
        .full_experiment_summary_filename
    )
    full_summary["summary_json"] = (
        relative_output_path(
            full_summary_path,
            full_root,
        )
    )

    save_json(
        full_summary,
        full_summary_path,
    )

    print("\n" + "=" * 100)
    print("FULL REPLICATE EXPERIMENT COMPLETE")
    print(f"Output directory: {full_root}")
    print(f"Summary JSON: {full_summary_path}")
    print(
        f"Total plots: "
        f"{len(full_summary['plot_manifest'])}"
    )
    print("=" * 100)

    return full_summary