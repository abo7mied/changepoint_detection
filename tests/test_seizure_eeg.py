from types import SimpleNamespace

import numpy as np
import pandas as pd

from cpd.data.seizure_eeg import (
    get_binary_annotation_column,
    map_annotation_changepoints_to_samples,
    single_changepoint_sections,
)
from cpd.experiments import seizure_eeg
from cpd.results import dataset_config_summary
from cpd import runner


def test_annotation_column_trims_trailing_missing_values():
    dataframe = pd.DataFrame({
        "1": [0, 0, 1, 1, np.nan, np.nan],
    })

    annotation = get_binary_annotation_column(
        dataframe,
        patient_number=1,
    )

    assert np.array_equal(
        annotation,
        np.array([0, 0, 1, 1]),
    )


def test_annotation_column_rejects_internal_missing_values():
    dataframe = pd.DataFrame({
        "1": [0, np.nan, 1, np.nan],
    })

    try:
        get_binary_annotation_column(
            dataframe,
            patient_number=1,
        )
    except ValueError as exc:
        assert "missing values before the end" in str(exc)
    else:
        raise AssertionError(
            "Internal missing annotations were accepted."
        )


def test_annotation_changepoints_map_at_256_hz():
    mapped = map_annotation_changepoints_to_samples(
        change_points=[2, 5],
        annotation_length=10,
        num_samples=2560,
        sample_frequency=256,
    )

    assert np.array_equal(
        mapped,
        np.array([512, 1280]),
    )


def test_single_changepoint_sections_exclude_neighbors():
    sections = single_changepoint_sections(
        num_samples=1000,
        change_points=[200, 500, 800],
    )

    assert sections == [
        {
            "section_number": 1,
            "start": 0,
            "end": 500,
            "tau_star": 200,
        },
        {
            "section_number": 2,
            "start": 201,
            "end": 800,
            "tau_star": 299,
        },
        {
            "section_number": 3,
            "start": 501,
            "end": 1000,
            "tau_star": 299,
        },
    ]


def test_dataset_configs_represent_observed_sections(monkeypatch):
    X = np.zeros((5000, 3))

    monkeypatch.setattr(
        seizure_eeg,
        "load_seizure_eeg",
        lambda *args, **kwargs: {
            "time_series": X,
            "sample_frequency": 256.0,
            "channel_labels": ["a", "b", "c"],
            "sections": [
                {
                    "section_number": 1,
                    "start": 100,
                    "end": 4101,
                    "tau_star": 2000,
                },
            ],
        },
    )

    parameters = (
        seizure_eeg.SeizureEEGExperimentParameters(
            raw_data_dir="unused",
            patient_numbers=[1],
            section_numbers=None,
            min_annotators=2,
            channel_labels=None,
            model_lag_order=2,
            window_size=1000,
        )
    )

    configs = list(
        seizure_eeg.iter_dataset_configs(parameters)
    )

    assert len(configs) == 1
    assert configs[0].observation_id == "eeg1_section1"
    assert configs[0].dimension == 3
    assert configs[0].tau_star == 2000


def test_dataset_summary_includes_eeg_metadata():
    config = seizure_eeg.DatasetConfig(
        raw_data_dir="unused",
        patient_number=7,
        section_number=3,
        section_start=100,
        section_end=2100,
        target_changepoint=1100,
        sample_frequency=256.0,
        channel_labels=("a", "b"),
        min_annotators=2,
        model_lag_order=2,
        window_size=500,
    )

    summary = dataset_config_summary(config)

    assert summary["observation_id"] == "eeg7_section3"
    assert summary["patient_number"] == 7
    assert summary["section_number"] == 3
    assert summary["channel_labels"] == ["a", "b"]


def test_full_runner_routes_observed_dataset_mode(monkeypatch):
    expected = {"mode": "observed"}

    monkeypatch.setattr(
        runner,
        "run_full_observed_dataset_experiment",
        lambda runner_config, dependencies: expected,
    )

    runner_config = SimpleNamespace(
        full_experiment_mode="observed_datasets"
    )

    result = runner.run_full_replicate_experiment(
        runner_config,
        dependencies=object(),
    )

    assert result is expected
