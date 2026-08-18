"""Seizure EEG experiment configuration and data loading."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from cpd.data.seizure_eeg import load_seizure_eeg


# Classes

@dataclass(frozen=True)
class SeizureEEGExperimentParameters:
    """Parameters controlling the seizure EEG experiment."""

    raw_data_dir: Path
    patient_numbers: Sequence[int]
    section_numbers: Optional[Sequence[int]]
    min_annotators: int
    channel_labels: Optional[Sequence[str]]
    model_lag_order: int
    window_size: int
    base_seed: int = 42


@dataclass(frozen=True)
class DatasetConfig:
    """One seizure EEG section and scan configuration."""

    raw_data_dir: Path
    patient_number: int
    section_number: int
    section_start: int
    section_end: int
    target_changepoint: int
    sample_frequency: float
    channel_labels: Optional[Sequence[str]]
    min_annotators: int
    model_lag_order: int
    window_size: int
    sensitivity_factor: str = "patient_section"
    sensitivity_value: str = "default"
    seed: int = 42

    @property
    def T(self):
        return self.section_end - self.section_start

    @property
    def tau_star(self):
        return self.target_changepoint - self.section_start

    @property
    def dimension(self):
        if self.channel_labels is None:
            return 0

        return len(self.channel_labels)

    @property
    def sigma(self):
        return 0.0

    @property
    def changepoint_percentile(self):
        return self.tau_star / self.T

    @property
    def empirical_quantile_match(self):
        return False

    @property
    def data_generating_process(self):
        return "seizure_eeg"

    @property
    def data_lag_order(self):
        return 0

    @property
    def label(self):
        return (
            f"patient={self.patient_number}, "
            f"section={self.section_number}, "
            f"d={self.dimension}, T={self.T}, "
            f"tau*={self.tau_star}, "
            f"fs={self.sample_frequency:g}, "
            f"fit_L={self.model_lag_order}, "
            f"h={self.window_size}"
        )

    @property
    def observation_id(self):
        return (
            f"eeg{self.patient_number}"
            f"_section{self.section_number}"
        )


# Functions

@lru_cache(maxsize=1)
def _load_patient(
    raw_data_dir,
    patient_number,
    min_annotators,
    channel_labels,
):
    """Load and retain only the current patient's recording."""

    return load_seizure_eeg(
        raw_data_dir=Path(raw_data_dir),
        patient_number=patient_number,
        min_annotators=min_annotators,
        channel_labels=channel_labels,
    )


def _validate_parameters(parameters):
    """Validate seizure EEG experiment parameters."""
    if parameters.min_annotators not in {1, 2, 3}:
        raise ValueError(
            "min_annotators must be 1, 2, or 3."
        )

    if parameters.model_lag_order <= 0:
        raise ValueError("model_lag_order must be positive.")

    if parameters.window_size <= 0:
        raise ValueError("window_size must be positive.")

    if not parameters.patient_numbers:
        raise ValueError("patient_numbers cannot be empty.")

    patient_numbers = [
        int(patient_number)
        for patient_number in parameters.patient_numbers
    ]

    if len(patient_numbers) != len(set(patient_numbers)):
        raise ValueError(
            "patient_numbers must be unique."
        )

    if any(
        patient_number < 1 or patient_number > 79
        for patient_number in patient_numbers
    ):
        raise ValueError(
            "patient_numbers must lie in [1, 79]."
        )

    if parameters.section_numbers is not None:
        section_numbers = [
            int(section_number)
            for section_number in parameters.section_numbers
        ]

        if any(
            section_number <= 0
            for section_number in section_numbers
        ):
            raise ValueError(
                "section_numbers must be positive."
            )


def iter_dataset_configs(
    parameters: SeizureEEGExperimentParameters,
) -> Iterable[DatasetConfig]:
    """Yield every searchable single-changepoint EEG section."""
    _validate_parameters(parameters)

    requested_sections = (
        None
        if parameters.section_numbers is None
        else set(parameters.section_numbers)
    )
    yielded = 0

    for patient_number in parameters.patient_numbers:
        loaded = _load_patient(
            str(Path(parameters.raw_data_dir).resolve()),
            int(patient_number),
            int(parameters.min_annotators),
            (
                None
                if parameters.channel_labels is None
                else tuple(parameters.channel_labels)
            ),
        )

        resolved_labels = tuple(
            loaded["channel_labels"]
        )

        for section in loaded["sections"]:
            section_number = section["section_number"]

            if (
                requested_sections is not None
                and section_number not in requested_sections
            ):
                continue

            T = section["end"] - section["start"]
            tau_star = section["tau_star"]

            if T < 2 * parameters.window_size + 1:
                continue

            if not (
                parameters.window_size
                <= tau_star
                <= T - parameters.window_size
            ):
                continue

            yielded += 1
            yield DatasetConfig(
                raw_data_dir=Path(parameters.raw_data_dir),
                patient_number=int(patient_number),
                section_number=int(section_number),
                section_start=int(section["start"]),
                section_end=int(section["end"]),
                target_changepoint=(
                    int(section["start"] + tau_star)
                ),
                sample_frequency=float(
                    loaded["sample_frequency"]
                ),
                channel_labels=resolved_labels,
                min_annotators=parameters.min_annotators,
                model_lag_order=parameters.model_lag_order,
                window_size=parameters.window_size,
                sensitivity_value=(
                    f"eeg{patient_number}_section{section_number}"
                ),
                seed=parameters.base_seed,
            )

    if yielded == 0:
        raise ValueError(
            "The seizure EEG configuration produced no searchable "
            "single-changepoint sections."
        )


def generate_dataset(
    config: DatasetConfig,
    parameters: SeizureEEGExperimentParameters,
):
    """Load one configured single-changepoint EEG section."""
    del parameters

    loaded = _load_patient(
        str(Path(config.raw_data_dir).resolve()),
        int(config.patient_number),
        int(config.min_annotators),
        (
            None
            if config.channel_labels is None
            else tuple(config.channel_labels)
        ),
    )

    X_raw = loaded["time_series"][
        config.section_start:config.section_end
    ]

    if X_raw.shape != (config.T, config.dimension):
        raise ValueError(
            "Loaded EEG section does not match its configuration. "
            f"Expected {(config.T, config.dimension)}; "
            f"got {X_raw.shape}."
        )

    if not np.all(np.isfinite(X_raw)):
        raise ValueError(
            "Loaded EEG section contains NaN or infinite values."
        )

    generation_details = {
        "data_generating_process": "seizure_eeg",
        "patient_number": config.patient_number,
        "section_number": config.section_number,
        "section_start": config.section_start,
        "section_end": config.section_end,
        "target_changepoint": config.target_changepoint,
        "sample_frequency": config.sample_frequency,
        "channel_labels": list(config.channel_labels),
    }

    return X_raw.copy(), X_raw, generation_details
