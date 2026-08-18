"""Neonatal seizure EEG data loading."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# Configuration

ANNOTATION_FILENAMES = (
    "annotations_2017_A_fixed.csv",
    "annotations_2017_B.csv",
    "annotations_2017_C.csv",
)


# Functions

def get_binary_annotation_column(
    dataframe,
    patient_number,
):
    """Return one patient's binary annotation column."""
    column = str(patient_number)

    if column not in dataframe.columns:
        raise ValueError(
            f"Annotation column {column!r} was not found."
        )

    numeric = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    valid_positions = np.flatnonzero(
        numeric.notna().to_numpy()
    )

    if len(valid_positions) == 0:
        raise ValueError(
            f"Annotation column {column!r} contains no values."
        )

    numeric = numeric.iloc[
        :valid_positions[-1] + 1
    ]

    if numeric.isna().any():
        raise ValueError(
            f"Annotation column {column!r} contains "
            "missing values before the end of the recording."
        )

    observed = numeric.to_numpy(dtype=float)

    if not np.all(np.isin(observed, [0.0, 1.0])):
        raise ValueError(
            "Annotations must contain only 0, 1, or missing values."
        )

    return numeric.to_numpy(
        dtype=int,
    )


def read_seizure_annotations(
    raw_data_dir,
    patient_number,
    min_annotators=2,
):
    """Read and aggregate the three expert annotations."""
    if min_annotators not in {1, 2, 3}:
        raise ValueError(
            "min_annotators must be 1, 2, or 3. "
            f"Got {min_annotators}."
        )

    raw_data_dir = Path(raw_data_dir)
    annotations = []

    for filename in ANNOTATION_FILENAMES:
        path = raw_data_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Annotation file was not found: {path}"
            )

        dataframe = pd.read_csv(path)
        annotations.append(
            get_binary_annotation_column(
                dataframe,
                patient_number,
            )
        )

    lengths = {
        len(annotation)
        for annotation in annotations
    }

    if len(lengths) != 1:
        raise ValueError(
            "All annotation files must have the same number of rows."
        )

    votes = np.sum(
        np.stack(annotations, axis=0),
        axis=0,
    )

    return (
        votes >= min_annotators
    ).astype(int)


def annotation_changepoints(annotation):
    """Return indices at which a binary annotation changes value."""
    annotation = np.asarray(annotation)

    if annotation.ndim != 1:
        raise ValueError(
            "annotation must be one-dimensional. "
            f"Got shape {annotation.shape}."
        )

    if len(annotation) == 0:
        raise ValueError("annotation cannot be empty.")

    if not np.all(np.isin(annotation, [0, 1])):
        raise ValueError("annotation must be binary.")

    indicator = np.concatenate(
        (
            [False],
            annotation[1:] != annotation[:-1],
        )
    )

    return np.flatnonzero(indicator).astype(int)


def map_annotation_changepoints_to_samples(
    change_points,
    annotation_length,
    num_samples,
    sample_frequency=None,
):
    """Map annotation coordinates to EEG sample coordinates."""
    if annotation_length <= 0:
        raise ValueError("annotation_length must be positive.")

    if num_samples <= 1:
        raise ValueError("num_samples must exceed one.")

    if sample_frequency is None:
        samples_per_annotation = (
            num_samples / annotation_length
        )
    else:
        samples_per_annotation = float(
            sample_frequency
        )

        if (
            not np.isfinite(samples_per_annotation)
            or samples_per_annotation <= 0
        ):
            raise ValueError(
                "sample_frequency must be positive and finite."
            )

    mapped = [
        int(round(
            int(change_point)
            * samples_per_annotation
        ))
        for change_point in change_points
    ]

    return np.asarray(
        sorted(
            set(
                change_point
                for change_point in mapped
                if 0 < change_point < num_samples
            )
        ),
        dtype=int,
    )


def read_edf_time_series(
    edf_path,
    channel_labels: Optional[Sequence[str]] = None,
):
    """Read an EDF recording as a time-by-channel NumPy array."""
    try:
        from pyedflib import highlevel
    except ImportError as exc:
        raise ImportError(
            "Reading seizure EEG files requires pyedflib."
        ) from exc

    edf_path = Path(edf_path)

    if not edf_path.is_file():
        raise FileNotFoundError(
            f"EDF file was not found: {edf_path}"
        )

    eeg, signal_headers, header = highlevel.read_edf(
        str(edf_path)
    )

    eeg = np.asarray(
        eeg,
        dtype=float,
    )

    if eeg.ndim != 2:
        raise ValueError(
            "EDF signals must have shape (channels, samples). "
            f"Got {eeg.shape}."
        )

    labels = [
        str(signal_header.get("label", index))
        for index, signal_header in enumerate(signal_headers)
    ]

    sample_frequencies = np.asarray([
        float(signal_header["sample_frequency"])
        for signal_header in signal_headers
    ])

    if not np.allclose(
        sample_frequencies,
        sample_frequencies[0],
    ):
        raise ValueError(
            "All selected EDF channels must have the same "
            "sample frequency."
        )

    if channel_labels is not None:
        missing = [
            label
            for label in channel_labels
            if label not in labels
        ]

        if missing:
            raise ValueError(
                f"EDF channels were not found: {missing}."
            )

        indices = [
            labels.index(label)
            for label in channel_labels
        ]
        eeg = eeg[indices]
        labels = [labels[index] for index in indices]

    if not np.all(np.isfinite(eeg)):
        raise ValueError("EDF signals contain NaN or infinite values.")

    return (
        eeg.T,
        float(sample_frequencies[0]),
        labels,
        header,
    )


def single_changepoint_sections(
    num_samples,
    change_points,
):
    """Return sections containing one target changepoint each."""
    change_points = np.asarray(
        change_points,
        dtype=int,
    )

    if np.any(change_points <= 0) or np.any(
        change_points >= num_samples
    ):
        raise ValueError(
            "Every changepoint must lie strictly inside the series."
        )

    if np.any(np.diff(change_points) <= 0):
        raise ValueError(
            "change_points must be sorted and unique."
        )

    sections = []

    for index, change_point in enumerate(change_points):
        start = (
            0
            if index == 0
            else int(change_points[index - 1]) + 1
        )
        end = (
            num_samples
            if index == len(change_points) - 1
            else int(change_points[index + 1])
        )

        sections.append({
            "section_number": index + 1,
            "start": start,
            "end": end,
            "tau_star": int(change_point) - start,
        })

    return sections


def load_seizure_eeg(
    raw_data_dir,
    patient_number,
    min_annotators=2,
    channel_labels=None,
):
    """Load one EEG recording and its seizure changepoints."""
    raw_data_dir = Path(raw_data_dir)
    edf_path = raw_data_dir / f"eeg{patient_number}.edf"

    X, sample_frequency, labels, header = (
        read_edf_time_series(
            edf_path,
            channel_labels=channel_labels,
        )
    )

    annotation = read_seizure_annotations(
        raw_data_dir,
        patient_number,
        min_annotators=min_annotators,
    )
    annotation_points = annotation_changepoints(
        annotation
    )
    change_points = map_annotation_changepoints_to_samples(
        annotation_points,
        annotation_length=len(annotation),
        num_samples=len(X),
        sample_frequency=sample_frequency,
    )

    return {
        "time_series": X,
        "annotation": annotation,
        "annotation_changepoints": annotation_points,
        "change_points": change_points,
        "sections": single_changepoint_sections(
            len(X),
            change_points,
        ),
        "sample_frequency": sample_frequency,
        "channel_labels": labels,
        "edf_header": header,
    }
