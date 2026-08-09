"""Spectral-mixture data generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from cpd.data.var import validate_change_points


try:
    from scipy.signal import butter, sosfiltfilt
except ImportError:
    butter = None
    sosfiltfilt = None


# Domains

SPECTRAL_BANDS = {
    "delta": (0.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}


# Functions

def simulate_frequency_band(
    f_l: float,
    f_h: float,
    f_s: float,
    T: int = 2000,
    M: float = 2.0,
    sigma: float = 1.0,
    filter_order: int = 4,
    burn_in: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate, filter, and standardize an AR(2) frequency band."""
    if butter is None or sosfiltfilt is None:
        raise ImportError(
            "Spectral-mixture generation requires "
            "scipy.signal.butter and scipy.signal.sosfiltfilt."
        )

    nyquist = f_s / 2.0

    if not (0 <= f_l < f_h <= nyquist):
        raise ValueError(
            f"Need 0 <= f_l < f_h <= Nyquist={nyquist}."
        )

    if M <= 1:
        raise ValueError(
            "M must be greater than 1 for a stable AR(2) process."
        )

    if T <= 1:
        raise ValueError("T must exceed 1.")

    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    f_0 = (f_l + f_h) / 2.0
    psi = f_0 / f_s

    phi_1 = (
        2.0
        * np.cos(2.0 * np.pi * psi)
        / M
    )
    phi_2 = -1.0 / M**2

    rng = np.random.default_rng(seed)
    n = T + burn_in

    innovations = rng.normal(
        0.0,
        sigma,
        size=n,
    )
    z = np.zeros(n)

    for t in range(2, n):
        z[t] = (
            phi_1 * z[t - 1]
            + phi_2 * z[t - 2]
            + innovations[t]
        )

    z = z[burn_in:]

    if f_l == 0:
        sos = butter(
            filter_order,
            f_h,
            btype="lowpass",
            fs=f_s,
            output="sos",
        )
    elif np.isclose(f_h, nyquist):
        sos = butter(
            filter_order,
            f_l,
            btype="highpass",
            fs=f_s,
            output="sos",
        )
    else:
        sos = butter(
            filter_order,
            [f_l, f_h],
            btype="bandpass",
            fs=f_s,
            output="sos",
        )

    z_band = sosfiltfilt(sos, z)
    z_std = float(z_band.std(ddof=0))

    if not np.isfinite(z_std) or z_std <= 0:
        raise RuntimeError(
            f"The simulated [{f_l}, {f_h}] Hz band "
            "has invalid standard deviation."
        )

    return (z_band - z_band.mean()) / z_std


def spectral_band_limits(
    f_s: float,
) -> Dict[str, Tuple[float, float]]:
    """Return the five simulated frequency intervals."""
    if f_s <= 60:
        raise ValueError(
            "f_s must exceed 60 Hz to include a gamma band."
        )

    return {
        **SPECTRAL_BANDS,
        "gamma": (30.0, f_s / 2.0),
    }


def validate_spectral_coefficients(
    coefficients: np.ndarray,
    name: str = "coefficients",
    require_unit_total: bool = False,
) -> np.ndarray:
    """Validate spectral coefficients without renormalizing them."""
    C = np.asarray(
        coefficients,
        dtype=float,
    ).copy()

    if C.shape[-1] != 6:
        raise ValueError(
            f"{name} must have six entries along its last axis, "
            "ordered as "
            "[delta, theta, alpha, beta, gamma, noise]."
        )

    if not np.all(np.isfinite(C)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if np.any(C < 0):
        raise ValueError(
            f"All band and noise coefficients in {name} "
            "must be nonnegative."
        )

    if require_unit_total:
        totals = C.sum(axis=-1)

        if not np.allclose(
            totals,
            1.0,
            atol=1e-12,
            rtol=0.0,
        ):
            bad_index = tuple(
                np.argwhere(
                    ~np.isclose(
                        totals,
                        1.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                )[0]
            )

            raise ValueError(
                f"Every regime-variable row in {name} "
                "must sum to 1 across "
                "[delta, theta, alpha, beta, gamma, noise]. "
                f"First invalid row {bad_index} has total "
                f"{totals[bad_index]:g}."
            )

    return C


def simulate_spectral_mixture(
    coefficients: Sequence[Sequence[float]],
    f_s: float,
    T: int = 2000,
    M: float = 2.0,
    filter_order: int = 4,
    burn_in: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate one spectral-mixture regime."""
    C = np.asarray(coefficients, dtype=float)

    if C.ndim == 1:
        C = C[None, :]

    if C.ndim != 2 or C.shape[1] != 6:
        raise ValueError(
            "coefficients must have shape (d, 6), "
            "with each row ordered as "
            "[delta, theta, alpha, beta, gamma, noise]."
        )

    input_coefficients = C.copy()

    C = validate_spectral_coefficients(
        C,
        name="coefficients",
    )

    d = C.shape[0]
    rng = np.random.default_rng(seed)
    band_limits = spectral_band_limits(f_s)

    band_sources = {
        band_name: np.zeros((T, d))
        for band_name in band_limits
    }

    for j in range(d):
        for band_name, (f_l, f_h) in band_limits.items():
            band_seed = int(
                rng.integers(
                    0,
                    np.iinfo(np.uint32).max,
                )
            )

            band_sources[band_name][:, j] = (
                simulate_frequency_band(
                    f_l=f_l,
                    f_h=f_h,
                    f_s=f_s,
                    T=T,
                    M=M,
                    sigma=1.0,
                    filter_order=filter_order,
                    burn_in=burn_in,
                    seed=band_seed,
                )
            )

    weighted_components: Dict[str, np.ndarray] = {}
    spectral_signal = np.zeros((T, d))

    for b, band_name in enumerate(band_limits):
        component = (
            band_sources[band_name]
            * C[None, :, b]
        )
        weighted_components[band_name] = component
        spectral_signal += component

    standard_noise = rng.normal(size=(T, d))
    scaled_noise = (
        standard_noise
        * C[None, :, 5]
    )
    X = spectral_signal + scaled_noise

    details: Dict[str, Any] = {
        "data_generating_process": "spectral_mixture",
        "band_sources": band_sources,
        "weighted_components": weighted_components,
        "spectral_signal": spectral_signal,
        "standard_noise": standard_noise,
        "scaled_noise": scaled_noise,
        "input_coefficient_matrix": input_coefficients,
        "coefficient_matrix": C,
    }

    return X, details


def controlled_spectral_mixture(
    num_variables: int,
    num_timepoints: int,
    num_regimes: int,
    change_points: Sequence[int],
    regime_coefficients: Sequence[
        Sequence[Sequence[float]]
    ],
    f_s: float = 512.0,
    M: float = 2.0,
    filter_order: int = 4,
    burn_in: int = 1000,
    seed: Optional[int] = None,
    return_details: bool = False,
):
    """Simulate a multiregime spectral-mixture process."""
    change_points_array = np.asarray(
        change_points,
        dtype=int,
    )
    coefficients = np.asarray(
        regime_coefficients,
        dtype=float,
    )

    validate_change_points(
        num_timepoints=num_timepoints,
        num_regimes=num_regimes,
        change_points=change_points_array.tolist(),
    )

    expected_shape = (
        num_regimes,
        num_variables,
        6,
    )

    if coefficients.shape != expected_shape:
        raise ValueError(
            "regime_coefficients must have shape "
            f"{expected_shape}, but received "
            f"{coefficients.shape}."
        )

    input_coefficients = coefficients.copy()

    coefficients = validate_spectral_coefficients(
        coefficients,
        name="regime_coefficients",
    )

    rng = np.random.default_rng(seed)
    band_limits = spectral_band_limits(f_s)

    band_sources = {
        band_name: np.zeros(
            (num_timepoints, num_variables)
        )
        for band_name in band_limits
    }

    for band_name, (f_l, f_h) in band_limits.items():
        for j in range(num_variables):
            source_seed = int(
                rng.integers(
                    0,
                    np.iinfo(np.uint32).max,
                )
            )

            band_sources[band_name][:, j] = (
                simulate_frequency_band(
                    f_l=f_l,
                    f_h=f_h,
                    f_s=f_s,
                    T=num_timepoints,
                    M=M,
                    sigma=1.0,
                    filter_order=filter_order,
                    burn_in=burn_in,
                    seed=source_seed,
                )
            )

    standard_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            num_timepoints,
            num_variables,
        ),
    )

    regime_index = np.searchsorted(
        change_points_array,
        np.arange(num_timepoints),
        side="right",
    )

    spectral_signal = np.zeros(
        (num_timepoints, num_variables)
    )
    scaled_noise = np.zeros_like(spectral_signal)

    weighted_components = {
        band_name: np.zeros_like(spectral_signal)
        for band_name in band_limits
    }

    for regime in range(num_regimes):
        mask = regime_index == regime
        C_r = coefficients[regime]

        for b, band_name in enumerate(band_limits):
            component = (
                band_sources[band_name][mask]
                * C_r[None, :, b]
            )
            weighted_components[band_name][mask] = (
                component
            )
            spectral_signal[mask] += component

        scaled_noise[mask] = (
            standard_noise[mask]
            * C_r[None, :, 5]
        )

    X = spectral_signal + scaled_noise

    if not return_details:
        return X

    details: Dict[str, Any] = {
        "data_generating_process": "spectral_mixture",
        "band_sources": band_sources,
        "weighted_components": weighted_components,
        "spectral_signal": spectral_signal,
        "standard_noise": standard_noise,
        "scaled_noise": scaled_noise,
        "regime_index": regime_index,
        "change_points": change_points_array,
        "input_regime_coefficients": input_coefficients,
        "regime_coefficients": coefficients,
    }

    return X, details