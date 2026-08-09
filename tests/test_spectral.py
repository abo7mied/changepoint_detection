"""Tests for spectral-mixture data generation."""

import numpy as np
import pytest

from cpd.data.spectral import (
    controlled_spectral_mixture,
    simulate_frequency_band,
    spectral_band_limits,
    validate_spectral_coefficients,
)


def test_spectral_band_limits():
    assert spectral_band_limits(128.0) == {
        "delta": (0.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
        "gamma": (30.0, 64.0),
    }


def test_frequency_band_matches_original():
    z = simulate_frequency_band(
        f_l=8.0,
        f_h=12.0,
        f_s=128.0,
        T=128,
        M=2.0,
        sigma=1.0,
        filter_order=4,
        burn_in=64,
        seed=123,
    )

    expected_start = np.array([
        0.28776623404391344,
        0.03245260572480265,
        -0.21901017233395126,
        -0.40420231225747527,
        -0.481153204266986,
        -0.43869544891423873,
        -0.2977831899136095,
        -0.10369451347125262,
        0.0883278030058695,
        0.2295708709861787,
        0.29150374441893206,
        0.27255840070983395,
    ])

    np.testing.assert_allclose(
        z[:12],
        expected_start,
        rtol=0,
        atol=1e-12,
    )

    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=0) - 1.0) < 1e-12


def test_controlled_spectral_mixture_matches_original():
    coefficients = _spectral_coefficients()

    X, details = controlled_spectral_mixture(
        num_variables=2,
        num_timepoints=128,
        num_regimes=2,
        change_points=[64],
        regime_coefficients=coefficients,
        f_s=128.0,
        M=2.0,
        filter_order=4,
        burn_in=64,
        seed=123,
        return_details=True,
    )

    expected_start = np.array([
        [0.22908199902746437, -0.14192667409419832],
        [0.6926499065624664, 0.3898470417661635],
        [0.24577420608409314, 0.540695487921192],
        [0.32672913695152805, 1.2285826144377858],
        [0.05748728432181273, 0.9911380326622543],
        [0.19297380230207037, 0.9603117620543663],
        [-0.1047204073615338, 0.3075706810412963],
        [0.19645108720326, 0.12421741305547093],
    ])

    expected_around_changepoint = np.array([
        [-0.1214673341083448, -0.483967700369079],
        [-0.38012417652649744, -0.6664370022425606],
        [-0.4553588105416752, -0.39199991373311555],
        [-0.7210939970638982, 0.050301875793378956],
        [-0.49839348792006133, 0.2850555920110653],
    ])

    np.testing.assert_allclose(
        X[:8],
        expected_start,
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        X[62:67],
        expected_around_changepoint,
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        details["regime_index"][:64],
        np.zeros(64, dtype=int),
    )

    np.testing.assert_array_equal(
        details["regime_index"][64:],
        np.ones(64, dtype=int),
    )

    np.testing.assert_array_equal(
        details["regime_coefficients"],
        coefficients,
    )

    reconstructed = (
        details["spectral_signal"]
        + details["scaled_noise"]
    )

    np.testing.assert_allclose(
        reconstructed,
        X,
        rtol=0,
        atol=1e-15,
    )


def test_return_details_switch():
    coefficients = _spectral_coefficients()

    X = controlled_spectral_mixture(
        num_variables=2,
        num_timepoints=128,
        num_regimes=2,
        change_points=[64],
        regime_coefficients=coefficients,
        f_s=128.0,
        M=2.0,
        filter_order=4,
        burn_in=64,
        seed=123,
        return_details=False,
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (128, 2)


def test_spectral_coefficients_are_not_normalized():
    coefficients = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
    ])

    validated = validate_spectral_coefficients(
        coefficients,
        require_unit_total=False,
    )

    np.testing.assert_array_equal(
        validated,
        coefficients,
    )


def test_unit_total_validation():
    valid = np.array([
        [0.2, 0.2, 0.2, 0.15, 0.15, 0.1],
    ])

    validate_spectral_coefficients(
        valid,
        require_unit_total=True,
    )

    invalid = valid.copy()
    invalid[0, 5] = 0.2

    with pytest.raises(
        ValueError,
        match="must sum to 1",
    ):
        validate_spectral_coefficients(
            invalid,
            require_unit_total=True,
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([[0.2, 0.2, 0.2]]),
        np.array([[0.2, 0.2, np.nan, 0.2, 0.1, 0.1]]),
        np.array([[0.2, 0.2, -0.1, 0.3, 0.2, 0.2]]),
    ],
)
def test_invalid_spectral_coefficients(coefficients):
    with pytest.raises(ValueError):
        validate_spectral_coefficients(coefficients)


def test_sampling_rate_must_include_gamma():
    with pytest.raises(
        ValueError,
        match="must exceed 60 Hz",
    ):
        spectral_band_limits(60.0)


def _spectral_coefficients():
    return np.array([
        [
            [0.20, 0.20, 0.20, 0.15, 0.15, 0.10],
            [0.10, 0.20, 0.30, 0.15, 0.15, 0.10],
        ],
        [
            [0.10, 0.15, 0.20, 0.20, 0.25, 0.10],
            [0.20, 0.15, 0.15, 0.20, 0.20, 0.10],
        ],
    ])