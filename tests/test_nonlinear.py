"""Tests for nonlinear synthetic datasets."""

import numpy as np
import pytest

from cpd.data.nonlinear import (
    sign_flip_quadratic_dataset,
)


def test_sign_flip_quadratic_shape():
    X = sign_flip_quadratic_dataset(
        num_variables=5,
        num_timepoints=100,
        change_point=50,
        coefficient=3.0,
        noise_std_dev=0.1,
        rng=np.random.default_rng(42),
    )

    assert X.shape == (100, 5)
    assert np.all(np.isfinite(X))


def test_sign_flip_quadratic_reproducibility():
    parameters = {
        "num_variables": 6,
        "num_timepoints": 100,
        "change_point": 50,
        "coefficient": 3.0,
        "noise_std_dev": 0.1,
    }

    X_1 = sign_flip_quadratic_dataset(
        **parameters,
        rng=np.random.default_rng(42),
    )

    X_2 = sign_flip_quadratic_dataset(
        **parameters,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(
        X_1,
        X_2,
    )


def test_sign_flip_quadratic_mechanism():
    num_variables = 5
    num_timepoints = 100
    change_point = 50
    coefficient = 3.0

    X = sign_flip_quadratic_dataset(
        num_variables=num_variables,
        num_timepoints=num_timepoints,
        change_point=change_point,
        coefficient=coefficient,
        noise_std_dev=0.0,
        rng=np.random.default_rng(42),
    )

    num_response_variables = (
        num_variables // 2
    )

    num_driver_variables = (
        num_variables
        - num_response_variables
    )

    drivers = X[
        :,
        :num_driver_variables,
    ]

    responses = X[
        :,
        num_driver_variables:,
    ]

    quadratic_signal = (
        drivers[
            :-1,
            :num_response_variables,
        ]
        ** 2
        - 1.0 / 3.0
    )

    left_end = change_point - 1

    np.testing.assert_allclose(
        responses[1:change_point],
        (
            coefficient
            * quadratic_signal[:left_end]
        ),
    )

    np.testing.assert_allclose(
        responses[change_point:],
        (
            -coefficient
            * quadratic_signal[left_end:]
        ),
    )


def test_sign_flip_quadratic_odd_dimension():
    X = sign_flip_quadratic_dataset(
        num_variables=5,
        num_timepoints=20,
        change_point=10,
        coefficient=3.0,
        noise_std_dev=0.1,
        rng=np.random.default_rng(42),
    )

    num_drivers = 3
    num_responses = 2

    assert X[:, :num_drivers].shape == (
        20,
        3,
    )

    assert X[:, num_drivers:].shape == (
        20,
        2,
    )

    assert num_drivers + num_responses == 5


@pytest.mark.parametrize(
    (
        "parameter_name",
        "parameter_value",
    ),
    [
        (
            "num_variables",
            1,
        ),
        (
            "num_timepoints",
            1,
        ),
        (
            "change_point",
            0,
        ),
        (
            "change_point",
            100,
        ),
        (
            "coefficient",
            np.inf,
        ),
        (
            "noise_std_dev",
            -0.1,
        ),
        (
            "noise_std_dev",
            np.inf,
        ),
    ],
)
def test_sign_flip_quadratic_invalid_parameters(
    parameter_name,
    parameter_value,
):
    parameters = {
        "num_variables": 5,
        "num_timepoints": 100,
        "change_point": 50,
        "coefficient": 3.0,
        "noise_std_dev": 0.1,
        "rng": np.random.default_rng(42),
    }

    parameters[parameter_name] = (
        parameter_value
    )

    with pytest.raises(ValueError):
        sign_flip_quadratic_dataset(
            **parameters
        )


def test_sign_flip_quadratic_requires_generator():
    with pytest.raises(
        TypeError,
        match="numpy.random.Generator",
    ):
        sign_flip_quadratic_dataset(
            num_variables=5,
            num_timepoints=100,
            change_point=50,
            coefficient=3.0,
            noise_std_dev=0.1,
            rng=None,
        )