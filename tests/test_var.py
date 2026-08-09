"""Tests for VAR data generation."""

import numpy as np
import pytest

from cpd.data.var import (
    empirical_quantile_match,
    make_causal_adjacency_graphs,
    make_var_lag_coefficient_graphs,
    piecewise_controlled_nonlinear_var_p,
    piecewise_linear_var,
    validate_change_points,
    var_companion_spectral_radius,
)


def test_causal_adjacency_graphs():
    A1, A2 = make_causal_adjacency_graphs(4)

    expected_A1 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ], dtype=float)

    expected_A2 = np.array([
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=float)

    np.testing.assert_array_equal(A1, expected_A1)
    np.testing.assert_array_equal(A2, expected_A2)


def test_var_lag_coefficient_graphs_match_original():
    coefficient_graphs = make_var_lag_coefficient_graphs(
        d=3,
        p=2,
        alpha=0.2,
        beta=0.5,
        rng=np.random.default_rng(123),
        target_radius=0.75,
    )

    expected = [
        [
            np.array([
                [0.40470555897444305, 0, -0.2661079618317834],
                [0.2553115432096009, -0.2527717703255091, 0],
                [0, 0.28297231933913186, 0.4459263684779006],
            ]),
            np.array([
                [-0.21122005783520645, 0, 0.23911108877877504],
                [-0.13478622829096282, -0.21986876929301244, 0],
                [0, -0.12488559898611169, -0.17466834527466907],
            ]),
        ],
        [
            np.array([
                [-0.41847299845497854, 0, 0],
                [0, 0.45940707532813246, 0],
                [0, 0, 0.28335958708967834],
            ]),
            np.array([
                [-0.18748576152265872, 0, 0],
                [0, -0.14691133843761503, 0],
                [0, 0, -0.17024688016398642],
            ]),
        ],
    ]

    for actual_regime, expected_regime in zip(
        coefficient_graphs,
        expected,
    ):
        for actual, target in zip(
            actual_regime,
            expected_regime,
        ):
            np.testing.assert_allclose(
                actual,
                target,
                rtol=0,
                atol=1e-15,
            )

        assert (
            var_companion_spectral_radius(actual_regime)
            <= 0.75
        )


def test_piecewise_linear_var_matches_original():
    coefficient_graphs = _coefficient_graphs()

    X = piecewise_linear_var(
        num_variables=2,
        num_timepoints=8,
        num_regimes=2,
        change_points=[4],
        coefficient_graphs=coefficient_graphs,
        noise_std_dev=0.1,
        rng=np.random.default_rng(123),
    )

    expected = np.array([
        [-0.09891213503478509, -0.03677866514678832],
        [0.09785114538142892, 0.005257168874824988],
        [0.1158540604565248, 0.06518496885056324],
        [-0.019830825824227807, 0.07191732689490743],
        [-0.02676920049629062, 0.006164023523179636],
        [-0.0002575761922526884, -0.1464541297829896],
        [0.08927074567241206, -0.11338816770655263],
        [0.10538533325379629, -0.03665987660227],
    ])

    np.testing.assert_allclose(
        X,
        expected,
        rtol=0,
        atol=1e-15,
    )


def test_piecewise_nonlinear_var_matches_original():
    coefficient_graphs = _coefficient_graphs()

    X = piecewise_controlled_nonlinear_var_p(
        num_variables=2,
        num_timepoints=8,
        num_regimes=2,
        change_points=[4],
        coefficient_graphs=coefficient_graphs,
        control=0.5,
        noise_std_dev=0.1,
        rng=np.random.default_rng(123),
    )

    expected = np.array([
        [-0.09891213503478509, -0.03677866514678832],
        [0.11332677091862005, 0.012327776574645464],
        [0.10628940594675863, 0.06271822312667415],
        [-0.042527862884913725, 0.06247469777080385],
        [-0.031737576211569966, -0.013867162439981005],
        [0.0028263987419641272, -0.1536507603294905],
        [0.1047609021237172, -0.09221033963660535],
        [0.10634517734637808, -0.009121705889268411],
    ])

    np.testing.assert_allclose(
        X,
        expected,
        rtol=0,
        atol=1e-15,
    )


def test_empirical_quantile_match():
    X = np.array([
        [0, 3],
        [10, 2],
        [20, 1],
        [30, 0],
        [100, 6],
        [90, 5],
        [80, 4],
        [70, 7],
    ], dtype=float)

    matched = empirical_quantile_match(
        time_series=X,
        num_regimes=2,
        change_points=[4],
    )

    expected = np.array([
        [0, 3],
        [10, 2],
        [20, 1],
        [30, 0],
        [30, 2],
        [20, 1],
        [10, 0],
        [0, 3],
    ], dtype=float)

    np.testing.assert_array_equal(matched, expected)
    np.testing.assert_array_equal(X[:4], matched[:4])


@pytest.mark.parametrize(
    "num_timepoints,num_regimes,change_points",
    [
        (10, 2, []),
        (10, 3, [5]),
        (10, 3, [7, 4]),
        (10, 2, [0]),
        (10, 2, [10]),
    ],
)
def test_invalid_change_points(
    num_timepoints,
    num_regimes,
    change_points,
):
    with pytest.raises(ValueError):
        validate_change_points(
            num_timepoints,
            num_regimes,
            change_points,
        )


def _coefficient_graphs():
    return [
        [
            np.array([
                [0.35, -0.10],
                [0.05, 0.25],
            ]),
            np.array([
                [0.10, 0],
                [-0.05, 0.10],
            ]),
        ],
        [
            np.array([
                [0.15, 0.20],
                [-0.10, 0.30],
            ]),
            np.array([
                [0, -0.10],
                [0.10, 0.05],
            ]),
        ],
    ]