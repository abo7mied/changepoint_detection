import math

import numpy as np
import pytest

from cpd.design import CompetitorConfig
from cpd.methods.cross_regime import (
    compute_log_normalized_score,
    estimate_causal_cpd_log_normalized_cross_regime,
)
from cpd.methods.kmo import (
    compute_kmo_varscore_curve,
    estimate_kmo_varscore_amoc,
    make_flat_var_design_for_kmo,
)

import cpd.methods.cross_regime as cross_regime


# --- 1: Testing Methods on CPU ---

def make_piecewise_var(seed=42):
    rng = np.random.default_rng(seed)

    T = 120
    tau = 60
    X = np.zeros((T, 2))

    for t in range(1, T):
        coefficient = 0.75 if t < tau else -0.55

        X[t, 0] = (
            coefficient * X[t - 1, 0]
            + 0.15 * X[t - 1, 1]
            + rng.normal(scale=0.2)
        )
        X[t, 1] = (
            coefficient * X[t - 1, 1]
            - 0.10 * X[t - 1, 0]
            + rng.normal(scale=0.2)
        )

    return X


def make_cross_regime_competitor(
    score_direction="forward_backward",
):
    return CompetitorConfig(
        model_choice="VARModel",
        score_direction=score_direction,
        epsilon=1e-8,
        model_lag_order=1,
        window_size=30,
    )


def make_kmo_competitor(
    h_choice="diag_inv_cov",
):
    return CompetitorConfig(
        model_choice="KMOVARScore",
        score_direction=f"kmo_{h_choice}",
        epsilon=1e-8,
        model_lag_order=1,
        window_size=30,
    )


def test_forward_only_score():
    score = compute_log_normalized_score(
        left_on_right_error=4.0,
        left_on_left_error=2.0,
        right_on_left_error=9.0,
        right_on_right_error=3.0,
        score_direction="forward_only",
        epsilon=1e-8,
        use_epsilon=True,
    )

    expected = math.log(
        (4.0 + 1e-8) / (2.0 + 1e-8)
    )

    assert score == pytest.approx(expected)


def test_forward_backward_score():
    score = compute_log_normalized_score(
        left_on_right_error=4.0,
        left_on_left_error=2.0,
        right_on_left_error=9.0,
        right_on_right_error=3.0,
        score_direction="forward_backward",
        epsilon=1e-8,
        use_epsilon=True,
    )

    expected = (
        math.log((4.0 + 1e-8) / (2.0 + 1e-8))
        + math.log((9.0 + 1e-8) / (3.0 + 1e-8))
    )

    assert score == pytest.approx(expected)


def test_score_without_epsilon():
    score = compute_log_normalized_score(
        left_on_right_error=4.0,
        left_on_left_error=2.0,
        right_on_left_error=9.0,
        right_on_right_error=3.0,
        score_direction="forward_backward",
        epsilon=1e-8,
        use_epsilon=False,
    )

    expected = math.log(4.0 / 2.0) + math.log(9.0 / 3.0)

    assert score == pytest.approx(expected)


def test_default_cross_regime_options_match_explicit_defaults():
    X = make_piecewise_var()
    candidates = [45, 60, 75]
    competitor = make_cross_regime_competitor()

    default_tau, default_scores = (
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=30,
            L=1,
            candidates=candidates,
            competitor=competitor,
            device='cpu'
        )
    )

    explicit_tau, explicit_scores = (
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=30,
            L=1,
            candidates=candidates,
            competitor=competitor,
            use_epsilon=True,
            reverse_right_to_left=True,
            split_train_validation=False,
            device='cpu'
        )
    )

    assert default_tau == explicit_tau
    assert default_scores == pytest.approx(explicit_scores)


@pytest.mark.parametrize(
    (
        "score_direction",
        "use_epsilon",
        "reverse_right_to_left",
        "split_train_validation",
    ),
    [
        ("forward_only", True, True, False),
        ("forward_backward", True, True, False),
        ("forward_backward", False, True, False),
        ("forward_backward", True, False, False),
        ("forward_backward", True, True, True),
        ("forward_backward", False, False, True),
    ],
)
def test_cross_regime_options_produce_valid_scores(
    score_direction,
    use_epsilon,
    reverse_right_to_left,
    split_train_validation,
):
    X = make_piecewise_var()
    candidates = [45, 60, 75]

    competitor = make_cross_regime_competitor(
        score_direction=score_direction
    )

    tau_hat, scores = (
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=30,
            L=1,
            candidates=candidates,
            competitor=competitor,
            use_epsilon=use_epsilon,
            reverse_right_to_left=reverse_right_to_left,
            split_train_validation=split_train_validation,
            device='cpu'
        )
    )

    assert tau_hat in candidates
    assert set(scores) == set(candidates)
    assert all(
        np.isfinite(score)
        for score in scores.values()
    )


def test_cross_regime_is_deterministic_for_var():
    X = make_piecewise_var()
    candidates = [45, 60, 75]
    competitor = make_cross_regime_competitor()

    first_tau, first_scores = (
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=30,
            L=1,
            candidates=candidates,
            competitor=competitor,
            device='cpu'
        )
    )
    second_tau, second_scores = (
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=30,
            L=1,
            candidates=candidates,
            competitor=competitor,
            device='cpu'
        )
    )

    assert first_tau == second_tau
    assert first_scores == pytest.approx(second_scores)


def test_cross_regime_rejects_empty_candidates():
    X = make_piecewise_var()
    competitor = make_cross_regime_competitor()

    with pytest.raises(ValueError, match="candidates"):
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=30,
            L=1,
            candidates=[],
            competitor=competitor,
            device='cpu'
        )


def test_cross_regime_rejects_window_not_larger_than_lag():
    X = make_piecewise_var()
    competitor = make_cross_regime_competitor()

    with pytest.raises(ValueError, match="window_size"):
        estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=1,
            L=1,
            candidates=[60],
            competitor=competitor,
            device='cpu'
        )


def test_kmo_flat_design_order():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    Phi, Y, target_times = make_flat_var_design_for_kmo(
        X=X,
        L=2,
        include_intercept=False,
    )

    expected_Phi = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )
    expected_Y = np.array(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    np.testing.assert_array_equal(Phi, expected_Phi)
    np.testing.assert_array_equal(Y, expected_Y)
    np.testing.assert_array_equal(
        target_times,
        np.array([2, 3]),
    )


def test_kmo_flat_design_with_intercept():
    X = np.arange(12, dtype=float).reshape(6, 2)

    Phi, _, _ = make_flat_var_design_for_kmo(
        X=X,
        L=2,
        include_intercept=True,
    )

    assert Phi.shape == (4, 5)
    np.testing.assert_array_equal(
        Phi[:, 0],
        np.ones(4),
    )


@pytest.mark.parametrize(
    "h_choice",
    [
        "identity",
        "diag_inv_cov",
        "pinv_cov",
    ],
)
def test_kmo_score_choices(h_choice):
    X = make_piecewise_var()
    candidates = [30, 45, 60, 75, 90]

    tau_hat, scores = compute_kmo_varscore_curve(
        series=X,
        L=1,
        candidates=candidates,
        h_choice=h_choice,
        include_intercept=True,
        ridge=1e-8,
    )

    assert tau_hat in candidates
    assert set(scores) == set(candidates)
    assert all(
        np.isfinite(score)
        for score in scores.values()
    )
    assert tau_hat == max(scores, key=scores.get)


def test_kmo_wrapper_matches_direct_function():
    X = make_piecewise_var()
    candidates = [30, 45, 60, 75, 90]
    competitor = make_kmo_competitor("diag_inv_cov")

    wrapper_tau, wrapper_scores = (
        estimate_kmo_varscore_amoc(
            series=X,
            L=1,
            candidates=candidates,
            competitor=competitor,
            include_intercept=True,
            ridge=1e-8,
        )
    )

    direct_tau, direct_scores = (
        compute_kmo_varscore_curve(
            series=X,
            L=1,
            candidates=candidates,
            h_choice="diag_inv_cov",
            include_intercept=True,
            ridge=1e-8,
        )
    )

    assert wrapper_tau == direct_tau
    assert wrapper_scores == pytest.approx(direct_scores)


def test_kmo_rejects_unknown_h_choice():
    X = make_piecewise_var()

    with pytest.raises(ValueError, match="H choice"):
        compute_kmo_varscore_curve(
            series=X,
            L=1,
            candidates=[45, 60, 75],
            h_choice="unknown",
            include_intercept=True,
            ridge=1e-8,
        )


def test_kmo_wrapper_requires_kmo_direction():
    X = make_piecewise_var()
    competitor = make_cross_regime_competitor()

    with pytest.raises(ValueError, match="kmo_"):
        estimate_kmo_varscore_amoc(
            series=X,
            L=1,
            candidates=[45, 60, 75],
            competitor=competitor,
            include_intercept=True,
            ridge=1e-8,
        )


# --- 2: Testing Device Forwarding ---


def test_cross_regime_forwards_device(monkeypatch):
    devices = []

    class DummyModel:
        def fit(self, X, Y):
            pass

        def test(self, X, Y):
            return np.zeros_like(Y), 1.0

    def fake_make_model(model_choice, L, device):
        devices.append(device)
        return DummyModel()

    monkeypatch.setattr(
        cross_regime,
        "make_model",
        fake_make_model,
    )

    X = make_piecewise_var()

    competitor = make_cross_regime_competitor(
        score_direction="forward_backward"
    )

    tau_hat, scores = (
        cross_regime
        .estimate_causal_cpd_log_normalized_cross_regime(
            series=X,
            window_size=30,
            L=1,
            candidates=[45, 60, 75],
            competitor=competitor,
            device="cuda",
        )
    )

    assert tau_hat in scores
    assert devices
    assert all(
        device == "cuda"
        for device in devices
    )