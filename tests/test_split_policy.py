"""Physical split orientation and experiment-policy integration."""

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cpd.design import CompetitorConfig, expand_cross_regime_variants
from cpd.methods import cross_regime


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def competitor(**kwargs):
    return CompetitorConfig(
        model_choice="VARModel",
        score_direction="forward_backward",
        epsilon=1e-10,
        model_lag_order=2,
        window_size=20,
        **kwargs,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_inner_training_and_outer_evaluation_in_physical_time(monkeypatch, reverse):
    models = []

    class RecordingModel:
        fit_mode = "adaptive"

        def fit(self, examples, labels, validation_data):
            self.training = labels[:, 0].copy()
            self.validation = validation_data[1][:, 0].copy()
            self.tests = []

        def test(self, examples, labels):
            self.tests.append(labels[:, 0].copy())
            return labels, 1.0

    def make_model(**kwargs):
        model = RecordingModel()
        models.append(model)
        return model

    monkeypatch.setattr(cross_regime, "make_model", make_model)
    cross_regime.estimate_causal_cpd_log_normalized_cross_regime(
        series=np.arange(80.0).reshape(-1, 1),
        window_size=20,
        L=2,
        candidates=[40],
        competitor=competitor(),
        split_train_validation=True,
        split_policy="inner_train",
        reverse_right_to_left=reverse,
    )
    left, right = models
    np.testing.assert_array_equal(left.training, np.arange(27, 40))
    np.testing.assert_array_equal(left.validation, np.arange(22, 27))
    np.testing.assert_array_equal(left.tests[0], np.arange(55, 60))
    np.testing.assert_array_equal(left.tests[1], left.validation)
    if reverse:
        np.testing.assert_array_equal(right.training, np.arange(52, 39, -1))
        np.testing.assert_array_equal(right.validation, np.arange(57, 52, -1))
        np.testing.assert_array_equal(right.tests[0], np.arange(24, 19, -1))
    else:
        np.testing.assert_array_equal(right.training, np.arange(42, 55))
        np.testing.assert_array_equal(right.validation, np.arange(55, 60))
        np.testing.assert_array_equal(right.tests[0], np.arange(22, 27))
    np.testing.assert_array_equal(right.tests[1], right.validation)


@pytest.mark.parametrize("policy", ["chronological", "inner_train"])
def test_forward_score_is_independent_of_reversal(policy):
    series = np.random.default_rng(42).normal(size=(100, 2))
    kwargs = dict(
        series=series, window_size=20, L=2, candidates=[30, 40, 50, 60],
        competitor=replace(competitor(), score_direction="forward_only"),
        split_train_validation=True, split_policy=policy,
    )
    assert cross_regime.estimate_causal_cpd_log_normalized_cross_regime(
        **kwargs, reverse_right_to_left=False,
    ) == cross_regime.estimate_causal_cpd_log_normalized_cross_regime(
        **kwargs, reverse_right_to_left=True,
    )


def test_policy_validation_and_unsplit_deduplication():
    variants = [("forward_only", False, False), ("forward_only", True, False)]
    expanded = expand_cross_regime_variants(
        variants, ["chronological", "inner_train"], default_split=True,
    )
    assert len(expanded) == 3
    assert expanded[0] == ("forward_only", False, False, "chronological")
    with pytest.raises(ValueError, match="split_policies"):
        expand_cross_regime_variants(variants, ["typo"], True)
    with pytest.raises(ValueError, match="split_policy"):
        cross_regime.estimate_causal_cpd_log_normalized_cross_regime(
            np.zeros((80, 1)), 20, 2, [40], competitor(), split_policy="typo",
        )


@pytest.mark.parametrize("module_name", ["run_causal", "run_spectral", "run_seizure_eeg"])
def test_script_sweeps_reuse_variants_for_all_models(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "MODEL_CHOICES", ["VARModel", "MLPModel"])
    monkeypatch.setattr(module, "FIT_MODE", "fixed")
    monkeypatch.setattr(module, "SWEEP_CROSS_REGIME_VARIANTS", True)
    monkeypatch.setattr(module, "SWEEP_WITHOUT_WITHIN_REGIME_DIVISION", False)
    monkeypatch.setattr(module, "CROSS_REGIME_SPLIT_OPTIONS", [False, True])
    monkeypatch.setattr(module, "CROSS_REGIME_SPLIT_POLICY_OPTIONS", ["chronological", "inner_train"])
    monkeypatch.setattr(module, "CROSS_REGIME_REVERSAL_OPTIONS", [False])
    direction_setting = "CROSS_REGIME_SCORE_DIRECTIONS" if module_name == "run_seizure_eeg" else "SCORE_DIRECTIONS"
    monkeypatch.setattr(module, direction_setting, ["forward_only"])
    config = SimpleNamespace(model_lag_order=2, window_size=20)
    entries = list(module.configured_competitors(config))
    assert len(entries) == 6
    for model in module.MODEL_CHOICES:
        selected = [c for c in entries if c.model_choice == model]
        assert len(selected) == 3
        assert sum(c.split_policy == "inner_train" for c in selected) == 1
        assert "split_policy=inner_train" in selected[-1].label

    monkeypatch.setattr(module, "SWEEP_CROSS_REGIME_VARIANTS", False)
    monkeypatch.setattr(module, "SPLIT_TRAIN_VALIDATION", True)
    monkeypatch.setattr(module, "SPLIT_POLICY", "inner_train")
    entries = list(module.configured_competitors(config))
    assert all(c.split_policy == "inner_train" for c in entries)
    assert all("split_policy=inner_train" in c.label for c in entries)


def test_hyperparameter_sweep_preserves_policy_in_execution_and_ranking(monkeypatch, tmp_path):
    module = importlib.import_module("hyperparameter_sweep")
    monkeypatch.setattr(module, "SWEEP_CROSS_REGIME_VARIANTS", True)
    monkeypatch.setattr(module, "SWEEP_WITHOUT_WITHIN_REGIME_DIVISION", False)
    monkeypatch.setattr(module, "CROSS_REGIME_SCORE_DIRECTIONS", ["forward_only"])
    monkeypatch.setattr(module, "CROSS_REGIME_SPLIT_OPTIONS", [True])
    monkeypatch.setattr(module, "CROSS_REGIME_REVERSAL_OPTIONS", [False])
    monkeypatch.setattr(module, "CROSS_REGIME_SPLIT_POLICY_OPTIONS", ["chronological", "inner_train"])
    monkeypatch.setattr(module, "get_sweep_grid", lambda: dict(
        sizes=[2], learning_rates=[0.1], batch_sizes=[8], epoch_counts=[1], size_name="hidden_size",
    ))
    monkeypatch.setattr(module, "validate_configuration", lambda: None)
    monkeypatch.setattr(module, "SWEEP_MODE", "plot")
    monkeypatch.setattr(module, "T", 100)
    monkeypatch.setattr(module, "WINDOW_SIZE", 20)
    monkeypatch.setattr(module, "make_dataset_config", lambda seed: SimpleNamespace(tau_star=50, T=100))
    monkeypatch.setattr(module, "generate_dataset", lambda config: np.zeros((100, 1)))
    monkeypatch.setattr(module, "set_model_seed", lambda seed: None)
    monkeypatch.setattr(module, "make_model_parameters", lambda **kwargs: {})
    received = []

    def estimator(**kwargs):
        policy = kwargs["split_policy"]
        assert kwargs["competitor"].split_policy == policy
        received.append(policy)
        tau = 50 if policy == "inner_train" else 40
        return tau, {40: 0.5, 50: 1.0}

    monkeypatch.setattr(module, "estimate_causal_cpd_log_normalized_cross_regime", estimator)
    raw, ranking, scores = module.run_sweep()
    assert received == ["chronological", "inner_train"]
    assert len(raw) == len(ranking) == len(scores) == 2
    assert ranking.iloc[0]["split_policy"] == "inner_train"
    assert {key[-1] for key in scores} == {"chronological", "inner_train"}
    titles = []
    from matplotlib.axes import Axes

    original_set_title = Axes.set_title

    def record_title(self, label, *args, **kwargs):
        titles.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", record_title)
    module.plot_score_curves(scores, tmp_path / "scores.png")
    assert len(titles) == 2
    assert all("ERROR" not in title for title in titles)
    assert any("split_policy=inner_train" in title for title in titles)

    monkeypatch.setattr(module, "SWEEP_CROSS_REGIME_VARIANTS", False)
    monkeypatch.setattr(module, "SPLIT_POLICY", "inner_train")
    monkeypatch.setattr(module, "SPLIT_TRAIN_VALIDATION", True)
    monkeypatch.setattr(module, "REVERSE_RIGHT_TO_LEFT", True)
    variants = list(module.configured_cross_regime_variants())
    assert variants[0][0][1:] == (True, True, "inner_train")


def test_runner_forwards_competitor_policy(monkeypatch):
    from cpd import runner

    received = {}

    def estimator(**kwargs):
        received.update(kwargs)
        return 40, {40: 1.0}

    monkeypatch.setattr(runner, "estimate_causal_cpd_log_normalized_cross_regime", estimator)
    monkeypatch.setattr(runner, "save_score_curve_plot", lambda **kwargs: None)
    monkeypatch.setattr(runner, "save_score_values", lambda **kwargs: None)
    config = SimpleNamespace(
        experiment_kind="causal", cross_regime_kwargs={"split_policy": "chronological"},
        verbose_candidate_progress=False, plot_config=None, saving_config=None,
        print_scores=False,
    )
    result = runner.evaluate_one_competitor(
        X=np.zeros((80, 1)), dataset_config=SimpleNamespace(tau_star=40, T=80, label="test"),
        competitor=competitor(split_train_validation=True, split_policy="inner_train"),
        series_domain="time", candidates=[40], runner_config=config,
        output_paths=SimpleNamespace(score_plots_dir=None, score_values_dir=None),
    )
    assert result.status == "OK"
    assert received["split_policy"] == "inner_train"
    assert "split_policy=inner_train" in result.competitor_label
