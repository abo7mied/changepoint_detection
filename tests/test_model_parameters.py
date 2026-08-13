"""Tests for model hyperparameter handling."""

from types import SimpleNamespace

import numpy as np

import cpd.methods.cross_regime as cross_regime
from cpd.models import (
    AEModel,
    MLPModel,
    RNNModel,
    make_model,
)


def test_model_constructors_store_custom_parameters():
    mlp = MLPModel(
        L=2,
        device="cpu",
        hidden_size=64,
        lr=0.05,
        epochs=100,
        batch_size=16,
        loss_type="mse",
    )

    assert mlp.L == 2
    assert mlp.device == "cpu"
    assert mlp.hidden_size == 64
    assert mlp.lr == 0.05
    assert mlp.epochs == 100
    assert mlp.batch_size == 16
    assert mlp.loss_type == "mse"

    rnn = RNNModel(
        L=3,
        hidden_size=32,
        lr=0.01,
        epochs=50,
        batch_size=8,
        loss_type="mse",
    )

    assert rnn.L == 3
    assert rnn.hidden_size == 32
    assert rnn.lr == 0.01
    assert rnn.epochs == 50
    assert rnn.batch_size == 8
    assert rnn.loss_type == "mse"

    ae = AEModel(
        L=4,
        device="cpu",
        bottleneck_size=8,
        lr=0.02,
        epochs=75,
        batch_size=32,
        loss_type="mse",
    )

    assert ae.L == 4
    assert ae.device == "cpu"
    assert ae.bottleneck_size == 8
    assert ae.lr == 0.02
    assert ae.epochs == 75
    assert ae.batch_size == 32
    assert ae.loss_type == "mse"


def test_make_model_forwards_custom_parameters():
    model = make_model(
        model_choice="MLPModel",
        L=2,
        device="cpu",
        model_parameters={
            "hidden_size": 64,
            "lr": 0.05,
            "epochs": 100,
            "batch_size": 16,
        },
    )

    assert isinstance(model, MLPModel)
    assert model.hidden_size == 64
    assert model.lr == 0.05
    assert model.epochs == 100
    assert model.batch_size == 16


def test_make_model_preserves_original_defaults():
    mlp = make_model(
        model_choice="MLPModel",
        L=2,
        device="cpu",
    )
    rnn = make_model(
        model_choice="RNNModel",
        L=2,
        device="cpu",
    )
    ae = make_model(
        model_choice="AEModel",
        L=2,
        device="cpu",
    )

    assert mlp.hidden_size == 32
    assert mlp.lr == 0.5
    assert mlp.epochs == 500
    assert mlp.batch_size == 8

    assert rnn.hidden_size == 16
    assert rnn.lr == 0.1
    assert rnn.epochs == 200
    assert rnn.batch_size == 16

    assert ae.bottleneck_size == 4
    assert ae.lr == 0.1
    assert ae.epochs == 200
    assert ae.batch_size == 16


def test_cross_regime_selects_current_model_parameters(
    monkeypatch,
):
    received_calls = []

    class FakeModel:
        def fit(self, X, Y=None):
            pass

        def test(self, X, Y=None):
            return np.zeros_like(Y), 1.0

    def fake_make_model(
        model_choice,
        L,
        device,
        model_parameters=None,
    ):
        received_calls.append(
            {
                "model_choice": model_choice,
                "L": L,
                "device": device,
                "model_parameters": model_parameters,
            }
        )

        return FakeModel()

    monkeypatch.setattr(
        cross_regime,
        "make_model",
        fake_make_model,
    )

    competitor = SimpleNamespace(
        model_choice="MLPModel",
        score_direction="forward_backward",
        epsilon=1e-10,
        label="MLPModel",
    )

    model_parameters = {
        "MLPModel": {
            "hidden_size": 64,
            "lr": 0.05,
            "epochs": 20,
            "batch_size": 8,
        },
        "AEModel": {
            "bottleneck_size": 4,
        },
    }

    series = np.arange(
        40,
        dtype=float,
    ).reshape(20, 2)

    tau_hat, scores = (
        cross_regime
        .estimate_causal_cpd_log_normalized_cross_regime(
            series=series,
            window_size=5,
            L=1,
            candidates=[10],
            competitor=competitor,
            device="cpu",
            model_parameters=model_parameters,
        )
    )

    assert tau_hat == 10
    assert 10 in scores
    assert len(received_calls) == 2

    for call in received_calls:
        assert call["model_choice"] == "MLPModel"
        assert call["L"] == 1
        assert call["device"] == "cpu"
        assert call["model_parameters"] == (
            model_parameters["MLPModel"]
        )