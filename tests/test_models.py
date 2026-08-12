"""Tests for forecasting and reconstruction models."""

import numpy as np
import pytest

import cpd.models as models
from cpd.models import (
    AEModel,
    MLPModel,
    RNNModel,
    VARModel,
    construct_supervised_prediction_pairs,
    make_model,
)


class FakeMLP:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.train_kwargs = None
        FakeMLP.instances.append(self)

    def train(self, **kwargs):
        self.train_kwargs = kwargs

    def predict(self, X):
        return X[:self.init_kwargs["n_y"]]


class FakeRNN:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.train_kwargs = None
        FakeRNN.instances.append(self)

    def train(self, **kwargs):
        self.train_kwargs = kwargs

    def predict(self, X):
        m = len(X)
        d = self.init_kwargs["d_y"]
        return np.zeros((m, 1, d))


def test_construct_supervised_prediction_pairs():
    series = np.array([
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ], dtype=float)

    examples, labels = construct_supervised_prediction_pairs(
        series,
        L=2,
    )

    expected_examples = np.array([
        [[0, 1], [2, 3]],
        [[2, 3], [4, 5]],
        [[4, 5], [6, 7]],
    ], dtype=float)

    expected_labels = np.array([
        [4, 5],
        [6, 7],
        [8, 9],
    ], dtype=float)

    np.testing.assert_array_equal(
        examples,
        expected_examples,
    )
    np.testing.assert_array_equal(
        labels,
        expected_labels,
    )


@pytest.mark.parametrize(
    "series,L",
    [
        (np.arange(5), 2),
        (np.zeros((5, 2)), 0),
        (np.zeros((2, 2)), 2),
    ],
)
def test_invalid_supervised_prediction_pairs(series, L):
    with pytest.raises(ValueError):
        construct_supervised_prediction_pairs(series, L)


def test_var_model_matches_least_squares():
    X = np.array([
        [[1, 2], [3, 4]],
        [[2, 1], [4, 3]],
        [[1, 3], [2, 4]],
        [[3, 1], [4, 2]],
        [[2, 4], [1, 3]],
    ], dtype=float)

    Y = np.array([
        [1, 2],
        [2, 1],
        [1, 3],
        [3, 1],
        [2, 4],
    ], dtype=float)

    model = VARModel(L=2)
    model.fit(X, Y)

    X_flat = X.reshape(len(X), -1)
    expected_B, *_ = np.linalg.lstsq(
        X_flat,
        Y,
        rcond=None,
    )
    expected_predictions = X_flat @ expected_B
    expected_error = np.mean(
        (expected_predictions - Y) ** 2
    )

    predictions, error = model.test(X, Y)

    np.testing.assert_allclose(
        model.B,
        expected_B,
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        predictions,
        expected_predictions,
        rtol=0,
        atol=1e-15,
    )
    assert error == pytest.approx(expected_error)
    assert isinstance(error, float)


def test_var_model_requires_labels():
    model = VARModel(L=2)
    X = np.zeros((4, 2, 1))

    with pytest.raises(
        ValueError,
        match="VARModel.fit requires labels Y",
    ):
        model.fit(X, None)

    model.fit(X, np.zeros((4, 1)))

    with pytest.raises(
        ValueError,
        match="VARModel.test requires labels Y",
    ):
        model.test(X, None)


def test_mlp_model_matches_original_wrapper(monkeypatch):
    FakeMLP.instances.clear()
    monkeypatch.setattr(models, "MLP", FakeMLP)

    X = np.arange(
        24,
        dtype=float,
    ).reshape(6, 2, 2)

    Y = np.arange(
        12,
        dtype=float,
    ).reshape(6, 2)

    model = MLPModel(
        L=2,
        device="gpu",
    )
    model.fit(X, Y)

    fake = FakeMLP.instances[-1]

    assert fake.init_kwargs == {
        "n_x": 4,
        "n_h": 32,
        "n_y": 2,
        "device": "gpu",
    }

    np.testing.assert_array_equal(
        fake.train_kwargs["X"],
        X.reshape(len(X), -1).T,
    )
    np.testing.assert_array_equal(
        fake.train_kwargs["Y"],
        Y.T,
    )

    assert fake.train_kwargs["lr"] == 0.5
    assert fake.train_kwargs["epochs"] == 500
    assert fake.train_kwargs["batch_size"] == 8
    assert fake.train_kwargs["loss_type"] == "mse"
    assert fake.train_kwargs["print_loss"] is False

    predictions, error = model.test(X, Y)

    expected_predictions = (
        X.reshape(len(X), -1).T[:2]
    ).T
    expected_error = np.mean(
        (expected_predictions - Y) ** 2
    )

    np.testing.assert_array_equal(
        predictions,
        expected_predictions,
    )
    assert error == pytest.approx(expected_error)
    assert isinstance(error, float)


def test_mlp_model_requires_labels(monkeypatch):
    monkeypatch.setattr(models, "MLP", FakeMLP)

    model = MLPModel(
        L=2,
        device="gpu",
    )
    X = np.zeros((4, 2, 1))

    with pytest.raises(
        ValueError,
        match="MLPModel.fit requires labels Y",
    ):
        model.fit(X, None)


def test_autoencoder_matches_original_wrapper(monkeypatch):
    FakeMLP.instances.clear()
    monkeypatch.setattr(models, "MLP", FakeMLP)

    X = np.arange(
        24,
        dtype=float,
    ).reshape(6, 2, 2)

    model = AEModel(
        L=2,
        device="gpu",
    )
    model.fit(X)

    fake = FakeMLP.instances[-1]

    assert fake.init_kwargs == {
        "n_x": 4,
        "n_h": 4,
        "n_y": 4,
        "device": "gpu",
    }

    X_flat = X.reshape(len(X), -1)

    np.testing.assert_array_equal(
        fake.train_kwargs["X"],
        X_flat.T,
    )
    np.testing.assert_array_equal(
        fake.train_kwargs["Y"],
        X_flat.T,
    )

    assert fake.train_kwargs["lr"] == 0.1
    assert fake.train_kwargs["epochs"] == 200
    assert fake.train_kwargs["batch_size"] == 16
    assert fake.train_kwargs["loss_type"] == "mse"
    assert fake.train_kwargs["print_loss"] is False

    predictions, error = model.test(X)

    np.testing.assert_array_equal(
        predictions,
        X_flat,
    )
    assert error == 0.0
    assert isinstance(error, float)


def test_rnn_model_matches_original_wrapper(monkeypatch):
    FakeRNN.instances.clear()
    monkeypatch.setattr(models, "RNN", FakeRNN)

    X = np.arange(
        24,
        dtype=float,
    ).reshape(6, 2, 2)

    Y = np.arange(
        12,
        dtype=float,
    ).reshape(6, 2)

    model = RNNModel(L=2)
    model.fit(X, Y)

    fake = FakeRNN.instances[-1]

    assert fake.init_kwargs == {
        "d_x": 2,
        "d_h": 16,
        "d_y": 2,
        "n_x": 2,
        "n_y": 1,
        "first_output_step": 1,
    }

    np.testing.assert_array_equal(
        fake.train_kwargs["X"],
        X,
    )
    np.testing.assert_array_equal(
        fake.train_kwargs["Y"],
        Y,
    )

    assert fake.train_kwargs["lr"] == 0.1
    assert fake.train_kwargs["epochs"] == 200
    assert fake.train_kwargs["batch_size"] == 16
    assert fake.train_kwargs["loss_type"] == "mse"
    assert fake.train_kwargs["print_loss"] is False

    predictions, error = model.test(X, Y)

    assert predictions.shape == Y.shape
    np.testing.assert_array_equal(
        predictions,
        np.zeros_like(Y),
    )
    assert error == pytest.approx(np.mean(Y**2))
    assert isinstance(error, float)


def test_model_factory():
    assert isinstance(
        make_model("VARModel", L=2, device="gpu"),
        VARModel,
    )
    assert isinstance(
        make_model("RNNModel", L=2, device="gpu"),
        RNNModel,
    )

    mlp = make_model(
        "MLPModel",
        L=2,
        device="gpu",
    )
    autoencoder = make_model(
        "AEModel",
        L=2,
        device="gpu",
    )

    assert isinstance(mlp, MLPModel)
    assert isinstance(autoencoder, AEModel)
    assert mlp.device == "gpu"
    assert autoencoder.device == "gpu"


def test_unknown_model():
    with pytest.raises(
        ValueError,
        match="Unknown model_choice",
    ):
        make_model(
            "UnknownModel",
            L=2,
            device="gpu",
        )