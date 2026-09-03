"""Forecasting and reconstruction models."""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

try:
    from machine_learning_templates.mlp.mlp import MLP
except ImportError:
    MLP = None

try:
    from machine_learning_templates.rnn.rnn import RNN
except ImportError:
    RNN = None


def construct_supervised_prediction_pairs(X, L):
    """Construct lagged prediction examples."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must have shape (T, d). Got {X.shape}."
        )

    if L <= 0:
        raise ValueError(
            f"L must be positive. Got {L}."
        )

    if len(X) <= L:
        raise ValueError(
            f"Need len(X) > L. Got len(X)={len(X)}, L={L}."
        )

    exs = np.array([
        X[t - L:t]
        for t in range(L, len(X))
    ])

    labels = np.array([
        X[t]
        for t in range(L, len(X))
    ])

    return exs, labels


def _validate_fit_policy(model):
    if model.fit_mode not in {"fixed", "adaptive"}:
        raise ValueError(
            "fit_mode must be 'fixed' or 'adaptive'. "
            f"Got {model.fit_mode!r}."
        )
    if model.patience <= 0:
        raise ValueError("patience must be positive.")
    if model.min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    if model.max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")


def _train_with_fit_policy(
    wrapper,
    train_X,
    train_Y,
    validation_data,
    prepare_validation,
):
    """Train for a fixed duration or to validation-loss convergence."""
    train_kwargs = {
        "X": train_X,
        "Y": train_Y,
        "lr": wrapper.lr,
        "batch_size": wrapper.batch_size,
        "loss_type": wrapper.loss_type,
        "print_loss": wrapper.print_loss,
    }

    if wrapper.fit_mode == "fixed":
        wrapper.model.train(
            epochs=wrapper.epochs,
            **train_kwargs,
        )
        wrapper.epochs_trained = wrapper.epochs
        return

    if validation_data is None:
        raise ValueError(
            "adaptive fit_mode requires validation_data."
        )

    validation_X, validation_Y = prepare_validation(
        validation_data
    )
    best_parameters = None
    best_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(1, wrapper.max_epochs + 1):
        wrapper.model.train(epochs=1, **train_kwargs)
        validation_loss = wrapper.validation_loss(
            validation_X,
            validation_Y,
        )

        if validation_loss < best_loss - wrapper.min_delta:
            best_loss = float(validation_loss)
            best_parameters = deepcopy(
                wrapper.model.parameters
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= wrapper.patience:
            break

    if best_parameters is None:
        raise RuntimeError(
            "Adaptive fitting produced no finite validation loss."
        )

    wrapper.model.parameters = best_parameters
    wrapper.epochs_trained = epoch
    wrapper.best_validation_loss = best_loss


class ForecastModel(ABC):
    """Common interface for competitor models."""

    def __init__(self, L):
        self.L = L

    @abstractmethod
    def fit(self, X, Y=None, validation_data=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def test(self, X, Y=None):
        pass


class VARModel(ForecastModel):
    """Linear least-squares VAR model."""

    def fit(self, X, Y, validation_data=None):
        if Y is None:
            raise ValueError("VARModel.fit requires labels Y.")

        X = np.asarray(X, dtype=float)
        X = X.reshape(len(X), -1)
        Y = np.asarray(Y, dtype=float)

        self.B, *_ = np.linalg.lstsq(
            X,
            Y,
            rcond=None,
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = X.reshape(len(X), -1)

        return X @ self.B

    def test(self, X, Y):
        if Y is None:
            raise ValueError("VARModel.test requires labels Y.")

        Y = np.asarray(Y, dtype=float)
        preds = self.predict(X)

        return preds, np.mean((preds - Y) ** 2)


class MLPModel(ForecastModel):
    """MLP forecasting model."""

    def __init__(
        self,
        L,
        device="cpu",
        hidden_size=32,
        lr=0.5,
        epochs=500,
        batch_size=8,
        loss_type="mse",
        print_loss=False,
        fit_mode="fixed",
        patience=20,
        min_delta=1e-4,
        max_epochs=None,
    ):
        super().__init__(L)

        self.device = device
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.print_loss = print_loss
        self.fit_mode = fit_mode
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = epochs if max_epochs is None else max_epochs
        self.epochs_trained = None
        self.best_validation_loss = None

        _validate_fit_policy(self)

    def fit(self, X, Y, validation_data=None):
        if Y is None:
            raise ValueError("MLPModel.fit requires labels Y.")

        if MLP is None:
            raise ImportError(
                "MLPModel requires "
                "machine_learning_templates.mlp.mlp.MLP."
            )

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        d = len(X[0][0])
        n_x = self.L * d
        n_y = d

        self.model = MLP(
            n_x=n_x,
            n_h=self.hidden_size,
            n_y=n_y,
            device=self.device,
        )

        X = X.reshape(len(X), -1)

        _train_with_fit_policy(
            wrapper=self,
            train_X=X.T,
            train_Y=Y.T,
            validation_data=validation_data,
            prepare_validation=lambda values: (
                np.asarray(values[0], dtype=float),
                np.asarray(values[1], dtype=float),
            ),
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = X.reshape(len(X), -1)

        return np.asarray(
            self.model.predict(X=X.T)
        ).T

    def validation_loss(self, X, Y):
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        Y = np.asarray(Y, dtype=float)
        predictions, _ = self.model.forward_propagation(X.T)
        if self.device == "gpu":
            predictions = self.model.xp.asnumpy(predictions)
        predictions = np.asarray(predictions).T
        return float(np.mean((predictions - Y) ** 2))

    def test(self, X, Y):
        if Y is None:
            raise ValueError("MLPModel.test requires labels Y.")

        Y = np.asarray(Y, dtype=float)
        preds = self.predict(X)

        return preds, np.mean((preds - Y) ** 2)


class RNNModel(ForecastModel):
    """RNN forecasting model."""

    def __init__(
        self,
        L,
        hidden_size=16,
        lr=0.1,
        epochs=200,
        batch_size=16,
        loss_type="mse",
        print_loss=False,
        fit_mode="fixed",
        patience=20,
        min_delta=1e-4,
        max_epochs=None,
    ):
        super().__init__(L)

        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.print_loss = print_loss
        self.fit_mode = fit_mode
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = epochs if max_epochs is None else max_epochs
        self.epochs_trained = None
        self.best_validation_loss = None

        _validate_fit_policy(self)

    def fit(self, X, Y, validation_data=None):
        if Y is None:
            raise ValueError("RNNModel.fit requires labels Y.")

        if RNN is None:
            raise ImportError(
                "RNNModel requires "
                "machine_learning_templates.rnn.rnn.RNN."
            )

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        self.model = RNN(
            d_x=len(X[0][0]),
            d_h=self.hidden_size,
            d_y=len(X[0][0]),
            n_x=self.L,
            n_y=1,
            first_output_step=self.L - 1,
        )

        _train_with_fit_policy(
            wrapper=self,
            train_X=X,
            train_Y=Y,
            validation_data=validation_data,
            prepare_validation=lambda values: (
                np.asarray(values[0], dtype=float),
                np.asarray(values[1], dtype=float),
            ),
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.asarray(
            self.model.predict(X=X)
        )

        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = preds[:, 0, :]

        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0, :, :]

        return preds

    def validation_loss(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        predictions, _ = self.model.forward_propagation(X)
        predictions = np.asarray(predictions)
        if predictions.ndim == 3 and predictions.shape[1] == 1:
            predictions = predictions[:, 0, :]
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0, :, :]
        return float(np.mean((predictions - Y) ** 2))

    def test(self, X, Y):
        if Y is None:
            raise ValueError("RNNModel.test requires labels Y.")

        Y = np.asarray(Y, dtype=float)
        preds = self.predict(X)

        return preds, np.mean((preds - Y) ** 2)


class AEModel(ForecastModel):
    """Autoencoder reconstruction model."""

    def __init__(
        self,
        L,
        device="cpu",
        bottleneck_size=4,
        lr=0.1,
        epochs=200,
        batch_size=16,
        loss_type="mse",
        print_loss=False,
        fit_mode="fixed",
        patience=20,
        min_delta=1e-4,
        max_epochs=None,
    ):
        super().__init__(L)

        self.device = device
        self.bottleneck_size = bottleneck_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.print_loss = print_loss
        self.fit_mode = fit_mode
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = epochs if max_epochs is None else max_epochs
        self.epochs_trained = None
        self.best_validation_loss = None

        _validate_fit_policy(self)

    def fit(self, X, Y=None, validation_data=None):
        if MLP is None:
            raise ImportError(
                "AEModel requires "
                "machine_learning_templates.mlp.mlp.MLP."
            )

        X = np.asarray(X, dtype=float)

        d = len(X[0][0])
        n_x = self.L * d
        n_y = n_x

        self.model = MLP(
            n_x=n_x,
            n_h=self.bottleneck_size,
            n_y=n_y,
            device=self.device,
        )

        X = X.reshape(len(X), -1)

        _train_with_fit_policy(
            wrapper=self,
            train_X=X.T,
            train_Y=X.T,
            validation_data=validation_data,
            prepare_validation=lambda values: (
                np.asarray(values[0], dtype=float),
                None,
            ),
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = X.reshape(len(X), -1)

        return np.asarray(
            self.model.predict(X=X.T)
        ).T

    def validation_loss(self, X, Y=None):
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        predictions, _ = self.model.forward_propagation(X.T)
        if self.device == "gpu":
            predictions = self.model.xp.asnumpy(predictions)
        predictions = np.asarray(predictions).T
        return float(np.mean((predictions - X) ** 2))

    def test(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        X_flat = X.reshape(len(X), -1)
        preds = self.predict(X)

        return preds, np.mean((preds - X_flat) ** 2)


def make_model(
    model_choice,
    L,
    device="cpu",
    model_parameters=None,
):
    """Construct a competitor model."""
    parameters = (
        {}
        if model_parameters is None
        else dict(model_parameters)
    )

    if model_choice == "VARModel":
        return VARModel(L)

    if model_choice == "MLPModel":
        parameters.setdefault("device", device)
        return MLPModel(
            L=L,
            **parameters,
        )

    if model_choice == "RNNModel":
        return RNNModel(
            L=L,
            **parameters,
        )

    if model_choice == "AEModel":
        parameters.setdefault("device", device)
        return AEModel(
            L=L,
            **parameters,
        )

    valid = [
        "VARModel",
        "AEModel",
        "MLPModel",
        "RNNModel",
        "KMOVARScore",
    ]

    raise ValueError(
        f"Unknown model_choice={model_choice!r}. "
        f"Valid choices are {valid}."
    )
