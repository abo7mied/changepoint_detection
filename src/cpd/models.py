"""Forecasting and reconstruction models."""

from abc import ABC, abstractmethod

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


class ForecastModel(ABC):
    """Common interface for competitor models."""

    def __init__(self, L):
        self.L = L

    @abstractmethod
    def fit(self, X, Y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def test(self, X, Y=None):
        pass


class VARModel(ForecastModel):
    """Linear least-squares VAR model."""

    def fit(self, X, Y):
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

    def __init__(self, L, device):
        super().__init__(L)
        self.device = device

    def fit(self, X, Y):
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
        n_h = 32
        n_y = d

        self.model = MLP(
            n_x=n_x,
            n_h=n_h,
            n_y=n_y,
            device=self.device,
        )

        X = X.reshape(len(X), -1)

        lr = 0.5
        epochs = 500
        batch_size = 8
        loss_type = "mse"

        self.model.train(
            X=X.T,
            Y=Y.T,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            loss_type=loss_type,
            print_loss=False,
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = X.reshape(len(X), -1)

        return np.asarray(
            self.model.predict(X=X.T)
        ).T

    def test(self, X, Y):
        if Y is None:
            raise ValueError("MLPModel.test requires labels Y.")
        Y = np.asarray(Y, dtype=float)
        preds = self.predict(X)

        return preds, np.mean((preds - Y) ** 2)


class RNNModel(ForecastModel):
    """RNN forecasting model."""

    def fit(self, X, Y):
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
            d_h=16,
            d_y=len(X[0][0]),
            n_x=self.L,
            n_y=1,
            first_output_step=self.L - 1,
        )

        lr = 0.1
        epochs = 200
        batch_size = 16
        loss_type = "mse"

        self.model.train(
            X=X,
            Y=Y,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            loss_type=loss_type,
            print_loss=False,
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

    def test(self, X, Y):
        if Y is None:
            raise ValueError("RNNModel.test requires labels Y.")
        Y = np.asarray(Y, dtype=float)
        preds = self.predict(X)

        return preds, np.mean((preds - Y) ** 2)


class AEModel(ForecastModel):
    """Autoencoder reconstruction model."""

    def __init__(self, L, device):
        super().__init__(L)
        self.device = device

    def fit(self, X, Y=None):
        if MLP is None:
            raise ImportError(
                "AEModel requires "
                "machine_learning_templates.mlp.mlp.MLP."
            )

        X = np.asarray(X, dtype=float)

        d = len(X[0][0])
        n_x = self.L * d
        n_h = 4
        n_y = n_x

        self.model = MLP(
            n_x=n_x,
            n_h=n_h,
            n_y=n_y,
            device=self.device,
        )

        X = X.reshape(len(X), -1)

        lr = 0.1
        epochs = 200
        batch_size = 16
        loss_type = "mse"

        self.model.train(
            X=X.T,
            Y=X.T,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            loss_type=loss_type,
            print_loss=False,
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = X.reshape(len(X), -1)

        return np.asarray(
            self.model.predict(X=X.T)
        ).T

    def test(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        X_flat = X.reshape(len(X), -1)
        preds = self.predict(X)

        return preds, np.mean((preds - X_flat) ** 2)


def make_model(model_choice, L, device):
    """Construct a competitor model."""
    if model_choice == "VARModel":
        return VARModel(L)

    if model_choice == "MLPModel":
        return MLPModel(L, device)

    if model_choice == "RNNModel":
        return RNNModel(L)

    if model_choice == "AEModel":
        return AEModel(L, device)

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