"""Tests for the hyperparameter-sweep script."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


MODULE_PATH = (
    SCRIPT_DIR
    / "hyperparameter_sweep.py"
)

SPEC = importlib.util.spec_from_file_location(
    "hyperparameter_sweep",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(
        f"Could not load {MODULE_PATH}."
    )

hyperparameter_sweep = (
    importlib.util.module_from_spec(SPEC)
)

SPEC.loader.exec_module(
    hyperparameter_sweep
)


def base_model_parameters():
    """Return model parameters used by the tests."""
    return {
        "MLPModel": {
            "hidden_size": 32,
            "lr": 0.5,
            "epochs": 500,
            "batch_size": 8,
            "loss_type": "mse",
        },
        "RNNModel": {
            "hidden_size": 16,
            "lr": 0.1,
            "epochs": 200,
            "batch_size": 16,
            "loss_type": "mse",
        },
        "AEModel": {
            "bottleneck_size": 4,
            "lr": 0.1,
            "epochs": 200,
            "batch_size": 16,
            "loss_type": "mse",
        },
    }


def test_plot_mode_normalizes_batch_and_epoch_selection(
    monkeypatch,
):
    monkeypatch.setattr(
        hyperparameter_sweep,
        "MODEL_CHOICE",
        "MLPModel",
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "SWEEP_MODE",
        "plot",
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "MLP_BATCH_SIZES",
        [8, 16],
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "MLP_EPOCH_COUNTS",
        [50, 100],
    )

    hyperparameter_sweep.validate_configuration()

    grid = (
        hyperparameter_sweep
        .get_sweep_grid()
    )

    assert grid["batch_sizes"] == [8]
    assert grid["epoch_counts"] == [50]


@pytest.mark.parametrize(
    (
        "model_choice",
        "size_name",
        "size",
    ),
    [
        (
            "MLPModel",
            "hidden_size",
            64,
        ),
        (
            "AEModel",
            "bottleneck_size",
            8,
        ),
        (
            "RNNModel",
            "hidden_size",
            32,
        ),
    ],
)
def test_make_model_parameters_updates_selected_model(
    monkeypatch,
    model_choice,
    size_name,
    size,
):
    parameters = base_model_parameters()

    monkeypatch.setattr(
        hyperparameter_sweep,
        "MODEL_CHOICE",
        model_choice,
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "get_base_model_parameters",
        lambda: parameters,
    )

    result = (
        hyperparameter_sweep
        .make_model_parameters(
            size=size,
            learning_rate=0.05,
            batch_size=24,
            epochs=75,
        )
    )

    selected = result[model_choice]

    assert selected[size_name] == size
    assert selected["lr"] == 0.05
    assert selected["batch_size"] == 24
    assert selected["epochs"] == 75
    assert selected["loss_type"] == "mse"


@pytest.mark.parametrize(
    "model_choice",
    [
        "MLPModel",
        "AEModel",
        "RNNModel",
    ],
)
def test_make_model_parameters_does_not_mutate_defaults(
    monkeypatch,
    model_choice,
):
    parameters = base_model_parameters()

    original = {
        name: values.copy()
        for name, values in parameters.items()
    }

    monkeypatch.setattr(
        hyperparameter_sweep,
        "MODEL_CHOICE",
        model_choice,
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "get_base_model_parameters",
        lambda: parameters,
    )

    (
        hyperparameter_sweep
        .make_model_parameters(
            size=64,
            learning_rate=0.01,
            batch_size=32,
            epochs=10,
        )
    )

    assert parameters == original


def test_make_model_parameters_preserves_other_models(
    monkeypatch,
):
    parameters = base_model_parameters()

    monkeypatch.setattr(
        hyperparameter_sweep,
        "MODEL_CHOICE",
        "MLPModel",
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "get_base_model_parameters",
        lambda: parameters,
    )

    result = (
        hyperparameter_sweep
        .make_model_parameters(
            size=64,
            learning_rate=0.05,
            batch_size=32,
            epochs=100,
        )
    )

    assert (
        result["AEModel"]
        == parameters["AEModel"]
    )

    assert (
        result["RNNModel"]
        == parameters["RNNModel"]
    )


def test_make_model_parameters_uses_lr_key(
    monkeypatch,
):
    parameters = base_model_parameters()

    monkeypatch.setattr(
        hyperparameter_sweep,
        "MODEL_CHOICE",
        "MLPModel",
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "get_base_model_parameters",
        lambda: parameters,
    )

    result = (
        hyperparameter_sweep
        .make_model_parameters(
            size=64,
            learning_rate=0.05,
            batch_size=8,
            epochs=50,
        )
    )

    selected = result["MLPModel"]

    assert selected["lr"] == 0.05
    assert "learning_rate" not in selected


def test_make_model_parameters_rejects_unknown_model(
    monkeypatch,
):
    monkeypatch.setattr(
        hyperparameter_sweep,
        "MODEL_CHOICE",
        "UnknownModel",
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "get_base_model_parameters",
        base_model_parameters,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported model",
    ):
        (
            hyperparameter_sweep
            .make_model_parameters(
                size=10,
                learning_rate=0.1,
                batch_size=8,
                epochs=10,
            )
        )


def test_run_one_configuration_forwards_parameters(
    monkeypatch,
):
    expected_parameters = (
        base_model_parameters()
    )

    received = {}

    def fake_make_model_parameters(
        size,
        learning_rate,
        batch_size,
        epochs,
    ):
        received[
            "sweep_parameters"
        ] = {
            "size": size,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
        }

        return expected_parameters

    def fake_estimator(**kwargs):
        received["estimator_kwargs"] = kwargs

        return (
            60,
            {
                40: 0.5,
                60: 1.0,
            },
        )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "make_model_parameters",
        fake_make_model_parameters,
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        (
            "estimate_causal_cpd_"
            "log_normalized_cross_regime"
        ),
        fake_estimator,
    )

    monkeypatch.setattr(
        hyperparameter_sweep,
        "set_model_seed",
        lambda seed: None,
    )

    dataset_config = SimpleNamespace(
        tau_star=50,
        T=100,
    )

    result = (
        hyperparameter_sweep
        .run_one_configuration(
            X=[[0.0], [1.0]],
            dataset_config=dataset_config,
            candidates=[40, 60],
            size=64,
            learning_rate=0.05,
            batch_size=8,
            epochs=50,
            model_seed=42,
        )
    )

    assert received[
        "sweep_parameters"
    ] == {
        "size": 64,
        "learning_rate": 0.05,
        "batch_size": 8,
        "epochs": 50,
    }

    estimator_kwargs = received[
        "estimator_kwargs"
    ]

    assert (
        estimator_kwargs["model_parameters"]
        is expected_parameters
    )

    assert (
        estimator_kwargs[
            "validation_fraction"
        ]
        == hyperparameter_sweep
        .VALIDATION_FRACTION
    )

    assert (
        estimator_kwargs["min_train_size"]
        == hyperparameter_sweep
        .MIN_TRAIN_EXAMPLES
    )

    assert (
        estimator_kwargs[
            "min_validation_size"
        ]
        == hyperparameter_sweep
        .MIN_VALIDATION_EXAMPLES
    )

    assert "val_frac" not in estimator_kwargs
    assert "min_train" not in estimator_kwargs
    assert "min_val" not in estimator_kwargs

    assert result["tau_hat"] == 60
    assert result["absolute_error"] == 10
    assert result["percentage_error"] == 10.0

    assert result["scores"] == {
        40: 0.5,
        60: 1.0,
    }


def test_validate_configuration_rejects_unknown_model(
    monkeypatch,
):
    monkeypatch.setattr(
        hyperparameter_sweep,
        "MODEL_CHOICE",
        "VARModel",
    )

    with pytest.raises(
        ValueError,
        match="MODEL_CHOICE",
    ):
        (
            hyperparameter_sweep
            .validate_configuration()
        )