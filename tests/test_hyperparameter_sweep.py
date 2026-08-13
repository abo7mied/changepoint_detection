import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = SCRIPT_DIR / "hyperparameter_sweep.py"
SPEC = importlib.util.spec_from_file_location(
    "hyperparameter_sweep",
    MODULE_PATH,
)
hyperparameter_sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hyperparameter_sweep)


def test_plot_mode_normalizes_batch_and_epoch_selection(monkeypatch):
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

    grid = hyperparameter_sweep.get_sweep_grid()
    assert grid["batch_sizes"] == [8]
    assert grid["epoch_counts"] == [50]
