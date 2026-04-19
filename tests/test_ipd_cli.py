# MIT License - 2026
import json
from pathlib import Path
import sys

from typer.testing import CliRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from uu.ai.thesis.cli.play import app, build_game_config


runner = CliRunner()


def test_build_ipd_config_uses_defaults_without_suffixes() -> None:
    """Build the default configuration without adding optional suffixes."""
    config = build_game_config(title1="QLS", title2="AD")

    assert config.destination_folder == "QLS_AD"
    assert config.master_seed == 1
    assert config.num_iterations == 10_000
    assert config.num_runs == 100
    assert config.eps_theta == 0.05
    assert config.eps_decay is False
    assert config.alpha_theta == 0.01
    assert config.decay == 0.0
    assert config.gamma == 0.9
    assert config.mixed_beta == 0.5


def test_ipd_cli_prints_resolved_configuration() -> None:
    """Print the resolved configuration for the provided command arguments."""
    result = runner.invoke(
        app,
        [
            "--title1",
            "QLUT",
            "--title2",
            "AD",
            "--master-seed",
            "7",
            "--num-iterations",
            "250",
            "--num-runs",
            "5",
            "--eps-theta",
            "0.1",
            "--eps-decay",
            "--alpha-theta",
            "0.02",
            "--decay",
            "0.001",
            "--gamma",
            "0.95",
            "--beta",
            "0.25",
            "--extra",
            "pilot",
        ],
    )

    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload == {
        "alpha_theta": 0.02,
        "decay": 0.001,
        "destination_folder": "QLUT_AD_seed7_iter250_runs5_pilot_eps_theta0.1_eps_decay_alpha_theta0.02_decay0.001_gamma0.95_beta0.25",
        "eps_theta": 0.1,
        "eps_decay": True,
        "extra": "pilot",
        "gamma": 0.95,
        "master_seed": 7,
        "mixed_beta": 0.25,
        "num_iterations": 250,
        "num_runs": 5,
        "title1": "QLUT",
        "title2": "AD",
    }
