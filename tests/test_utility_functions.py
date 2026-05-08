# MIT License - 2026
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from uu.ai.thesis.core.utility.functions import NonLinearUtility
from uu.ai.thesis.core.rl.agent import Player
from uu.ai.thesis.core.rl.types import Strategy
import numpy as np


def test_non_linear_utility_scalarises_moral_non_linearly_and_individual_linearly() -> None:
    utility = NonLinearUtility(phi=2.0, moral_weight=0.5, individual_weight=3.0)

    result = utility.scalarise({"moral": 4.0, "individual": 2.0})

    assert result == 14.0


def test_non_linear_utility_keeps_negative_moral_returns_real() -> None:
    utility = NonLinearUtility(phi=0.5)

    result = utility.scalarise({"moral": -5.0, "individual": 4.0})

    assert result == 4.0 - 5.0 ** 0.5


def test_player_scalarised_q_values_can_include_accumulated_return() -> None:
    player = Player(
        strategy=Strategy.QLS,
        eps_theta=0.0,
        eps_decay=False,
        utility=NonLinearUtility(phi=1.0),
    )
    player.q_values = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
    ])

    result = player.scalarised_q_values(
        accumulated_return=np.array([10.0, 20.0]),
        discount_power=0.5,
    )

    np.testing.assert_allclose(result, np.array([[31.5, 33.5]]))
