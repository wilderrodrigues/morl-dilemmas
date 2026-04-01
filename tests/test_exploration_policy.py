# MIT License - 2026
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from uu.ai.core.policy.exploration import EpsilonGreedy, ExplorationPolicy, Policy


class StubPolicy(Policy):
    """Test double for verifying policy delegation."""

    def __init__(self, result: tuple[int, float, str]) -> None:
        self.result = result
        self.calls: list[tuple[int, int, float, bool]] = []

    def update(
        self,
        iteration: int,
        total_iterations: int,
        eps_zero: float,
        eps_decay: float,
    ) -> tuple[int, float, str]:
        """Record the incoming arguments and return a predefined result."""
        self.calls.append((iteration, total_iterations, eps_zero, eps_decay))
        return self.result


def test_use_policy_delegates_to_configured_policy() -> None:
    """Delegate action selection to the wrapped policy instance."""
    expected = (1, 0.15, "greedy")
    stub_policy = StubPolicy(expected)
    exploration_policy = ExplorationPolicy(stub_policy)

    result = exploration_policy.use_policy(
        iteration=10,
        total_iterations=100,
        eps_zero=0.15,
        eps_decay=False,
    )

    assert result == expected
    assert stub_policy.calls == [(10, 100, 0.15, False)]


def test_use_policy_with_epsilon_greedy_returns_greedy_action() -> None:
    """Use the wrapped epsilon-greedy policy to select the greedy action."""
    epsilon_greedy_policy = EpsilonGreedy(
        random_numbers=[0.2, 0.9, 0.1],
        q_values=np.array([[0.1, 0.9], [0.8, 0.2]]),
        state_index=0,
    )
    exploration_policy = ExplorationPolicy(epsilon_greedy_policy)

    result = exploration_policy.use_policy(
        iteration=10,
        total_iterations=100,
        eps_zero=0.15,
        eps_decay=False,
    )

    assert result == (1, 0.15, "greedy, according to learnt Q-values")
