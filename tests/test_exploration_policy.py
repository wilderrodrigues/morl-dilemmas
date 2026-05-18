# MIT License - 2026
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from uu.ai.thesis.core.policy.exploration import EpsilonGreedy, ExplorationPolicy, Policy


class StubPolicy(Policy):
    """Test double for verifying policy delegation."""

    def __init__(self, result: tuple[int, float, str]) -> None:
        self.result = result
        self.calls: list[tuple[int, int, float, float]] = []

    def update(
        self,
        iteration: int,
        total_iterations: int,
        eps_theta: float,
        eps_decay: float,
    ) -> tuple[int, float, str]:
        """Record the incoming arguments and return a predefined result."""
        self.calls.append((iteration, total_iterations, eps_theta, eps_decay))
        return self.result


def test_use_policy_delegates_to_configured_policy() -> None:
    """Delegate action selection to the wrapped policy instance."""
    expected = (1, 0.15, "greedy")
    stub_policy = StubPolicy(expected)
    exploration_policy = ExplorationPolicy(stub_policy)

    result = exploration_policy.use_policy(
        iteration=10,
        total_iterations=100,
        eps_theta=0.15,
        eps_decay=False,
    )

    assert result == expected
    assert stub_policy.calls == [(10, 100, 0.15, False)]


def test_use_policy_with_epsilon_greedy_returns_greedy_action() -> None:
    """Use the wrapped epsilon-greedy policy to select the greedy action."""
    epsilon_greedy_policy = EpsilonGreedy(
        random_numbers=np.array([0.2, 0.9, 0.1]),
        q_values=np.array([[0.1, 0.9], [0.8, 0.2]]),
        state_index=0,
    )
    exploration_policy = ExplorationPolicy(epsilon_greedy_policy)

    result = exploration_policy.use_policy(
        iteration=10,
        total_iterations=100,
        eps_theta=0.15,
        eps_decay=False,
    )

    assert result == (1, 0.15, "greedy, according to learnt Q-values")


def test_plot():
    import matplotlib.pyplot as plt

    # ==========================================
    # Data
    # x = Dot-counting performance
    # y = Tracking performance
    # ==========================================

    young_x = [0, 67, 82, 92, 100]
    young_y = [100, 91, 87, 75, 0]

    old_x = [0, 77, 87, 91, 100]
    old_y = [100, 80, 70, 59, 0]

    # Dual-task labels
    labels = ["DT-ETR", "DT-NE", "DT-EDC"]

    # ==========================================
    # Plot
    # ==========================================

    plt.figure(figsize=(9, 9))

    # POC curves
    plt.plot(
        young_x,
        young_y,
        marker="D",
        linewidth=2,
        color="tab:blue",
        label="Young adults"
    )

    plt.plot(
        old_x,
        old_y,
        marker="s",
        linewidth=2,
        color="tab:red",
        label="Old adults"
    )

    # ==========================================
    # Annotate CoC
    # ==========================================

    # Young adults
    for x, y, label in zip(young_x[1:-1], young_y[1:-1], labels):
        coc_dc = 100 - x
        coc_tr = 100 - y

        # helper lines
        plt.plot([x, x], [y, 100], "--", color="tab:blue", alpha=0.4)
        plt.plot([x, 100], [y, y], "--", color="tab:blue", alpha=0.4)

        # point label
        plt.text(x + 1, y + 1, label, fontsize=9)

        # CoC annotation
        plt.text(
            x + 2,
            y - 6,
            f"CoC DC={coc_dc}\nCoC TR={coc_tr}",
            fontsize=9,
            color="tab:blue"
        )

    # Old adults
    for x, y, label in zip(old_x[1:-1], old_y[1:-1], labels):
        coc_dc = 100 - x
        coc_tr = 100 - y

        # helper lines
        plt.plot([x, x], [y, 100], "--", color="tab:red", alpha=0.4)
        plt.plot([x, 100], [y, y], "--", color="tab:red", alpha=0.4)

        # point label
        plt.text(x + 1, y - 4, label, fontsize=9)

        # CoC annotation
        plt.text(
            x - 18,
            y - 10,
            f"CoC DC={coc_dc}\nCoC TR={coc_tr}",
            fontsize=9,
            color="tab:red"
        )

    # ==========================================
    # Axes and style
    # ==========================================

    plt.xlabel("Dot-Counting Task (%)", fontsize=12)
    plt.ylabel("Tracking Task (%)", fontsize=12)

    plt.title(
        "Performance Operating Characteristic (POC) Curve",
        fontsize=14
    )

    plt.xlim(0, 105)
    plt.ylim(0, 105)

    plt.xticks(range(0, 101, 10))
    plt.yticks(range(0, 101, 10))

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()
    plt.show()