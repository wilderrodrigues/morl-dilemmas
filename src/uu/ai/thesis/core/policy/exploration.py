# MIT License - 2026
"""Exploration-policy abstractions for learning agents."""
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Policy(ABC):
    """Abstract interface for exploration policies."""

    @abstractmethod
    def update(self, iteration: int, total_iterations: int, eps_theta: float, eps_decay: float) -> tuple[
        int, float, str]:
        """Update the policy and select an action.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        total_iterations : int
            Total number of training iterations.
        eps_theta : float
            Initial epsilon value used for exploration.
        eps_decay : float
            Flag-like value indicating whether epsilon decay should be applied.

        Returns
        -------
        tuple[int, float, str]
            Selected action, effective epsilon value, and the reason for the
            action choice.
        """
        raise NotImplementedError


class EpsilonGreedy(Policy):
    """Epsilon-greedy exploration policy."""

    def __init__(self, random_numbers: npt.NDArray[np.float32], q_values: npt.NDArray[np.float32], state_index: int):
        """Initialize the epsilon-greedy policy.

        Parameters
        ----------
        random_numbers : npt.NDArray[np.float32]
            Precomputed random values used to sample exploratory actions.
        q_values : npt.NDArray[np.float32]
            Estimated action values for each state-action pair.
        state_index : int
            Index of the current state in the Q-value table.
        """
        self.random_numbers = random_numbers
        self.q_values = q_values
        self.state_index = state_index

    def update(self, iteration: int, total_iterations: int, eps_theta: float, eps_decay: float) -> tuple[
        int, float, str]:
        """Select an action using an epsilon-greedy decision rule.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        total_iterations : int
            Total number of training iterations.
        eps_theta : float
            Initial epsilon value used for exploration.
        eps_decay : float
            Flag-like value indicating whether epsilon should decay over time.

        Returns
        -------
        tuple[int, float, str]
            Selected action, effective epsilon value, and the reason for the
            action choice.
        """
        prob = self.random_numbers[1]
        if not eps_decay:
            eps = eps_theta  # try 0.05 #0.01 #0.001
        else:  # if I need to implement eps_decay and eps_theta has been pre-defined
            eps_initial = eps_theta
            eps_final = 0
            r = max((int(total_iterations) - int(iteration)) / int(total_iterations), 0)
            eps = (eps_initial - eps_final) * r + eps_final

        if prob <= eps:
            # make a random move with probability eps
            reason = 'random, due to eps'
            return int(self.random_numbers[2] < 0.5), eps, reason
        else:
            # move optimally based on current Q-value estimates, if they are not empty
            if not np.any(self.q_values[self.state_index]):  # if Q-values for this state are empty
                reason = 'random, due to empty Q-values'
                return int(self.random_numbers[0] < 0.5), eps, reason  # make a random move
            else:
                optimal_policy = np.argmax(self.q_values, axis=1)  # list(np.argmax(self.q_values, axis=1))
                reason = 'greedy, according to learnt Q-values'
                return int(bool(optimal_policy[self.state_index])), eps, reason


class ExplorationPolicy:
    """Wrapper that delegates action selection to a concrete policy."""

    def __init__(self, policy: Policy) -> None:
        """Initialize the exploration policy wrapper.

        Parameters
        ----------
        policy : Policy
            Concrete policy implementation used for action selection.
        """
        self.policy = policy

    def use_policy(self, iteration: int, total_iterations: int, eps_theta: float, eps_decay: float) -> tuple[
        int, float, str]:
        """Execute the configured exploration policy.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        total_iterations : int
            Total number of training iterations.
        eps_theta : float
            Initial epsilon value used for exploration.
        eps_decay : float
            Flag-like value indicating whether epsilon decay should be applied.

        Returns
        -------
        tuple[int, float, str]
            Selected action, effective epsilon value, and the reason for the
            action choice.
        """
        return self.policy.update(iteration, total_iterations, eps_theta, eps_decay)
