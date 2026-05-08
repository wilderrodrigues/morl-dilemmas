# MIT License - 2026
"""Agent abstractions for fixed and exploratory IPD players.

This module defines the player wrapper used by the experiments to hold
strategy-specific configuration, Q-values for learning agents, and action
selection helpers for both exploratory and fixed-strategy settings.
"""
from uu.ai.thesis.core.policy.exploration import ExplorationPolicy, EpsilonGreedy
from uu.ai.thesis.core.rl.types import Strategy
import numpy as np
import numpy.typing as npt

from uu.ai.thesis.core.utility.functions import NonLinearUtility


class Player:
    """Represent a player agent in the social dilemma experiments.

    Parameters
    ----------
    strategy : Strategy
        Strategy configuration used by the player.
    eps_theta : float
        Initial epsilon value for epsilon-greedy exploration.
    eps_decay : bool
        Flag indicating whether epsilon decay is enabled.
    mixed_beta : int | None, optional
        Mixing parameter used only by the mixed virtue ethics agent.
    """

    def __init__(self, strategy: Strategy, eps_theta: float, eps_decay: bool, mixed_beta: float | None = None,
                 utility: NonLinearUtility | None = None) -> None:
        """Initialize the player configuration and learning state.

        Parameters
        ----------
        strategy : Strategy
            Strategy configuration used by the player.
        eps_theta : float
            Initial epsilon value for epsilon-greedy exploration.
        eps_decay : bool
            Flag indicating whether epsilon decay is enabled.
        mixed_beta : float | None, optional
            Mixing parameter used only by the mixed virtue ethics agent.
        utility : NonLinearUtility | None, optional
            Non-linear utility used to scalarise vector rewards when present.
        """
        # Cooperate on the first move if the strategy is either Tit-for-Tat or Q-Learning
        self.strategy = strategy
        self.eps_theta = eps_theta
        self.eps_decay = eps_decay
        # Will only be used for QLVE_m agent
        self.mixed_beta = mixed_beta
        self.utility = utility

        self.q_values: npt.NDArray | None = None

    def make_exploratory_move(self, state: int, iteration: int, num_iter: int,
                              random_numbers: npt.NDArray,
                              accumulated_return: npt.NDArray | None = None,
                              discount_power: float = 1.0) -> tuple[
        int, float, str, npt.NDArray]:
        """Select an action using epsilon-greedy exploration.

        Parameters
        ----------
        state : int
            Index of the current environment state.
        iteration : int
            Current training iteration.
        num_iter : int
            Total number of training iterations.
        random_numbers : npt.NDArray[np.float32]
            Random values used by the exploration policy.
        accumulated_return : npt.NDArray | None, optional
            Discounted vector return accumulated so far in the current episode.
            In MORL/SER mode this is added before scalarising action values.
        discount_power : float, optional
            Discount factor power applied to the vector Q-values before
            scalarisation. This corresponds to ``gamma ** c`` in the
            accumulated-return action-selection rule.

        Returns
        -------
        tuple[int, float, str, npt.NDArray[np.float32]]
            Selected action, effective epsilon value, selection reason, and
            the random numbers used for the choice.
        """
        q_values = self.q_values
        if q_values is not None and q_values.ndim == 3:
            q_values = self.scalarised_q_values(
                accumulated_return=accumulated_return,
                discount_power=discount_power,
            )

        policy = EpsilonGreedy(random_numbers=random_numbers, q_values=q_values, state_index=state)
        exploration_policy = ExplorationPolicy(policy=policy)
        move, eps, reason = exploration_policy.use_policy(iteration=iteration, total_iterations=num_iter,
                                                          eps_theta=self.eps_theta,
                                                          eps_decay=self.eps_decay)
        return int(bool(move)), eps, reason, random_numbers

    def make_fixed_move(self, state: tuple[int, int], player_rn_spawn_4: float) -> int:
        """Select an action from a non-learning fixed strategy.

        Parameters
        ----------
        state : tuple[int, int]
            Current state representation. For tit-for-tat, the first element is
            interpreted as the opponent's previous move.
        player_rn_spawn_4 : float
            Random value used when the configured strategy is ``Strategy.Random``.

        Returns
        -------
        int
            Encoded action selected by the configured fixed strategy.

        Raises
        ------
        ValueError
            If the configured strategy does not support fixed-move selection.
        """

        if self.strategy == Strategy.Random:
            return int(player_rn_spawn_4 < 0.5)
        elif self.strategy == Strategy.TFT:
            # Take the first element of the state, i.e. the opponent's previous move
            return int(bool(state[0]))
        elif self.strategy == Strategy.AC:
            return int(False)
        elif self.strategy == Strategy.AD:
            return int(True)
        else:
            raise ValueError(f"Strategy {self.strategy} not supported by this agent when making a fixed move.")

    def scalarise_reward(self, reward_vector: dict[str, float]) -> float:
        """Scalarise a reward vector, or return the individual reward directly.

        Parameters
        ----------
        reward_vector : dict[str, float]
            Mapping containing at least an ``"individual"`` objective and,
            when using MORL, a ``"moral"`` objective.

        Returns
        -------
        float
            Scalar reward value. If the player has no utility function, the
            individual objective is used unchanged.
        """
        if self.utility is None:
            return reward_vector["individual"]

        return self.utility.scalarise(reward_vector)

    def scalarise_value(self, value_vector: npt.NDArray) -> float:
        """Scalarise an expected return vector for SER action selection.

        Parameters
        ----------
        value_vector : npt.NDArray
            Objective vector ordered as ``[moral, individual]``.

        Returns
        -------
        float
            Scalarised utility value.
        """
        reward_vector = {
            "moral": float(value_vector[0]),
            "individual": float(value_vector[1]),
        }

        return self.scalarise_reward(reward_vector)

    def scalarised_q_values(
        self,
        accumulated_return: npt.NDArray | None = None,
        discount_power: float = 1.0,
    ) -> npt.NDArray:
        """Return scalar action values derived from vector-valued Q-values.

        Parameters
        ----------
        accumulated_return : npt.NDArray | None, optional
            Discounted vector return accumulated in the current episode. When
            provided, the method scalarises ``r_acc + discount_power * Q(s,a)``
            for each state-action pair.
        discount_power : float, optional
            Multiplicative discount applied to vector Q-values before
            scalarisation.

        Returns
        -------
        npt.NDArray
            Two-dimensional scalar Q-table when the player stores vector-valued
            Q-values. Single-objective Q-values are returned unchanged.

        Raises
        ------
        ValueError
            If the player does not currently hold any Q-values.
        """
        if self.q_values is None:
            raise ValueError("Cannot scalarise missing Q-values.")
        if self.q_values.ndim != 3:
            return self.q_values

        scalar_q_values = np.zeros(self.q_values.shape[:2])
        for state in range(self.q_values.shape[0]):
            for action in range(self.q_values.shape[1]):
                value_vector = discount_power * self.q_values[state, action]
                if accumulated_return is not None:
                    value_vector = accumulated_return + value_vector
                scalar_q_values[state, action] = self.scalarise_value(value_vector)

        return scalar_q_values
