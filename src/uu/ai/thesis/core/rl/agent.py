# MIT License - 2026
"""Agent abstractions for fixed and exploratory IPD players.

This module defines the player wrapper used by the experiments to hold
strategy-specific configuration, Q-values for learning agents, and action
selection helpers for both exploratory and fixed-strategy settings.
"""
from uu.ai.thesis.core.policy.exploration import ExplorationPolicy, EpsilonGreedy
from uu.ai.thesis.core.rl.types import Strategy
import numpy.typing as npt


class Player:
    """Represent a player agent in the social dilemma experiments.

    Parameters
    ----------
    strategy : Strategy
        Strategy configuration used by the player.
    eps_theta : float
        Initial epsilon value for epsilon-greedy exploration.
    eps_decay : float
        Flag-like value indicating whether epsilon decay is enabled.
    mixed_beta : int | None, optional
        Mixing parameter used only by the mixed virtue ethics agent.
    """

    def __init__(self, strategy: Strategy, eps_theta: float, eps_decay: float, mixed_beta: float | None = None) -> None:
        """Initialize the player configuration and learning state.

        Parameters
        ----------
        strategy : Strategy
            Strategy configuration used by the player.
        eps_theta : float
            Initial epsilon value for epsilon-greedy exploration.
        eps_decay : float
            Flag-like value indicating whether epsilon decay is enabled.
        mixed_beta : float | None, optional
            Mixing parameter used only by the mixed virtue ethics agent.
        """
        # Cooperate on the first move if the strategy is either Tit-for-Tat or Q-Learning
        self.strategy = strategy
        self.eps_theta = eps_theta
        self.eps_decay = eps_decay
        # Will only be used for QLVE_m agent
        self.mixed_beta = mixed_beta

        self.q_values: npt.NDArray | None = None

    def make_exploratory_move(self, state: int, iteration: int, num_iter: int,
                              random_numbers: npt.NDArray) -> tuple[
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

        Returns
        -------
        tuple[int, float, str, npt.NDArray[np.float32]]
            Selected action, effective epsilon value, selection reason, and
            the random numbers used for the choice.
        """
        policy = EpsilonGreedy(random_numbers=random_numbers, q_values=self.q_values, state_index=state)
        exploration_policy = ExplorationPolicy(policy=policy)
        move, eps, reason = exploration_policy.use_policy(iteration=iteration, total_iterations=num_iter,
                                                          eps_theta=self.eps_theta,
                                                          eps_decay=self.eps_decay)
        return int(bool(move)), eps, reason, random_numbers

    def make_fixed_move(self, state, player_rn_spawn_4) -> int:
        """Select an action from a non-learning fixed strategy.

        Parameters
        ----------
        state : Any
            Current state representation. For tit-for-tat, the first element is
            interpreted as the opponent's previous move.
        player_rn_spawn_4 : Any
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
