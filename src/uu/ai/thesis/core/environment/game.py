# MIT License - 2026
"""Game environments and reward models for iterative social dilemmas.

This module defines reward helpers, the abstract game interface, and concrete
or placeholder iterative dilemma environments used by the thesis experiments.
It centralizes step logic for learning, mixed, and fixed-strategy matchups.
"""
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame

from uu.ai.thesis.core.functions import RandomNumberGenerator
from uu.ai.thesis.core.rl.agent import Player
from uu.ai.thesis.core.rl.types import Morality


class Reward(ABC):
    """Abstract base class for game reward functions.

    Parameters
    ----------
    payoff_matrix : list[list[tuple[int, int]]]
        Payoff matrix indexed by the actions of player 1 and player 2.
    """

    def __init__(self, payoff_matrix: list[list[tuple[int, int]]]) -> None:
        """Store the payoff matrix used by the reward function.

        Parameters
        ----------
        payoff_matrix : list[list[tuple[int, int]]]
            Payoff matrix indexed by the actions of player 1 and player 2.
        """
        self.payoff_matrix = payoff_matrix

    @abstractmethod
    def reward(self, action_p1: int, action_p2: int) -> list[int]:
        """Compute a reward value from a pair of actions.

        Parameters
        ----------
        action_p1 : int
            Action selected by player 1.
        action_p2 : int
            Action selected by player 2.

        Returns
        -------
        list[int]
            Reward value or values derived from the payoff matrix.
        """
        raise NotImplementedError


class ExtrinsicReward(Reward):
    """Reward function that returns the game payoffs directly."""

    def reward(self, action_p1: int, action_p2: int) -> list[int]:
        """Return the extrinsic rewards for both players.

        Parameters
        ----------
        action_p1 : int
            Action selected by player 1.
        action_p2 : int
            Action selected by player 2.

        Returns
        -------
        list[int]
            Extrinsic rewards for player 1 and player 2.
        """
        pay1, pay2 = self.payoff_matrix[action_p1][action_p2][0], self.payoff_matrix[action_p1][action_p2][1]
        return [pay1, pay2]


class IntrinsicReward(Reward):
    """Reward function derived from the player's configured moral type."""

    def __init__(self, payoff_matrix: list[list[tuple[int, int]]], player: Player) -> None:
        """Initialize the intrinsic reward calculator.

        Parameters
        ----------
        payoff_matrix : list[list[tuple[int, int]]]
            Payoff matrix indexed by the actions of player 1 and player 2.
        player : Player
            Player whose morality determines the intrinsic reward formulation.
        """
        super().__init__(payoff_matrix=payoff_matrix)
        self.player = player
        self.state: tuple[int, int] | None = None
        # The constant that defines reward & punishment values for norm-based agents
        self.reward_base = 5

    def update_state(self, state: tuple[int, int]) -> None:
        """Update the state used by state-dependent moral rewards.

        Parameters
        ----------
        state : tuple[int, int]
            State representation used to evaluate intrinsic rewards.
        """
        self.state = state

    def reward(self, action_p1: int, action_p2: int) -> int | None:
        """Compute the intrinsic reward for the configured player.

        Parameters
        ----------
        action_p1 : int
            Action selected by the player associated with this reward object.
        action_p2 : int
            Action selected by the opponent.

        Returns
        -------
        int | None
            Intrinsic reward value, or ``None`` when the player is selfish and
            does not use an intrinsic reward component.
        """
        # Create the baseline individual payoffs, as defined in the IPD game
        pay1, pay2 = self.payoff_matrix[action_p1][action_p2][0], self.payoff_matrix[action_p1][action_p2][1]

        pay1_intrinsic = None
        if self.player.strategy.value[1] == Morality.UTILITARIAN:
            pay1_intrinsic = pay1 + pay2
        elif self.player.strategy.value[1] == Morality.DEONTOLOGICAL:
            if self.state[0] == 0:
                # if I (player1) defected against a cooperator (based on 1 previous move of the opponent), get punished
                if action_p1 == 1:
                    pay1_intrinsic = -self.reward_base
                else:
                    pay1_intrinsic = 0
            else:
                pay1_intrinsic = 0
        elif self.player.strategy.value[1] == Morality.VIRTUE_ETHICS_EQUALITY:
            # A simplification of the Gini coefficient for 2 players
            pay1_intrinsic = 1 - ((abs(pay1 - pay2)) / (pay1 + pay2))
        elif self.player.strategy.value[1] == Morality.VIRTUE_ETHICS_KINDNESS:
            # If this agent cooperated, get rewarded
            if action_p1 == 0:
                pay1_intrinsic = self.reward_base
            else:
                pay1_intrinsic = 0
        elif self.player.strategy.value[1] == Morality.VIRTUE_ETHICS_MIXED:
            mixed_beta = int(self.player.mixed_beta)
            k_normalised = self.reward_base / self.reward_base
            # If this agent cooperated
            if action_p1 == 0:
                pay1_intrinsic = mixed_beta * (1 - ((abs(pay1 - pay2)) / (pay1 + pay2))) + (
                        1 - mixed_beta) * k_normalised
            else:
                pay1_intrinsic = mixed_beta * (1 - ((abs(pay1 - pay2)) / (pay1 + pay2)))
        elif self.player.strategy.value[1] == Morality.SELFISH:
            pay1_intrinsic = None

        return pay1_intrinsic


class UtilitarianReward(Reward):
    """Collective reward based on the sum of both players' payoffs."""

    def reward(self, action_p1: int, action_p2: int) -> float:
        """Compute the utilitarian reward.

        Parameters
        ----------
        action_p1 : int
            Action selected by player 1.
        action_p2 : int
            Action selected by player 2.

        Returns
        -------
        float
            Sum of both players' extrinsic rewards.
        """
        pay1, pay2 = self.payoff_matrix[action_p1][action_p2][0], self.payoff_matrix[action_p1][action_p2][1]
        pay_final = pay1 + pay2
        return pay_final


class VirtueReward(Reward):
    """Collective reward based on payoff ratio equality."""

    def reward(self, action_p1: int, action_p2: int) -> float:
        """Compute the virtue-style ratio reward.

        Parameters
        ----------
        action_p1 : int
            Action selected by player 1.
        action_p2 : int
            Action selected by player 2.

        Returns
        -------
        float
            Ratio-based reward reflecting payoff balance.
        """
        pay1, pay2 = self.payoff_matrix[action_p1][action_p2][0], self.payoff_matrix[action_p1][action_p2][1]
        pay_final = (min(pay1, pay2) + 1) / (max(pay1, pay2) + 1)
        return pay_final


class GiniReward(Reward):
    """Collective reward based on a simplified two-player Gini measure."""

    def reward(self, action_p1: int, action_p2: int) -> float:
        """Compute the equality-based Gini reward.

        Parameters
        ----------
        action_p1 : int
            Action selected by player 1.
        action_p2 : int
            Action selected by player 2.

        Returns
        -------
        float
            Equality reward derived from the players' payoffs.
        """
        pay1, pay2 = self.payoff_matrix[action_p1][action_p2][0], self.payoff_matrix[action_p1][action_p2][1]
        pay_final = 1 - ((abs(pay1 - pay2)) / (pay1 + pay2))
        return pay_final


class MinimumReward(Reward):
    """Collective reward based on the minimum payoff achieved."""

    def reward(self, action_p1: int, action_p2: int) -> float:
        """Compute the minimum-payoff reward.

        Parameters
        ----------
        action_p1 : int
            Action selected by player 1.
        action_p2 : int
            Action selected by player 2.

        Returns
        -------
        float
            Minimum of the two players' extrinsic rewards.
        """
        pay1, pay2 = self.payoff_matrix[action_p1][action_p2][0], self.payoff_matrix[action_p1][action_p2][1]
        pay_final = min(pay1, pay2)
        return pay_final


class Game(ABC):
    """Abstract base class for iterative two-player dilemma games.

    Parameters
    ----------
    player1 : Player
        First player in the game.
    player2 : Player
        Second player in the game.
    payoff_matrix : list[list[tuple[int, int]]]
        Payoff matrix indexed by the actions of player 1 and player 2.
    """
    PLAYER_1: int = 0
    PLAYER_2: int = 1

    def __init__(self, player1: Player, player2: Player, payoff_matrix: list[list[tuple[int, int]]]) -> None:
        """Initialize the game state and shared bookkeeping.

        Parameters
        ----------
        player1 : Player
            First player in the game.
        player2 : Player
            Second player in the game.
        payoff_matrix : list[list[tuple[int, int]]]
            Payoff matrix indexed by the actions of player 1 and player 2.
        """
        self.player1 = player1
        self.player2 = player2
        self.payoff_matrix = payoff_matrix
        self.history = list()  # TODO make this a circular array / queue instead
        self.opponents = {player1: player2, player2: player1}
        self.state_index_converter = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    @abstractmethod
    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, tuple[int, int], tuple[int, int], int | None, int | None]:
        """Execute one step with two learning players.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics are recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number generator streams used for sampling actions.

        Returns
        -------
        tuple[int, int, tuple[int, int], tuple[int, int], int | None, int | None]
            Selected actions, next states, and learning rewards for both
            players.
        """
        raise NotImplementedError

    @abstractmethod
    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                   global_history: DataFrame,
                   num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, tuple[int, int], tuple[int, int], int | None]:
        """Execute one step with one learning player and one fixed player.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by the learning player.
        state_p2 : tuple[int, int]
            Current state perceived by the fixed player.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics are recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number streams used for stochastic action selection.

        Returns
        -------
        tuple[int, tuple[int, int], tuple[int, int], int | None]
            Learning player's selected action, next states for both players,
            and the reward signal used for learning.
        """
        raise NotImplementedError

    @abstractmethod
    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                    global_history: DataFrame,
                    random_numbers_stream: RandomNumberGenerator) -> tuple[
        tuple[int, int], tuple[int, int]]:
        """Execute one step with two fixed-strategy players.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics are recorded.
        random_numbers_stream : RandomNumberGenerator
            Random number streams used by stochastic fixed strategies.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int]]
            Next states for player 1 and player 2 after the fixed-strategy
            interaction.
        """
        raise NotImplementedError


class IterativePrisonersDilemma(Game):
    """Iterated Prisoner's Dilemma with extrinsic and moral reward tracking."""

    def __init__(self, player1: Player, player2: Player, payoff_matrix: list[list[tuple[int, int]]]) -> None:
        """Initialize reward calculators for the iterated prisoner's dilemma.

        Parameters
        ----------
        player1 : Player
            First player in the game.
        player2 : Player
            Second player in the game.
        payoff_matrix : list[list[tuple[int, int]]]
            Payoff matrix indexed by the actions of player 1 and player 2.
        """
        super().__init__(player1=player1, player2=player2, payoff_matrix=payoff_matrix)
        self.extrinsic_reward = ExtrinsicReward(payoff_matrix=payoff_matrix)
        self.intrinsic_reward_p1 = IntrinsicReward(payoff_matrix=payoff_matrix, player=player1)
        self.intrinsic_reward_p2 = IntrinsicReward(payoff_matrix=payoff_matrix, player=player2)
        self.utilitarian_reward = UtilitarianReward(payoff_matrix=payoff_matrix)
        self.virtue_reward = VirtueReward(payoff_matrix=payoff_matrix)
        self.gini_reward = GiniReward(payoff_matrix=payoff_matrix)
        self.minimum_reward = MinimumReward(payoff_matrix=payoff_matrix)

    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, tuple[int, int], tuple[int, int], int | None, int | None]:
        """Execute one step with two exploratory learning players.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics are recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number streams used for exploratory action selection.

        Returns
        -------
        tuple[int, int, tuple[int, int], tuple[int, int], int | None, int | None]
            Selected actions, next states, and the rewards used for learning.
        """
        # Generate 2 * 4 random numbers, then use whichever is needed. Generate all of them to make sure we go through the RN list consistently
        player1_rn_1 = random_numbers_stream.player_streams[Game.PLAYER_1][0].uniform(0,
                                                                                      1)  # Random move when Q-table is empty
        player2_rn_1 = random_numbers_stream.player_streams[Game.PLAYER_2][0].uniform(0,
                                                                                      1)  # Random move when Q-table is empty
        player1_rn_2 = random_numbers_stream.player_streams[Game.PLAYER_1][1].uniform(0,
                                                                                      1)  # Probability to compare against eps
        player2_rn_2 = random_numbers_stream.player_streams[Game.PLAYER_2][1].uniform(0,
                                                                                      1)  # Probability to compare against eps
        player1_rn_3 = random_numbers_stream.player_streams[Game.PLAYER_1][2].uniform(0, 1)  # Random move due to eps
        player2_rn_3 = random_numbers_stream.player_streams[Game.PLAYER_2][2].uniform(0, 1)  # Random move due to eps
        # There is also a 4th random number, used to generate move for a static agent with strategy==’random’, and a
        # 5th random number -  used to generate the initial state within the main script

        state_index_player1 = self.state_index_converter[state_p1]
        state_index_player2 = self.state_index_converter[state_p2]

        action_player1, eps_player1, reason_player1, rns_player1 = self.player1.make_exploratory_move(
            state=state_index_player1, iteration=iteration, num_iter=num_iter,
            random_numbers=np.array([player1_rn_1, player1_rn_2, player1_rn_3], dtype=object))
        action_player2, eps_player2, reason_player2, rns_player2 = self.player2.make_exploratory_move(
            state=state_index_player2, iteration=iteration, num_iter=num_iter,
            random_numbers=np.array([player2_rn_1, player2_rn_2, player2_rn_3], dtype=object))

        # Save the key information as next_state for each agent - of shape (action_opponent, action_own)
        next_state_player1 = (action_player2, action_player1)
        next_state_player2 = (action_player1, action_player2)

        # Calculate reward - extrinsic (from the game scores), intrinsic (based on moral rule of the player), collective
        reward_game_player1 = self.extrinsic_reward.reward(action_p1=action_player1, action_p2=action_player2)[0]
        reward_game_player2 = self.extrinsic_reward.reward(action_p1=action_player2, action_p2=action_player1)[0]

        self.intrinsic_reward_p1.update_state(state=state_p1)
        reward_intrinsic_player1 = self.intrinsic_reward_p1.reward(action_p1=action_player1, action_p2=action_player2)
        self.intrinsic_reward_p2.update_state(state=state_p2)
        reward_intrinsic_player2 = self.intrinsic_reward_p2.reward(action_p1=action_player2, action_p2=action_player1)

        reward_collective = self.utilitarian_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_ratio = self.virtue_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_gini = self.gini_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_min = self.minimum_reward.reward(action_p1=action_player1, action_p2=action_player2)

        # append values to the history dataframe - used for plotting later
        global_history.loc[iteration, ['state_player1', 'action_player1', 'state_player2', 'action_player2']] = [
            state_p1, action_player1, state_p2, action_player2]

        global_history.loc[
            iteration, ['reward_game_player1', 'next_state_player1', 'reward_game_player2', 'next_state_player2']] = [
            reward_game_player1, next_state_player1, reward_game_player2, next_state_player2]

        global_history.loc[
            iteration, ['reward_intrinsic_player1', 'reward_intrinsic_player2', 'reward_collective', 'reward_ratio',
                        'reward_gini', 'reward_min']] = [
            reward_intrinsic_player1, reward_intrinsic_player2, reward_collective, reward_ratio, reward_gini,
            reward_min]

        global_history.loc[iteration, ['eps_player1', 'eps_player2', 'reason_player1', 'reason_player2']] = [
            eps_player1, eps_player2, reason_player1, reason_player2]

        global_history.loc[iteration, ['RNs_player1', 'RNs_player2']] = [
            str(rns_player1), str(rns_player2)]

        if self.player1.strategy.value[1] == Morality.SELFISH:
            reward_learning_player1 = reward_game_player1
        else:
            # In case the moral type is one of Utilitarian, Deontological or VirtueEthics
            reward_learning_player1 = reward_intrinsic_player1

        if self.player2.strategy.value[1] == Morality.SELFISH:
            reward_learning_player2 = reward_game_player2
        else:
            # In case the moral type is one of Utilitarian, Deontological or VirtueEthics
            reward_learning_player2 = reward_intrinsic_player2

        global_history.loc[iteration, ['reward_learning_player1', 'reward_learning_player2']] = [
            reward_learning_player1, reward_learning_player2]

        return int(action_player1), int(
            action_player2), next_state_player1, next_state_player2, reward_learning_player1, reward_learning_player2

    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                   global_history: DataFrame, num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, tuple[int, int], tuple[int, int], int | None]:
        """Execute one step with a learning player against a fixed player.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by the learning player.
        state_p2 : tuple[int, int]
            Current state perceived by the fixed player.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics are recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number streams used for action selection.

        Returns
        -------
        tuple[int, tuple[int, int], tuple[int, int], int | None]
            Learning player's action, both next states, and the learning reward.
        """
        # Generate 2 * 4 random numbers, then use whichever is needed. Generate all of them to make sure we go through the RN list consistently
        # Only using some of the RNs generated, since one of the players is static
        player1_rn_1 = random_numbers_stream.player_streams[Game.PLAYER_1][0].uniform(0,
                                                                                      1)  # Random move when Q-table is empty
        player1_rn_2 = random_numbers_stream.player_streams[Game.PLAYER_1][1].uniform(0,
                                                                                      1)  # Probability to compare against eps
        player1_rn_3 = random_numbers_stream.player_streams[Game.PLAYER_1][2].uniform(0, 1)  # Random move due to eps
        player2_rn_4 = random_numbers_stream.player_streams[Game.PLAYER_2][3].uniform(0,
                                                                                      1)  # Move for a static agent with strategy==’random’

        state_index_player1 = self.state_index_converter[state_p1]

        action_player1, eps_player1, reason_player1, rns_player1 = self.player1.make_exploratory_move(
            state=state_index_player1, iteration=iteration, num_iter=num_iter,
            random_numbers=np.array([player1_rn_1, player1_rn_2, player1_rn_3]))
        action_player2 = self.player2.make_fixed_move(state=state_p2, player_rn_spawn_4=player2_rn_4)

        # Save the key information as next_state for each agent
        # Note we record state with opponent's move first, then own movement
        next_state_player1 = (action_player2, action_player1)
        next_state_player2 = (action_player1, action_player2)  # Note this does not really get used as player2 is static

        # calculate reward - extrinsic (from the game scores), intrinsic (based on moral rule of the player), collective
        reward_game_player1 = self.extrinsic_reward.reward(action_p1=action_player1, action_p2=action_player2)[0]
        reward_game_player2 = self.extrinsic_reward.reward(action_p1=action_player2, action_p2=action_player1)[0]

        self.intrinsic_reward_p1.update_state(state=state_p1)
        reward_intrinsic_player1 = self.intrinsic_reward_p1.reward(action_p1=action_player1, action_p2=action_player2)

        reward_collective = self.utilitarian_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_ratio = self.virtue_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_gini = self.gini_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_min = self.minimum_reward.reward(action_p1=action_player1, action_p2=action_player2)

        # append values to the history dataframe - used for plotting later
        global_history.loc[iteration, ['state_player1', 'action_player1', 'state_player2', 'action_player2']] = [
            state_p1, action_player1, state_p2, action_player2]

        global_history.loc[
            iteration, ['reward_game_player1', 'next_state_player1', 'reward_game_player2', 'next_state_player2']] = [
            reward_game_player1, next_state_player1, reward_game_player2, next_state_player2]

        global_history.loc[iteration, ['reward_intrinsic_player1', 'reward_collective', 'reward_ratio', 'reward_gini',
                                       'reward_min']] = [
            reward_intrinsic_player1, reward_collective, reward_ratio, reward_gini, reward_min]

        global_history.loc[iteration, ['eps_player1', 'reason_player1', 'RNs_player1']] = [
            eps_player1, reason_player1, str(rns_player1)]
        # Note if we do not use str() here, this throws and error about creating np arrray from ragged nested sequences - ignore for now

        if self.player1.strategy.value[1] == Morality.SELFISH:
            reward_learning_player1 = reward_game_player1
        else:
            # In case the moral type is one of Utilitarian, Deontological or VirtueEthics
            reward_learning_player1 = reward_intrinsic_player1

        global_history.loc[iteration, ['reward_learning_player1']] = [reward_learning_player1]

        return int(action_player1), next_state_player1, next_state_player2, reward_learning_player1

    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                    global_history: DataFrame, random_numbers_stream: RandomNumberGenerator) -> tuple[
        tuple[int, int], tuple[int, int]]:
        """Execute one step with two fixed-strategy players.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics are recorded.
        random_numbers_stream : RandomNumberGenerator
            Random number streams used by random fixed strategies.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int]]
            Next states for player 1 and player 2.
        """
        # Generate 2 * 4 random numbers, then use whichever is needed. Generate all of them to make sure we go through the RN list consistently
        # RN_1 to RN_3 not used by player1 or player2
        player1_rn_4 = random_numbers_stream.player_streams[Game.PLAYER_1][3].uniform(0,
                                                                                      1)  # move for a static agent with strategy==’random’
        player2_rn_4 = random_numbers_stream.player_streams[Game.PLAYER_2][3].uniform(0,
                                                                                      1)  # move for a static agent with strategy==’random’

        action_player1 = self.player1.make_fixed_move(state=state_p1, player_rn_spawn_4=player1_rn_4)
        action_player2 = self.player2.make_fixed_move(state=state_p2, player_rn_spawn_4=player2_rn_4)

        # Save the key information as next_state for each agent #NOTE we record state with opponent's move first, then own movement
        next_state_player1 = (action_player2, action_player1)
        next_state_player2 = (action_player1, action_player2)

        # Calculate reward - extrinsic (from the game scores), collective
        reward_game_player1 = self.extrinsic_reward.reward(action_p1=action_player1, action_p2=action_player2)[0]
        reward_game_player2 = self.extrinsic_reward.reward(action_p1=action_player2, action_p2=action_player1)[0]

        reward_collective = self.utilitarian_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_ratio = self.virtue_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_gini = self.gini_reward.reward(action_p1=action_player1, action_p2=action_player2)
        reward_min = self.minimum_reward.reward(action_p1=action_player1, action_p2=action_player2)

        # Append values to the history dataframe - used for plotting later
        global_history.loc[iteration, ['state_player1', 'action_player1', 'state_player2', 'action_player2']] = [
            state_p1, action_player1, state_p2, action_player2]

        global_history.loc[
            iteration, ['reward_game_player1', 'next_state_player1', 'reward_game_player2', 'next_state_player2']] = [
            reward_game_player1, next_state_player1, reward_game_player2, next_state_player2]

        global_history.loc[iteration, ['reward_collective', 'reward_ratio', 'reward_gini', 'reward_min']] = [
            reward_collective, reward_ratio, reward_gini, reward_min]

        return next_state_player1, next_state_player2


class IterativeVolunteersDilemma(Game):
    """Placeholder for an iterated volunteer's dilemma environment."""

    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, tuple[int, int], tuple[int, int], int | None, int | None]:
        """Execute one learning step for the volunteer's dilemma.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics would be recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number streams intended for stochastic action selection.

        Returns
        -------
        tuple[int, int, tuple[int, int], tuple[int, int], int | None, int | None]
            Selected actions, next states, and learning rewards for both
            players.

        Notes
        -----
        This method is currently a placeholder and has not been implemented.
        """
        pass

    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                   global_history: DataFrame,
                   num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, int, int, float, float]:
        """Execute one mixed-strategy step for the volunteer's dilemma.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by the learning player.
        state_p2 : tuple[int, int]
            Current state perceived by the fixed player.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics would be recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number streams intended for stochastic action selection.

        Returns
        -------
        tuple[int, int, int, int, float, float]
            Placeholder return signature for action, state, and reward values.

        Notes
        -----
        This method is currently a placeholder and has not been implemented.
        """
        pass

    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                    global_history: DataFrame,
                    random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        """Execute one fixed-strategy step for the volunteer's dilemma.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics would be recorded.
        random_numbers_stream : RandomNumberGenerator
            Random number streams intended for stochastic fixed strategies.

        Returns
        -------
        tuple[int, int, int, int, float, float]
            Placeholder return signature for action, state, and reward values.

        Notes
        -----
        This method is currently a placeholder and has not been implemented.
        """
        pass


class IterativeStagHuntDilemma(Game):
    """Placeholder for an iterated stag hunt environment."""

    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, tuple[int, int], tuple[int, int], int | None, int | None]:
        """Execute one learning step for the stag hunt dilemma.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics would be recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number streams intended for stochastic action selection.

        Returns
        -------
        tuple[int, int, tuple[int, int], tuple[int, int], int | None, int | None]
            Selected actions, next states, and learning rewards for both
            players.

        Notes
        -----
        This method is currently a placeholder and has not been implemented.
        """
        pass

    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                   global_history: DataFrame,
                   num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, int, int, float, float]:
        """Execute one mixed-strategy step for the stag hunt dilemma.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by the learning player.
        state_p2 : tuple[int, int]
            Current state perceived by the fixed player.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics would be recorded.
        num_iter : int
            Total number of training iterations.
        random_numbers_stream : RandomNumberGenerator
            Random number streams intended for stochastic action selection.

        Returns
        -------
        tuple[int, int, int, int, float, float]
            Placeholder return signature for action, state, and reward values.

        Notes
        -----
        This method is currently a placeholder and has not been implemented.
        """
        pass

    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                    global_history: DataFrame,
                    random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        """Execute one fixed-strategy step for the stag hunt dilemma.

        Parameters
        ----------
        state_p1 : tuple[int, int]
            Current state perceived by player 1.
        state_p2 : tuple[int, int]
            Current state perceived by player 2.
        iteration : int
            Current iteration index.
        global_history : DataFrame
            DataFrame where step-level diagnostics would be recorded.
        random_numbers_stream : RandomNumberGenerator
            Random number streams intended for stochastic fixed strategies.

        Returns
        -------
        tuple[int, int, int, int, float, float]
            Placeholder return signature for action, state, and reward values.

        Notes
        -----
        This method is currently a placeholder and has not been implemented.
        """
        pass
