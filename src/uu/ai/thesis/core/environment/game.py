# MIT License - 2026
from abc import ABC, abstractmethod

from pandas import DataFrame

from uu.ai.thesis.core.functions import RandomNumberGenerator


class Reward(ABC):

    @abstractmethod
    def reward(self, action_p1: int, action_p2: int) -> float:
        raise NotImplementedError


class ExtrinsicReward(Reward):

    def reward(self, action_p1: int, action_p2: int) -> float:
        pass


class IntrinsicReward(Reward):

    def reward(self, action_p1: int, action_p2: int) -> float:
        pass


class UtilitarianReward(Reward):

    def reward(self, action_p1: int, action_p2: int) -> float:
        pass


class VirtueReward(Reward):

    def reward(self, action_p1: int, action_p2: int) -> float:
        pass


class GiniReward(Reward):

    def reward(self, action_p1: int, action_p2: int) -> float:
        pass


class MinimumReward(Reward):

    def reward(self, action_p1: int, action_p2: int) -> float:
        pass


class Game(ABC):

    def __init__(self, player1, player2, payoff_format) -> None:
        self.player1 = player1
        self.player2 = player2
        self.payoff_format = payoff_format
        self.history = list()  # TODO make this a circular array / queue instead
        self.opponents = {player1: player2, player2: player1}
        self.state_index_converter = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    @abstractmethod
    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        raise NotImplementedError

    @abstractmethod
    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        raise NotImplementedError

    @abstractmethod
    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        raise NotImplementedError


class IterativePrisonersDilemma(Game):

    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        pass

    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        pass

    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        pass


class IterativeVolunteersDilemma(Game):

    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        pass

    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                   global_history: DataFrame,
                   num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, int, int, float, float]:
        pass

    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                    global_history: DataFrame,
                    random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        pass


class IterativeStagHuntDilemma(Game):

    def step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int, global_history: DataFrame,
             num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        pass

    def mixed_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                   global_history: DataFrame,
                   num_iter: int, random_numbers_stream: RandomNumberGenerator) -> tuple[
        int, int, int, int, float, float]:
        pass

    def static_step(self, state_p1: tuple[int, int], state_p2: tuple[int, int], iteration: int,
                    global_history: DataFrame,
                    random_numbers_stream: RandomNumberGenerator) -> tuple[int, int, int, int, float, float]:
        pass
