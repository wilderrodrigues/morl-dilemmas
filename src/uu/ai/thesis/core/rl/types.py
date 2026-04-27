# MIT License - 2026
"""Enumerations describing agent strategies and moral reward models."""
from enum import Enum


class Morality(Enum):
    """Enumerate the supported moral choice models for an agent.

    The enum values identify the intrinsic moral reward formulation used by an
    agent in the social dilemma experiments.
    """

    UTILITARIAN = 1
    DEONTOLOGICAL = 2
    VIRTUE_ETHICS_EQUALITY = 3
    VIRTUE_ETHICS_KINDNESS = 4
    VIRTUE_ETHICS_MIXED = 5
    SELFISH = 6


class Strategy(Enum):
    """Enumerate the supported agent strategy configurations.

    Each enum value stores a tuple with the human-readable strategy label and
    the associated :class:`Morality` used to parameterize the agent.
    """

    AC = ("Always-Cooperate", Morality.SELFISH)
    AD = ("Always-Defect", Morality.SELFISH)
    TFT = ("Tit-For-Tat", Morality.SELFISH)
    Random = ("Random", Morality.SELFISH)
    QLS = ("Q-Learning eps-greedy", Morality.SELFISH)
    QLUT = ("Q-Learning eps-greedy", Morality.UTILITARIAN)
    QLDE = ("Q-Learning eps-greedy", Morality.DEONTOLOGICAL)
    QLVE_e = ("Q-Learning eps-greedy", Morality.VIRTUE_ETHICS_EQUALITY)
    QLVE_k = ("Q-Learning eps-greedy", Morality.VIRTUE_ETHICS_KINDNESS)
    QLVM = ("Q-Learning eps-greedy", Morality.VIRTUE_ETHICS_MIXED)

    @staticmethod
    def get_strategy(strategy: str) -> "Strategy":
        """Return the enum member for a strategy key.

        Parameters
        ----------
        strategy : str
            Name of the :class:`Strategy` enum member to resolve.

        Returns
        -------
        Strategy
            The matching strategy enum member.
        """
        return Strategy[strategy]

    @staticmethod
    def get_morality(strategy: str) -> Morality:
        """Return the morality associated with a strategy key.

        Parameters
        ----------
        strategy : str
            Name of the :class:`Strategy` enum member whose morality should be
            returned.

        Returns
        -------
        Morality
            The morality associated with the resolved strategy.
        """
        return Strategy.get_strategy(strategy).value[1]
