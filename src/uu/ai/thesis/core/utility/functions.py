# MIT License - 2026
"""Utility functions for multi-objective thesis domain logic."""

from typing import Mapping
import numpy.typing as npt
import numpy as np


class NonLinearUtility:
    """Scalarisation utility for MORL moral and individual returns.

    Parameters
    ----------
    phi : float
        Curvature parameter applied to the moral objective. Values below one
        flatten the moral component, while values above one amplify larger
        moral magnitudes. Negative moral values are handled by signed power so
        fractional exponents do not produce complex numbers.
    moral_weight : float, optional
        Multiplicative weight applied to the non-linear moral component.
    individual_weight : float, optional
        Multiplicative weight applied to the individual payoff component.
    """

    def __init__(self, phi: float, moral_weight: float = 1.0, individual_weight: float = 1.0) -> None:
        """Initialize the utility parameters.

        Parameters
        ----------
        phi : float
            Curvature parameter for the signed moral objective.
        moral_weight : float, optional
            Weight applied to the transformed moral objective.
        individual_weight : float, optional
            Weight applied to the individual objective.
        """
        self.phi = phi
        self.moral_weight = moral_weight
        self.individual_weight = individual_weight

    @staticmethod
    def signed_power(value: float, exponent: float) -> float:
        """Raise a value to a power while preserving its sign.

        Parameters
        ----------
        value : float
            Input value to transform.
        exponent : float
            Exponent applied to the absolute magnitude of ``value``.

        Returns
        -------
        float
            ``sign(value) * abs(value) ** exponent``.
        """
        return float(np.sign(value) * (abs(value) ** exponent))

    def scalarise(self, return_vector: Mapping[str, float]) -> float:
        """Scalarise a moral-individual return vector.

        Parameters
        ----------
        return_vector : Mapping[str, float]
            Mapping containing ``"moral"`` and ``"individual"`` objective
            values.

        Returns
        -------
        float
            Scalar utility value used for SER action selection.
        """
        moral = return_vector["moral"]
        individual = return_vector["individual"]

        return self.moral_weight * self.signed_power(moral, self.phi) + self.individual_weight * individual

    def greedy_ser_action(self, scalar_values: npt.NDArray, state_index: int) -> int:
        """Select the greedy action from scalarised action values.

        Parameters
        ----------
        scalar_values : npt.NDArray
            Two-dimensional array indexed by state and action.
        state_index : int
            State row from which to select the greedy action.

        Returns
        -------
        int
            Action index with maximum scalarised value in the given state.
        """
        return int(np.argmax(scalar_values[state_index]))

    def optimal_ser_policy(self, scalar_q_values: npt.NDArray, q_values: npt.NDArray, num_states: int) -> np.ndarray:
        """Build a greedy policy from scalarised vector Q-values.

        Parameters
        ----------
        scalar_q_values : npt.NDArray
            Two-dimensional scalarised Q-table indexed by state and action.
        q_values : npt.NDArray
            Original Q-table used to detect states without learned values.
        num_states : int
            Number of states to include in the policy.

        Returns
        -------
        np.ndarray
            Greedy action per state. States with no learned values are assigned
            ``None``, which NumPy stores as ``nan`` for the current float array.
        """
        result = np.zeros(num_states)
        for state in range(num_states):
            if not np.any(q_values[state]):
                result[state] = None
            else:
                result[state] = np.argmax(scalar_q_values[state])

        return result
