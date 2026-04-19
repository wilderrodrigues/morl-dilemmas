# MIT License - 2026
"""Payoff-matrix definitions for the CLI experiment entry points.

Attributes
----------
PAYOFF_MATRIX_IPD : list[list[tuple[int, int]]]
    Payoff matrix for the iterated prisoner's dilemma configuration.
PAYOFF_MATRIX_VOLUNTEER : list[list[tuple[int, int]]]
    Payoff matrix for the iterated volunteer's dilemma configuration.
PAYOFF_MATRIX_STAGHUNT : list[list[tuple[int, int]]]
    Payoff matrix for the iterated stag hunt configuration.
payoff_matrices : dict[str, list[list[tuple[int, int]]]]
    Mapping from short CLI game identifiers to their corresponding payoff
    matrices.
"""
PAYOFF_MATRIX_IPD = [ [(3,3),(1,4)] , [(4,1),(2,2)] ]
PAYOFF_MATRIX_VOLUNTEER = [ [(4,4),(2,5)] , [(5,2),(1,1)] ]
PAYOFF_MATRIX_STAGHUNT = [ [(5,5),(1,4)] , [(4,1),(2,2)] ]

payoff_matrices = {
    "ipd": PAYOFF_MATRIX_IPD,
    "ivd": PAYOFF_MATRIX_VOLUNTEER,
    "ish": PAYOFF_MATRIX_STAGHUNT,
}
