# MIT License - 2026
"""Configuration models and default values for game experiments."""
from dataclasses import dataclass


DEFAULT_MASTER_SEED = 1
DEFAULT_NUM_ITERATIONS = 10000
DEFAULT_NUM_RUNS = 100
DEFAULT_EPS_THETA = 0.05
DEFAULT_ALPHA_THETA = 0.01
DEFAULT_DECAY = 0.0
DEFAULT_GAMMA = 0.9
DEFAULT_BETA = 0.5


@dataclass(frozen=True)
class GameConfig:
    """Resolved configuration for an iterated dilemma experiment.

    Parameters
    ----------
    game_type : str
        The game type to run, e.g. 'ipd', 'ish', 'ivd'.
    title1 : str
        Short title identifying player 1.
    title2 : str
        Short title identifying player 2.
    destination_folder : str
        Output folder name derived from the selected experiment parameters.
    master_seed : int
        Root seed used to initialize reproducible random number streams.
    num_iterations : int
        Number of iterations executed per run.
    num_runs : int
        Number of experiment replicas to execute.
    eps_theta : float
        Initial epsilon value used for exploration.
    eps_decay : bool
        Indicates whether epsilon decays over time.
    alpha_theta : float
        Initial learning rate used by Q-learning agents.
    decay : float
        Learning-rate decay factor.
    gamma : float
        Discount factor used in temporal-difference updates.
    mixed_beta : float
        Weighting coefficient for mixed virtue-ethics rewards.
    extra : str | None, optional
        Optional extra label appended to the destination folder name.
    """

    game_type: str
    title1: str
    title2: str
    destination_folder: str
    master_seed: int
    num_iterations: int
    num_runs: int
    eps_theta: float
    eps_decay: bool
    alpha_theta: float
    decay: float
    gamma: float
    mixed_beta: float
    extra: str | None = None


def build_game_config(
    game_type: str,
    title1: str,
    title2: str,
    master_seed: int | None = None,
    num_iterations: int | None = None,
    num_runs: int | None = None,
    eps_theta: float | None = None,
    eps_decay: bool = False,
    alpha_theta: float | None = None,
    decay: float | None = None,
    gamma: float | None = None,
    beta: float | None = None,
    extra: str | None = None,
) -> GameConfig:
    """Build a resolved game configuration and destination folder name.

    Parameters
    ----------
    game_type : str
        The game type to run, e.g. 'ipd', 'ish', 'ivd'.
    title1 : str
        Short title identifying player 1.
    title2 : str
        Short title identifying player 2.
    master_seed : int | None, optional
        Root seed used to initialize reproducible random number streams. If
        omitted, :data:`DEFAULT_MASTER_SEED` is used.
    num_iterations : int | None, optional
        Number of iterations executed per run. If omitted,
        :data:`DEFAULT_NUM_ITERATIONS` is used.
    num_runs : int | None, optional
        Number of experiment replicas to execute. If omitted,
        :data:`DEFAULT_NUM_RUNS` is used.
    eps_theta : float | None, optional
        Initial epsilon value used for exploration. If omitted,
        :data:`DEFAULT_EPS_THETA` is used.
    eps_decay : bool, optional
        Indicates whether epsilon decays over time.
    alpha_theta : float | None, optional
        Initial learning rate used by Q-learning agents. If omitted,
        :data:`DEFAULT_ALPHA_THETA` is used.
    decay : float | None, optional
        Learning-rate decay factor. If omitted, :data:`DEFAULT_DECAY` is used.
    gamma : float | None, optional
        Discount factor used in temporal-difference updates. If omitted,
        :data:`DEFAULT_GAMMA` is used.
    beta : float | None, optional
        Weighting coefficient for mixed virtue-ethics rewards. If omitted,
        :data:`DEFAULT_BETA` is used.
    extra : str | None, optional
        Optional extra label appended to the destination folder name.

    Returns
    -------
    GameConfig
        Immutable configuration object with resolved defaults and a derived
        destination folder name.
    """
    resolved_master_seed = DEFAULT_MASTER_SEED if master_seed is None else master_seed
    resolved_num_iterations = DEFAULT_NUM_ITERATIONS if num_iterations is None else num_iterations
    resolved_num_runs = DEFAULT_NUM_RUNS if num_runs is None else num_runs
    resolved_eps_theta = DEFAULT_EPS_THETA if eps_theta is None else eps_theta
    resolved_alpha_theta = DEFAULT_ALPHA_THETA if alpha_theta is None else alpha_theta
    resolved_decay = DEFAULT_DECAY if decay is None else decay
    resolved_gamma = DEFAULT_GAMMA if gamma is None else gamma
    resolved_beta = DEFAULT_BETA if beta is None else beta

    destination_folder = f"{title1}_{title2}"

    if master_seed is not None:
        destination_folder += f"_seed{resolved_master_seed}"
    if num_iterations is not None:
        destination_folder += f"_iter{resolved_num_iterations}"
    if num_runs is not None:
        destination_folder += f"_runs{resolved_num_runs}"
    if extra:
        destination_folder += f"_{extra}"
    if eps_theta is not None:
        destination_folder += f"_eps_theta{resolved_eps_theta}"
    if eps_decay:
        destination_folder += "_eps_decay"
    if alpha_theta is not None:
        destination_folder += f"_alpha_theta{resolved_alpha_theta}"
    if decay is not None:
        destination_folder += f"_decay{resolved_decay}"
    if gamma is not None:
        destination_folder += f"_gamma{resolved_gamma}"
    if beta is not None:
        destination_folder += f"_beta{resolved_beta}"

    return GameConfig(
        game_type=game_type,
        title1=title1,
        title2=title2,
        destination_folder=destination_folder,
        master_seed=resolved_master_seed,
        num_iterations=resolved_num_iterations,
        num_runs=resolved_num_runs,
        eps_theta=resolved_eps_theta,
        eps_decay=eps_decay,
        alpha_theta=resolved_alpha_theta,
        decay=resolved_decay,
        gamma=resolved_gamma,
        mixed_beta=resolved_beta,
        extra=extra,
    )
