# MIT License - 2026
"""
Typer-based CLI for configuring iterated prisoner's dilemma runs.

This module exposes a small command-line interface that resolves experiment
parameters into a serialized configuration payload.
"""
from dataclasses import asdict
import json
from pathlib import Path
from typing import Annotated

import typer

from uu import logger
from uu.ai.thesis.cli import payoff_matrices
from uu.ai.thesis.cli.setup.match import create_pair_of_players, store_raw_data, save_history, run_one_episode_static, \
    run_one_episode_mixed, store_learning_data, run_one_episode
from uu.ai.thesis.core.data.model import build_game_config, GameConfig
from uu.ai.thesis.core.environment.game import Game, IterativeMultiObjectiveGame, IterativeSingleObjectiveGame
from uu.ai.thesis.core.functions import RandomNumberGenerator
from uu.ai.thesis.core.rl.agent import Player
from uu.ai.thesis.core.rl.types import Strategy

app = typer.Typer(add_completion=False, help="Plays iterative matches with 2 players and different learning algorithms "
                                             "given a game type and a set of parameters.")


def create_game(config: GameConfig, player1: Player, player2: Player) -> Game:
    """Create the concrete game environment for an experiment.

    Parameters
    ----------
    config : GameConfig
        Resolved experiment configuration. The ``morl`` flag determines whether
        the game returns scalar learning rewards or vector-valued MORL rewards.
    player1 : Player
        First player in the game.
    player2 : Player
        Second player in the game.

    Returns
    -------
    Game
        Configured single-objective or multi-objective iterative game.
    """
    game_class = IterativeMultiObjectiveGame if config.morl else IterativeSingleObjectiveGame
    return game_class(player1, player2, payoff_matrices[config.game_type])


def run_static(config: GameConfig) -> None:
    """Run a static-strategy IPD experiment and persist per-run results.

    Parameters
    ----------
    config : GameConfig
        Resolved experiment configuration. The function uses the player titles,
        number of runs, number of iterations, destination folder, and master
        seed to create player pairs, generate shared random-number streams, run
        each episode, and store the resulting histories and aggregated raw
        data.

    Returns
    -------
    None
        This function writes experiment outputs to disk and does not return a
        value.

    Raises
    ------
    ValueError
        If either configured player title contains ``"QL"``, indicating a
        Q-learning strategy. This runner only supports static-strategy matchups.

    Notes
    -----
    Results are written under ``results/<destination_folder>``. A single
    :class:`~uu.ai.thesis.core.functions.RandomNumberGenerator` instance is
    created from ``config.master_seed`` and shared sequentially across all
    runs so each run consumes reproducible random streams from the same source.
    """
    title1 = config.title1
    title2 = config.title2
    num_runs = config.num_runs
    num_iterations = config.num_iterations
    destination_folder = config.destination_folder
    master_seed = config.master_seed

    logger.info(
        f"Running {title1} vs {title2}, {num_runs} runs, {num_iterations} iterations each, storing in {destination_folder}")

    if 'QL' in title1 or 'QL' in title2:
        raise ValueError("This is not the right function for these player types!")

    results_path = Path("results") / config.game_type / destination_folder
    results_path.mkdir(parents=True, exist_ok=True)

    strategy_p1 = Strategy[title1]
    strategy_p2 = Strategy[title2]
    pairs_of_players = create_pair_of_players(game_config=config, strategy_p1=strategy_p1, strategy_p2=strategy_p2,
                                              num_runs=num_runs)

    # Instantiate the RandomNumberGenerator before I run my n runs - so that all n runs share a single set of RN streams (4, to be exact) and read from it sequentially
    rng = RandomNumberGenerator(master_seed)
    rng.generate(results_path)

    counter = 0
    for player1, player2 in pairs_of_players:
        counter += 1
        game = create_game(config, player1, player2)
        global_history = run_one_episode_static(destination_folder=results_path, game=game,
                                                num_iter=num_iterations, random_numbers_stream=rng)
        save_history(history=global_history, run_idx=counter, destination_folder=results_path)
        logger.info(f"finished run {counter}, {title1} vs {title2}")

    store_raw_data(destination_folder=results_path, num_runs=num_runs)


def run_qlearning_vs_static(config: GameConfig) -> None:
    """Run mixed-strategy IPD experiments and persist histories and learning outputs.

    Parameters
    ----------
    config : GameConfig
        Resolved experiment configuration. The function uses the configured
        player titles, number of runs, number of iterations, destination
        folder, and master seed to build player pairs, initialize reproducible
        random-number streams, execute one mixed episode per run, and store
        both per-run histories and aggregated learning artifacts.

    Returns
    -------
    None
        This function writes experiment results to disk and does not return a
        value.

    Raises
    ------
    ValueError
        If player 1 is not a Q-learning strategy, identified by the absence of
        ``"QL"`` in ``config.title1``. This runner expects a mixed matchup with
        a learning player in the first position.

    Notes
    -----
    Results are written under ``results/<destination_folder>``. For each run,
    the function stores the episode history, accumulates the learned policy and
    player-1 Q-value traces, then writes aggregated raw data and learning data
    after all runs complete.
    """
    title1 = config.title1
    title2 = config.title2
    num_runs = config.num_runs
    num_iterations = config.num_iterations
    destination_folder = config.destination_folder
    master_seed = config.master_seed

    logger.info(
        f"Running {title1} vs {title2}, {num_runs} runs, {num_iterations} iterations each, storing in {destination_folder}")

    if 'QL' not in title1:
        raise ValueError("This is not the right function for these player types!")

    results_path = Path("results") / config.game_type / destination_folder
    results_path.mkdir(parents=True, exist_ok=True)

    strategy_p1 = Strategy[title1]
    strategy_p2 = Strategy[title2]
    pairs_of_players = create_pair_of_players(game_config=config, strategy_p1=strategy_p1, strategy_p2=strategy_p2,
                                              num_runs=num_runs)

    # Instantiate the RandomNumberGenerator before I run my n runs - so that all n runs share a single set of RN streams (4, to be exact) and read from it sequentially
    rng = RandomNumberGenerator(master_seed)
    rng.generate(results_path)

    optimal_policies = list()
    q_values_player1 = list()
    counter = 0
    for player1, player2 in pairs_of_players:
        counter += 1
        game = create_game(config, player1, player2)
        global_history, result, history_q_values_player1 = run_one_episode_mixed(config=config, counter=counter,
                                                                                 destination_folder=results_path,
                                                                                 game=game, num_iter=num_iterations,
                                                                                 random_numbers_stream=rng)
        save_history(history=global_history, run_idx=counter, destination_folder=results_path)
        optimal_policies.append(result)  # Save the optimal policies
        q_values_player1.append(history_q_values_player1)
        logger.info(f"Finished run {counter}, {title1} vs {title2}.")

    ## Store raw data - all 100 data points for each type of reward
    store_raw_data(destination_folder=results_path, num_runs=num_runs)

    # Store learnt optimal policies and learnt Q-values over time
    store_learning_data(optimal_policies=optimal_policies, q_values_player_1=q_values_player1, q_values_player_2=None,
                        destination_folder=results_path)


def run_qlearning_vs_qlearning(config: GameConfig) -> None:
    """Run experiments with two Q-learning players and persist all outputs.

    Parameters
    ----------
    config : GameConfig
        Resolved experiment configuration. The function uses the configured
        player titles, game type, number of runs, number of iterations,
        destination folder, and master seed to build player pairs, initialize
        reproducible random-number streams, execute one learning episode per
        run, and store both per-run histories and aggregated learning
        artifacts.

    Returns
    -------
    None
        This function writes experiment results to disk and does not return a
        value.

    Raises
    ------
    ValueError
        If either player title does not identify a Q-learning strategy,
        determined by the absence of ``"QL"`` in ``config.title1`` or
        ``config.title2``. This runner expects both players to learn.

    Notes
    -----
    Results are written under ``results/<destination_folder>``. For each run,
    the function stores the episode history, collects the learned policies for
    both players, and records both Q-value trajectories before writing the
    aggregated raw and learning outputs.
    """
    title1 = config.title1
    title2 = config.title2
    num_runs = config.num_runs
    num_iterations = config.num_iterations
    destination_folder = config.destination_folder
    master_seed = config.master_seed

    logger.info(
        f"Running {title1} vs {title2}, {num_runs} runs, {num_iterations} iterations each, storing in {destination_folder}")

    if 'QL' not in title1 or 'QL' not in title2:
        raise ValueError("This is not the right function for these player types!")

    results_path = Path("results") / config.game_type / destination_folder
    results_path.mkdir(parents=True, exist_ok=True)

    strategy_p1 = Strategy[title1]
    strategy_p2 = Strategy[title2]
    pairs_of_players = create_pair_of_players(game_config=config, strategy_p1=strategy_p1, strategy_p2=strategy_p2,
                                              num_runs=num_runs)

    # Instantiate the RN_generator before I run my n runs - so that all n runs share a single set of RN streams (4, to be exact) and read from it sequentially
    rng = RandomNumberGenerator(master_seed)
    rng.generate(results_path)

    optimal_policies = list()
    q_values_player1 = list()
    q_values_player2 = list()
    counter = 0
    for player1, player2 in pairs_of_players:
        counter += 1
        game = create_game(config, player1, player2)
        global_history, result, history_q_values_player1, history_q_values_player2 = run_one_episode(config=config,
                                                                                                   counter=counter,
                                                                                                   destination_folder=results_path,
                                                                                                   game=game,
                                                                                                   num_iter=num_iterations,
                                                                                                   random_numbers_stream=rng)
        optimal_policies.append(result)  # save the optimal policies
        q_values_player1.append(history_q_values_player1)
        q_values_player2.append(history_q_values_player2)
        save_history(history=global_history, run_idx=counter, destination_folder=results_path)
        logger.info(f"Finished run {counter}, {title1} vs {title2}")

    ## Store raw data - all 100 data points for each type of reward:
    store_raw_data(destination_folder=results_path, num_runs=num_runs)

    # Store learnt optimal policies and learnt Q-values over time
    store_learning_data(optimal_policies=optimal_policies, q_values_player_1=q_values_player1,
                        q_values_player_2=q_values_player2, destination_folder=results_path)


@app.command()
def main(
        title1: Annotated[str, typer.Option(help="Short title for player 1.")],
        title2: Annotated[str, typer.Option(help="Short title for player 2.")],
        game_type: Annotated[str, typer.Option(help="Game to run, e.g. 'ipd', 'ish', 'ivd'.")] = "ipd",
        morl: Annotated[bool, typer.Option(help="Enable Multi Object Reinforcement Learning.")] = False,
        num_states: Annotated[int, typer.Option(help="Number of states in the game.")] = 4,
        num_actions: Annotated[int, typer.Option(help="Number of actions available to each player.")] = 2,
        num_objectives: Annotated[int, typer.Option(help="Number of objectives available to each player.")] = 2,
        master_seed: Annotated[int | None, typer.Option(help="Master seed for reproducible random streams.")] = None,
        num_iterations: Annotated[int | None, typer.Option(help="Iterations per run.")] = None,
        num_runs: Annotated[int | None, typer.Option(help="Number of runs with different seeds.")] = None,
        eps_theta: Annotated[float | None, typer.Option(help="Initial exploration rate.")] = None,
        eps_decay: Annotated[bool, typer.Option("--eps-decay/--no-eps-decay", help="Enable epsilon decay.")] = False,
        alpha_theta: Annotated[float | None, typer.Option(help="Initial Q-learning rate.")] = None,
        decay: Annotated[float | None, typer.Option(help="Learning-rate decay for Q-learning.")] = None,
        gamma: Annotated[float | None, typer.Option(help="Discount factor for Q-learning.")] = None,
        beta: Annotated[float | None, typer.Option(help="Relative weighting for mixed virtue rewards.")] = None,
        phi: Annotated[float | None, typer.Option(help="Curvature parameter for non-linear moral utility.")] = None,
        extra: Annotated[str | None, typer.Option(help="Extra label to append to the destination folder.")] = None,
) -> None:
    """Print the resolved IPD configuration as JSON.

    Parameters
    ----------
    game_type: str
        The game type to run, e.g. 'ipd', 'ish', 'ivd'.
    title1 : str
        Short title identifying player 1.
    title2 : str
        Short title identifying player 2.
    morl : bool
        If True, the game is configured for Multi Objective Reinforcement Learning.
    num_states : int
        The number of states in the game.
    num_actions : int
        The number of actions available to each player.
    num_objectives : int
        The number of objectives available to each player.
    master_seed : int | None, optional
        Root seed used to initialize reproducible random number streams.
    num_iterations : int | None, optional
        Number of iterations executed per run.
    num_runs : int | None, optional
        Number of experiment replicas to execute.
    eps_theta : float | None, optional
        Initial epsilon value used for exploration.
    eps_decay : bool, optional
        Indicates whether epsilon decays over time.
    alpha_theta : float | None, optional
        Initial learning rate used by Q-learning agents.
    decay : float | None, optional
        Learning-rate decay factor.
    gamma : float | None, optional
        Discount factor used in temporal-difference updates.
    beta : float | None, optional
        Weighting coefficient for mixed virtue-ethics rewards.
    phi : float | None, optional
        Curvature parameter for the non-linear moral utility component.
    extra : str | None, optional
        Optional extra label appended to the destination folder name.
    """
    config = build_game_config(
        game_type=game_type,
        title1=title1,
        title2=title2,
        morl=morl,
        num_states=num_states,
        num_actions=num_actions,
        num_objectives=num_objectives,
        master_seed=master_seed,
        num_iterations=num_iterations,
        num_runs=num_runs,
        eps_theta=eps_theta,
        eps_decay=eps_decay,
        alpha_theta=alpha_theta,
        decay=decay,
        gamma=gamma,
        beta=beta,
        phi=phi,
        extra=extra,
    )
    typer.echo(json.dumps(asdict(config), indent=2, sort_keys=True))

    if 'QL' in title1:
        if 'QL' in title2:
            logger.info("Both players are Q-learning, running mixed strategy.")
            run_qlearning_vs_qlearning(config=config)
        else:
            logger.info("Player 2 is not a Q-learning player, running static strategy.")
            run_qlearning_vs_static(config=config)
    else:
        logger.info("Player 1 is not a Q-learning player, running static strategy.")
        run_static(config=config)


if __name__ == "__main__":
    app()
