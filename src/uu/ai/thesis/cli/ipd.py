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
from uu.ai.thesis.cli import PAYOFF_MATRIX_IPD
from uu.ai.thesis.cli.setup.match import create_pair_of_players, store_raw_data, save_history, run_one_episode_static
from uu.ai.thesis.core.data.model import build_game_config, GameConfig
from uu.ai.thesis.core.environment.game import IterativePrisonersDilemma
from uu.ai.thesis.core.functions import RandomNumberGenerator
from uu.ai.thesis.core.rl.types import Strategy

app = typer.Typer(add_completion=False, help="Configure IPD experiment parameters.")


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

    results_path = Path("results") / destination_folder
    results_path.mkdir(parents=True, exist_ok=True)

    strategy_p1 = Strategy[title1]
    strategy_p2 = Strategy[title2]
    pairs_of_players = create_pair_of_players(game_config=config, strategy_p1=strategy_p1, strategy_p2=strategy_p2, num_runs=num_runs)

    # Instantiate the RandomNumberGenerator before I run my n runs - so that all n runs share a single set of RN streams (4, to be exact) and read from it sequentially
    rng = RandomNumberGenerator(master_seed)
    rng.generate(results_path)

    counter = 0
    for player1, player2 in pairs_of_players:
        counter += 1
        game = IterativePrisonersDilemma(player1, player2, PAYOFF_MATRIX_IPD)
        global_history = run_one_episode_static(destination_folder=results_path, game=game,
                                                num_iter=num_iterations, random_numbers_stream=rng)
        save_history(history=global_history, run_idx=counter, destination_folder=results_path)
        logger.info(f"finished run {counter}, {title1} vs {title2}")

    store_raw_data(destination_folder=results_path, num_runs=num_runs)


@app.command()
def main(
    title1: Annotated[str, typer.Option(help="Short title for player 1.")],
    title2: Annotated[str, typer.Option(help="Short title for player 2.")],
    master_seed: Annotated[int | None, typer.Option(help="Master seed for reproducible random streams.")] = None,
    num_iterations: Annotated[int | None, typer.Option(help="Iterations per run.")] = None,
    num_runs: Annotated[int | None, typer.Option(help="Number of runs with different seeds.")] = None,
    eps_theta: Annotated[float | None, typer.Option(help="Initial exploration rate.")] = None,
    eps_decay: Annotated[bool, typer.Option("--eps-decay/--no-eps-decay", help="Enable epsilon decay.")] = False,
    alpha_theta: Annotated[float | None, typer.Option(help="Initial Q-learning rate.")] = None,
    decay: Annotated[float | None, typer.Option(help="Learning-rate decay for Q-learning.")] = None,
    gamma: Annotated[float | None, typer.Option(help="Discount factor for Q-learning.")] = None,
    beta: Annotated[float | None, typer.Option(help="Relative weighting for mixed virtue rewards.")] = None,
    extra: Annotated[str | None, typer.Option(help="Extra label to append to the destination folder.")] = None,
) -> None:
    """Print the resolved IPD configuration as JSON.

    Parameters
    ----------
    title1 : str
        Short title identifying player 1.
    title2 : str
        Short title identifying player 2.
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
    extra : str | None, optional
        Optional extra label appended to the destination folder name.
    """
    config = build_game_config(
        title1=title1,
        title2=title2,
        master_seed=master_seed,
        num_iterations=num_iterations,
        num_runs=num_runs,
        eps_theta=eps_theta,
        eps_decay=eps_decay,
        alpha_theta=alpha_theta,
        decay=decay,
        gamma=gamma,
        beta=beta,
        extra=extra,
    )
    typer.echo(json.dumps(asdict(config), indent=2, sort_keys=True))

    run_static(config=config)


if __name__ == "__main__":
    app()
