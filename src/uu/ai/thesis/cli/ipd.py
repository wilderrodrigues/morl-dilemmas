# MIT License - 2026
"""
Typer-based CLI for configuring iterated prisoner's dilemma runs.

This module exposes a small command-line interface that resolves experiment
parameters into a serialized configuration payload.
"""

from dataclasses import asdict
import json
from typing import Annotated

import typer

from uu.ai.thesis.core.data.model import build_game_config

app = typer.Typer(add_completion=False, help="Configure IPD experiment parameters.")


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


if __name__ == "__main__":
    app()
