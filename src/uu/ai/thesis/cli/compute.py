# MIT License - 2026
"""CLI helpers for post-processing and plotting experiment results.

This module exposes a Typer command that normalizes result-folder names and
generates the standard plotting outputs for supported game-strategy pairs.
"""
import os
import shutil
from pathlib import Path
from typing import Annotated

import typer

from uu import logger
from uu.ai.thesis.plot.pair_results import plot_results, plot_actions, plot_action_types_area, plot_first_n_actions, \
    visualise_first_n_actions_matrix, plot_last_n_actions, visualise_last_n_actions_matrix, plot_action_pairs

app = typer.Typer(add_completion=False, help="Computes the results iterative experiments for different strategies.")

title_mapping = {'AC': 'AlwaysCooperate', 'AD': 'AlwaysDefect', 'TFT': 'TitForTat', 'Random': 'random',
                 'QLS': 'Selfish', 'QLUT': 'Utilitarian', 'QLDE': 'Deontological', 'QLVE_e': 'VirtueEthics_equality',
                 'QLVE_k': 'VirtueEthics_kindness', 'QLVM': 'VirtueEthics_mixed'}


def clear_previous_runs(results_path: Path) -> None:
    """Normalize result-folder names produced by earlier parameterized runs.

    Parameters
    ----------
    results_path : Path
        Root directory containing experiment result subdirectories. Any
        subdirectory whose name contains ``"_eps_theta1.0_eps_decay"`` is
        renamed to remove that suffix.
    """
    for dirname in os.listdir(results_path):
        if "_eps_theta1.0_eps_decay" in dirname:
            # Get rid of formatting that reflects our parameter choice
            new_dir = results_path / dirname.replace("_eps_theta1.0_eps_decay", '')
            if new_dir.exists():
                logger.info(f"Old run exists. Removing directory '{new_dir}' to be able to rename the new run.")
                shutil.rmtree(os.fspath(new_dir))
            os.rename(results_path / dirname, results_path / dirname.replace('_eps_theta1.0_eps_decay', ''))


def plot_results_for_pair(results_path: Path, game_type: str, num_runs: int) -> None:
    """Generate the standard plot suite for predefined strategy pairings.

    Parameters
    ----------
    results_path : Path
        Root directory containing one subdirectory per evaluated strategy
        pairing.
    game_type : str
        Short game identifier used when titling result plots, such as
        ``"ipd"``, ``"ivd"``, or ``"ish"``.
    num_runs : int
        Number of runs represented in the stored result files for each pairing.
    """
    ql_static_opponents = ['QLS_QLS', 'QLUT_QLS', 'QLDE_QLS', 'QLVE_e_QLS', 'QLVE_k_QLS']
    ext_moral_opponents = ['QLUT_QLUT', 'QLDE_QLUT', 'QLDE_QLDE', 'QLVE_e_QLUT']
    int_moral_opponents = ['QLVE_e_QLDE', 'QLVE_e_QLVE_e', 'QLVE_k_QLUT', 'QLVE_k_QLDE', 'QLVE_k_QLVE_e',
                           'QLVE_k_QLVE_k']
    static_opponents = ['QLS_AC', 'QLS_AD', 'QLS_TFT', 'QLS_Random', 'QLUT_AC', 'QLUT_AD', 'QLUT_TFT', 'QLUT_Random',
                        'QLDE_AC', 'QLDE_AD', 'QLDE_TFT', 'QLDE_Random', 'QLVE_e_AC', 'QLVE_e_AD', 'QLVE_e_TFT',
                        'QLVE_e_Random', 'QLVE_k_AC', 'QLVE_k_AD', 'QLVE_k_TFT', 'QLVE_k_Random']

    ql_static_opponents_path = [results_path / games_pair for games_pair in ql_static_opponents]
    ext_moral_opponents_path = [results_path / games_pair for games_pair in ext_moral_opponents]
    int_moral_opponents_path = [results_path / games_pair for games_pair in int_moral_opponents]
    static_opponents_path = [results_path / games_pair for games_pair in static_opponents]

    all_learners_paths = ql_static_opponents_path + ext_moral_opponents_path + int_moral_opponents_path + static_opponents_path
    pairs_index = 2
    for destination_folder in all_learners_paths:
        path_to_split = os.fspath(destination_folder)
        if "QLVE" not in path_to_split.split('/')[pairs_index]:
            short_titles = path_to_split.split('/')[pairs_index].split('_')[0:2]
        else:
            # Manually split the 'QLVE_' types
            short_titles = path_to_split.split('/')[2][0:6], path_to_split.split('/')[pairs_index][7:]
        long_titles = [title_mapping[title] for title in short_titles]
        logger.info(f"plotting results for: {long_titles} for games under the '{destination_folder}' folder.")

        plot_action_pairs(destination_folder=destination_folder, player1_title=long_titles[0],
                          player2_title=long_titles[1], n_runs=num_runs)
        plot_results(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1],
                     n_runs=num_runs, game_title=game_type.upper())
        plot_actions(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1],
                     n_runs=num_runs)
        plot_action_types_area(destination_folder=destination_folder, player1_title=short_titles[0],
                               player2_title=short_titles[1], n_runs=num_runs)
        plot_first_n_actions(destination_folder=destination_folder, player1_title=short_titles[0],
                             player2_title=short_titles[1], n_runs=num_runs)
        visualise_first_n_actions_matrix(destination_folder=destination_folder)
        plot_last_n_actions(destination_folder=destination_folder, player1_title=short_titles[0],
                            player2_title=short_titles[1], n_runs=num_runs)
        visualise_last_n_actions_matrix(destination_folder=destination_folder)


@app.command()
def main(
        game_type: Annotated[str, typer.Option(help="Game to run, e.g. 'ipd', 'ish', 'ivd'.")] = "ipd",
        num_runs: Annotated[int | None, typer.Option(help="Number of runs with different seeds.")] = 100,
) -> None:
    """Run the result post-processing pipeline for one game family.

    Parameters
    ----------
    game_type : str, optional
        Short game identifier used to locate the results directory and label
        generated plots.
    num_runs : int | None, optional
        Number of experiment runs expected in each result bundle.
    """
    results_path = Path("results") / game_type
    clear_previous_runs(results_path)

    plot_results_for_pair(results_path=results_path, game_type=game_type, num_runs=num_runs)


if __name__ == "__main__":
    app()
