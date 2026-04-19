# MIT License - 2026
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame

from uu import logger
from uu.ai.thesis.core.data.model import GameConfig
from uu.ai.thesis.core.environment.game import Game
from uu.ai.thesis.core.functions import RandomNumberGenerator
from uu.ai.thesis.core.rl.agent import Player
from uu.ai.thesis.core.rl.types import Strategy


def reset_learning_parameters(game_config: GameConfig, counter: int, destination_folder: Path,
                              random_numbers_stream: RandomNumberGenerator) -> tuple[tuple[int], tuple[int]]:
    """Initialize starting states and persist Q-learning parameters on the first run.

    Parameters
    ----------
    game_config : GameConfig
        Resolved experiment configuration containing the Q-learning
        hyperparameters to persist.
    counter : int
        One-based run counter used to detect the first iteration.
    destination_folder : Path
        Output folder for the current experiment run.
    random_numbers_stream : RandomNumberGenerator
        Reproducible random-number streams for both players.

    Returns
    -------
    tuple[tuple[int], tuple[int]]
        Initial state tuples for player 1 and player 2.
    """
    state_p1 = random_numbers_stream.player_streams[Game.PLAYER_1][4].choice([(0, 0), (0, 1), (1, 0), (1, 1)], 1)
    state_p1 = tuple(state_p1[0])

    if "TFT" in os.fspath(destination_folder):  # Specifically, if player2 is TFT
        # Hard-code as though TFT's opponent cooperated on the previous move, to force TFT to cooperate at first
        state_p2 = (0, 0)
    else:
        state_p2 = tuple(
            random_numbers_stream.player_streams[Game.PLAYER_2][4].choice([(0, 0), (0, 1), (1, 0), (1, 1)], 1))
        state_p2 = tuple(state_p2[0])

    # On the first iteration, save QL parameters to a file for future reference.
    if counter == 1:
        ql_parameters_path = destination_folder / "QL_parameters.txt"
        with open(ql_parameters_path, "w") as fp:
            fp.write("run 0, Q-learning parameters: \n")  # Write each item on a new line
            fp.write(f"Initial exploration rate eps_theta: {str(game_config.eps_theta)} \n")
            fp.write(f"Is eps (exploration) decay to 0 present: {str(game_config.eps_decay)} \n")
            fp.write(f"Initial learning rate alpha_theta: {str(game_config.alpha_theta)} \n")
            fp.write(f"Learning rate decay: {str(game_config.decay)} \n")
            fp.write(f"Discount factor gamma: {str(game_config.gamma)} \n")

            logger.info(f"Saved parameters for player1 & player2 under '{os.fspath(ql_parameters_path)}'.")

    return state_p1, state_p2


def create_pair_of_players(game_config: GameConfig, strategy_p1: Strategy, strategy_p2: Strategy, num_runs: int) -> \
        list[tuple[Player, Player]]:
    """Create one player pair per run using the configured strategies and exploration settings.

    Parameters
    ----------
    game_config : GameConfig
        Resolved experiment configuration containing player hyperparameters.
    strategy_p1 : Strategy
        Strategy assigned to player 1.
    strategy_p2 : Strategy
        Strategy assigned to player 2.
    num_runs : int
        Number of player pairs to instantiate.

    Returns
    -------
    list[tuple[Player, Player]]
        Player pairs for all requested runs.
    """
    pairs_of_players = [
        (Player(strategy=strategy_p1, eps_theta=game_config.eps_theta, eps_decay=game_config.eps_decay,
                mixed_beta=game_config.mixed_beta),
         Player(strategy=strategy_p2, eps_theta=game_config.eps_theta, eps_decay=game_config.eps_decay,
                mixed_beta=game_config.mixed_beta))
        for _ in range(num_runs)]

    return pairs_of_players


def store_raw_data(destination_folder: Path, num_runs: int) -> None:
    """Aggregate per-run CSV histories into experiment-level raw data outputs.

    Parameters
    ----------
    destination_folder : Path
        Output folder containing the per-run history files and target export
        locations.
    num_runs : int
        Number of run history files to load and aggregate.

    Returns
    -------
    None
        This function writes aggregated data files to disk and does not return
        a value.
    """
    run_history_path = destination_folder / "history"
    rewards_dataframes = [pd.read_csv(os.fspath(run_history_path / f"run{run_idx + 1}.csv"), index_col=0) for run_idx in
                          range(num_runs)]

    # First, make sure folders 'player1' and 'player2' exist within each results folder before storing data
    player1_path = destination_folder / "player1"
    if not player1_path.exists():
        player1_path.mkdir(parents=True, exist_ok=True)

    player2_path = destination_folder / "player2"
    if not player2_path.exists():
        player2_path.mkdir(parents=True, exist_ok=True)

    # Store game [extrinsic] reward
    axis_labels = ["episode" for i in range(num_runs)]

    df_reward_game_player1 = pd.concat([df["reward_game_player1"] for df in rewards_dataframes], axis=1)
    df_reward_game_player1 = df_reward_game_player1.set_axis(axis_labels, axis=1)
    df_reward_game_player1.to_csv(os.fspath(player1_path / "df_reward_game.csv"))

    df_reward_game_player2 = pd.concat([df['reward_game_player2'] for df in rewards_dataframes], axis=1)
    df_reward_game_player2 = df_reward_game_player2.set_axis(axis_labels, axis=1)
    df_reward_game_player2.to_csv(os.fspath(player2_path / "df_reward_game.csv"))
    logger.info("Saved game [extrinsic] rewards for players 1 and 2.")

    # Store intrinsic reward
    df_reward_intrinsic_player1 = pd.concat([df["reward_intrinsic_player1"] for df in rewards_dataframes], axis=1)
    df_reward_intrinsic_player1 = df_reward_intrinsic_player1.set_axis(axis_labels, axis=1)
    df_reward_intrinsic_player1.to_csv(os.fspath(player1_path / "df_reward_intrinsic.csv"))

    df_reward_intrinsic_player2 = pd.concat([df["reward_intrinsic_player2"] for df in rewards_dataframes], axis=1)
    df_reward_intrinsic_player2 = df_reward_intrinsic_player2.set_axis(axis_labels, axis=1)
    df_reward_intrinsic_player2.to_csv(os.fspath(player2_path / "df_reward_intrinsic.csv"))
    logger.info("Saved intrinsic rewards for players 1 and 2.")

    # Store cumulative_reward_game [extrinsic]
    cumulative_reward_game_player1 = np.cumsum(df_reward_game_player1["episode"])
    cumulative_reward_game_player2 = np.cumsum(df_reward_game_player2["episode"])

    np.savetxt(os.fspath(player1_path / "df_cumulative_reward_game.csv"), cumulative_reward_game_player1, delimiter=',')
    np.savetxt(os.fspath(player2_path / "df_cumulative_reward_game.csv"), cumulative_reward_game_player2, delimiter=',')
    logger.info("Saved cumulative game [extrinsic] rewards for players 1 and 2.")

    # Store cumulative_reward_intrinsic
    cumulative_reward_intrinsic_player1 = np.cumsum(df_reward_intrinsic_player1["episode"])
    cumulative_reward_intrinsic_player2 = np.cumsum(df_reward_intrinsic_player2["episode"])

    np.savetxt(os.fspath(player1_path / "df_cumulative_reward_intrinsic.csv"), cumulative_reward_intrinsic_player1,
               delimiter=',')
    np.savetxt(os.fspath(player2_path / "df_cumulative_reward_intrinsic.csv"), cumulative_reward_intrinsic_player2,
               delimiter=',')
    logger.info("Saved cumulative intrinsic reward")

    # Store collective reward
    df_reward_collective = pd.concat([df["reward_collective"] for df in rewards_dataframes], axis=1)
    df_reward_collective = df_reward_collective.set_axis(axis_labels, axis=1)
    df_reward_collective.to_csv(os.fspath(destination_folder / "df_reward_collective.csv"))
    logger.info("Saved collective reward")

    # Store cumulative_reward_collective
    cumulative_reward_collective = np.cumsum(df_reward_collective["episode"])

    np.savetxt(os.fspath(destination_folder / "df_cumulative_reward_collective.csv"), cumulative_reward_collective,
               delimiter=',')
    logger.info("Saved cumulative collective reward")

    # Store gini reward
    df_reward_gini = pd.concat([df["reward_gini"] for df in rewards_dataframes], axis=1)
    df_reward_gini = df_reward_gini.set_axis(axis_labels, axis=1)
    df_reward_gini.to_csv(os.fspath(destination_folder / "df_reward_gini.csv"))
    logger.info("Saved gini reward")

    # Store cumulative_reward_gini
    cumulative_reward_gini = np.cumsum(df_reward_gini["episode"])

    np.savetxt(os.fspath(destination_folder / "df_cumulative_reward_gini.csv"), cumulative_reward_gini, delimiter=',')
    logger.info("Saved cumulative gini reward")

    # Store min reward
    df_reward_min = pd.concat([df["reward_min"] for df in rewards_dataframes], axis=1)
    df_reward_min = df_reward_min.set_axis(axis_labels, axis=1)
    df_reward_min.to_csv(os.fspath(destination_folder / "df_reward_min.csv"))
    logger.info("Saved min reward")

    # Store cumulative_reward_min
    cumulative_reward_min = np.cumsum(df_reward_min["episode"])

    np.savetxt(os.fspath(destination_folder / "df_cumulative_reward_min.csv"), cumulative_reward_min, delimiter=',')
    logger.info("Saved cumulative min reward")

    # Store state
    df_state_player1 = pd.concat([df["state_player1"] for df in rewards_dataframes], axis=1)
    df_state_player1 = df_state_player1.set_axis(axis_labels, axis=1)
    df_state_player1.to_csv(os.fspath(destination_folder / "player1/state.csv"))

    df_state_player2 = pd.concat([df["state_player2"] for df in rewards_dataframes], axis=1)
    df_state_player2 = df_state_player2.set_axis(axis_labels, axis=1)
    df_state_player2.to_csv(os.fspath(destination_folder / "player2/state.csv"))
    logger.info("Saved state")

    # Store action
    df_action_player1 = pd.concat([df["action_player1"] for df in rewards_dataframes], axis=1)
    df_action_player1 = df_action_player1.set_axis(axis_labels, axis=1)
    df_action_player1.to_csv(os.fspath(destination_folder / "player1/action.csv"))

    df_action_player2 = pd.concat([df["action_player2"] for df in rewards_dataframes], axis=1)
    df_action_player2 = df_action_player2.set_axis(axis_labels, axis=1)
    df_action_player2.to_csv(os.fspath(destination_folder / "player2/action.csv"))
    logger.info("Saved action")

    logger.info("Done storing all raw data")


def save_history(history: DataFrame, run_idx: int, destination_folder: Path) -> None:
    """Persist the per-run history dataframe to the experiment history folder.

    Parameters
    ----------
    history : DataFrame
        Tabular run history containing the recorded episode data to export.
    run_idx : int
        One-based run index used to name the output CSV file.
    destination_folder : Path
        Output folder for the current experiment run where the ``history``
        subdirectory is created if needed.
    """
    history_path = destination_folder / "history"
    history_path.mkdir(parents=True, exist_ok=True)

    history.to_csv(os.fspath(destination_folder / "history" / f"run{run_idx}.csv"))


def run_one_episode_static(destination_folder: Path, game: Game, num_iter: int,
                           random_numbers_stream: RandomNumberGenerator):
    global_history = pd.DataFrame.from_dict({"state_player1": [None], "action_player1": [None],
                                             "state_player2": [None], "action_player2": [None],
                                             "reward_game_player1": [None], "next_state_player1": [None],
                                             "reward_game_player2": [None], "next_state_player2": [None],
                                             "reward_intrinsic_player1": [None], "reward_intrinsic_player2": [None],
                                             "reward_collective": [None], "reward_ratio": [None], "reward_gini": [None],
                                             "reward_min": [None]})

    state_player1 = random_numbers_stream.player_streams[Game.PLAYER_1][4].choice([(0, 0), (0, 1), (1, 0), (1, 1)], 1)
    state_player1 = tuple(state_player1[0])

    if "TFT" in os.fspath(destination_folder):  # Specifically, if player2 is TFT
        state_player2 = (0,
                         0)  # Hard-code to force TFT to cooperate at first (following the traditional definition of TFT)
    else:
        state_player2 = tuple(
            random_numbers_stream.player_streams[Game.PLAYER_2][4].choice([(0, 0), (0, 1), (1, 0), (1, 1)], 1))
        state_player2 = tuple(state_player2[0])

    for iteration in range(num_iter):
        # Executes a step that interacts with the environment & updates global_history behind the scenes
        next_state_player1, next_state_player2 = game.static_step(state_p1=state_player1, state_p2=state_player2,
                                                                  iteration=iteration, global_history=global_history,
                                                                  random_numbers_stream=random_numbers_stream)

        state_player1 = next_state_player1
        state_player2 = next_state_player2

    return global_history
