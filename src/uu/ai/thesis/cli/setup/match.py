# MIT License - 2026
"""Helpers for running IPD matches and persisting experiment outputs.

This module contains utilities for creating player pairs, executing episode
loops for static and mixed settings, and exporting raw histories and learning
artifacts to disk.
"""
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
from uu.ai.thesis.core.rl.types import Strategy, Morality
from uu.ai.thesis.core.utility.functions import NonLinearUtility


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
    utility_p1 = NonLinearUtility(phi=game_config.phi) if game_config.morl and strategy_p1.value[1] != Morality.SELFISH else None
    utility_p2 = NonLinearUtility(phi=game_config.phi) if game_config.morl and strategy_p2.value[1] != Morality.SELFISH else None

    pairs_of_players = [
        (Player(strategy=strategy_p1, eps_theta=game_config.eps_theta, eps_decay=game_config.eps_decay,
                mixed_beta=game_config.mixed_beta, utility=utility_p1),
         Player(strategy=strategy_p2, eps_theta=game_config.eps_theta, eps_decay=game_config.eps_decay,
                mixed_beta=game_config.mixed_beta, utility=utility_p2))
        for _ in range(num_runs)]

    return pairs_of_players


def _reward_vector_to_array(reward_vector: dict[str, float], num_objectives: int) -> np.ndarray:
    """Convert a named MORL reward vector to the Q-table objective order.

    Parameters
    ----------
    reward_vector : dict[str, float]
        Mapping containing the MORL objectives. The expected objective names
        are ``"moral"`` and ``"individual"``.
    num_objectives : int
        Number of objectives to include from the canonical objective order.

    Returns
    -------
    np.ndarray
        Reward vector ordered as ``[moral, individual]`` and truncated to
        ``num_objectives``.
    """
    objective_names = ("moral", "individual")
    return np.array([float(reward_vector[name]) for name in objective_names[:num_objectives]])


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
    # TODO [Wilder]: We have to use an index for the column names to avoid issues.
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

    # # Store cumulative_reward_game [extrinsic]
    # cumulative_reward_game_player1 = np.cumsum(df_reward_game_player1["episode"])
    # cumulative_reward_game_player2 = np.cumsum(df_reward_game_player2["episode"])
    #
    # np.savetxt(os.fspath(player1_path / "df_cumulative_reward_game.csv"), cumulative_reward_game_player1, delimiter=',')
    # np.savetxt(os.fspath(player2_path / "df_cumulative_reward_game.csv"), cumulative_reward_game_player2, delimiter=',')
    # logger.info("Saved cumulative game [extrinsic] rewards for players 1 and 2.")
    #
    # # Store cumulative_reward_intrinsic
    # cumulative_reward_intrinsic_player1 = np.cumsum(df_reward_intrinsic_player1["episode"])
    # cumulative_reward_intrinsic_player2 = np.cumsum(df_reward_intrinsic_player2["episode"])
    #
    # np.savetxt(os.fspath(player1_path / "df_cumulative_reward_intrinsic.csv"), cumulative_reward_intrinsic_player1,
    #            delimiter=',')
    # np.savetxt(os.fspath(player2_path / "df_cumulative_reward_intrinsic.csv"), cumulative_reward_intrinsic_player2,
    #            delimiter=',')
    # logger.info("Saved cumulative intrinsic reward")

    # Store collective reward
    df_reward_collective = pd.concat([df["reward_collective"] for df in rewards_dataframes], axis=1)
    df_reward_collective = df_reward_collective.set_axis(axis_labels, axis=1)
    df_reward_collective.to_csv(os.fspath(destination_folder / "df_reward_collective.csv"))
    logger.info("Saved collective reward")

    # # Store cumulative_reward_collective
    # cumulative_reward_collective = np.cumsum(df_reward_collective["episode"])
    #
    # np.savetxt(os.fspath(destination_folder / "df_cumulative_reward_collective.csv"), cumulative_reward_collective,
    #            delimiter=',')
    # logger.info("Saved cumulative collective reward")

    # Store gini reward
    df_reward_gini = pd.concat([df["reward_gini"] for df in rewards_dataframes], axis=1)
    df_reward_gini = df_reward_gini.set_axis(axis_labels, axis=1)
    df_reward_gini.to_csv(os.fspath(destination_folder / "df_reward_gini.csv"))
    logger.info("Saved gini reward")

    # # Store cumulative_reward_gini
    # cumulative_reward_gini = np.cumsum(df_reward_gini["episode"])
    #
    # np.savetxt(os.fspath(destination_folder / "df_cumulative_reward_gini.csv"), cumulative_reward_gini, delimiter=',')
    # logger.info("Saved cumulative gini reward")

    # Store min reward
    df_reward_min = pd.concat([df["reward_min"] for df in rewards_dataframes], axis=1)
    df_reward_min = df_reward_min.set_axis(axis_labels, axis=1)
    df_reward_min.to_csv(os.fspath(destination_folder / "df_reward_min.csv"))
    logger.info("Saved min reward")
    #
    # # Store cumulative_reward_min
    # cumulative_reward_min = np.cumsum(df_reward_min["episode"])
    #
    # np.savetxt(os.fspath(destination_folder / "df_cumulative_reward_min.csv"), cumulative_reward_min, delimiter=',')
    # logger.info("Saved cumulative min reward")

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


def store_learning_data(optimal_policies: list, q_values_player_1: list, q_values_player_2: list | None,
                        destination_folder: Path) -> None:
    """Persist learned policies and Q-value histories for mixed-strategy runs.

    Parameters
    ----------
    optimal_policies : list
        Learned policy summaries collected across runs, typically one entry per
        run describing the greedy action selected for each state.
    q_values_player_1 : list
        Sequence of player-1 Q-value histories accumulated during training,
        where each entry contains the per-iteration Q-table snapshots for one
        run.
    q_values_player_2 : list | None
        Optional sequence of player-2 Q-value histories. When ``None`` or
        empty, only player-1 learning data are written.
    destination_folder : Path
        Output folder where the NumPy and text representations of the learning
        artifacts are stored.

    Returns
    -------
    None
        This function writes learning artifacts to disk and does not return a
        value.

    Notes
    -----
    The function stores both ``.npy`` and ``.txt`` versions of the learned
    policies and Q-value traces. Player-2 artifacts are only exported when
    data are provided.
    """
    # Save RESULTS_list to numpy
    # TODO [Wilder]: Disable NPY saving for now.
    # np.save(destination_folder / "RESULTS_list.npy", optimal_policies, allow_pickle=True)

    # Save RESULTS_list to txt
    with open(destination_folder / "RESULTS_list.txt", 'w') as fp:
        for item in optimal_policies:
            # Write each item on a new line
            fp.write(f"\n{str(item)}")

    # Save Q_VALUES_list for player1 (learning over time) to .npy file
    # TODO [Wilder]: Disable NPY saving for now.
    # np.save(destination_folder / "Q_VALUES_player1_list.npy", q_values_player_1, allow_pickle=True)


    # Save Q_VALUES_list for each player (learning over time) to txt file
    with open(destination_folder / "Q_VALUES_player1_list.txt", 'w') as fp:
        for item in q_values_player_1:
            fp.write(f"\n{str(item)}")

    # Save Q_VALUES_list for player2 if available - if player2 is a QL player
    if q_values_player_2:
        # TODO [Wilder]: Disable NPY saving for now.
        # np.save(destination_folder / "Q_VALUES_player2_list.npy", q_values_player_2, allow_pickle=True)

        with open(destination_folder / "Q_VALUES_player2_list.txt", 'w') as fp:
            for item in q_values_player_2:
                fp.write(f"\n{str(item)}")
    else:
        logger.info("Player 2 is not a QL player, no Q Values to save.")
    logger.info("Done storing all available learning data.")


def run_one_episode_mixed(config: GameConfig, counter: int, destination_folder: Path, game: Game, num_iter: int,
                          random_numbers_stream: RandomNumberGenerator) -> tuple[DataFrame, np.ndarray, np.ndarray]:
    """Run one mixed episode with a learning player and record learning outputs.

    Parameters
    ----------
    config : GameConfig
        Resolved experiment configuration containing the learning-rate,
        discount-factor, and exploration settings used during Q-value updates.
    counter : int
        One-based run index used when initializing learning parameters and
        deciding whether to persist the Q-learning configuration file.
    destination_folder : Path
        Experiment output folder used for initialization side effects such as
        storing the Q-learning parameter snapshot on the first run.
    game : Game
        Configured game environment. The function uses its players, state-index
        converter, and :meth:`mixed_step` method to advance the episode and
        append step-level diagnostics to the shared history dataframe.
    num_iter : int
        Number of interactions to execute in the episode.
    random_numbers_stream : RandomNumberGenerator
        Reproducible random-number streams used for state initialization and
        any stochastic choices inside the environment dynamics.

    Returns
    -------
    tuple[DataFrame, np.ndarray, np.ndarray]
        Three-element tuple containing the episode history dataframe, the
        learned greedy policy per state for player 1, and the sequence of
        player-1 Q-table snapshots collected over training.

    Notes
    -----
    Player 1 is treated as the learning agent. Its Q-table is reset to zeros at
    the start of the episode, updated online after each call to
    :meth:`game.mixed_step <uu.ai.thesis.core.environment.game.Game.mixed_step>`,
    and copied into the returned history array before every iteration.
    """
    global_history = pd.DataFrame.from_dict({'state_player1': [None], 'action_player1': [None],
                                             'state_player2': [None], 'action_player2': [None],
                                             'reward_game_player1': [None], 'next_state_player1': [None],
                                             'reward_game_player2': [None], 'next_state_player2': [None],
                                             'reward_intrinsic_player1': [None], 'reward_intrinsic_player2': [None],
                                             # NB reward_intrinsic_player2 will remian empty
                                             'reward_collective': [None], 'reward_ratio': [None], 'reward_gini': [None],
                                             'reward_min': [None],
                                             'reward_learning_player1': [None], 'reward_learning_player2': [None],
                                             'eps_player1': [None], 'reason_player1': [None], 'RNs_player1': [None]})

    player_1 = game.player1

    if config.morl and player_1.utility is not None:
        player_1.q_values = np.zeros((config.num_states, config.num_actions, config.num_objectives))
    else:
        player_1.q_values = np.zeros((config.num_states, config.num_actions))
    state_index_converter = game.state_index_converter

    # Store myVars to allow the code to refer to a previously defined variable name - used to look up Q-value table for each agent
    state_player1, state_player2 = reset_learning_parameters(game_config=config, counter=counter,
                                                             destination_folder=destination_folder,
                                                             random_numbers_stream=random_numbers_stream)
    history_q_values_player_1 = []
    accumulated_return_player1 = np.zeros(config.num_objectives) if config.morl else None
    discount_step = 0

    for iteration in range(num_iter):  # e.g. num_iter - e.g. encounters
        history_q_values_player_1.append(player_1.q_values.copy())

        # Execute a step that interacts with the environment & updates global_history behind the scenes
        if config.morl:
            action_player1, next_state_player1, next_state_player2, reward_learning_player1 = game.mixed_step(
                state_player1,
                state_player2,
                iteration,
                global_history,
                num_iter,
                random_numbers_stream,
                accumulated_return_p1=accumulated_return_player1,
                discount_power=config.gamma ** discount_step,
            )
        else:
            action_player1, next_state_player1, next_state_player2, reward_learning_player1 = game.mixed_step(
                state_player1,
                state_player2,
                iteration,
                global_history,
                num_iter,
                random_numbers_stream,
            )

        state_index_player1 = state_index_converter[state_player1]

        next_state_index_player1 = state_index_converter[next_state_player1]

        alpha = config.alpha_theta / (1 + iteration * config.decay)

        if config.morl and player_1.utility is not None:
            # TODO [Wilder]: This is where the Q-learning update happens.
            scalar_values = player_1.scalarised_q_values(discount_power=config.gamma)
            next_action_player1 = NonLinearUtility.greedy_ser_action(scalar_values, next_state_index_player1)
            reward_vector_player1 = _reward_vector_to_array(reward_learning_player1, config.num_objectives)
            target_player1 = reward_vector_player1 + config.gamma * player_1.q_values[
                next_state_index_player1, next_action_player1
            ]
            player_1.q_values[state_index_player1, action_player1] *= 1 - alpha
            player_1.q_values[state_index_player1, action_player1] += alpha * target_player1
            accumulated_return_player1 += (config.gamma ** discount_step) * reward_vector_player1
            discount_step += 1
        else:
            reward_scalar_player1 = player_1.scalarise_reward(reward_learning_player1)
            next_value_player1 = np.max(player_1.q_values[next_state_index_player1])
            player_1.q_values[state_index_player1, action_player1] *= 1 - alpha
            player_1.q_values[state_index_player1, action_player1] += alpha * (
                    reward_scalar_player1 + config.gamma * next_value_player1)
        state_player1 = next_state_player1
        state_player2 = next_state_player2

    history_q_values_player_1 = np.array(history_q_values_player_1)

    if config.morl and player_1.utility is not None:
        scalar_values = player_1.scalarised_q_values()
        q_values = player_1.q_values
        result = NonLinearUtility.optimal_ser_policy(scalar_values, q_values, config.num_states)
    else:
        result = np.zeros(config.num_states)
        for state in range(len(result)):
            if not np.any(player_1.q_values[state]):
                result[state] = None
            else:
                result[state] = np.argmax(player_1.q_values[state])

    return global_history, result, history_q_values_player_1


def run_one_episode_static(destination_folder: Path, game: Game, num_iter: int,
                           random_numbers_stream: RandomNumberGenerator) -> DataFrame:
    """Run one static-strategy episode and collect the recorded interaction history.

    Parameters
    ----------
    destination_folder : Path
        Experiment output folder. Its name is inspected to detect whether the
        second player uses TFT, in which case the initial state for player 2 is
        forced to cooperation-compatible ``(0, 0)``.
    game : Game
        Configured game environment used to execute each static interaction
        step. Its :meth:`static_step` method appends step-level data to the
        shared history dataframe and returns the next states for both players.
    num_iter : int
        Number of iterations to execute in the episode.
    random_numbers_stream : RandomNumberGenerator
        Reproducible random-number streams used to sample the initial states
        and any stochastic choices inside the game dynamics.

    Returns
    -------
    DataFrame
        Episode history containing the recorded states, actions, and reward
        signals accumulated across all iterations.

    Notes
    -----
    The returned dataframe is initialized with a placeholder first row of
    ``None`` values before the episode loop begins. During the loop,
    :meth:`game.static_step <uu.ai.thesis.core.environment.game.Game.static_step>`
    mutates this dataframe in place while the local player states are updated
    from the returned next-state tuples.
    """
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


def run_one_episode(config: GameConfig, counter: int, destination_folder: Path, game: Game, num_iter: int,
                    random_numbers_stream: RandomNumberGenerator) -> tuple[
    DataFrame, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Run one episode with two learning players and record training artifacts.

    Parameters
    ----------
    config : GameConfig
        Resolved experiment configuration containing the Q-learning
        hyperparameters used for online value updates, including the learning
        rate, decay, and discount factor.
    counter : int
        One-based run index used during state initialization and for deciding
        whether to persist the Q-learning parameter snapshot on the first run.
    destination_folder : Path
        Experiment output folder used by the initialization routine when
        writing auxiliary run metadata.
    game : Game
        Configured game environment with two learning players. The function
        uses its players, state-index converter, and :meth:`step` method to
        advance the episode and append step-level diagnostics to the shared
        history dataframe.
    num_iter : int
        Number of training interactions to execute in the episode.
    random_numbers_stream : RandomNumberGenerator
        Reproducible random-number streams used for state initialization and
        stochastic exploration inside the environment.

    Returns
    -------
    tuple[DataFrame, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]
        Four-element tuple containing the episode history dataframe, the
        learned greedy policies for player 1 and player 2, the sequence of
        player-1 Q-table snapshots, and the sequence of player-2 Q-table
        snapshots.

    Notes
    -----
    Both players are treated as Q-learning agents. Their Q-tables are reset to
    zeros at the start of the episode, copied before each iteration, and then
    updated online from the rewards and next-state values returned by
    :meth:`game.step <uu.ai.thesis.core.environment.game.Game.step>`.
    """
    global_history = pd.DataFrame.from_dict({'state_player1': [None], 'action_player1': [None],
                                             'state_player2': [None], 'action_player2': [None],
                                             'reward_game_player1': [None], 'next_state_player1': [None],
                                             'reward_game_player2': [None], 'next_state_player2': [None],
                                             'reward_intrinsic_player1': [None], 'reward_intrinsic_player2': [None],
                                             'reward_collective': [None], 'reward_ratio': [None], 'reward_gini': [None],
                                             'reward_min': [None],
                                             'reward_learning_player1': [None], 'reward_learning_player2': [None],
                                             'eps_player1': [None], 'eps_player2': [None], 'reason_player1': [None],
                                             'reason_player2': [None],
                                             'RNs_player1': [None], 'RNs_player2': [None]})

    # Q-Learning:
    player_1 = game.player1
    player_2 = game.player2

    num_states = config.num_states
    num_actions = config.num_actions
    num_objectives = config.num_objectives
    if config.morl and num_objectives == 0 or num_objectives is None:
        raise ValueError("MORL requires num_objectives.")

    if config.morl and player_1.utility is not None:
        player_1.q_values = np.zeros((num_states, num_actions, num_objectives))
    else:
        player_1.q_values = np.zeros((num_states, num_actions))

    if config.morl and player_2.utility is not None:
        player_2.q_values = np.zeros((num_states, num_actions, num_objectives))
    else:
        player_2.q_values = np.zeros((num_states, num_actions))

    state_index_converter = game.state_index_converter

    # Store myVars to allow the code to refer to a previously defined variable name - used to look up Q-value table for each agent
    state_player1, state_player2 = reset_learning_parameters(game_config=config, counter=counter,
                                                             destination_folder=destination_folder,
                                                             random_numbers_stream=random_numbers_stream)

    history_q_values_player_1 = []
    history_q_values_player_2 = []
    accumulated_return_player1 = np.zeros(config.num_objectives) if config.morl else None
    accumulated_return_player2 = np.zeros(config.num_objectives) if config.morl else None
    discount_step = 0

    for iteration in range(num_iter):  # Default=10000 encounters of the game

        history_q_values_player_1.append(player_1.q_values.copy())
        history_q_values_player_2.append(player_2.q_values.copy())

        # Execute a step that interacts with the environment & updates global_history behind the scenes
        if config.morl:
            action_player1, action_player2, next_state_player1, next_state_player2, reward_learning_player1, reward_learning_player2 = game.step(
                state_player1,
                state_player2,
                iteration,
                global_history,
                num_iter,
                random_numbers_stream,
                accumulated_return_p1=accumulated_return_player1,
                accumulated_return_p2=accumulated_return_player2,
                discount_power=config.gamma ** discount_step,
            )
        else:
            action_player1, action_player2, next_state_player1, next_state_player2, reward_learning_player1, reward_learning_player2 = game.step(
                state_player1, state_player2, iteration, global_history, num_iter, random_numbers_stream)

        state_index_player1 = state_index_converter[state_player1]
        state_index_player2 = state_index_converter[state_player2]

        next_state_index_player1 = state_index_converter[next_state_player1]
        next_state_index_player2 = state_index_converter[next_state_player2]

        alpha = config.alpha_theta / (1 + iteration * config.decay)

        if config.morl and player_1.utility is not None:
            scalar_values = player_1.scalarised_q_values(discount_power=config.gamma)
            next_action_player1 = NonLinearUtility.greedy_ser_action(scalar_values, next_state_index_player1)
            reward_vector_player1 = _reward_vector_to_array(reward_learning_player1, config.num_objectives)
            target_player1 = reward_vector_player1 + config.gamma * player_1.q_values[
                next_state_index_player1, next_action_player1
            ]
            player_1.q_values[state_index_player1, action_player1] *= 1 - alpha
            player_1.q_values[state_index_player1, action_player1] += alpha * target_player1
        else:
            reward_scalar_player1 = player_1.scalarise_reward(reward_learning_player1)
            next_value_player1 = np.max(player_1.q_values[next_state_index_player1])
            player_1.q_values[state_index_player1, action_player1] *= 1 - alpha
            player_1.q_values[state_index_player1, action_player1] += alpha * (
                    reward_scalar_player1 + config.gamma * next_value_player1)
        state_player1 = next_state_player1

        if config.morl and player_2.utility is not None:
            scalar_values = player_2.scalarised_q_values(discount_power=config.gamma)
            next_action_player2 = NonLinearUtility.greedy_ser_action(scalar_values, next_state_index_player2)
            reward_vector_player2 = _reward_vector_to_array(reward_learning_player2, config.num_objectives)
            target_player2 = reward_vector_player2 + config.gamma * player_2.q_values[
                next_state_index_player2, next_action_player2
            ]
            player_2.q_values[state_index_player2, action_player2] *= 1 - alpha
            player_2.q_values[state_index_player2, action_player2] += alpha * target_player2
            accumulated_return_player1 += (config.gamma ** discount_step) * reward_vector_player1
            accumulated_return_player2 += (config.gamma ** discount_step) * reward_vector_player2
            discount_step += 1
        else:
            reward_scalar_player2 = player_2.scalarise_reward(reward_learning_player2)
            next_value_player2 = np.max(player_2.q_values[next_state_index_player2])
            player_2.q_values[state_index_player2, action_player2] *= 1 - alpha
            player_2.q_values[state_index_player2, action_player2] += alpha * (
                    reward_scalar_player2 + config.gamma * next_value_player2)
        state_player2 = next_state_player2

    history_q_values_player_1 = np.array(history_q_values_player_1)
    history_q_values_player_2 = np.array(history_q_values_player_2)

    if config.morl and player_1.utility is not None:
        scalar_values_p1 = player_1.scalarised_q_values()
        q_values_p1 = player_1.q_values
        result_player1 = NonLinearUtility.optimal_ser_policy(scalar_values_p1, q_values_p1, num_states)
    else:
        result_player1 = np.zeros(num_states)
        for state in range(len(result_player1)):
            if not np.any(player_1.q_values[state]):
                result_player1[state] = None
            else:
                result_player1[state] = np.argmax(player_1.q_values[state])
    if config.morl and player_2.utility is not None:
        scalar_values_p2 = player_2.scalarised_q_values()
        q_values_p2 = player_2.q_values
        result_player2 = NonLinearUtility.optimal_ser_policy(scalar_values_p2, q_values_p2, num_states)
    else:
        result_player2 = np.zeros(num_states)
        for state in range(len(result_player2)):
            if not np.any(player_2.q_values[state]):
                result_player2[state] = None
            else:
                result_player2[state] = np.argmax(player_2.q_values[state])

    result = (result_player1, result_player2)

    return global_history, result, history_q_values_player_1, history_q_values_player_2
