# MIT License - 2026
"""Plotting utilities for visualizing iterative game experiment outputs.

This module reads stored CSV artifacts from experiment runs and produces a
range of reward, action, and state-based visualizations for downstream
analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import dataframe_image as dfi

from uu import logger


def color_condition(map_value: str) -> str:
    """Map an action-state label to a cell background color.

    Parameters
    ----------
    map_value : str
        String label describing an action and state combination, such as
        ``"C | (C, C)"``.

    Returns
    -------
    str
        CSS style fragment specifying the background color for the given label.
    """
    # TODO [Wilder]: This function must be refactored into a more elegant solution.
    color = "#000000"
    if map_value == "C | (C, C)":
        color = "#28641E"
    elif map_value == "C | (C, D)":
        color = "#63A336"
    elif map_value == "C | (D, C)":
        color = "#B0DC82"
    elif map_value == "C | (D, D)":
        color = "#EBF6DC"
    elif map_value == "D | (C, C)":
        color = "#FBE6F1"
    elif map_value == "D | (C, D)":
        color = "#EEAED4"
    elif map_value == "D | (D, C)":
        color = "#CE4591"
    elif map_value == "D | (D, D)":
        color = "#8E0B52"
    return f"background-color: {color}"


def plot_action_pairs(destination_folder: Path, player1_title: str, player2_title: str, n_runs: int,
                      option: bool = False) -> None:
    """Plot the frequency of simultaneous action pairs over time.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing the per-player action histories.
    player1_title : str
        Human-readable title for player 1 used in the plot title.
    player2_title : str
        Human-readable title for player 2 used in the plot title.
    n_runs : int
        Number of runs aggregated in the stored action histories.
    option : bool, optional
        Flag controlling how the output filename prefix is derived.
    """
    # NOTE: This will only work for 10.000 iterations right now, not fewer!
    # NOTE: We plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    action_pairs_path = destination_folder / "action_pairs.csv"
    if action_pairs_path.exists():
        action_pairs_results = pd.read_csv(destination_folder / "action_pairs.csv", index_col=0)
    else:
        actions_player1 = pd.read_csv(destination_folder / "player1" / "action.csv", index_col=0)
        actions_player2 = pd.read_csv(destination_folder / "player2" / "action.csv", index_col=0)

        col_names = ["run" + str(i) for i in range(n_runs)]
        actions_player1.columns = col_names
        actions_player2.columns = col_names

        action_pairs_results = pd.DataFrame(columns=col_names)

        for col_name in col_names:
            str_value = actions_player1[col_name].astype(str) + ", " + actions_player2[col_name].astype(str)
            str_value = str_value.str.replace("1", "D")
            str_value = str_value.str.replace("0", "C")
            action_pairs_results[col_name] = str_value

        action_pairs_results.to_csv(destination_folder / "action_pairs.csv")

    results_counts = action_pairs_results.transpose().apply(lambda s: s.value_counts()).transpose()[1:]
    # Plot after the first episode, as in the first episode they are reacting to default state=0.
    results_counts.dropna(axis=1, how='all', inplace=True)

    plt.figure(dpi=80, figsize=(5, 4))
    plt.rcParams.update({'font.size': 20})
    results_counts.plot.area(stacked=True, ylabel='# action pairs observed',  # rot=45,
                             xlabel='Iteration',  # colormap='PiYG_r',
                             color={'C, C': '#28641E', 'C, D': '#B0DC82', 'D, C': '#EEAED4', 'D, D': '#8E0B52'},
                             linewidth=0.05, alpha=0.9,
                             # color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
                             title=str(
                                 player1_title.replace('Ethics', '').replace('_', '-') + ' vs ' + player2_title.replace(
                                     'Ethics', '').replace('_', '-')))  # Pairs of simultaneous actions over time: \n '+

    results_path = destination_folder.parent
    actions_results_path = results_path / "outcome_plots" / "actions"
    if not actions_results_path.exists():
        actions_results_path.mkdir(parents=True, exist_ok=True)

    if not option:
        # If plotting main action_pairs_results
        pair = os.fspath(destination_folder).split("/")[2]
    else:
        # If plotting extra parameter-search for phi in QLVM
        pair = destination_folder

    plt.savefig(actions_results_path / f"pairs_{pair}.pdf", bbox_inches="tight")
    plt.close()


def plot_results(destination_folder: Path, player1_title: str, player2_title: str, n_runs: int, game_title: str) -> None:
    """Generate reward plots for one experiment matchup.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing the aggregated reward CSV files.
    player1_title : str
        Human-readable title for player 1 used in legends and plot titles.
    player2_title : str
        Human-readable title for player 2 used in legends and plot titles.
    n_runs : int
        Number of runs aggregated in the reward summaries.
    game_title : str
        Game identifier used when selecting plot limits and titles.
    """
    plots_path = destination_folder / "plots"
    if not plots_path.exists():
        plots_path.mkdir(parents=True)

    outcome_plots_path = plots_path / "outcome_plots"
    if not outcome_plots_path.exists():
        outcome_plots_path.mkdir(parents=True)

    z_score = 1.96
    ##################################
    ####  Cumulative Game reward  ####
    ##################################
    try:
        df_player1 = pd.read_csv(destination_folder / "player1" / "df_cumulative_reward_game.csv", index_col=0)
    except FileNotFoundError:
        df_player1 = pd.read_csv(destination_folder / "player1" / "df_reward_game.csv", index_col=0)
        episode_cols = [col for col in df_player1.columns if col.startswith("episode")]
        df_player1 = df_player1[episode_cols].cumsum(axis=0)
        df_player1.to_csv(
            destination_folder / "player1" / "df_cumulative_reward_game.csv",
            index=False,
        )
        logger.info("Saved the cumulative game [extrinsic] rewards for player 1.")

    mean_player1 = df_player1.mean(axis=1)
    std_player1 = df_player1.std(axis=1)
    confidence_interval_player1 = z_score * std_player1 / np.sqrt(n_runs)

    try:
        df_player2 = pd.read_csv(destination_folder / "player2" / "df_cumulative_reward_game.csv", index_col=0)
    except FileNotFoundError:
        df_player2 = pd.read_csv(destination_folder / "player2" / "df_reward_game.csv", index_col=0)
        episode_cols = [col for col in df_player2.columns if col.startswith("episode")]
        df_player2 = df_player2[episode_cols].cumsum(axis=0)
        df_player2.to_csv(
            destination_folder / "player2" / "df_cumulative_reward_game.csv",
            index=False,
        )
        logger.info("Saved the cumulative game [extrinsic] rewards for player 2.")

    mean_player2 = df_player2.mean(axis=1)
    std_player2 = df_player2.std(axis=1)
    confidence_interval_player2 = z_score * std_player2 / np.sqrt(n_runs)

    plt.figure(dpi=80)  # figsize=(10, 6)
    plt.plot(df_player1.index[:], mean_player1[:], label=f"player1 - {player1_title}", color="blue")
    plt.plot(df_player2.index[:], mean_player2[:], label=f"player2 - {player2_title}", color="orange")
    plt.fill_between(df_player1.index[:], mean_player1 - confidence_interval_player1,
                     mean_player1 + confidence_interval_player1,
                     facecolor="#95d0fc", alpha=0.7)
    plt.fill_between(df_player2.index[:], mean_player2 - confidence_interval_player2,
                     mean_player2 + confidence_interval_player2,
                     facecolor="#fed8b1", alpha=0.7)

    plot_title = "Cumulative Game Reward (Mean " + r"$\pm$ 95% CI over " + str(
        n_runs) + " runs), " + "\n" + player1_title + " vs " + player2_title
    plt.title(plot_title)
    plt.ylabel("Cumulative Game reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "reward_cumulative_Game.png", bbox_inches="tight")
    plt.close()

    ######################################
    #### Non-cumulative Game reward ####
    ######################################
    df_player1 = pd.read_csv(destination_folder / "player1" / "df_reward_game.csv", index_col=0)
    mean_player1 = df_player1.mean(axis=1)
    std_player1 = df_player1.std(axis=1)
    confidence_interval_player1 = z_score * std_player1 / np.sqrt(n_runs)

    df_player2 = pd.read_csv(destination_folder / "player2" / "df_reward_game.csv", index_col=0)
    mean_player2 = df_player2.mean(axis=1)
    std_player2 = df_player2.std(axis=1)
    confidence_interval_player2 = z_score * std_player2 / np.sqrt(n_runs)

    plt.figure(dpi=80)  # figsize=(10, 6)
    plt.plot(df_player1.index[:], mean_player1[:], lw=0.5, label=f"player1 - {player1_title}", color="blue")
    plt.plot(df_player2.index[:], mean_player2[:], lw=0.5, label=f"player2 - {player2_title}", color="orange")
    plt.fill_between(df_player1.index[:], mean_player1 - confidence_interval_player1,
                     mean_player1 + confidence_interval_player1,
                     facecolor="#95d0fc", alpha=0.7)
    plt.fill_between(df_player2.index[:], mean_player2 - confidence_interval_player2,
                     mean_player2 + confidence_interval_player2,
                     facecolor="#fed8b1", alpha=0.7)

    plot_title = 'Game Reward (Mean ' + r'$\pm$ 95% CI over ' + str(
        n_runs) + ' runs), ' + '\n' + player1_title + ' vs ' + player2_title
    plt.title(plot_title)
    # TODO: This IF statement block has to be removed. We can map the ylim couple based on the game title prior to calling this function.
    if game_title == "IPD":
        plt.gca().set_ylim((1, 4))
    elif game_title == "IVD":
        plt.gca().set_ylim((1, 5))
    elif game_title == "ISH":
        plt.gca().set_ylim((1, 5))

    plt.ylabel("Game reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "reward_Game.png", bbox_inches="tight")
    plt.close()

    ######################################
    #### Cumulative - Intrinsic reward ###
    ######################################
    try:
        df_player1 = pd.read_csv(destination_folder / "player1" / "df_cumulative_reward_intrinsic.csv", index_col=0)
    except FileNotFoundError:
        df_player1 = pd.read_csv(destination_folder / "player1" / "df_reward_intrinsic.csv", index_col=0)
        episode_cols = [col for col in df_player1.columns if col.startswith("episode")]
        df_player1 = df_player1[episode_cols].cumsum(axis=0)
        df_player1.to_csv(
            destination_folder / "player1" / "df_cumulative_reward_intrinsic.csv",
            index=False,
        )
        logger.info("Saved the cumulative intrinsic rewards for player 1.")

    mean_player1 = df_player1.mean(axis=1)
    std_player1 = df_player1.std(axis=1)

    try:
        df_player2 = pd.read_csv(destination_folder / "player2" / "df_cumulative_reward_intrinsic.csv", index_col=0)
    except FileNotFoundError:
        df_player2 = pd.read_csv(destination_folder / "player2" / "df_reward_intrinsic.csv", index_col=0)
        episode_cols = [col for col in df_player2.columns if col.startswith("episode")]
        df_player2 = df_player2[episode_cols].cumsum(axis=0)
        df_player2.to_csv(
            destination_folder / "player2" / "df_cumulative_reward_intrinsic.csv",
            index=False,
        )
        logger.info("Saved the cumulative intrinsic rewards for player 2.")

    mean_player2 = df_player2.mean(axis=1)
    std_player2 = df_player2.std(axis=1)

    # Plot player1 and player2 separately
    # Only plot player1 intrinsic reward if they are a QL player
    if "QL" in player1_title:
        if "Selfish" not in player1_title:
            plt.figure(dpi=80)
            plt.plot(df_player1.index[:], mean_player1[:], label=f"player1 - {player1_title}", color="blue")
            plt.fill_between(df_player1.index[:], mean_player1 - std_player1, mean_player1 + std_player1,
                             facecolor="#95d0fc", alpha=0.7)
            plot_title = "Cumulative Intrinsic Reward (Mean " + r"$\pm$ 95% CI over " + str(
                n_runs) + " runs), " + "\n" + player1_title
            plt.title(plot_title)
            plt.ylabel("Cumulative Intrinsic reward")
            plt.xlabel("Iteration")
            leg = plt.legend()
            for line in leg.get_lines():
                line.set_linewidth(4.0)

            plt.savefig(destination_folder / "plots" / "reward_cumulative_Intrinsic_player1.png", bbox_inches="tight")
            plt.close()

    # Only plot player2 intrinsic reward if they are a QL player
    if "QL" in player2_title:
        if "Selfish" not in player2_title:
            plt.figure(dpi=80)  # figsize=(10, 6)
            plt.plot(df_player2.index[:], mean_player2[:], label=f"player2 - {player2_title}", color="orange")
            plt.fill_between(df_player2.index[:], mean_player2 - std_player2, mean_player2 + std_player2,
                             facecolor="#fed8b1", alpha=0.7)

            plot_title = "Cumulative Intrinsic Reward (Mean " + r"$\pm$ 95% CI over " + str(
                n_runs) + " runs), " + "\n" + player2_title
            plt.title(plot_title)
            plt.ylabel("Cumulative Intrinsic reward")
            plt.xlabel("Iteration")
            leg = plt.legend()
            for line in leg.get_lines():
                line.set_linewidth(4.0)

            plt.savefig(destination_folder / "plots" / "reward_cumulative_Intrinsic_player2.png", bbox_inches="tight")
            plt.close()

    ##########################################
    #### Non-cumulative - Intrinsic reward ###
    ##########################################
    # Only plot player1 intrinsic reward if they are a QL player
    if "QL" in player1_title:
        if "Selfish" not in player1_title:
            df_player1 = pd.read_csv(destination_folder / "player1" / "df_reward_intrinsic.csv", index_col=0)
            mean_player1 = df_player1.mean(axis=1)
            std_player1 = df_player1.std(axis=1)

            plt.figure(dpi=80)
            plt.plot(df_player1.index[:], mean_player1[:], lw=0.5, label=f"player1 - {player1_title}", color="blue")
            plt.fill_between(df_player1.index[:], mean_player1 - std_player1, mean_player1 + std_player1,
                             facecolor="#95d0fc", alpha=0.7)
            plot_title = "Intrinsic Reward (Mean " + r"$\pm$ 95% CI over " + str(
                n_runs) + " runs), " + "\n" + player1_title
            plt.title(plot_title)

            # TODO: This IF statement block has to be removed. We can map the ylim couple based on the game title prior to calling this function.
            if game_title == "IPD":
                plt.gca().set_ylim((-5, 6))
            elif game_title == "IVD":
                plt.gca().set_ylim((-5, 8))
            elif game_title == "ISH":
                plt.gca().set_ylim((-5, 10))

            plt.ylabel("Intrinsic reward")
            plt.xlabel("Iteration")
            leg = plt.legend()
            for line in leg.get_lines():
                line.set_linewidth(4.0)

            plt.savefig(destination_folder / "plots" / "reward_Intrinsic_player1.png", bbox_inches="tight")
            plt.close()

    # Only plot player2 intrinsic reward if they are a QL player
    if "QL" in player2_title:
        if "Selfish" not in player2_title:
            df_player2 = pd.read_csv(destination_folder / "player2" / "df_reward_intrinsic.csv", index_col=0)
            mean_player2 = df_player2.mean(axis=1)
            std_player2 = df_player2.std(axis=1)

            plt.figure(dpi=80)
            plt.plot(df_player2.index[:], mean_player2[:], lw=0.5, label=f"player2 - {player2_title}",
                     color="orange")
            plt.fill_between(df_player2.index[:], mean_player2 - std_player2, mean_player2 + std_player2,
                             facecolor="#fed8b1", alpha=0.7)

            plot_title = "Intrinsic Reward (Mean " + r"$\pm$ 95% CI over " + str(
                n_runs) + " runs), " + "\n" + player2_title
            plt.title(plot_title)

            # TODO: This IF statement block has to be removed. We can map the ylim couple based on the game title prior to calling this function.
            if game_title == "IPD":
                plt.gca().set_ylim((-5, 6))
            elif game_title == "IVD":
                plt.gca().set_ylim((-5, 8))
            elif game_title == "ISH":
                plt.gca().set_ylim((-5, 10))

            plt.ylabel("Intrinsic reward")
            plt.xlabel("Iteration")
            leg = plt.legend()
            for line in leg.get_lines():
                line.set_linewidth(4.0)

            plt.savefig(destination_folder / "plots" / "reward_Intrinsic_player2.png", bbox_inches="tight")
            plt.close()

    #############################################
    #### Cumulative - Collective Game reward ####
    #############################################
    try:
        df_cumulative = pd.read_csv(destination_folder / "df_cumulative_reward_collective.csv", index_col=0)
    except FileNotFoundError:
        df_cumulative = pd.read_csv(destination_folder / "df_reward_collective.csv", index_col=0)
        episode_cols = [col for col in df_cumulative.columns if col.startswith("episode")]
        df_cumulative = df_cumulative[episode_cols].cumsum(axis=0)
        df_cumulative.to_csv(
            destination_folder / "df_cumulative_reward_collective.csv",
            index=False,
        )
        logger.info("Saved the cumulative collective reward")

    mean_cumulative = df_cumulative.mean(axis=1)
    std_cumulative = df_cumulative.std(axis=1)
    confidence_interval_cumulative = z_score * std_cumulative / np.sqrt(n_runs)

    plt.figure(dpi=80)  # figsize=(10, 6)
    plt.plot(df_cumulative.index[:], mean_cumulative[:], lw=0.5, label=f"both players", color="purple")
    plt.fill_between(df_cumulative.index[:], mean_cumulative - confidence_interval_cumulative,
                     mean_cumulative + confidence_interval_cumulative, facecolor="#bf92e4", alpha=0.7)

    plot_title = "Cumulative Collective Reward (Mean " + r"$\pm$ 95% CI over " + str(
        n_runs) + " runs), " + "\n" + player1_title + " vs " + player2_title
    plt.title(plot_title)
    plt.ylabel("Cumulative Collective reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "outcome_plots" / "reward_cumulative_Collective.png",
                bbox_inches="tight")
    plt.close()

    #################################################
    #### Non-cumulative - Collective Game reward ####
    #################################################
    df_collective = pd.read_csv(destination_folder / "df_reward_collective.csv", index_col=0)
    mean_collective = df_collective.mean(axis=1)
    std_collective = df_collective.std(axis=1)
    confidence_interval_collective = z_score * std_collective / np.sqrt(n_runs)

    plt.figure(dpi=80)  # figsize=(10, 6)
    plt.plot(df_collective.index[:], mean_collective[:], lw=0.5, label=f"both players", color="purple")
    plt.fill_between(df_collective.index[:], mean_collective - confidence_interval_collective,
                     mean_collective + confidence_interval_collective, facecolor="#bf92e4", alpha=0.7)

    plot_title = "Collective Reward (Mean " + r"$\pm$ 95% CI over " + str(
        n_runs) + " runs), " + "\n" + player1_title + " vs " + player2_title
    plt.title(plot_title)

    # TODO: This IF statement block has to be removed. We can map the ylim couple based on the game title prior to calling this function.
    if game_title == "IPD":
        plt.gca().set_ylim((4, 6))
    elif game_title == "IVD":
        plt.gca().set_ylim((2, 8))
    elif game_title == "ISH":
        plt.gca().set_ylim((4, 10))

    plt.ylabel("Collective reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "outcome_plots" / "reward_Collective.png", bbox_inches="tight")
    plt.close()

    #######################################
    #### Cumulative - Gini Game reward ####
    #######################################
    try:
        df_gini = pd.read_csv(destination_folder / "df_cumulative_reward_gini.csv", index_col=0)
    except FileNotFoundError:
        df_gini = pd.read_csv(destination_folder / "df_reward_gini.csv", index_col=0)
        episode_cols = [col for col in df_gini.columns if col.startswith("episode")]
        df_gini = df_gini[episode_cols].cumsum(axis=0)
        df_gini.to_csv(
            destination_folder / "df_cumulative_reward_gini.csv",
            index=False,
        )
        logger.info("Saved the cumulative gini reward")

    mean_gini = df_gini.mean(axis=1)
    std_gini = df_gini.std(axis=1)
    confidence_interval_gini = z_score * std_gini / np.sqrt(n_runs)

    plt.figure(dpi=80)  # figsize=(10, 6)
    plt.plot(df_gini.index[:], mean_gini[:], lw=0.5, label=f'both players', color='purple')
    plt.fill_between(df_gini.index[:], mean_gini - confidence_interval_gini,
                     mean_gini + confidence_interval_gini, facecolor='#bf92e4', alpha=0.7)

    plot_title = r"Cumulative Gini Reward (Mean over " + str(
        n_runs) + r" runs $\pm$ SD), " + "\n" + player1_title + " vs " + player2_title
    plt.title(plot_title)
    plt.ylabel("Cumulative Gini reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "outcome_plots" / "reward_cumulative_Gini.png", bbox_inches="tight")
    plt.close()

    ###########################################
    #### Non-cumulative - Gini Game reward ####
    ###########################################
    df_gini_non_cumulative = pd.read_csv(destination_folder / "df_reward_gini.csv", index_col=0)
    mean_gini_non_cumulative = df_gini_non_cumulative.mean(axis=1)
    std_mean_gini_non_cumulative = df_gini_non_cumulative.std(axis=1)
    confidence_interval_std_mean_gini_non_cumulative = z_score * std_mean_gini_non_cumulative / np.sqrt(n_runs)

    plt.figure(dpi=80)
    plt.plot(df_gini_non_cumulative.index[:], mean_gini_non_cumulative[:], lw=0.5, label="both players", color="purple")
    plt.fill_between(df_gini_non_cumulative.index[:],
                     mean_gini_non_cumulative - confidence_interval_std_mean_gini_non_cumulative,
                     mean_gini_non_cumulative + confidence_interval_std_mean_gini_non_cumulative, facecolor="#bf92e4",
                     alpha=0.7)

    plot_title = r"Gini Reward (Mean over " + str(
        n_runs) + r" runs $\pm$ SD), " + "\n" + player1_title + " vs " + player2_title
    plt.title(plot_title)
    plt.gca().set_ylim((0, 1))
    plt.ylabel("Gini reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "outcome_plots" / "reward_Gini.png", bbox_inches="tight")
    plt.close()

    ######################################
    #### Cumulative - Min Game reward ####
    ######################################
    try:
        df_min_reward = pd.read_csv(destination_folder / "df_cumulative_reward_min.csv", index_col=0)
    except FileNotFoundError:
        df_min_reward = pd.read_csv(destination_folder / "df_reward_min.csv", index_col=0)
        episode_cols = [col for col in df_min_reward.columns if col.startswith("episode")]
        df_min_reward = df_min_reward[episode_cols].cumsum(axis=0)
        df_min_reward.to_csv(
            destination_folder / "df_cumulative_reward_min.csv",
            index=False,
        )
        logger.info("Saved the cumulative min reward")

    mean_min_reward = df_min_reward.mean(axis=1)
    std_min_reward = df_min_reward.std(axis=1)
    confidence_interval_min_reward = z_score * std_min_reward / np.sqrt(n_runs)

    plt.figure(dpi=80)
    plt.plot(df_min_reward.index[:], mean_min_reward[:], lw=0.5, label=f'both players', color='purple')
    plt.fill_between(df_min_reward.index[:], mean_min_reward - confidence_interval_min_reward,
                     mean_min_reward + confidence_interval_min_reward, facecolor='#bf92e4', alpha=0.7)

    plot_title = r"Cumulative Min Reward (Mean over " + str(
        n_runs) + r" runs $\pm$ SD), " + "\n" + player1_title + " vs " + player2_title
    plt.title(plot_title)
    plt.ylabel("Cumulative Min reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "outcome_plots" / "reward_cumulative_Min.png", bbox_inches="tight")
    plt.close()

    ##########################################
    #### Non-cumulative - Min Game reward ####
    ##########################################
    df_min_reward_on_cumulative = pd.read_csv(destination_folder / "df_reward_min.csv", index_col=0)
    mean_min_reward_on_cumulative = df_min_reward_on_cumulative.mean(axis=1)
    std_min_reward_on_cumulative = df_min_reward_on_cumulative.std(axis=1)
    confidence_interval_min_reward_on_cumulative = z_score * std_min_reward_on_cumulative / np.sqrt(n_runs)

    plt.figure(dpi=80)  # figsize=(10, 6)
    plt.plot(df_min_reward_on_cumulative.index[:], mean_min_reward_on_cumulative[:], lw=0.5, label="both players",
             color="purple")
    plt.fill_between(df_min_reward_on_cumulative.index[:],
                     mean_min_reward_on_cumulative - confidence_interval_min_reward_on_cumulative,
                     mean_min_reward_on_cumulative + confidence_interval_min_reward_on_cumulative, facecolor="#bf92e4",
                     alpha=0.7)

    plot_title = r"Min Reward (Mean over " + str(
        n_runs) + r" runs $\pm$ SD), " + "\n" + player1_title + " vs " + player2_title
    plt.title(plot_title)

    # TODO: This IF statement block has to be removed. We can map the ylim couple based on the game title prior to calling this function.
    if game_title == "IPD":
        plt.gca().set_ylim((1, 3))
    elif game_title == "IVD":
        plt.gca().set_ylim((1, 4))
    elif game_title == "ISH":
        plt.gca().set_ylim((1, 5))

    plt.ylabel("Min reward")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "outcome_plots" / "reward_Min.png", bbox_inches="tight")
    plt.close()


def plot_actions(destination_folder: Path, player1_title: str, player2_title: str, n_runs: int) -> None:
    """Plot the cooperation rate over time for each player.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing the per-player action histories.
    player1_title : str
        Human-readable title for player 1 used in the output plot.
    player2_title : str
        Human-readable title for player 2 used in the output plot.
    n_runs : int
        Number of runs aggregated in the action histories.
    """
    actions_player1 = pd.read_csv(destination_folder / "player1" / "action.csv", index_col=0)

    # Calculate % of 100 agents (runs) that cooperate at every step out of the 10.000
    actions_player1["%_defect"] = actions_player1[actions_player1[:] == 1].count(axis="columns")
    actions_player1["%_cooperate"] = n_runs - actions_player1["%_defect"]

    # Convert to %
    actions_player1["%_defect"] = (actions_player1["%_defect"] / n_runs) * 100.0
    actions_player1["%_cooperate"] = (actions_player1["%_cooperate"] / n_runs) * 100.0

    # Plot results
    plt.figure(dpi=80)
    plt.plot(actions_player1.index[:], actions_player1["%_cooperate"], label=f"player1 - {player1_title}", color="blue")

    plot_title = "The actions of Player 1 at every step of the episode \n (percentage cooperated over " + str(
        n_runs) + r" runs)"
    plt.title(plot_title)
    plt.gca().set_ylim((0, 100))
    plt.ylabel("Percentage cooperating")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "actions_player1.png", bbox_inches="tight")
    plt.close()

    # Repeat for player2. TODO [Wilder]: This have to be parameterised to avoid code duplication.
    actions_player2 = pd.read_csv(destination_folder / "player2" / "action.csv", index_col=0)

    actions_player2["%_defect"] = actions_player2[actions_player2[:] == 1].count(axis="columns")
    actions_player2["%_cooperate"] = n_runs - actions_player2["%_defect"]

    # Convert to %
    actions_player2["%_defect"] = (actions_player2["%_defect"] / n_runs) * 100.0
    actions_player2["%_cooperate"] = (actions_player2["%_cooperate"] / n_runs) * 100.0

    plt.figure(dpi=80)
    plt.plot(actions_player2.index[:], actions_player2['%_cooperate'], label=f'player2 - {player2_title}',
             color="orange")

    plot_title = "The actions of Player 2 at every step of the episode \n (percentage cooperated over " + str(
        n_runs) + r" runs)"
    plt.title(plot_title)
    plt.gca().set_ylim((0, 100))
    plt.ylabel("Percentage cooperating")
    plt.xlabel("Iteration")
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(destination_folder / "plots" / "actions_player2.png", bbox_inches="tight")
    plt.close()


def plot_action_types_area(destination_folder: Path, player1_title: str, player2_title: str, n_runs: int) -> None:
    """Plot stacked action-type frequencies from each player's perspective.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing action and state histories.
    player1_title : str
        Title used when plotting player 1's perspective.
    player2_title : str
        Title used when plotting player 2's perspective.
    n_runs : int
        Number of runs aggregated in the histories.
    """
    # NOTE: This will only work for 10.000 iterations right now, not fewer!
    # NOTE: We plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    ###################################################
    ### Plot from the perspective of player1 first ####
    ###################################################

    actions_player1 = pd.read_csv(destination_folder / "player1" / "action.csv", index_col=0)
    state_player1 = pd.read_csv(destination_folder / "player1" / "state.csv", index_col=0)

    col_names = ["run" + str(i) for i in range(n_runs)]
    actions_player1.columns = col_names
    state_player1.columns = col_names
    results = pd.DataFrame(columns=col_names)

    for col_name in col_names:
        str_value = actions_player1[col_name].astype(str) + " | " + state_player1[col_name].astype(str)
        str_value = str_value.str.replace("1", "D")
        str_value = str_value.str.replace("0", "C")
        results[col_name] = str_value

    results_counts = results.transpose().apply(lambda s: s.value_counts()).transpose()[1:]
    # Plot after the first episode, as in the first episode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    results_counts.plot.area(stacked=True,
                             ylabel="# agents taking this type of action \n (across " + str(n_runs) + " runs)", rot=45,
                             xlabel="Iteration", colormap="PiYG_r",
                             title="Types of actions over time: \n " + player1_title + " agent (player1) against " + player2_title + " agent")

    plt.savefig(destination_folder / "plots" / "action_types_area_player1.png", bbox_inches="tight")
    plt.close()

    #############################################
    ### Plot from the perspective of player2 ####
    #############################################
    # TODO [Wilder]: This have to be parameterised to avoid code duplication.

    actions_player2 = pd.read_csv(destination_folder / "player2" / "action.csv", index_col=0)
    state_player2 = pd.read_csv(destination_folder / "player2" / "state.csv", index_col=0)
    # -> Use col_names form above. TODO [Wilder]: This must go!
    actions_player2.columns = col_names
    state_player2.columns = col_names
    results = pd.DataFrame(columns=col_names)

    for col_name in col_names:
        str_value = actions_player2[col_name].astype(str) + ' | ' + state_player2[col_name].astype(str)
        str_value = str_value.str.replace("1", "D")
        str_value = str_value.str.replace("0", "C")
        results[col_name] = str_value

    results_counts = results.transpose().apply(lambda s: s.value_counts()).transpose()[1:]
    results_counts.dropna(axis=1, how="all", inplace=True)

    results_counts.plot.area(stacked=True,
                             ylabel="# agents taking this type of action \n (across " + str(n_runs) + " runs)", rot=45,
                             xlabel="Iteration", colormap="PiYG_r",
                             title="Types of actions over time: \n " + player2_title + " agent (player2) against " + player1_title + " agent")

    plt.savefig(destination_folder / "plots" / "action_types_area_player2.png", bbox_inches="tight")
    plt.close()


def plot_first_n_actions(destination_folder: Path, player1_title: str, player2_title: str, n_runs: int,
                         n_actions: int = 20) -> None:
    """Plot the distribution of action-state types for the first interactions.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing action and state histories.
    player1_title : str
        Title used when plotting player 1's perspective.
    player2_title : str
        Title used when plotting player 2's perspective.
    n_runs : int
        Number of runs aggregated in the histories.
    n_actions : int, optional
        Number of initial actions to include in the summary plots.
    """
    ###################################################
    ### Plot from the perspective of player1 first ####
    ###################################################

    actions_player1 = pd.read_csv(destination_folder / "player1" / "action.csv", index_col=0)[0:n_actions]
    state_player1 = pd.read_csv(destination_folder / "player1" / "state.csv", index_col=0)[0:n_actions]

    col_names = ["run" + str(i) for i in range(n_runs)]
    actions_player1.columns = col_names
    state_player1.columns = col_names
    results = pd.DataFrame(columns=col_names)

    for col_name in col_names:
        str_value = actions_player1[col_name].astype(str) + " | " + state_player1[col_name].astype(str)
        str_value = str_value.str.replace("1", "D")
        str_value = str_value.str.replace("0", "C")
        results[col_name] = str_value

    results.to_csv(destination_folder / "player1" / "first_20_actions.csv")
    results_counts = results.transpose().apply(lambda s: s.value_counts()).transpose()[1:]
    # Plot after the first episode, as in the first episode they are reacting to default state=0
    results_counts.dropna(axis=1, how="all", inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True,
                            ylabel="# agents taking this type of action \n (across " + str(n_runs) + ' runs)', rot=45,
                            xlabel="Iteration", colormap="PiYG_r",
                            title=f"First {n_actions} action types: \n {player2_title} agent (player2) against {player1_title} agent")
    plt.savefig(destination_folder / "plots" / "first_20_actions_player1.png", bbox_inches="tight")
    plt.close()

    #############################################
    ### Plot from the perspective of player2 ####
    #############################################
    # TODO [Wilder]: This have to be parameterised to avoid code duplication.

    actions_player2 = pd.read_csv(destination_folder / "player2" / "action.csv", index_col=0)[0:n_actions]
    state_player2 = pd.read_csv(destination_folder / "player2" / "state.csv", index_col=0)[0:n_actions]
    # -> Use col_names form above. TODO [Wilder]: This must go!
    actions_player2.columns = col_names
    state_player2.columns = col_names
    results = pd.DataFrame(columns=col_names)

    for col_name in col_names:
        str_value = actions_player2[col_name].astype(str) + " | " + state_player2[col_name].astype(str)
        str_value = str_value.str.replace("1", "D")
        str_value = str_value.str.replace("0", "C")
        results[col_name] = str_value

    results.to_csv(destination_folder / "player2" / "first_20_actions.csv")
    results_counts = results.transpose().apply(lambda s: s.value_counts()).transpose()[1:]
    results_counts.dropna(axis=1, how="all", inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True,
                            ylabel="# agents taking this type of action \n (across " + str(n_runs) + ' runs)', rot=45,
                            xlabel="Iteration", colormap="PiYG_r",
                            title=f"First {n_actions} action types: \n {player2_title} agent (player2) against {player1_title} agent")
    plt.savefig(f'{destination_folder}/plots/first_20_actions_player2.png', bbox_inches='tight')
    plt.close()


def visualise_first_n_actions_matrix(destination_folder: Path, n_actions: int = 20) -> None:
    """Export styled tables for the first recorded action-state combinations.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing the intermediate first-action CSV
        files.
    n_actions : int, optional
        Number of initial actions represented in the exported tables.
    """
    results_player1 = pd.read_csv(destination_folder / "player1" / f"first_{n_actions}_actions.csv",
                                  index_col=0).transpose()
    results_player2 = pd.read_csv(destination_folder / "player2" / f"first_{n_actions}_actions.csv",
                                  index_col=0).transpose()

    caption = os.fspath(destination_folder).replace("results/", "")
    caption = caption.replace("QL", "")
    results_player1 = results_player1.style.map(color_condition).set_caption(f"Player1 from {caption}")
    dfi.export(results_player1, destination_folder / "plots" / f"table_export_player1_first{n_actions}.png")

    results_player2 = results_player2.style.map(color_condition).set_caption(f"Player2 from {caption}")
    dfi.export(results_player2, destination_folder / "plots" / f"table_export_player2_first{n_actions}.png")


def plot_last_n_actions(destination_folder: Path, player1_title: str, player2_title: str, n_runs: int,
                        n_actions: int = 20) -> None:
    """Plot the distribution of action-state types for the last interactions.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing action and state histories.
    player1_title : str
        Title used when plotting player 1's perspective.
    player2_title : str
        Title used when plotting player 2's perspective.
    n_runs : int
        Number of runs aggregated in the histories.
    n_actions : int, optional
        Number of final actions to include in the summary plots.
    """
    ###################################################
    ### Plot from the perspective of player1 first ####
    ###################################################

    actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)[-n_actions:]
    state_player1 = pd.read_csv(f'{destination_folder}/player1/state.csv', index_col=0)[-n_actions:]
    # Rename columns
    col_names = ["run" + str(i) for i in range(n_runs)]
    actions_player1.columns = col_names
    state_player1.columns = col_names
    results = pd.DataFrame(columns=col_names)

    for col_name in col_names:
        str_value = actions_player1[col_name].astype(str) + " | " + state_player1[col_name].astype(str)
        str_value = str_value.str.replace("1", "D")
        str_value = str_value.str.replace("0", "C")
        results[col_name] = str_value

    results.to_csv(destination_folder / "player1" / "last_20_actions.csv")
    results_counts = results.transpose().apply(lambda s: s.value_counts()).transpose()[1:]
    # Plot after the first episode, as in the first episode they are reacting to default state=0
    results_counts.dropna(axis=1, how="all", inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True,
                            ylabel="# agents taking this type of action \n (across " + str(n_runs) + " runs)", rot=45,
                            xlabel="Iteration", colormap="PiYG_r",
                            title=f"Last {n_actions} action types: \n " + player1_title + " agent (player1) against " + player2_title + " agent")
    plt.savefig(destination_folder / "plots" / "last_20_actions_player1.png", bbox_inches="tight")
    plt.close()

    #############################################
    ### Plot from the perspective of player2 ####
    #############################################
    # TODO [Wilder]: This have to be parameterised to avoid code duplication.

    actions_player2 = pd.read_csv(destination_folder / "player2" / "action.csv", index_col=0)[-n_actions:]
    state_player2 = pd.read_csv(destination_folder / "player2" / "state.csv", index_col=0)[-n_actions:]
    # -> Use col_names form above. TODO [Wilder]: This must go!
    actions_player2.columns = col_names
    state_player2.columns = col_names
    results = pd.DataFrame(columns=col_names)

    for col_name in col_names:
        str_value = actions_player2[col_name].astype(str) + " | " + state_player2[col_name].astype(str)
        str_value = str_value.str.replace("1", "D")
        str_value = str_value.str.replace("0", "C")
        results[col_name] = str_value

    results.to_csv(destination_folder / "player2" / "last_20_actions.csv")
    results_counts = results.transpose().apply(lambda s: s.value_counts()).transpose()[1:]
    results_counts.dropna(axis=1, how="all", inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True,
                            ylabel="# agents taking this type of action \n (across " + str(n_runs) + ' runs)', rot=45,
                            xlabel="Iteration", colormap="PiYG_r",
                            title=f"Last {n_actions} action types: \n " + player2_title + " agent (player2) against " + player1_title + " agent")
    plt.savefig(destination_folder / "plots" / "last_20_actions_player2.png", bbox_inches="tight")
    plt.close()


def visualise_last_n_actions_matrix(destination_folder: Path, n_actions: int = 20) -> None:
    """Export styled tables for the last recorded action-state combinations.

    Parameters
    ----------
    destination_folder : Path
        Experiment directory containing the intermediate last-action CSV
        files.
    n_actions : int, optional
        Number of final actions represented in the exported tables.
    """
    results_player1 = pd.read_csv(destination_folder / "player1" / f"last_{n_actions}_actions.csv",
                                  index_col=0).transpose()
    results_player2 = pd.read_csv(destination_folder / "player2" / f"last_{n_actions}_actions.csv",
                                  index_col=0).transpose()

    caption = os.fspath(destination_folder).replace("results/", "")
    caption = caption.replace("QL", "")
    results_player1 = results_player1.style.map(color_condition).set_caption(f"Player1 from {caption}")
    dfi.export(results_player1, destination_folder / "plots" / f"table_export_player1_last{n_actions}.pdf")

    results_player2 = results_player2.style.map(color_condition).set_caption(f"Player2 from {caption}")
    dfi.export(results_player2, destination_folder / "plots" / f"table_export_player2_last{n_actions}.pdf")
