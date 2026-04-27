# MIT License - 2026
"""Utilities for exporting heatmaps of terminal action-pair frequencies."""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from uu import logger


def plot_matrix_action_pairs(destination_path: Path, num_runs: int) -> None:
    """Plot heatmaps summarizing final action-pair frequencies across matchups.

    Parameters
    ----------
    destination_path : Path
        Root directory containing per-matchup ``action_pairs.csv`` files and
        the target output directory for the generated heatmaps.
    num_runs : int
        Number of runs represented in each stored ``action_pairs.csv`` file.
    """
    # NOTE: This will only work for 10.000 iterations right now, not fewer!
    # NOTE: We plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    fig_size = (8, 6.5)
    types = ["S", "UT", "DE", "Ve", "Vk", "Vm"]

    matrix_CC = pd.DataFrame(columns=types, index=types)  # shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_DD = pd.DataFrame(columns=types, index=types)  # shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_DC = pd.DataFrame(columns=types, index=types)  # shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_CD = pd.DataFrame(columns=types, index=types)  # shape = vs S,UT,DE,Ve,Vk(,VEm)

    for player1_title in ["QLS", "QLUT", "QLDE", "QLVE_e", "QLVE_k", "QLVM"]:
        if not (destination_path / f"{player1_title}_QLS" / "action_pairs.csv").exists():
            continue

        action_pairs_against_QLS = \
            pd.read_csv(destination_path / f"{player1_title}_QLS" / "action_pairs.csv", index_col=0).iloc[9999]
        try:
            action_pairs_against_QLUT = \
                pd.read_csv(destination_path / f"{player1_title}_QLUT" / "action_pairs.csv", index_col=0).iloc[
                    9999]
            order_QLUT = ""
        except:
            action_pairs_against_QLUT = \
                pd.read_csv(destination_path / f"QLUT_{player1_title}" / "action_pairs.csv", index_col=0).iloc[
                    9999]
            order_QLUT = "QLUT_first"
        try:
            action_pairs_against_QLDE = \
                pd.read_csv(destination_path / f"{player1_title}_QLDE" / "action_pairs.csv", index_col=0).iloc[
                    9999]
            order_QLDE = ""
        except:
            action_pairs_against_QLDE = \
                pd.read_csv(destination_path / f"QLDE_{player1_title}" / "action_pairs.csv", index_col=0).iloc[
                    9999]
            order_QLDE = "QLDE_first"
        try:
            action_pairs_against_QLVE_e = \
                pd.read_csv(destination_path / f"{player1_title}_QLVE_e" / "action_pairs.csv", index_col=0).iloc[9999]
            order_QLVE_e = ""
        except:
            action_pairs_against_QLVE_e = \
                pd.read_csv(destination_path / f"QLVE_e_{player1_title}" / "action_pairs.csv", index_col=0).iloc[
                    9999]
            order_QLVE_e = "QLVE_e_first"
        try:
            action_pairs_against_QLVE_k = \
                pd.read_csv(destination_path / f"{player1_title}_QLVE_k" / "action_pairs.csv", index_col=0).iloc[9999]
            order_QLVE_k = ""
        except:
            action_pairs_against_QLVE_k = \
                pd.read_csv(destination_path / f"QLVE_k_{player1_title}" / "action_pairs.csv", index_col=0).iloc[9999]
            order_QLVE_k = "QLVE_k_first"
        try:
            action_pairs_against_QLVM = \
                pd.read_csv(destination_path / f"{player1_title}_QLVM" / "action_pairs.csv", index_col=0).iloc[
                    9999]
            order_QLVM = ""
        except:
            action_pairs_against_QLVM = \
                pd.read_csv(destination_path / f"QLVM_{player1_title}" / "action_pairs.csv", index_col=0).iloc[
                    9999]
            order_QLVM = "QLVM_first"

        logger.info(
            f"Creating action matrices plots for player {player1_title} against learning agents QLS, QLUT, QLDE, QLVE_e, QLVE_k, QLVM.")

        action_pairs_against_QLS["%_CC"] = (action_pairs_against_QLS[
                                                action_pairs_against_QLS[:] == "C, C"].count() / num_runs) * 100
        action_pairs_against_QLUT["%_CC"] = (action_pairs_against_QLUT[
                                                 action_pairs_against_QLUT[:] == "C, C"].count() / num_runs) * 100
        action_pairs_against_QLDE["%_CC"] = (action_pairs_against_QLDE[
                                                 action_pairs_against_QLDE[:] == "C, C"].count() / num_runs) * 100
        action_pairs_against_QLVE_e["%_CC"] = (action_pairs_against_QLVE_e[
                                                   action_pairs_against_QLVE_e[:] == "C, C"].count() / num_runs) * 100
        action_pairs_against_QLVE_k["%_CC"] = (action_pairs_against_QLVE_k[
                                                   action_pairs_against_QLVE_k[:] == "C, C"].count() / num_runs) * 100
        action_pairs_against_QLVM["%_CC"] = (action_pairs_against_QLVM[
                                                 action_pairs_against_QLVM[:] == "C, C"].count() / num_runs) * 100

        short_title = player1_title.replace("QL", "").replace("VE_e", "Ve").replace("VE_k", "Vk").replace("VM", "Vm")
        matrix_CC.loc[short_title] = [action_pairs_against_QLS["%_CC"], action_pairs_against_QLUT["%_CC"],
                                      action_pairs_against_QLDE["%_CC"], action_pairs_against_QLVE_e["%_CC"],
                                      action_pairs_against_QLVE_k["%_CC"], action_pairs_against_QLVM["%_CC"]]

        action_pairs_against_QLS["%_DD"] = (action_pairs_against_QLS[
                                                action_pairs_against_QLS[:] == "D, D"].count() / num_runs) * 100
        action_pairs_against_QLUT["%_DD"] = (action_pairs_against_QLUT[
                                                 action_pairs_against_QLUT[:] == "D, D"].count() / num_runs) * 100
        action_pairs_against_QLDE["%_DD"] = (action_pairs_against_QLDE[
                                                 action_pairs_against_QLDE[:] == "D, D"].count() / num_runs) * 100
        action_pairs_against_QLVE_e["%_DD"] = (action_pairs_against_QLVE_e[
                                                   action_pairs_against_QLVE_e[:] == "D, D"].count() / num_runs) * 100
        action_pairs_against_QLVE_k["%_DD"] = (action_pairs_against_QLVE_k[
                                                   action_pairs_against_QLVE_k[:] == "D, D"].count() / num_runs) * 100
        action_pairs_against_QLVM["%_DD"] = (action_pairs_against_QLVM[
                                                 action_pairs_against_QLVM[:] == "D, D"].count() / num_runs) * 100

        matrix_DD.loc[short_title] = [action_pairs_against_QLS["%_DD"], action_pairs_against_QLUT["%_DD"],
                                      action_pairs_against_QLDE["%_DD"], action_pairs_against_QLVE_e["%_DD"],
                                      action_pairs_against_QLVE_k["%_DD"], action_pairs_against_QLVM["%_DD"]]

        action_pairs_against_QLS["%_DC"] = (action_pairs_against_QLS[
                                                action_pairs_against_QLS[:] == "D, C"].count() / num_runs) * 100
        if order_QLUT == "QLUT_first":
            action_pairs_against_QLUT["%_DC"] = (action_pairs_against_QLUT[
                                                     action_pairs_against_QLUT[:] == "C, D"].count() / num_runs) * 100
        else:
            action_pairs_against_QLUT["%_DC"] = (action_pairs_against_QLUT[
                                                     action_pairs_against_QLUT[:] == "D, C"].count() / num_runs) * 100
        if order_QLDE == "QLDE_first":
            action_pairs_against_QLDE["%_DC"] = (action_pairs_against_QLDE[
                                                     action_pairs_against_QLDE[:] == "C, D"].count() / num_runs) * 100
        else:
            action_pairs_against_QLDE["%_DC"] = (action_pairs_against_QLDE[
                                                     action_pairs_against_QLDE[:] == "D, C"].count() / num_runs) * 100
        if order_QLVE_e == "QLVE_e_first":
            action_pairs_against_QLVE_e["%_DC"] = (action_pairs_against_QLVE_e[
                                                       action_pairs_against_QLVE_e[
                                                           :] == "C, D"].count() / num_runs) * 100
        else:
            action_pairs_against_QLVE_e["%_DC"] = (action_pairs_against_QLVE_e[
                                                       action_pairs_against_QLVE_e[
                                                           :] == "D, C"].count() / num_runs) * 100
        if order_QLVE_k == "QLVE_k_first":
            action_pairs_against_QLVE_k["%_DC"] = (action_pairs_against_QLVE_k[
                                                       action_pairs_against_QLVE_k[
                                                           :] == "C, D"].count() / num_runs) * 100
        else:
            action_pairs_against_QLVE_k["%_DC"] = (action_pairs_against_QLVE_k[
                                                       action_pairs_against_QLVE_k[
                                                           :] == "D, C"].count() / num_runs) * 100
        if order_QLVM == "QLVM_first":
            action_pairs_against_QLVM["%_DC"] = (action_pairs_against_QLVM[
                                                     action_pairs_against_QLVM[:] == "C, D"].count() / num_runs) * 100
        else:
            action_pairs_against_QLVM["%_DC"] = (action_pairs_against_QLVM[
                                                     action_pairs_against_QLVM[:] == "D, C"].count() / num_runs) * 100

        matrix_DC.loc[short_title] = [action_pairs_against_QLS["%_DC"], action_pairs_against_QLUT["%_DC"],
                                      action_pairs_against_QLDE["%_DC"], action_pairs_against_QLVE_e["%_DC"],
                                      action_pairs_against_QLVE_k["%_DC"], action_pairs_against_QLVM["%_DC"]]

        action_pairs_against_QLS["%_CD"] = (action_pairs_against_QLS[
                                                action_pairs_against_QLS[:] == "C, D"].count() / num_runs) * 100
        if order_QLUT == "QLUT_first":
            action_pairs_against_QLUT["%_CD"] = (action_pairs_against_QLUT[
                                                     action_pairs_against_QLUT[:] == "D, C"].count() / num_runs) * 100
        else:
            action_pairs_against_QLUT["%_CD"] = (action_pairs_against_QLUT[
                                                     action_pairs_against_QLUT[:] == "C, D"].count() / num_runs) * 100
        if order_QLDE == "QLDE_first":
            action_pairs_against_QLDE["%_CD"] = (action_pairs_against_QLDE[
                                                     action_pairs_against_QLDE[:] == "D, C"].count() / num_runs) * 100
        else:
            action_pairs_against_QLDE["%_CD"] = (action_pairs_against_QLDE[
                                                     action_pairs_against_QLDE[:] == "C, D"].count() / num_runs) * 100
        if order_QLVE_e == "QLVE_e_first":
            action_pairs_against_QLVE_e["%_CD"] = (action_pairs_against_QLVE_e[
                                                       action_pairs_against_QLVE_e[
                                                           :] == "D, C"].count() / num_runs) * 100
        else:
            action_pairs_against_QLVE_e["%_CD"] = (action_pairs_against_QLVE_e[
                                                       action_pairs_against_QLVE_e[
                                                           :] == "C, D"].count() / num_runs) * 100
        if order_QLVE_k == "QLVE_k_first":
            action_pairs_against_QLVE_k["%_CD"] = (action_pairs_against_QLVE_k[
                                                       action_pairs_against_QLVE_k[
                                                           :] == "D, C"].count() / num_runs) * 100
        else:
            action_pairs_against_QLVE_k["%_CD"] = (action_pairs_against_QLVE_k[
                                                       action_pairs_against_QLVE_k[
                                                           :] == "C, D"].count() / num_runs) * 100
        if order_QLVM == "QLVM_first":
            action_pairs_against_QLVM["%_CD"] = (action_pairs_against_QLVM[
                                                     action_pairs_against_QLVM[:] == "D, C"].count() / num_runs) * 100
        else:
            action_pairs_against_QLVM["%_CD"] = (action_pairs_against_QLVM[
                                                     action_pairs_against_QLVM[:] == "C, D"].count() / num_runs) * 100

        matrix_CD.loc[short_title] = [action_pairs_against_QLS["%_CD"], action_pairs_against_QLUT["%_CD"],
                                      action_pairs_against_QLDE["%_CD"], action_pairs_against_QLVE_e["%_CD"],
                                      action_pairs_against_QLVE_k["%_CD"], action_pairs_against_QLVM["%_CD"]]

    for label in types:
        for matrix in [matrix_CC, matrix_DD, matrix_DC, matrix_CD]:
            matrix[label] = matrix[label].astype(float)

    types_long = ["Selfish", "Utilitarian", "Deontolog.", "Virtue-eq.", "Virtue-kind.", "Virtue-mix."]

    actions_matrx_path = destination_path / "outcome_plots" / "actions_matrix"
    if not os.path.isdir(actions_matrx_path):
        actions_matrx_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=fig_size)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_CC, cmap="YlGn", xticklabels=types_long, yticklabels=types_long, annot=False,
                    linecolor="black", linewidths=0, square=False, vmin=0, vmax=100)
    s.set(xlabel="Opponent O", ylabel="Player M", title="Mutual Cooperation \n")
    plt.savefig(actions_matrx_path / "CC.pdf", bbox_inches='tight')
    matrix_CC.to_csv(actions_matrx_path / "matrix_CC.csv")

    plt.figure(figsize=fig_size)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_DD, cmap="PuRd", xticklabels=types_long, yticklabels=types_long, annot=False,
                    linecolor="black", linewidths=0, square=False, vmin=0, vmax=100)
    s.set(xlabel="Opponent O", ylabel="Player M", title="Mutual Defection \n")
    plt.savefig(actions_matrx_path / "DD.pdf", bbox_inches='tight')
    matrix_DD.to_csv(actions_matrx_path / "matrix_DD.csv")

    plt.figure(figsize=fig_size)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_DC, cmap="Oranges", xticklabels=types_long, yticklabels=types_long, annot=False,
                    linecolor="black", linewidths=0, square=False, vmin=0, vmax=100)
    s.set(xlabel="Opponent O", ylabel="Player M", title="M Defects, O Cooperates\n ")  # M exploits O
    plt.savefig(actions_matrx_path / "DC.pdf", bbox_inches='tight')
    matrix_DC.to_csv(actions_matrx_path / "matrix_DC.csv")

    plt.figure(figsize=fig_size)
    s = sns.heatmap(matrix_CD, cmap="Blues", xticklabels=types_long, yticklabels=types_long, annot=False,
                    linecolor="black", linewidths=0, square=False, vmin=0, vmax=100)
    sns.set(font_scale=4)
    s.set(xlabel="Opponent O", ylabel="Player M", title="M Cooperates, O Defects\n")  # M Getting Exploited
    plt.savefig(actions_matrx_path / "CD.pdf", bbox_inches='tight')
    matrix_CD.to_csv(actions_matrx_path / "matrix_CD.csv")
