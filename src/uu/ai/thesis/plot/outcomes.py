# MIT License - 2026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_matrix_social_outcomes(source_folder: Path, game_title: str, n_runs: int):
    """Plot social outcome heatmaps across MORL agent matchups.

    The function reads cumulative reward CSV files for each learning-agent
    pairing, extracts the terminal collective, Gini, and minimum returns, and
    writes one heatmap and one matrix CSV for each outcome type.

    Parameters
    ----------
    source_folder : Path
        Root directory containing the per-matchup MORL result folders. The
        generated heatmaps and matrices are written under
        ``source_folder / "outcome_plots" / "outcomes_matrix"``.
    game_title : str
        Lowercase game identifier used to select heatmap value ranges. Expected
        values are ``"ipd"``, ``"ivd"``, and ``"ish"``.
    n_runs : int
        Number of independent runs represented in each cumulative reward CSV.
        Used to compute 95% confidence intervals for each terminal return.
    """
    figsize = (8, 6.5)
    types = ["S", "UT", "DE", "Ve", "Vk", "Vm"]
    types_long = ["Selfish", "Utilitarian", "Deontolog.", "Virtue-eq.", "Virtue-kind.", "Virtue-mix."]

    matrix_collective = pd.DataFrame(columns=types, index=types)  # shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_gini = pd.DataFrame(columns=types, index=types)  # shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_min = pd.DataFrame(columns=types, index=types)  # shape = vs S,UT,DE,Ve,Vk(,VEm)

    ylabs = {}
    ylims = {}

    if not (source_folder / "outcome_plots" / "outcomes_matrix").exists():
        (source_folder / "outcome_plots" / "outcomes_matrix").mkdir(
            parents=True, exist_ok=True
        )

    for player1_title in ["QLS", "QLUT", "QLDE", "QLVE_e", "QLVE_k", "QLVM"]:
        short_title = player1_title.replace("QL", "").replace("VE_e", "Ve").replace("VE_k", "Vk").replace("VM", "Vm")

        for type in ["collective", "gini", "min"]:
            against_QLS = pd.read_csv(source_folder / f"{player1_title}_QLS_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[
                -1]
            against_QLS_mean = against_QLS.mean()
            against_QLS_sds = against_QLS.std()
            against_QLS_ci = 1.96 * against_QLS_sds / np.sqrt(n_runs)

            try:
                against_QLUT = \
                pd.read_csv(source_folder / f"{player1_title}_QLUT_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            except:
                against_QLUT = \
                pd.read_csv(source_folder / f"QLUT_{player1_title}_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            against_QLUT_mean = against_QLUT.mean()
            against_QLUT_sds = against_QLUT.std()
            against_QLUT_ci = 1.96 * against_QLUT_sds / np.sqrt(n_runs)

            try:
                against_QLDE = \
                pd.read_csv(source_folder / f"{player1_title}_QLDE_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            except:
                against_QLDE = \
                pd.read_csv(source_folder / f"QLDE_{player1_title}_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            against_QLDE_mean = against_QLDE.mean()
            against_QLDE_sds = against_QLDE.std()
            against_QLDE_ci = 1.96 * against_QLDE_sds / np.sqrt(n_runs)

            try:
                against_QLVE_e = \
                pd.read_csv(source_folder / f"{player1_title}_QLVE_e_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            except:
                against_QLVE_e = \
                pd.read_csv(source_folder / f"QLVE_e_{player1_title}_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            against_QLVE_e_mean = against_QLVE_e.mean()
            against_QLVE_e_sds = against_QLVE_e.std()
            against_QLVE_e_ci = 1.96 * against_QLVE_e_sds / np.sqrt(n_runs)

            try:
                against_QLVE_k = \
                pd.read_csv(source_folder / f"{player1_title}_QLVE_k_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            except:
                against_QLVE_k = \
                pd.read_csv(source_folder / f"QLVE_k_{player1_title}_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            against_QLVE_k_mean = against_QLVE_k.mean()
            against_QLVE_k_sds = against_QLVE_k.std()
            against_QLVE_k_ci = 1.96 * against_QLVE_k_sds / np.sqrt(n_runs)

            try:
                against_QLVM = \
                pd.read_csv(source_folder / f"{player1_title}_QLVM_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            except:
                against_QLVM = \
                pd.read_csv(source_folder / f"QLVM_{player1_title}_MORL/df_cumulative_reward_{type}.csv", index_col=0).iloc[-1]
            against_QLVM_mean = against_QLVM.mean()
            against_QLVM_sds = against_QLVM.std()
            against_QLVM_ci = 1.96 * against_QLVM_sds / np.sqrt(n_runs)

            ylabs[f"{type}"] = (f"$G_{{{type}}}$")

            if game_title == "IPD".lower():
                if type == "collective":
                    ylims[f"{type}"] = [40000, 60000]
                elif type == "gini":
                    ylims[f"{type}"] = [5000, 10000]  # 5714.285714285714 min vale for G_gini
                elif type == "min":
                    ylims[f"{type}"] = [10000, 30000]
            elif game_title == "IVD".lower():
                if type == "collective":
                    ylims[f"{type}"] = [20000, 80000]
                elif type == "gini":
                    ylims[f"{type}"] = [4000, 10000]
                elif type == "min":
                    ylims[f"{type}"] = [10000, 40000]
            elif game_title == "ISH".lower():
                if type == "collective":
                    ylims[f"{type}"] = [40000, 100000]
                elif type == "gini":
                    ylims[f"{type}"] = [4000, 10000]
                elif type == "min":
                    ylims[f"{type}"] = [10000, 40000]

            if type == "collective":
                matrix_collective.loc[short_title] = [against_QLS_mean, against_QLUT_mean, against_QLDE_mean,
                                                      against_QLVE_e_mean, against_QLVE_k_mean, against_QLVM_mean]
                for label in types:
                    matrix_collective[label] = matrix_collective[label].astype(float)

            elif type == "gini":
                matrix_gini.loc[short_title] = [against_QLS_mean, against_QLUT_mean, against_QLDE_mean,
                                                against_QLVE_e_mean, against_QLVE_k_mean, against_QLVM_mean]
                for label in types:
                    matrix_gini[label] = matrix_gini[label].astype(float)

            elif type == "min":
                matrix_min.loc[short_title] = [against_QLS_mean, against_QLUT_mean, against_QLDE_mean,
                                               against_QLVE_e_mean, against_QLVE_k_mean, against_QLVM_mean]
                for label in types:
                    matrix_min[label] = matrix_min[label].astype(float)

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_collective, cmap="BrBG", xticklabels=types_long, yticklabels=types_long, annot=False,
                    linecolor="black", linewidths=0, square=False, vmin=ylims["collective"][0],
                    vmax=ylims["collective"][1])
    s.set(xlabel="Opponent O", ylabel="Player M", title="Collective Return " + ylabs["collective"] + " \n")
    plt.savefig(source_folder / "outcome_plots/outcomes_matrix/collective.pdf", bbox_inches="tight")
    matrix_collective.to_csv(source_folder / "outcome_plots/outcomes_matrix/matrix_collective.csv")

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_gini, cmap="BrBG", xticklabels=types_long, yticklabels=types_long, annot=False,
                    linecolor="black", linewidths=0, square=False, vmin=ylims["gini"][0], vmax=ylims["gini"][1])
    s.set(xlabel="Opponent O", ylabel="Player M", title="Gini Return " + ylabs["gini"] + " \n")
    plt.savefig(source_folder / "outcome_plots/outcomes_matrix/gini.pdf", bbox_inches="tight")
    matrix_gini.to_csv(source_folder / "outcome_plots/outcomes_matrix/matrix_gini.csv")

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_min, cmap="BrBG", xticklabels=types_long, yticklabels=types_long, annot=False,
                    linecolor="black", linewidths=0, square=False, vmin=ylims["min"][0], vmax=ylims["min"][1])
    s.set(xlabel="Opponent O", ylabel="Player M", title="Minimum Return " + ylabs["min"] + " \n")
    plt.savefig(source_folder / "outcome_plots/outcomes_matrix/min.pdf", bbox_inches="tight")
    matrix_min.to_csv(source_folder / "outcome_plots/outcomes_matrix/matrix_min.csv")
