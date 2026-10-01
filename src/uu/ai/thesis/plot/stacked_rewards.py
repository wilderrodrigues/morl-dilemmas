import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def set_ylims(game_title, player1_title):
    '''this function sets ylims for the plots of cumulative moral reward across all 3 games & 6 player types'''
    ylims = []
    if game_title == "IPD".lower():
        if player1_title == "QLUT":
            ylims[0:1] = [0, 60000]
        elif player1_title == "QLDE":
            ylims[0:1] = [-50000, 0]
        elif player1_title == "QLVE_e":
            ylims[0:1] = [0, 10000]
        elif player1_title == "QLVE_k":
            ylims[0:1] = [0, 50000]
        elif player1_title == "QLVM":
            ylims[0:1] = [0, 10000]
    elif game_title == "IVD".lower():
        if player1_title == "QLUT":
            ylims[0:1] = [0, 80000]
        elif player1_title == "QLDE":
            ylims[0:1] = [-50000, 0]
        elif player1_title == "QLVE_e":
            ylims[0:1] = [0, 10000]
        elif player1_title == "QLVE_k":
            ylims[0:1] = [0, 50000]
        elif player1_title == "QLVM":
            ylims[0:1] = [0, 10000]
    elif game_title == "ISH".lower():
        if player1_title == "QLUT":
            ylims[0:2] = [0, 100000]
        elif player1_title == "QLDE":
            ylims[0:2] = [-50000, 0]
        elif player1_title == "QLVE_e":
            ylims[0:2] = [0, 10000]
        elif player1_title == "QLVE_k":
            ylims[0:2] = [0, 50000]
        elif player1_title == "QLVM":
            ylims[0:2] = [0, 10000]
    return ylims[0], ylims[1]


def plot_stacked_relative_reward(destination_folder: Path, game_title: str, n_runs: int) -> None:
    '''plot game reward - relatie cumulative (bar & over time) & per iteration
    - how well off did the players end up relative to each other on the game?'''

    data = dict()  # keys=['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM'], values=[]

    for player1_title in ['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM']:
        ##################################
        #### bar chart game cumulative reward for player1_tytle vs others  ####
        ##################################
        against_QLS = \
        pd.read_csv(destination_folder / f"{player1_title}_QLS_MORL/player1/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        against_QLS_means = against_QLS.mean()
        against_QLS_sds = against_QLS.std()
        against_QLS_ci = 1.96 * against_QLS_sds / np.sqrt(n_runs)

        try:
            against_QLUT = \
            pd.read_csv(destination_folder / f"{player1_title}_QLUT_MORL/player1/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        except:
            against_QLUT = \
            pd.read_csv(destination_folder / f"QLUT_{player1_title}_MORL/player2/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        against_QLUT_means = against_QLUT.mean()
        against_QLUT_sds = against_QLUT.std()
        against_QLUT_ci = 1.96 * against_QLUT_sds / np.sqrt(n_runs)

        try:
            against_QLDE = \
            pd.read_csv(destination_folder / f"{player1_title}_QLDE_MORL/player1/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        except:
            against_QLDE = \
            pd.read_csv(destination_folder/ f"QLDE_{player1_title}_MORL/player2/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        against_QLDE_means = against_QLDE.mean()
        against_QLDE_sds = against_QLDE.std()
        against_QLDE_ci = 1.96 * against_QLDE_sds / np.sqrt(n_runs)

        try:
            against_QLVE_e = \
            pd.read_csv(destination_folder / f"{player1_title}_QLVE_e_MORL/player1/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        except:
            against_QLVE_e = \
            pd.read_csv(destination_folder / f"QLVE_e_{player1_title}_MORL/player2/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        against_QLVE_e_means = against_QLVE_e.mean()
        against_QLVE_e_sds = against_QLVE_e.std()
        against_QLVE_e_ci = 1.96 * against_QLVE_e_sds / np.sqrt(n_runs)

        try:
            against_QLVE_k = \
            pd.read_csv(destination_folder / f"{player1_title}_QLVE_k_MORL/player1/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        except:
            against_QLVE_k = \
            pd.read_csv(destination_folder / f"QLVE_k_{player1_title}_MORL/player2/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        against_QLVE_k_means = against_QLVE_k.mean()
        against_QLVE_k_sds = against_QLVE_k.std()
        against_QLVE_k_ci = 1.96 * against_QLVE_k_sds / np.sqrt(n_runs)

        try:
            against_QLVM = \
            pd.read_csv(destination_folder / f"{player1_title}_QLVM_MORL/player1/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        except:
            against_QLVM = \
            pd.read_csv(destination_folder / f"QLVM_{player1_title}_MORL/player2/df_cumulative_reward_game.csv", index_col=0).iloc[-1]
        against_QLVM_means = against_QLVM.mean()
        against_QLVM_sds = against_QLVM.std()
        against_QLVM_ci = 1.96 * against_QLVM_sds / np.sqrt(n_runs)

        data[str('means_' + player1_title)] = [against_QLS_means, against_QLUT_means, against_QLDE_means,
                                               against_QLVE_e_means, against_QLVE_k_means, against_QLVM_means]
        data[str('CIs_' + player1_title)] = [against_QLS_ci, against_QLUT_ci, against_QLDE_ci, against_QLVE_e_ci,
                                             against_QLVE_k_ci, against_QLVM_ci]

    labels = ['S', 'UT', 'DE', 'V' + r'$_e$', 'V' + r'$_k$', 'V' + r'$_m$']
    # colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple', 'pink']
    colors = ['#984464', '#e6a176', '#556b2f', '#5eccab', '#537eff', '#c0affb']
    # colors = ['#e1562c', 'orange', '#556b2f', 'lightblue', '#00678a', 'pink']
    # plt.rcParams.update({'font.size':20})
    font = {'size': 15}
    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(1, 6, figsize=(11, 3.5), sharey=True)  # constrained_layout=False, tight_layout=False,
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    ax1.bar(labels, data['means_QLS'], yerr=data['CIs_QLS'], color=colors)  # , width = 0.8
    ax1.set_title('Selfish')  # vs others
    ax1.set_ylabel('Cumulative \n' + 'Game Reward')  # r'$R_{extr}$'
    # ax1.set_ylim()
    if game_title == "IPD".lower():  # NOTE game_title is set outside this function - in the overall environment - see code below
        plt.gca().set_ylim([0, 40000])
    elif game_title == "IVD".lower():
        plt.gca().set_ylim([0, 50000])
    elif game_title == "ISH".lower():
        plt.gca().set_ylim([0, 50000])
        # ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)
    # ax1.set_xticklabels(labels, rotation=90, ha='right')
    # ax1.yaxis.set_label(r'Cumulative Game Reward $R_{extr}$')

    ax2.bar(labels, data['means_QLUT'], yerr=data['CIs_QLUT'], color=colors)  # , width = 0.8
    ax2.set_title('Utilitarian')

    ax3.bar(labels, data['means_QLDE'], yerr=data['CIs_QLDE'], color=colors)  # , width = 0.8
    ax3.set_title('Deontolog.')

    ax4.bar(labels, data['means_QLVE_e'], yerr=data['CIs_QLVE_e'], color=colors)  # , width = 0.8
    ax4.set_title(' Virtue-equal.')

    ax5.bar(labels, data['means_QLVE_k'], yerr=data['CIs_QLVE_k'], color=colors)  # , width = 0.8
    ax5.set_title(' Virtue-kind.')

    ax6.bar(labels, data['means_QLVM'], yerr=data['CIs_QLVM'], color=colors)  # , width = 0.8
    ax6.set_title(' Virtue-mixed')

    fig.suptitle('Game Reward for player type M vs all others \n')

    # ax1.yaxis.set_label(r'Cumulative Game Reward $R_{extr}$')
    fig.autofmt_xdate(rotation=90, bottom=0.2, ha='center')
    # plt.xticks(rotation=45)
    fig.supxlabel('\n' + 'Opponent type')  # NOTE this requires matplotlib>3.4
    plt.tight_layout(pad=0.001, w_pad=0.8)
    fig.show()

    if not os.path.isdir(destination_folder / "outcome_plots/reward"):
        os.makedirs(destination_folder / "outcome_plots/reward")
    plt.savefig(destination_folder / f"outcome_plots/reward/cumulative_game_reward.png", bbox_inches='tight')


def plot_stacked_relative_moral_reward(destination_folder: Path, game_title: str, n_runs: int) -> None:
    '''plot game reward - relatie cumulative (bar & over time) & per iteration
    - how well off did the players end up relative to each other on the game?'''

    data = dict()  # keys=['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM'], values=[]

    for player1_title in ['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM']:
        ##################################
        #### bar chart game cumulative reward for player1_tytle vs others  ####
        ##################################
        against_QLS = \
        pd.read_csv(destination_folder / f'{player1_title}_QLS_MORL/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_QLS_means = against_QLS.mean()
        against_QLS_sds = against_QLS.std()
        against_QLS_ci = 1.96 * against_QLS_sds / np.sqrt(n_runs)

        try:
            against_QLUT = \
            pd.read_csv(destination_folder / f'{player1_title}_QLUT_MORL/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        except:
            against_QLUT = \
            pd.read_csv(destination_folder / f'QLUT_{player1_title}_MORL/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        against_QLUT_means = against_QLUT.mean()
        against_QLUT_sds = against_QLUT.std()
        against_QLUT_ci = 1.96 * against_QLUT_sds / np.sqrt(n_runs)

        try:
            against_QLDE = \
            pd.read_csv(destination_folder / f'{player1_title}_QLDE_MORL/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        except:
            against_QLDE = \
            pd.read_csv(destination_folder / f'QLDE_{player1_title}_MORL/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        against_QLDE_means = against_QLDE.mean()
        against_QLDE_sds = against_QLDE.std()
        against_QLDE_ci = 1.96 * against_QLDE_sds / np.sqrt(n_runs)

        try:
            against_QLVE_e = \
            pd.read_csv(destination_folder / f'{player1_title}_QLVE_e_MORL/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        except:
            against_QLVE_e = \
            pd.read_csv(destination_folder / f'QLVE_e_{player1_title}_MORL/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        against_QLVE_e_means = against_QLVE_e.mean()
        against_QLVE_e_sds = against_QLVE_e.std()
        against_QLVE_e_ci = 1.96 * against_QLVE_e_sds / np.sqrt(n_runs)

        try:
            against_QLVE_k = \
            pd.read_csv(destination_folder / f'{player1_title}_QLVE_k_MORL/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        except:
            against_QLVE_k = \
            pd.read_csv(destination_folder / f'QLVE_k_{player1_title}_MORL/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        against_QLVE_k_means = against_QLVE_k.mean()
        against_QLVE_k_sds = against_QLVE_k.std()
        against_QLVE_k_ci = 1.96 * against_QLVE_k_sds / np.sqrt(n_runs)

        try:
            against_QLVM = \
            pd.read_csv(destination_folder / f'{player1_title}_QLVM_MORL/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        except:
            against_QLVM = \
            pd.read_csv(destination_folder / f'QLVM_{player1_title}_MORL/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[
                -1]
        against_QLVM_means = against_QLVM.mean()
        against_QLVM_sds = against_QLVM.std()
        against_QLVM_ci = 1.96 * against_QLVM_sds / np.sqrt(n_runs)

        data[str('means_' + player1_title)] = [against_QLS_means, against_QLUT_means, against_QLDE_means,
                                               against_QLVE_e_means, against_QLVE_k_means, against_QLVM_means]
        data[str('CIs_' + player1_title)] = [against_QLS_ci, against_QLUT_ci, against_QLDE_ci, against_QLVE_e_ci,
                                             against_QLVE_k_ci, against_QLVM_ci]

    labels = ['S', 'UT', 'DE', 'V' + r'$_e$', 'V' + r'$_k$', 'V' + r'$_m$']
    # colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple', 'pink']
    colors = ['#984464', '#e6a176', '#556b2f', '#5eccab', '#537eff', '#c0affb']
    # colors = ['#e1562c', 'orange', '#556b2f', 'lightblue', '#00678a', 'pink']
    # plt.rcParams.update({'font.size':20})
    font = {'size': 15}
    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(1, 5, figsize=(11, 3.5), sharey=False)  # constrained_layout=False, tight_layout=False,
    ax2, ax3, ax4, ax5, ax6 = axes

    # ax1.bar(labels, data['means_QLS'], yerr=data['CIs_QLS'], color=colors) #, width = 0.8
    # ax1.set_title('Selfish') #vs others
    # ax1.set_ylabel('Cumulative \n' + 'Moral Reward') #r'$R_{extr}$'

    ax2.bar(labels, data['means_QLUT'], yerr=data['CIs_QLUT'], color=colors)  # , width = 0.8
    ax2.set_title('Utilitarian')
    ylims = set_ylims(game_title, 'QLUT')
    ax2.set_ylim(ylims)

    ax3.bar(labels, data['means_QLDE'], yerr=data['CIs_QLDE'], color=colors)  # , width = 0.8
    ax3.set_title('Deontolog.')
    ylims = set_ylims(game_title, 'QLDE')
    ax3.set_ylim(ylims)

    ax4.bar(labels, data['means_QLVE_e'], yerr=data['CIs_QLVE_e'], color=colors)  # , width = 0.8
    ax4.set_title(' Virtue-equal.')
    ylims = set_ylims(game_title, 'QLVE_e')
    ax4.set_ylim(ylims)

    ax5.bar(labels, data['means_QLVE_k'], yerr=data['CIs_QLVE_k'], color=colors)  # , width = 0.8
    ax5.set_title(' Virtue-kind.')
    ylims = set_ylims(game_title, 'QLVE_k')
    ax5.set_ylim(ylims)

    ax6.bar(labels, data['means_QLVM'], yerr=data['CIs_QLVM'], color=colors)  # , width = 0.8
    ax6.set_title(' Virtue-mixed')
    ylims = set_ylims(game_title, 'QLVM')
    ax6.set_ylim(ylims)

    for ax in axes:
        for label in ax.get_yticklabels():
            label.set_fontsize(11)

    fig.suptitle('Moral Reward for player type M vs all others \n')

    ax2.set_ylabel('Cumulative \n Moral Reward')
    fig.autofmt_xdate(rotation=90, bottom=0.2, ha='center')
    # plt.xticks(rotation=45)
    fig.supxlabel('\n' + 'Opponent type')  # NOTE this requires matplotlib>3.4
    plt.tight_layout(pad=0.001, w_pad=0.1)
    fig.show()

    if not os.path.isdir(destination_folder / "outcome_plots/reward"):
        os.makedirs(destination_folder / "outcome_plots/reward")
    plt.savefig(destination_folder / f"outcome_plots/reward/cumulative_intrinsic_reward.png", bbox_inches="tight")