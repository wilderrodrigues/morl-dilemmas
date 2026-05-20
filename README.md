# Incorporating Moral Choices in Social Dilemmas with Multi-Objective Reinforcement Learning

This repository contains the code and materials for my Bachelor's thesis in Artificial Intelligence.

## Disclaimer

Most of the code has been adapted from the works performerd by Elizaveta _et al_. (2023). The existing code adaptation and
structure of the framework has been fully developed by the author of this repository, Wilder Rodrigues (2023). The Python
_docstrings_ have been generated using OpenAI Codex, GPT 5.4 (medium).  

## Abstract

Multi-objective reinforcement learning (MORL) extends traditional reinforcement learning by allowing agents to optimize multiple, potentially conflicting objectives simultaneously. In this thesis, MORL is studied using a parametric non-linear utility function to model individual agents' risk preferences. This framework is combined with a predefined set of moral choice types, incorporated as intrinsic reward components grounded in moral theories.

The project investigates learning in a novel multi-objective formulation of cooperative social dilemma games, including iterated versions of the Prisoner's Dilemma, Stag Hunt, and the Volunteer's Dilemma. The analysis focuses on how moral reward components and non-linear utility functions shape agent behavior in settings with incentive misalignment, particularly regarding selfish strategies, cooperation, exploration, and the distribution of rewards.

The approach is evaluated by modeling interactions between learning moral agents in the selected iterated social dilemma games. The thesis also studies the interplay between collective and individual reward components in a multi-objective setting, and examines how different moral types influence cooperation and defection across cooperative and non-cooperative environments. Evaluation is based on three social outcome metrics computed from cumulative returns after a defined number of iterations.

## Research Focus

- Multi-objective reinforcement learning
- Moral reward modeling
- Social dilemma games
- Cooperation and defection dynamics
- Non-linear utility and risk preferences

## Planned Environments

- Iterated Prisoner's Dilemma
- Iterated Stag Hunt
- Iterated Volunteer's Dilemma

## Repository Status

This repository is currently in the early setup phase. Code, experiments, and results will be added as the thesis progresses.

## Setup

The project currently uses a PyEnv in combination with Poetry. To set up the environment, run the following commands:

* **NB:** _This setup is for MacOS only. Do some googling to get it working with Linux environments._

### For MacOS

```shell
brew install pyenv pyenv-virtualenv
```

### For Linux

```shell
curl -fsSL https://pyenv.run | bash
```

Then proceed with the following commands:

```shell
pyenv install 3.13.7
pyenv activate 3.13.7
```

Once that's done, please proceed and install poetry:

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

Now you are ready to install the project dependencies:

```shell
poetry install
```

### Quick test

To make sure all is in place, run the following command:

```shell
poetry run python -m uu.ai.thesis.cli.play --help
```

To get a quick experiment running, run the following command:

```shell
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLS --title2 AC --eps-theta 1.0 --eps-decay
```

## Running the Experiments

To run the experiments, use one of the `bash` scripts provided in the root directory of the repository. For example, 
to run all IPD experiments, execute the following command:

```shell
./run_ipd.sh morl 0.5
```

The `morl` argument is to enable multi-objective reinforcement learning. The `0.5` argument is the Phi parameter for the utility function.

Once the run is done, please clean the `run*.csv` to make room for the next run.

```shell
find . -type f -name 'run*.csv' -delete
```

## Pushing results with Git LFS - Large File Storage

Please follow the [Git LFS](https://git-lfs.com/) page for instructions on how to install and use Git LFS.

We already have .ZIP files mapped to the Git LFS attributes file. To push the files to Git LFS, please run the following command:

```shell
git add ipd-results.zip
git commit -m "Adds IPD results to Git LFS"
git push origin main
```

To fetch the files from Git LFS, please run the following command:

```shell
git lfs fetch
```

## Plotting results

To plot the results for the experiments, please make use of the `compute` CLI. To generate the plots for the IPD experient,
for example, run the following command:

```shell
poetry run python -m uu.ai.thesis.cli.compute --game-type ipd --num-runs 100
```

There are other two scripts available for the Iterative Stag Hunt and Iterative Volunteer's Dilemma experiments. Those can
be found under the root directory of the repository as well.

## Author

1. Wilder Rodrigues

## Supervisors

1. Dr. Roxana Radulescu
2. Dr. Gerard Vreeswijk

## Citations

Most of the work in the repository was originally implemented by Elizaveta _et al_. I have refactored the main features
out into the core package, where I abstracted the environment, agent, and policy components. The execution of the experiments
has been moved into a command line interface (CLI), fully parametrised for easy use.

```bibtex
@INPROCEEDINGS{Tennant-ijcai2023p36,
  title     = {Modeling Moral Choices in Social Dilemmas with Multi-Agent Reinforcement Learning},
  author    = {Tennant, Elizaveta and Hailes, Stephen and Musolesi, Mirco},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {317--325},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/36},
  url       = {https://doi.org/10.24963/ijcai.2023/36},
}
```