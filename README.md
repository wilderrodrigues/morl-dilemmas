# Incorporating Moral Choices in Social Dilemmas with Multi-Objective Reinforcement Learning

This repository contains the code and materials for my Bachelor's thesis in Artificial Intelligence.

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

The project currently uses a Conda environment defined in [`environment.yaml`](/Users/wilderrodrigues/dev/git/uni/thesis/morl-dilemmas/environment.yaml).

```bash
conda env create -f environment.yaml
conda activate morl-dilemmas
```

## Author

Wilder Rodrigues

## Supervisors

Dr. Roxana Radulescu
