#!/bin/bash
################################
#### MAIN BASH SCRIPT - IPD ####
################################

#Phi determines the degree of morality of the player. 0.5 = low morality agent. If phi = 1 agents put equal importance on the individual and moral dimenssions. If phi == 2 we have a highly moral agent.
#(base) ➜  ~ Understand the outcomes when working in the environemnt with different morality ... moral type ==

MORL=$1
PHI=$2

if [ -n "$MORL" ] && [ "$MORL" = "morl" ]; then
    MORL="--morl"
else
    MORL="--no-morl"
fi

if [ -n "PHI" ]; then
    echo "Using Phi param to regulate morality: $PHI"
else
    PHI=0.5
fi

# Run part 1 - QLS vs all other learners, some moral vs moral - DONE ON 3
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLS --title2 QLS --eps-theta 1.0 --eps-decay --phi "$PHI" "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLUT --title2 QLS --eps-theta 1.0 --eps-decay --phi "$PHI" "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLDE --title2 QLS --eps-theta 1.0 --eps-decay --phi "$PHI" "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLVM --title2 QLS --eps-theta 1.0 --eps-decay --phi "$PHI" "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLUT --title2 QLUT --eps-theta 1.0 --eps-decay --phi "$PHI" "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLDE --title2 QLUT --eps-theta 1.0 --eps-decay --phi "$PHI" "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ipd --title1 QLVM --title2 QLVM --eps-theta 1.0 --eps-decay --phi "$PHI" "$MORL"
