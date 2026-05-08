#!/bin/bash
#################################
#### BASH SCRIPT - VOLUNTEER ####
#################################

MORL=$1

if [ -n "$MORL" ] && [ "$MORL" = "morl" ]; then
    MORL="--morl"
else
    MORL="--no-morl"
fi

# Run part 1 - QLS vs all other learners, some moral vs moral - DONE
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLS --title2 QLS --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLUT --title2 QLS --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLDE --title2 QLS --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 QLS --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 QLS --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 QLS --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLUT --title2 QLUT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLDE --title2 QLUT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLDE --title2 QLDE --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 QLUT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 QLDE --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 QLVE_e --eps-theta 1.0 --eps-decay "$MORL"

# Run part 2 - remainder of moral vs moral; moral mixed vs. all others - DONE
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 QLUT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 QLDE --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 QLVE_e --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 QLVE_k --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 QLUT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 QLDE --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 QLVE_e --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 QLVE_k --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 QLVM --eps-theta 1.0 --eps-decay "$MORL"

# Run part 3 - all vs static - DONE
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLS --title2 AC --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLS --title2 AD --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLS --title2 TFT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLS --title2 Random --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLUT --title2 AC --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLUT --title2 AD --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLUT --title2 TFT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLUT --title2 Random --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLDE --title2 AC --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLDE --title2 AD --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLDE --title2 TFT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLDE --title2 Random --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 AC --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 AD --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 TFT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_e --title2 Random --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 AC --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 AD --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 TFT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVE_k --title2 Random --eps-theta 1.0 --eps-decayv
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 AC --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 AD --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 TFT --eps-theta 1.0 --eps-decay "$MORL"
poetry run python -m uu.ai.thesis.cli.play --game-type ivd --title1 QLVM --title2 Random --eps-theta 1.0 --eps-decay "$MORL"
