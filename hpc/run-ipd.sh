#!/bin/bash
#SBATCH --array=1-51
#SBATCH --job-name=mo-ipd
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --mail-type=END
#SBATCH --mail-user=roxana@ai.vub.ac.be

# Load the necessary modules.
module load Python/3.13.1-GCCcore-14.2.0
module load poetry/2.1.2-GCCcore-14.2.0

export OMP_NUM_THREADS=1

# Define variables.
EXPERIMENT_DIR="${VSC_DATA}/morl-dilemmas"
export POETRY_HOME="${VSC_DATA}/poetry"
export PATH=${POETRY_HOME}/bin:$PATH
export POETRY_CACHE_DIR=${VSC_DATA}/poetry-cache
export POETRY_VIRTUALENVS_IN_PROJECT=false
export POETRY_VIRTUALENVS_PATH=${VSC_DATA}/poetry-venvs
export PIP_CACHE_DIR=${VSC_DATA}/pip-cache
export TMPDIR=${VSC_DATA}/tmp

poetry install

MORL=$1
PHI=$2

if [ -n "$MORL" ] && [ "$MORL" = "morl" ]; then
    export MORL="--morl"
else
    export MORL="--no-morl"
fi


if [ -n "PHI" ]; then
    echo "Using Phi param to regulate morality: $PHI"
else
    export PHI=0.5
fi

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:$VSC_DATA/morl-dilemmas"

# Execute the line matching the array index from file *.list:
sleep ${SLURM_ARRAY_TASK_ID}
cmd=`head -${SLURM_ARRAY_TASK_ID} ${VSC_DATA}/morl-dilemmas/hpc/ipd.list | tail -1`

# Execute the command extracted from the file:
eval $cmd
