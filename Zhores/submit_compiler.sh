#!/bin/bash
#SBATCH -p htc
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --output=/trinity/home/guskov.dv/locality/logs/%j.out
#SBATCH --array=0-99

module load compilers/intel_2019.5.281

module load python/intel-3.7

export OMP_NUM_THREADS=1

R=/trinity/home/guskov.dv/locality/data_logs/

python3 ./QAOA_locality_Zhores.py ${1} ${2} ${3} ${4} ${R} $(( ${SLURM_ARRAY_TASK_ID} ))
