#!/bin/bash

#SBATCH --array=1
#SBATCH --time=120:00:00
#SBATCH --job-name=single_%j
#SBATCH --output=results/single_%j.out
#SBATCH --error=results/single_%j.err

module add julia/1.10.3
module add gurobi/10.0.3

sbatch --nodes=1 --ntasks=1 --mem-per-cpu=32G --time=4380 --cpus-per-task=4 --ntasks-per-node=1 --wrap "julia runSingle.jl $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK"