#!/bin/bash

#SBATCH --array=1
#SBATCH --time=120:00:00
#SBATCH --job-name=benders_%j
#SBATCH --output=results/benders_%j.out
#SBATCH --error=results/benders_%j.err

module add julia/1.10.3
module add gurobi/10.0.3

sbatch --nodes=7 --ntasks=7 --mem-per-cpu=32G --time=4380 --cpus-per-task=4 --ntasks-per-node=1 --wrap "julia runBenders_dist_v05.jl $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK"