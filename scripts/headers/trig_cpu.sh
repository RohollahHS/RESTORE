#!/bin/bash
#SBATCH --job-name=trig
#SBATCH --output=/scratch/rohhs/projects/slurm_outputs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mail-user=rohollah903@gmail.com
#SBATCH --mail-type=ALL
