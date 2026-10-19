#!/bin/bash
#SBATCH --job-name=narv
#SBATCH --output=/scratch/rohhs/projects/slurm_outputs/%j_%x.out
#SBATCH --nodes=2
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mail-user=rohollah903@gmail.com
#SBATCH --mail-type=ALL
