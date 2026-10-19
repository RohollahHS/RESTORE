#!/bin/bash
#SBATCH --job-name=ror
#SBATCH --output=/scratch/rohhs/projects/slurm_outputs/%j_%x.out
#SBATCH --error=/scratch/rohhs/projects/slurm_outputs/%j_%x.err
#SBATCH --nodes=2
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mail-user=rohollah903@gmail.com
#SBATCH --mail-type=ALL
