#!/bin/bash
#SBATCH --job-name=nibi
#SBATCH --output=/scratch/rohhs/projects/slurm_outputs/%j_%x.out
#SBATCH --error=/scratch/rohhs/projects/slurm_outputs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --gres=gpu:h100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mail-user=rohollah903@gmail.com
#SBATCH --mail-type=ALL
