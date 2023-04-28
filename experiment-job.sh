#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# Execute program
python python_test.py