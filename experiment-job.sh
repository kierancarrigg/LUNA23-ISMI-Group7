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

# Copy files to local scratch space
cp -r experiment/ /scratch-local/ssommers/

# Execute program
python /scratch-local/ssommers/experiment/python_test.py

cp -r /scratch-local/ssommers/experiment/ .
