#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=thin
#SBATCH --time=01:00:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

pip install lightning

# # Copy files to local scratch space
# cp -r experiment/ /scratch-shared/ssommers/

# # Set project directory to scratch space
# project_dir=/scratch-shared/ssommers/experiment/

# # Execute program
# python /scratch-shared/ssommers/experiment/python_test.py $project_dir

# # cp -r /scratch-shared/ssommers/experiment/cifar_net.pth .
