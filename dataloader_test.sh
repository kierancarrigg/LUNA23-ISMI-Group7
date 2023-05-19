#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -o ./logs/slurm-%j.out # STDOUT

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# Copy files to local scratch space
cp -r data/ /scratch-shared/ssommers/
cp data/luna23-ismi-train-set.csv /scratch-shared/ssommers/data/
cp dataloaderDuplicate.py /scratch-shared/ssommers/

# Set project directory to scratch space
project_dir=/scratch-shared/ssommers/

# Execute program
python /scratch-shared/ssommers/dataloaderDuplicate.py $project_dir