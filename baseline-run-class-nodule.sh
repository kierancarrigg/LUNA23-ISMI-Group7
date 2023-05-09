#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o ./logs/slurm-%j.out # STDOUT

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# Copy files to local scratch space
cp -r data/ /scratch-shared/ssommers/
cp -r luna23-ismi-train-set.csv /scratch-shared/ssommers/data/
cp -r bodyct-luna23-ismi-trainer/networks.py /scratch-shared/ssommers/
cp -r bodyct-luna23-ismi-trainer/dataloader.py /scratch-shared/ssommers/
cp -r bodyct-luna23-ismi-trainer/inference.py /scratch-shared/ssommers/
cp -r bodyct-luna23-ismi-trainer/train_classification_nodule.py /scratch-shared/ssommers/

# Set project directory to scratch space
project_dir=/scratch-shared/ssommers/

# Execute program
python /scratch-shared/ssommers/train_classification_nodule.py $project_dir

# Copy results from scratch space
cp -r /scratch-shared/ssommers/results/ .
