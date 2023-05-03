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

pip install SimpleITK

# Copy files to local scratch space
cp -r data/ /scratch-shared/ssommers/
cp -r luna23-ismi-train-set.csv /scratch-shared/ssommers/data/
cp -r bodyct-luna23-ismi-trainer/networks.py /scratch-shared/ssommers/
cp -r bodyct-luna23-ismi-trainer/dataloader.py /scratch-shared/ssommers/
cp -r bodyct-luna23-ismi-trainer/inference.py /scratch-shared/ssommers/
cp -r bodyct-luna23-ismi-trainer/train.py /scratch-shared/ssommers/

# Set project directory to scratch space
project_dir=/scratch-shared/ssommers/

# Execute program
python /scratch-shared/ssommers/train.py $project_dir