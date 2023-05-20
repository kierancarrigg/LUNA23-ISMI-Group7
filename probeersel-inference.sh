#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH -o ./logs/inference-%j.out # STDOUT

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# Copy files to local scratch space
cp -r data/ /scratch-shared/ssommers/
cp bodyct-luna23-ismi-trainer/dataloader.py /scratch-shared/ssommers/
cp probeersel.py /scratch-shared/ssommers/
cp multitask_inference.py /scratch-shared/ssommers/
cp -r results/ /scratch-shared/ssommers/
mkdir /scratch-shared/ssommers/results/20230519_1_multitask_model/fold0/test_set_predictions/

# Set project directory to scratch space
project_dir=/scratch-shared/ssommers/

# Execute program
python /scratch-shared/ssommers/multitask_inference.py $project_dir

# Copy results from scratch space
cp -r /scratch-shared/ssommers/results/20230519_1_multitask_model/fold0/test_set_predictions/ results/20230519_1_multitask_model/fold0/