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
cp dataloader.py /scratch-shared/ssommers/
cp multitask_network.py /scratch-shared/ssommers/
cp multitask_inference_ensembling.py /scratch-shared/ssommers/
cp -r results/20230529_22_multitask_model/ /scratch-shared/ssommers/results/
mkdir /scratch-shared/ssommers/results/20230529_22_multitask_model/test_set_predictions_mean/

# Set project directory to scratch space
project_dir=/scratch-shared/ssommers/

# Execute program
python /scratch-shared/ssommers/multitask_inference_ensembling.py $project_dir

# Copy results from scratch space
cp -r /scratch-shared/ssommers/results/20230529_22_multitask_model/test_set_predictions_mean/ results/20230529_22_multitask_model/