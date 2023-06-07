# LUNA23-ISMI-Group7

## Members
-Kieran Carigg (s4792882)
-Eline Braun (s1010232)
-Lisa Hensens (s1018583)
-Serah Sommers (s1015986)

## Abstract
Lung cancer is the most common cancer type and results in millions of cases and deaths worldwide. Deep learning techniques are a potential solution to speed up the diagnosis process, as they automate processes to decrease time and healthcare costs. 
This paper proposes a multi-task deep learning model for segmentation, nodule type classification, and malignancy risk classification using CT scan images of lungs. It uses a U-Net architecture including shared feature extraction in the contraction space. Data augmentation (i.e. translation, rotation, and a combination of translation and rotation) is applied to generate more data. Additionally, class weights are used to solve the class imbalance problem. The proposed model also includes batch normalization, dropout, weight decay, and parameter sharing for regularization. The results show potential for using a multi-task learning model in diagnostic image analysis.

## Directory structure
### [Literature/](Literature/)
This folder contains the literature we reviewed in order to gain a better understanding of the topic, and insights on effective methods to apply to this problem.

### [data/](data/)
This folder contains the LUNA23 dataset, consisting of a train set with both images and a csv file with corresponding labels, and a test set with just images.

### [plots/](plots/)
This folder contains plots of the results and training processes.

### [results/](results/)
This folder contains the results of training our model with various different parameters. Note that these results only contain the metrics saved during training and not the actual models themselves, as these files were too large to upload to this repository.

## File structure
### Notebooks
- [Exploratory data analysis](exploratory-data-analysis.ipynb): This notebook visualises the training data.
- [Visualise training plots](visualize-training-plots.ipynb): This notebook can be used to visualise the metrics and losses of the experiments during training.
- [Visualise predictions](visualize-predictions.ipynb): This notebook can be used to visualise the model's predictions on the test data.
- [Visualise experiment results](visualise_experiment_results_plots.ipynb): This notebook was used to visualise the results of various optimisation and regularisation experiments.

### .py files
- [Dataloader](dataloader.py): Can be called to load in and augment train or test data.
- [Multitask network](multitask_network.py): Contains the multitask model.
- [Multitask train](multitask_train.py): This file loads in the multitask model and trains it.
- [Multitask inference](multitask_inference.py): This file loads in a trained multitask model and runs inference on it.
- [Multitask ensembling inference](multitask_inference_ensembling.py): This file loads in all folds of a trained multitask model and runs inference on it using ensembling methods.

### bash files
- [Run multitask inference](run_multitask_inference.sh): Submits the [multitask inference](multitask_inference.py) file as a slurm job.
- [Run multitask training](run_multitask_training.sh): Submits the [multitask train](multitask_train.py) file as a slurm job.
