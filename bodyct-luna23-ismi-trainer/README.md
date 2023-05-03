# LUNA23-ISMI baseline training scripts

## Preparing your environment

Make sure to install the libraries listed in `requirements.txt`

## Preparing your dataset

1. Make sure to download the dataset: https://luna23-ismi.grand-challenge.org/datasets/
2. Copy the zip file into a new folder at the root of your git repository and name it `data/`
3. Extract the zip file inside this folder
4. Make sure that the training images are inside `data/train_set/images/`

## Inspecting your dataset

You can use the notebook, [`exploratory-data-analysis.ipynb`](./exploratory-data-analysis.ipynb) to visualize the nodules and understand the task. Needless to say (although I must reiterate, upon second thoughts), data inspection is crucial to understand what you are trying to solve, especially in machine learning and data science.

## Training the baseline algorithm

Ensure that the `workspace` argument is properly configured in [`train.py`](./train.py). Then, simply execute the following code:

```
python3 train.py
```

You can use [`visualize-training-plots.ipynb`](./visualize-training-plots.ipynb) to understand if your models are converging or not.

## Inference

A basic inference script is provided in [`inference.py`](./inference.py). Configure your models and the checkpoints inside the `perform_inference_on_test_set` function and then execute the following:

```
python3 inference.py
```

## Visualizing predictions

You can use [`visualize-predictions.ipynb`](./visualize-predictions.ipynb) to visualize the segmentations and the predictions from your trained models. 