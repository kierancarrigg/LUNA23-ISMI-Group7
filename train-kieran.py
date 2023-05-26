import sys
import pandas
import dataloaderDuplicate as dataloader
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import probeersel
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skl_metrics
from typing import List
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

logging = dataloader.logging
project_dir = sys.argv[1]

def dice_loss(input, target):
    """Function to compute dice loss
    source: https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398

    Args:
        input (torch.Tensor): predictions
        target (torch.Tensor): ground truth mask

    Returns:
        dice loss: 1 - dice coefficient
    """
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def make_development_splits(
    train_set: pandas.DataFrame,
    save_path: Path(project_dir+"splits/"),
    n_folds: int = 5,
):
    """Function to split your training set into 5 folds at a patient-level

    Args:
        train_set (pandas.DataFrame): pandas dataframe that contains list of nodules
        save_path (Path): path to save the splits
        n_folds (int, optional): number of folds. Defaults to 5.
    """

    np.random.seed(2023)

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    pids = train_set.patientid.unique()
    labs = [train_set[train_set.patientid == pid].malignancy.values[0] for pid in pids]
    labs = np.array(labs)

    assert len(pids) == len(labs)

    skf = StratifiedKFold(n_splits=n_folds)
    skf.get_n_splits(pids, labs)

    folds_missing = False

    for idx in range(n_folds):

        train_pd = save_path / f"train{idx}.csv"
        valid_pd = save_path / f"valid{idx}.csv"

        if not train_pd.is_file():
            folds_missing = True

        if not valid_pd.is_file():
            folds_missing = True

    if folds_missing:

        print(f"Making {n_folds} folds from the train set")

        for idx, (train_index, test_index) in enumerate(skf.split(pids, labs)):

            train_pids, valid_pids = pids[train_index], pids[test_index]

            train_pd = train_set[train_set.patientid.isin(train_pids)]
            valid_pd = train_set[train_set.patientid.isin(valid_pids)]

            train_pd = train_pd.reset_index(drop=True)
            valid_pd = valid_pd.reset_index(drop=True)

            train_pd.to_csv(save_path / f"train{idx}.csv", index=False)
            valid_pd.to_csv(save_path / f"valid{idx}.csv", index=False)
    
    if not folds_missing:
        print("folds found! :)")

class NoduleAnalyzer:
    """Class to train a multi-task nodule analyzer"""

    def __init__(
        self,
        best_metric_fn,
        workspace: Path(project_dir),
        experiment_id: int,
        fold: int = 0,
        batch_size: int = 4,
        num_workers: int = 16,
        max_epochs: int = 1000        
    ) -> None:

        self.workspace = workspace
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.size_mm = 50
        self.size_px = 64
        self.patch_size = np.array([64, 128, 128])
        self.max_rotation_degree = 20

        self.max_epochs = max_epochs
        self.learning_rate = 1e-3
        # self.learning_rate = 0.05

        self.best_metric_fn = best_metric_fn

        date = datetime.today().strftime("%Y%m%d")
        self.exp_id = f"{date}_{experiment_id}"

        self.fold = fold
        self.tasks = ['segmentation', 'nodule-type', 'malignancy']

        train_df_path = workspace / "data" / "luna23-ismi-train-set.csv"
        make_development_splits(
            train_set=pandas.read_csv(train_df_path),
            save_path=workspace / "data" / "train_set" / "folds",
        )

        self.train_df = pandas.read_csv(
            workspace / "data" / "train_set" / "folds" / f"train{fold}.csv"
        )
        self.valid_df = pandas.read_csv(
            workspace / "data" / "train_set" / "folds" / f"valid{fold}.csv"
        )

    def _initialize_model(self, model):

        torch.backends.cudnn.benchmark = True
        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not

        # define the GPU - ideally this is the first GPU, hence cuda:0
        self.device = torch.device("cuda:0")

        # transfer model to GPU
        self.model = model.to(self.device)

        # define the optimzer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # define the scheduler
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer=self.optimizer,
        #     step_size=50,
        #     gamma=0.5,
        # )

    def _initialize_data_loaders(self):

        
        x = self.train_df.malignancy.values
        x = dataloader.make_weights_for_balanced_classes(x)
        weights_malignancy = x

        y = self.train_df.noduletype.values
        y = [dataloader.NODULETYPE_MAPPING[t] for t in y]
        y = dataloader.make_weights_for_balanced_classes(y)
        weights_noduletype = y

        weights = weights_malignancy * weights_noduletype  # 🥚 Easter egg

        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights,
            num_samples = 4 * len(self.train_df),
        )

        self.train_loader = dataloader.get_data_loader(
            self.workspace / "data" / "train_set",
            self.train_df,
            sampler=sampler,
            workers=self.num_workers // 2,
            batch_size=self.batch_size,
            rotations=[(-self.max_rotation_degree, self.max_rotation_degree)] * 3,
            translations=True,
            size_mm=self.size_mm,
            size_px=self.size_px,
            patch_size=self.patch_size,
            train = True
        )

        self.valid_loader = dataloader.get_data_loader(
            self.workspace / "data" / "train_set",
            self.valid_df,
            workers=self.num_workers // 2,
            batch_size=self.batch_size,
            size_mm=self.size_mm,
            size_px=self.size_px,
            patch_size=self.patch_size
        )

    def forward(self, batch_data, update_weights=False):

        images, masks, noduletype_targets, malignancy_targets = (
            batch_data["image"].to(self.device),
            batch_data["mask"].to(self.device),
            batch_data["noduletype_target"].to(self.device),
            batch_data["malignancy_target"].to(self.device),
        )

        targets, losses = {}, {}

        if update_weights:
            self.optimizer.zero_grad()

        outputs = self.model(images)  # do the forward pass

        seg_loss, type_loss, malig_loss, overall_loss = self.model.loss(masks, noduletype_targets, malignancy_targets, outputs)

        losses["segmentation"] = seg_loss.item()
        outputs["segmentation"] = outputs["segmentation"]
        targets["segmentation"] = masks.data.cpu().numpy()
        
        losses["nodule-type"] = type_loss.item()
        outputs["nodule-type"] = (outputs["nodule-type"].data.cpu().numpy().reshape(-1, 4))
        targets["nodule-type"] = noduletype_targets.data.cpu().numpy().reshape(-1)
        
        losses["malignancy"] = malig_loss.item()
        outputs["malignancy"] = outputs["malignancy"].data.cpu().numpy().reshape(-1)
        targets["malignancy"] = malignancy_targets.data.cpu().numpy().reshape(-1)

        losses["total"] = overall_loss.item()

        if update_weights:
            overall_loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

        return outputs, targets, losses

    def train(self, model):

        self._initialize_data_loaders()
        self._initialize_model(model)

        save_dir = self.workspace / "results" / self.exp_id / f"fold{self.fold}"
        save_dir.mkdir(exist_ok=True, parents=True)

        epoch_metrics = {
            "training": [],
            "validation": [],
        }

        best_metric = 0
        best_epoch = 0
        print("Start of training model: ",self.exp_id)

        for epoch in range(self.max_epochs):

            for mode in ["training", "validation"]:

                print("\n\n")
                logging.info(
                    f" {mode}, epoch: {epoch + 1} / {self.max_epochs}, with fold: {self.fold}"
                )

                if mode == "training":
                    self.model.train()
                    data = self.train_loader
                else:
                    self.model.eval()
                    data = self.valid_loader

                metrics = {task: {"loss": [],}
                    for task in self.tasks
                }
                metrics["cumulative"] = {"loss": []}

                predictions = {task: [] for task in self.tasks}
                labels = {task: [] for task in self.tasks}

                for batch_data in tqdm(data):

                    if mode == "training":

                        outputs, targets, losses = self.forward(
                            batch_data,
                            update_weights=True,
                        )

                    else:

                        with torch.no_grad():
                            outputs, targets, losses = self.forward(
                                batch_data,
                                update_weights=False,
                            )
                            

                    for task in self.tasks:
                        metrics[task]["loss"].append(losses[task])

                    metrics["cumulative"]["loss"].append(losses["total"])

                    for task in self.tasks:
                        predictions[task].extend(outputs[task])
                        labels[task].extend(targets[task])

                for task in self.tasks:
                    loss = np.mean(metrics[task]["loss"])
                    metrics[task]["loss"] = loss  # aggregate the loss

                metrics["cumulative"]["loss"] = np.mean(metrics["cumulative"]["loss"])

                x = predictions["malignancy"]
                y = labels["malignancy"]
                metrics["malignancy"]["auc"] = skl_metrics.roc_auc_score(y, x)

                x = [p.argmax() for p in predictions["nodule-type"]]
                y = labels["nodule-type"]
                metrics["nodule-type"]["balanced_accuracy"] = skl_metrics.balanced_accuracy_score(y, x)

                dice = 1 - np.mean(metrics["segmentation"]["loss"])
                metrics["segmentation"]["dice"] = dice

                epoch_metrics[mode].append(metrics)

                if mode == "validation":

                    if self.best_metric_fn(metrics) > best_metric:
                    
                        print("\n===== Saving best model! =====\n")
                        best_metric = self.best_metric_fn(metrics)
                        best_epoch = epoch
                        torch.save(
                            self.model.state_dict(),
                            save_dir / "best_model.pth",
                        )
                        np.save(save_dir / "best_metrics.npy", metrics)
                        np.save(save_dir / "predictions.npy", predictions)
                        np.save(save_dir / "labels.npy", labels)

                    else:

                        print(f"Model has not improved since epoch {best_epoch + 1}")

                metrics = pandas.DataFrame(metrics).round(3)
                metrics.replace(np.nan, "", inplace=True)
                print(metrics.to_markdown(tablefmt="grid"))

            np.save(save_dir / "metrics.npy", epoch_metrics)
        # torch.save(self.model.state_dict(), save_dir / "last_model.pth")

if __name__ == "__main__":
    workspace = Path(project_dir)

    def best_metric_fn(metrics):
        return 0.5 * metrics["malignancy"]["auc"] + 0.25 * metrics["nodule-type"]["balanced_accuracy"] + 0.25 * metrics["segmentation"]["dice"]
        # return metrics["malignancy"]["auc"]  # 🥚 Easter egg


    
    
    for i in range(1):
        model = probeersel.MultiTaskNetwork(n_input_channels=1, n_filters=64)
        nodule_analyzer = NoduleAnalyzer(workspace=workspace, 
                                        best_metric_fn=best_metric_fn, 
                                        experiment_id="8_multitask_model", 
                                        batch_size=16, 
                                        fold=i, 
                                        max_epochs=400)
        nodule_analyzer.train(model)