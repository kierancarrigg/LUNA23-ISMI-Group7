
import sys
import torch
import multitask_network
import dataloader
import numpy as np
import SimpleITK as sitk
from typing import Tuple
import pandas
import scipy.ndimage as snd
from pathlib import Path
from tqdm import tqdm


def keep_central_connected_component(
    prediction: sitk.Image,
    patch_size: Tuple = (128, 128, 64),
) -> sitk.Image:
    """Function to post-process the prediction to keep only the central connected component in a patch

    Args:
        prediction (sitk.Image): prediction file (should be binary)
        patch_size (np.array, optional): patch size (x, y, z) to ensure the center is computed appropriately. Defaults to np.array([96, 96, 96]).

    Returns:
        sitk.Image: post-processed binary file with only the central connected component
    """

    origin = prediction.GetOrigin()
    spacing = prediction.GetSpacing()
    direction = prediction.GetDirection()

    prediction = sitk.GetArrayFromImage(prediction)

    c, n = snd.label(prediction)
    centroids = np.array(
        [np.array(np.where(c == i)).mean(axis=1) for i in range(1, n + 1)]
    ).astype(int)

    patch_size = np.array(list(reversed(patch_size)))

    if len(centroids) > 0:
        dists = np.sqrt(((centroids - patch_size // 2) ** 2).sum(axis=1))
        keep_idx = np.argmin(dists)
        output = np.zeros(c.shape)
        output[c == (keep_idx + 1)] = 1
        prediction = output.astype(np.uint8)

    prediction = sitk.GetImageFromArray(prediction)
    prediction.SetSpacing(spacing)
    prediction.SetOrigin(origin)
    prediction.SetDirection(direction)
    return prediction


def perform_inference_on_test_set(workspace: Path):

    models = {}
    for i in range(5):
        model_name = f"multitask_model_{i}"
        model = multitask_network.MultiTaskNetwork(n_input_channels=1, n_filters=64, dropout=True).cuda()
        model.eval()
        ckpt = torch.load(workspace / "results" / "20230529_22_multitask_model"/ f"fold{i}" / "best_model.pth")
        model.load_state_dict(ckpt)
        models[model_name] = model

    test_set_path = Path(workspace / "data" / "test_set" / "images")
    save_path = workspace / "results" / "20230529_22_multitask_model" / "test_set_predictions_max"

    segmentation_save_path = save_path / "segmentations"
    segmentation_save_path.mkdir(exist_ok=True, parents=True)

    patch_size = np.array([64, 128, 128])
    size_mm = 50
    size_px = 64

    predictions = []

    for idx, image_path in enumerate(tqdm(list(test_set_path.glob("*.mha")))):

        # load and pre-process input image

        sitk_image = sitk.ReadImage(str(image_path))

        noduleid = image_path.stem

        image = sitk_image
        metad = {
            "origin": np.flip(image.GetOrigin()),
            "spacing": np.flip(image.GetSpacing()),
            "transform": np.array(np.flip(image.GetDirection())).reshape(3, 3),
            "shape": np.flip(image.GetSize()),
        }
        image = sitk.GetArrayFromImage(image)

        image = dataloader.extract_patch(
            CTData=image,
            coord=tuple(patch_size // 2),
            srcVoxelOrigin=(0, 0, 0),
            srcWorldMatrix=metad["transform"],
            srcVoxelSpacing=metad["spacing"],
            output_shape=(size_px, size_px, size_px),
            voxel_spacing=(
                size_mm / size_px,
                size_mm / size_px,
                size_mm / size_px,
            ),
            coord_space_world=False,
        )

        image = image.reshape(1, 1, size_px, size_px, size_px).astype(np.float32)
        image = dataloader.clip_and_scale(image)

        image = torch.from_numpy(image).cuda()
        
        outputs = []
        with torch.no_grad():
            for model_name, model in models.items():
                outputs.append(model(image))

        # for output in outputs:
        #     output = {k: output[k].data.cpu().numpy().squeeze() for k in output.keys()}

        segmentations = []
        noduletypes = []
        malignancies = []
        for output in outputs:
            segmentations.append(output["segmentation"].data.cpu().numpy().squeeze())
            noduletypes.append(output["nodule-type"].data.cpu().numpy().squeeze())
            malignancies.append(output["malignancy"].data.cpu().numpy().squeeze())

        # post-process segmentation

        # resample image to original spacing
        for i in range(len(segmentations)):
            segmentations[i] = snd.zoom(
                segmentations[i],
                (size_mm / size_px) / metad["spacing"],
                order=1,
            )

            # pad image
            diff = metad["shape"] - segmentations[i].shape
            pad_widths = [
                (np.round(a), np.round(b))
                for a, b in zip(
                    diff // 2.0 + 1,
                    diff - diff // 2.0 - 1,
                )
            ]
            pad_widths = np.array(pad_widths).astype(int)
            pad_widths = np.clip(pad_widths, 0, pad_widths.max())
            segmentations[i] = np.pad(
                segmentations[i],
                pad_width=pad_widths,
                mode="constant",
                constant_values=0,
            )

            # crop, if necessary
            if diff.min() < 0:

                shape = np.array(segmentations[i].shape)
                center = shape // 2

                segmentations[i] = segmentations[i][
                    center[0] - patch_size[0] // 2 : center[0] + patch_size[0] // 2,
                    center[1] - patch_size[1] // 2 : center[1] + patch_size[1] // 2,
                    center[2] - patch_size[2] // 2 : center[2] + patch_size[2] // 2,
                ]
            print(segmentations[i].shape)

        # apply threshold
        segmentations = np.array(segmentations)
        print("shape before mean", segmentations.shape)
        segmentation = np.mean(segmentations, axis=0)
        segmentation = (segmentation > 0.5).astype(np.uint8)
        print("shape after mean", segmentation.shape)
        # set metadata
        segmentation = sitk.GetImageFromArray(segmentation)
        segmentation.SetOrigin(np.flip(metad["origin"]))
        segmentation.SetSpacing(np.flip(metad["spacing"]))
        segmentation.SetDirection(np.flip(metad["transform"].reshape(-1)))

        # keep central connected component
        segmentation = keep_central_connected_component(segmentation)

        # write as simpleitk image
        sitk.WriteImage(
            segmentation,
            str(segmentation_save_path / f"{noduleid}.mha"),
            True,
        )

        # Ensemble mean over classification results
        noduletypes = np.array(noduletypes)
        malignancies = np.array(malignancies)

        nodule_type = np.mean(noduletypes, axis=0)
        malignancy = np.mean(malignancies, axis=0)

        # combine predictions from other task models
        prediction = {
            "noduleid": noduleid,
            "malignancy": malignancy,
            "noduletype": nodule_type.argmax(),
            "ggo_probability": nodule_type[0],
            "partsolid_probability": nodule_type[1],
            "solid_probability": nodule_type[2],
            "calcified_probability": nodule_type[3],
        }

        predictions.append(pandas.Series(prediction))

    predictions = pandas.DataFrame(predictions)
    predictions.to_csv(save_path / "predictions.csv", index=False)


if __name__ == "__main__":

    project_dir = sys.argv[1]
    workspace = Path(project_dir)
    perform_inference_on_test_set(workspace=workspace)
