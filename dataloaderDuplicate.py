"""
PyTorch port of DeepR sampler from Setio16
Based on https://github.com/DIAGNijmegen/bodyct-multiview-nodule-detection/blob/30-retrain-setio16/dataloader/luna16.py
"""
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy.linalg as npl
import scipy.ndimage as ndi
import SimpleITK as sitk
import logging
import sys
import torchvision.transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

project_dir = sys.argv[1]


NODULETYPE_MAPPING = {
    "GroundGlassOpacity": 0,
    "SemiSolid": 1,
    "Solid": 2,
    "Calcified": 3,
}


def _calculateAllPermutations(itemList):
    if len(itemList) == 1:
        return [[i] for i in itemList[0]]
    else:
        sub_permutations = _calculateAllPermutations(itemList[1:])
        return [[i] + p for i in itemList[0] for p in sub_permutations]


def worker_init_fn(worker_id):
    """
    A worker initialization method for seeding the numpy random
    state using different random seeds for all epochs and workers
    """
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32)
    np.random.seed(seed=seed)


def volumeTransform(
    image,
    voxel_spacing,
    transform_matrix,
    center=None,
    output_shape=None,
    output_voxel_spacing=None,
    **argv,
):
    """
    Parameters
    ----------
      image : a numpy.ndarray
          The image that should be transformed

      voxel_spacing : a vector
          This vector describes the voxel spacing between individual pixels. Can
          be filled with (1,) * image.ndim if unknown.

      transform_matrix : a Nd x Nd matrix where Nd is the number of image dimensions
          This matrix governs how the output image will be oriented. The x-axis will be
          oriented along the last row vector of the transform_matrix, the y-Axis along
          the second-to-last row vector etc. (Note that numpy uses a matrix ordering
          of axes to index image axes). The matrix must be square and of the same
          order as the dimensions of the input image.

          Typically, this matrix is the transposed mapping matrix that maps coordinates
          from the projected image to the original coordinate space.

      center : vector (default: None)
          The center point around which the transform_matrix pivots to extract the
          projected image. If None, this defaults to the center point of the
          input image.

      output_shape : a list of integers (default None)
          The shape of the image projection. This can be used to limit the number
          of pixels that are extracted from the orignal image. Note that the number
          of dimensions must be equal to the number of dimensions of the
          input image. If None, this defaults to dimenions needed to enclose the
          whole inpput image given the transform_matrix, center, voxelSPacings,
          and the output_shape.

      output_voxel_spacing : a vector (default: None)
          The interleave at which points should be extracted from the original image.
          None, lets the function default to a (1,) * output_shape.ndim value.

      **argv : extra arguments
          These extra arguments are passed directly to scipy.ndimage.affine_transform
          to allow to modify its behavior. See that function for an overview of optional
          paramters (other than offset and output_shape which are used by this function
          already).
    """
    if "offset" in argv:
        raise ValueError(
            "Cannot supply 'offset' to scipy.ndimage.affine_transform - already used by this function"
        )
    if "output_shape" in argv:
        raise ValueError(
            "Cannot supply 'output_shape' to scipy.ndimage.affine_transform - already used by this function"
        )

    if image.ndim != len(voxel_spacing):
        raise ValueError("Voxel spacing must have the same dimensions")

    if center is None:
        voxelCenter = (np.array(image.shape) - 1) / 2.0
    else:
        if len(center) != image.ndim:
            raise ValueError(
                "center point has not the same dimensionality as the image"
            )

        # Transform center to voxel coordinates
        voxelCenter = np.asarray(center) / voxel_spacing

    transform_matrix = np.asarray(transform_matrix)
    if output_voxel_spacing is None:
        if output_shape is None:
            output_voxel_spacing = np.ones(transform_matrix.shape[0])
        else:
            output_voxel_spacing = np.ones(len(output_shape))
    else:
        output_voxel_spacing = np.array(output_voxel_spacing)

    if transform_matrix.shape[1] != image.ndim:
        raise ValueError(
            "transform_matrix does not have the correct number of columns (does not match image dimensionality)"
        )
    if transform_matrix.shape[0] != image.ndim:
        raise ValueError(
            "Only allowing square transform matrices here, even though this is unneccessary. However, one will need an algorithm here to create full rank-square matrices. 'QR decomposition with Column Pivoting' would probably be a solution, but the author currently does not know what exactly this is, nor how to do this..."
        )
    #  print (transform_matrix, transform_matrix, np.zeros((transform_matrix.shape[1], image.ndim - transform_matrix.shape[0])))
    #  transform_matrix = np.hstack((transform_matrix, np.zeros((transform_matrix.shape[1], image.ndim - transform_matrix.shape[0]))))

    # Normalize the transform matrix
    transform_matrix = np.array(transform_matrix)
    transform_matrix = (
        transform_matrix.T
        / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))
    ).T
    transform_matrix = np.linalg.inv(
        transform_matrix.T
    )  # Important normalization for shearing matrices!!

    # The forwardMatrix transforms coordinates from input image space into result image space
    forward_matrix = np.dot(
        np.dot(np.diag(1.0 / output_voxel_spacing), transform_matrix),
        np.diag(voxel_spacing),
    )

    if output_shape is None:
        # No output dimensions are specified
        # Therefore we calculate the region that will span the whole image
        # considering the transform matrix and voxel spacing.
        image_axes = [[0 - o, x - 1 - o] for o, x in zip(voxelCenter, image.shape)]
        image_corners = _calculateAllPermutations(image_axes)

        transformed_image_corners = map(
            lambda x: np.dot(forward_matrix, x), image_corners
        )
        output_shape = [
            1 + int(np.ceil(2 * max(abs(x_max), abs(x_min))))
            for x_min, x_max in zip(
                np.amin(transformed_image_corners, axis=0),
                np.amax(transformed_image_corners, axis=0),
            )
        ]
    else:
        # Check output_shape
        if len(output_shape) != transform_matrix.shape[1]:
            raise ValueError(
                "output dimensions must match dimensionality of the transform matrix"
            )
    output_shape = np.array(output_shape)

    # Calculate the backwards matrix which will be used for the slice extraction
    backwards_matrix = npl.inv(forward_matrix)
    target_image_offset = voxelCenter - backwards_matrix.dot((output_shape - 1) / 2.0)

    # print forward_matrix.dot(voxelCenter), voxelCenter, output_shape, target_image_offset, voxelCenter + backwards_matrix.dot((output_shape - 1) / 2.0)
    # print ndi.affine_transform(image, backwards_matrix, offset=target_image_offset, output_shape=output_shape, **argv)
    # def lookup(x): return ndi.affine_transform(image, np.eye(image.ndim), offset=np.array(x), output_shape=np.ones(image.ndim), order=0, cval=1137)
    # print target_image_offset + backwards_matrix.dot(np.array((0, 10))), lookup((5., 9.49)), lookup((5., 9.5)), lookup((5., 10.1)), image.shape
    # print "-------"

    return ndi.affine_transform(
        image,
        backwards_matrix,
        offset=target_image_offset,
        output_shape=output_shape,
        **argv,
    )


def clip_and_scale(npzarray, maxHU=400.0, minHU=-1000.0):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray


def rotateMatrixX(cosAngle, sinAngle):
    return np.asarray([[1, 0, 0], [0, cosAngle, -sinAngle], [0, sinAngle, cosAngle]])


def rotateMatrixY(cosAngle, sinAngle):
    return np.asarray([[cosAngle, 0, sinAngle], [0, 1, 0], [-sinAngle, 0, cosAngle]])


def rotateMatrixZ(cosAngle, sinAngle):
    return np.asarray([[cosAngle, -sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])


def getVoxelCoordinates(coordx, coordy, coordz, origin):
    interestPoint = (coordx, coordy, coordz)
    numpyInterestPoint = np.array(list(reversed(interestPoint)))
    stretchedVoxelSpaceCoordinate = np.absolute(numpyInterestPoint - origin)

    return stretchedVoxelSpaceCoordinate


class NoduleDataset(data.Dataset):
    """Nodule dataset"""

    def __init__(
        self,
        data_dir,
        dataset,
        translations=None,
        rotations=None,
        patch_size=(64, 128, 128),
        size_px=64,
        size_mm=50,
    ):

        self.logger = logging.getLogger(type(self).__name__)
        self.data_dir = Path(data_dir)
        self.rotations = rotations
        self.translations = translations
        self.patch_size = patch_size
        self.size_px = size_px
        self.size_mm = size_mm
        self.dataset = dataset

        self.cache = self.data_dir / "cache"
        self.cache.mkdir(exist_ok=True, parents=True)

    def _extract_patch(
        self,
        pd,
    ):

        image_path = self.cache / f"{pd.noduleid}_image.npy"
        label_path = self.cache / f"{pd.noduleid}_label.npy"
        metad_path = self.cache / f"{pd.noduleid}_metad.npy"

        if not image_path.is_file():

            image = sitk.ReadImage(str(self.data_dir / "images" / f"{pd.noduleid}.mha"))
            label = sitk.ReadImage(str(self.data_dir / "labels" / f"{pd.noduleid}.mha"))

            metad = {
                "origin": np.flip(image.GetOrigin()),
                "spacing": np.flip(image.GetSpacing()),
                "transform": np.array(np.flip(image.GetDirection())).reshape(3, 3),
                "shape": np.flip(image.GetSize()),
            }

            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)

            np.save(image_path, image)
            np.save(label_path, label)
            np.save(metad_path, metad)

        else:

            image = np.load(image_path)
            label = np.load(label_path)
            metad = np.load(metad_path, allow_pickle=True).item()

        translations = None
        if self.translations:
            radius = pd.diameter_mm / 2
            translations = radius if radius > 0 else None

        output_shape = (self.size_px, self.size_px, self.size_px)

        patch, mask = extract_patch(
            CTData=image,
            coord=tuple(np.array(self.patch_size) // 2),
            srcVoxelOrigin=(0, 0, 0),
            srcWorldMatrix=metad["transform"],
            srcVoxelSpacing=metad["spacing"],
            mask=label,
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            rotations=self.rotations,
            translations=translations,
            coord_space_world=False,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)
        mask = mask.astype(np.float32)

        # Enhance contrast
        # patch = enhance_contrast(patch)

        # clip and scale...
        patch = clip_and_scale(patch)

        return patch, mask, metad

    def __getitem__(self, idx):

        pd = self.dataset.iloc[idx]

        patch, mask, metad = self._extract_patch(pd)

        sample = {
            "image": torch.from_numpy(patch),
            "mask": torch.from_numpy(mask),
            "origin": torch.from_numpy(metad["origin"].copy()),
            "spacing": torch.from_numpy(metad["spacing"].copy()),
            "transform": torch.from_numpy(metad["transform"].copy()),
            "shape": torch.from_numpy(metad["shape"].copy()),
            "malignancy_target": torch.ones((1,)) * pd.malignancy,
            "noduletype_target": torch.ones((1,)) * NODULETYPE_MAPPING[pd.noduletype],
            "noduleid": pd.noduleid,
        }

        return sample

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str

    # def __merge__(self):
    #Hier misschien zelf een functie maken?



def get_data_loader(
    data_dir,
    dataset,
    sampler=None,
    workers=0,
    batch_size=16,
    patch_size=(64, 128, 128),
    size_px=64,
    size_mm=70,
    rotations=None,
    translations=None,
    pin_memory=True,
    train = False, 
):

    shuffle = False
    if sampler == None:
        shuffle = (True,)
    
    if train: 
        dataset_original = NoduleDataset(
            data_dir=data_dir,
            translations=False,
            dataset=dataset,
            rotations=None,
            patch_size=patch_size,
            size_mm=size_mm,
            size_px=size_px,
        )

        dataset_translation = NoduleDataset(
            data_dir=data_dir,
            translations=True,
            dataset=dataset,
            rotations=None,
            patch_size=patch_size,
            size_mm=size_mm,
            size_px=size_px,
        )


        dataset_rotation = NoduleDataset(
            data_dir=data_dir,
            translations=False,
            dataset=dataset,
            rotations=rotations,
            patch_size=patch_size,
            size_mm=size_mm,
            size_px=size_px,
        )

        dataset_trans_rot = NoduleDataset(
            data_dir=data_dir,
            translations=True,
            dataset=dataset,
            rotations=rotations,
            patch_size=patch_size,
            size_mm=size_mm,
            size_px=size_px,
        )

        concat_dataset = torch.utils.data.ConcatDataset((dataset_original, dataset_translation, dataset_rotation, dataset_trans_rot))

        data_loader = DataLoader(
            concat_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
        )


    if train == False:
        dataset = NoduleDataset(
            data_dir=data_dir,
            translations=False,
            dataset=dataset,
            rotations=None,
            patch_size=patch_size,
            size_mm=size_mm,
            size_px=size_px,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
        )

    return data_loader


def sample_random_coordinate_on_sphere(radius):
    # Generate three random numbers x,y,z using Gaussian distribution
    random_nums = np.random.normal(size=(3,))

    # You should handle what happens if x=y=z=0.
    if np.all(random_nums == 0):
        return np.zeros((3,))

    # Normalise numbers and multiply number by radius of sphere
    return random_nums / np.sqrt(np.sum(random_nums * random_nums)) * radius


def extract_patch(
    CTData,
    coord,
    srcVoxelOrigin,
    srcWorldMatrix,
    srcVoxelSpacing,
    mask=None,
    output_shape=(64, 64, 64),
    voxel_spacing=(50.0 / 64, 50.0 / 64, 50.0 / 64),
    rotations=None,
    translations=None,
    coord_space_world=False,
    transformMatrixAug=np.eye(3),
    offset=np.array([0, 0, 0]),
):

    transform_matrix = np.eye(3)

    # compute rotation matrices

    if rotations is not None:

        (zmin, zmax), (ymin, ymax), (xmin, xmax) = rotations

        # add random rotation
        angleX = np.multiply(np.pi / 180.0, np.random.randint(xmin, xmax, 1))[0]
        angleY = np.multiply(np.pi / 180.0, np.random.randint(ymin, ymax, 1))[0]
        angleZ = np.multiply(np.pi / 180.0, np.random.randint(zmin, zmax, 1))[0]

        transformMatrixAug = np.eye(3)
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixX(np.cos(angleX), np.sin(angleX))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixY(np.cos(angleY), np.sin(angleY))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixZ(np.cos(angleZ), np.sin(angleZ))
        )

    # multiply rotation matrices
    transform_matrix = np.dot(transform_matrix, transformMatrixAug)

    # compute random translation
    if translations is not None:
        # add random translation
        radius = np.random.random_sample() * translations
        offset = sample_random_coordinate_on_sphere(radius=radius)
        offset = offset * (1.0 / srcVoxelSpacing)

    # apply random translation
    coord = np.array(coord) + offset

    thisTransformMatrix = transform_matrix
    # Normalize transform matrix
    thisTransformMatrix = (
        thisTransformMatrix.T
        / np.sqrt(np.sum(thisTransformMatrix * thisTransformMatrix, axis=1))
    ).T

    invSrcMatrix = np.linalg.inv(srcWorldMatrix)

    # world coord sampling
    if coord_space_world:
        overrideCoord = invSrcMatrix.dot(coord - srcVoxelOrigin)
    else:
        # image coord sampling
        overrideCoord = coord * srcVoxelSpacing
    overrideMatrix = (invSrcMatrix.dot(thisTransformMatrix.T) * srcVoxelSpacing).T

    patch = volumeTransform(
        CTData,
        srcVoxelSpacing,
        overrideMatrix,
        center=coord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    )
    patch = np.expand_dims(patch, axis=0)

    if mask is not None:
        mask = volumeTransform(
            mask,
            srcVoxelSpacing,
            overrideMatrix,
            center=overrideCoord,
            output_shape=np.array(output_shape),
            output_voxel_spacing=np.array(voxel_spacing),
            order=0,
            prefilter=False,
        )
        mask = np.expand_dims(mask, axis=0)
        return patch, mask

    else:
        return patch


def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weigghts for dealing with class imbalance problem

    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))

    return np.array(weights)


def test(workspace: Path = Path("/code/bodyct-luna23-ismi-trainer/")):

    import pandas

    df = pandas.read_csv(workspace / "data/luna23-ismi-train-set.csv")
    noduletypes = [NODULETYPE_MAPPING[t] for t in df.noduletype]

    x = np.array(make_weights_for_balanced_classes(noduletypes))
    y = np.array(make_weights_for_balanced_classes(df.malignancy))

    weights = x * y

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights,
        len(df),
    )

    train_loader = get_data_loader(
        workspace / "data/train_set",
        df,
        sampler=sampler,
        workers=16,
        batch_size=16,
        rotations=[(-20, 20)] * 3,
        translations=True,
        size_mm=50,
        size_px=64,
        patch_size=(64, 128, 128),
        train = True, 
    )

    for batch_data in tqdm(train_loader):

        malignancy_targets = batch_data["malignancy_target"].numpy().squeeze()
        noduletype_targets = batch_data["noduletype_target"].numpy().squeeze()

        iets, c = np.unique(noduletype_targets, return_counts=True)

        print(malignancy_targets.sum())
        print(list(c))
        print(list(iets))
        print()

def enhance_contrast(patch):
    enhanced_patch = patch
    return enhanced_patch



if __name__ == "__main__":

    workspace = Path(project_dir)
    test(workspace)
