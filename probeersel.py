import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(n_in, n_out, padding=1):
    return [
        nn.Conv3d(n_in, n_out, kernel_size=3, padding=padding, bias=False),
        nn.BatchNorm3d(n_out, affine=True, track_running_stats=False),
        nn.ReLU(inplace=True),
    ]

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


class MultiTaskNetwork(nn.Module):
    class ContractionBlock(nn.Module):
        """
        Encoder block which represents the shared feature space. 
        This could be substituted by the contraction block from a pre-trained U-net in the future.
        """
        def __init__(self, n_input_channels, n_filters, dropout=None, pooling=True):
            super().__init__()

            layers = []
            if pooling:
                layers.append(nn.MaxPool3d(kernel_size=2))
            layers += conv3x3(n_input_channels, n_filters)
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers += conv3x3(n_filters, n_filters)
            self.pool_conv = nn.Sequential(*layers)

        def forward(self, incoming):
            return self.pool_conv(incoming)

    class SegmentationBlock(nn.Module):
        """
        Segmentation block which represents the decoder/expension block for the segmentation output
        """
        def __init__(self, n_input_channels, n_filters, dropout=None):
            super().__init__()

            self.upconv = nn.Sequential(
                nn.ConvTranspose3d(
                    n_input_channels, n_filters, kernel_size=2, stride=2, bias=False
                ),
                nn.BatchNorm3d(n_filters, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
            )

            layers = conv3x3(n_filters * 2, n_filters)
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers += conv3x3(n_filters, n_filters)
            self.conv = nn.Sequential(*layers)

        def forward(self, incoming, skip_connection):
            y = self.upconv(incoming)
            y = torch.cat([y, skip_connection], dim=1)
            return self.conv(y)

    class NoduleTypeBlock(nn.Module):
        def __init__(self, n_input_channels, n_filters, n_classes=4, dropout=None):
            super().__init__()
            """
            Nodule type decoder that is a fully connected network with the output being the probability per possible class
            Currently: 2 layer network where you have input layer, hidden layer and output layer with output being probability per class
            """
            layers = [nn.Linear(n_input_channels, n_filters)]
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(n_filters, n_filters))
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(n_filters, n_classes))
            self.conv = nn.Sequential(*layers)

        def forward(self, incoming):
            y = self.conv(incoming)
            return F.softmax(y)

    class MalignancyBlock(nn.Module):
        def __init__(self, n_input_channels, n_filters, dropout=None):
            super().__init__()
            """
            Malignancy block that is a fully connected network with the output being the label 0 or 1
            Currently: Fully connected 2 layers where you have input layer, hidden layer and output layer
            """
            layers = [nn.Linear(n_input_channels, n_filters)]
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(n_filters, n_filters))
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(n_filters, 1))
            self.conv = nn.Sequential(*layers)

        def forward(self, incoming):
            y = self.conv(incoming)
            return F.sigmoid(y)

    def __init__(self, n_input_channels, n_filters, dropout) -> None:
        super(MultiTaskNetwork, self).__init__()
        """
        Initiate Encoder and Decoder with appropiate input and output dimensions.

        feature_dim: number of input / ouput units. For example the total number of pixels.
        latent_dim: number of latent variables.
        hidden_size: number of hidden units

        """
        self.contraction = self.ContractionBlock(n_input_channels, n_filters, dropout)
        self.segmentation = self.SegmentationBlock(n_input_channels, n_filters, dropout)
        self.nodule_type = self.NoduleTypeBlock(n_input_channels, n_filters, dropout)
        self.malignant = self.MalignancyBlock(n_input_channels, n_filters, dropout)

    def forward(self, incoming):
        # Moet nog omgebouwd worden want nu missen we de meerdere hoeveelheden contraction en 
        # segmentation blocks, en gebeurt er niks met skip connections.
        # Originele U-net code gaf ook de intermediate features bij output dus wellicht die alsnog gebruiken?
        latent = self.contraction(incoming)
        seg = self.segmentation(latent)
        noduletype = self.nodule_type(latent)
        malignancy = self.malignant(latent)
        return {'segmentation': seg, 'nodule-type': noduletype, 'malignancy': malignancy}

    def loss(self, original_masks, noduletype_labels, malignancy_labels, result):
        """
        Combination of the losses that resembles the balance between the importance of the separate tasks in the overall score.
        The loss per task is taken from the baseline model.
        """
        result_segmentation = result['segmentation']
        result_noduletype = result['nodule-type']
        result_malignancy = result['malignancy']
        seg_loss = dice_loss(result_segmentation, original_masks)
        type_loss = F.cross_entropy(result_noduletype, noduletype_labels)
        malig_loss = F.binary_cross_entropy(result_malignancy, malignancy_labels)
        overall_loss = 2 * malig_loss + type_loss + seg_loss
        return seg_loss, type_loss, malig_loss, overall_loss


# class UNet(nn.Module):
#     def __init__(
#         self,
#         n_input_channels,
#         n_filters,
#         n_output_channels=1,
#         dropout=None,
#         sigmoid=True,
#     ):
#         super().__init__()

#         # Build contraction path
#         self.contraction = nn.ModuleList()
#         for i in range(1, 5):
#             n_in = n_filters if i > 1 else n_input_channels
#             self.contraction.append(
#                 ContractionBlock(
#                     n_in, n_filters, dropout=dropout if i > 1 else None, pooling=i > 1
#                 )
#             )

#         # Build expansion path
#         self.expansion = nn.ModuleList()
#         for i in range(1, 4):
#             self.expansion.append(ExpansionBlock(n_filters, n_filters, dropout=dropout))

#         output_layer = nn.Conv3d(
#             in_channels=n_filters,
#             out_channels=n_output_channels,
#             kernel_size=1,
#         )
#         if sigmoid:
#             self.segmentation = nn.Sequential(output_layer, nn.Sigmoid())
#         else:
#             self.segmentation = output_layer

#     def forward(self, image):
#         y = image

#         # Pass image through contraction path
#         cf = []
#         for contract in self.contraction:
#             y = contract(y)
#             cf.append(y)

#         print('Intermediate y', y.shape)

#         # Pass features through expansion path
#         for expand, features in zip(self.expansion, reversed(cf[:-1])):
#             y = expand(y, features)

#         # Collect final output
#         segmentation = self.segmentation(y)
#         features = cf[-1]

#         outputs = {"segmentation": segmentation, "features": cf[-1]}

#         print('outputs', segmentation.shape)
#         print('features', features.shape)

#         return outputs


# class Flatten(nn.Module):
#     def forward(self, y):
#         return y.view(y.size(0), -1)


# class CNN3D(nn.Module):
#     def __init__(
#         self,
#         n_input_channels,
#         n_output_channels=1,
#         task="malignancy",
#     ) -> None:
#         super().__init__()

#         assert task in ["noduletype", "malignancy"]

#         self.task = task

#         if task == "malignancy":
#             activation = nn.Sigmoid()
#         else:
#             activation = nn.Softmax(dim=1)

#         self.classification = nn.Sequential(
#             *conv3x3(n_input_channels, 32, padding=0),
#             nn.MaxPool3d(kernel_size=2),
#             *conv3x3(32, 64, padding=0),
#             nn.MaxPool3d(kernel_size=2),
#             *conv3x3(64, 64, padding=0),
#             nn.MaxPool3d(kernel_size=2),
#             *conv3x3(64, 128, padding=0),
#             nn.MaxPool3d(kernel_size=2),
#             Flatten(),
#             nn.Linear(1024, out_features=n_output_channels),
#             activation,
#         )

#     def forward(self, x):
#         outputs = self.classification(x)
#         outputs = {self.task: outputs}
#         return outputs
