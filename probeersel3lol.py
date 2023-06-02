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
        def __init__(self, n_input_channels, n_filters, dropout=None, pooling=True, prob = 0.5):
            super().__init__()

            layers = []
            if pooling:
                layers.append(nn.MaxPool3d(kernel_size=2))
            layers += conv3x3(n_input_channels, n_filters)
            if dropout:
                layers.append(nn.Dropout(p=prob))
            layers += conv3x3(n_filters, n_filters)
            self.pool_conv = nn.Sequential(*layers)

        def forward(self, incoming):
            return self.pool_conv(incoming)

    class SegmentationBlock(nn.Module):
        """
        Segmentation block which represents the decoder/expension block for the segmentation output
        """
        def __init__(self, n_input_channels, n_filters, dropout=None, prob = 0.5):
            super().__init__()

            self.upconv = nn.Sequential(
                nn.ConvTranspose3d(
                    n_input_channels, n_filters, kernel_size=2, stride=2, bias=False
                ),
                nn.BatchNorm3d(n_filters, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
            )

            layers = conv3x3(n_input_channels, n_filters)
            if dropout:
                layers.append(nn.Dropout(p=prob))
            layers += conv3x3(n_filters, n_filters)
            self.conv = nn.Sequential(*layers)

        def forward(self, incoming, skip_connection):
            y = self.upconv(incoming)
            y = torch.cat([y, skip_connection], dim=1)
            return self.conv(y)

    class ClassificationBlock(nn.Module):
        def __init__(self, n_input_channels, n_filters, dropout=None, prob = 0.5):
            super().__init__()
            """
            Classification weight sharing - let's take weight sharing a stop further!
            """
            layers = [nn.Linear(n_input_channels, n_filters)]
            if dropout:
                layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(n_filters, n_filters//2))
            if dropout:
                layers.append(nn.Dropout(p=prob))
            self.conv = nn.Sequential(*layers)

        def forward(self, incoming):
            y = self.conv(incoming)
            return y

    class NoduleTypeBlock(nn.Module):
        def __init__(self, n_filters):
            super().__init__()
            """
            Nodule type decoder that is a fully connected network with the output being the probability per possible class
            Currently: 2 layer network where you have input layer, hidden layer and output layer with output being probability per class
            """
            self.conv = nn.Linear(n_filters//2, 4)

        def forward(self, incoming):
            y = self.conv(incoming)
            # return F.softmax(y)
            return y

    class MalignancyBlock(nn.Module):
        def __init__(self, n_filters):
            super().__init__()
            """
            Malignancy block that is a fully connected network with the output being the label 0 or 1
            Currently: Fully connected 2 layers where you have input layer, hidden layer and output layer
            """
            self.conv = nn.Linear(n_filters//2, 1)

        def forward(self, incoming):
            y = self.conv(incoming)
            return F.sigmoid(y)

    def __init__(self, n_input_channels, n_filters, dropout=None, sigmoid=True) -> None:
        super(MultiTaskNetwork, self).__init__()
        """
        Initiate Encoder and Decoder with appropiate input and output dimensions.

        pas deze doc aan
        """
        self.contraction = nn.ModuleList()
        filters = n_filters
        for i in range(1, 5):
            if i == 1:
                n_in = n_input_channels
            self.contraction.append(
                self.ContractionBlock(
                    n_in, filters, dropout=dropout if i > 1 else None, pooling=i > 1
                )
            )
            if i<4:
                n_in = filters
                filters = filters * 2

        self.expansion = nn.ModuleList()
        f = filters
        for i in range(1, 4):
            f1 = f // 2
            self.expansion.append(self.SegmentationBlock(f, f1, dropout=dropout))
            f = f1

        output_layer = nn.Conv3d(
            in_channels=f1,
            out_channels=1,
            kernel_size=1,
        )
        if sigmoid:
            self.segmentation = nn.Sequential(output_layer, nn.Sigmoid())
        else:
            self.segmentation = output_layer

        self.classification = self.ClassificationBlock(64*64*64,128,dropout) # image/patch size x patch size x n_filters
        self.nodule_type = self.NoduleTypeBlock(128) #inspiratie voor n_filters=128 uit COVID-19 multitask model
        self.malignant = self.MalignancyBlock(128)

    def forward(self, incoming): 
        """
        Hupsakee
        """
        # Pass incoming through contraction path
        latent = incoming
        cf = []
        i = 0
        for contract in self.contraction:
            latent = contract(latent)
            cf.append(latent)
            i +=1

        # Three output paths: segmentation, nodule type classification, malignancy classification

        # Pass features through expansion path
        seg = latent
        for expand, features in zip(self.expansion, reversed(cf[:-1])):
            seg = expand(seg, features)

        # Collect final output
        segmentation = self.segmentation(seg)

        # Flatten latent features for fully connected layers
        latent = latent.view(latent.size(0), -1)

        intermediate = self.classification(latent)

        # Pass features through nodule type fully connected layers
        noduletype = self.nodule_type(intermediate)

        # Pass features through malignancy fully connected layers
        malignancy = self.malignant(intermediate)

        return {'segmentation': segmentation, 'nodule-type': noduletype, 'malignancy': malignancy}

    def loss(self, original_masks, noduletype_labels, malignancy_labels, result):
        """
        Combination of the losses that resembles the balance between the importance of the separate tasks in the overall score.
        The loss per task is taken from the baseline model.
        """
        result_segmentation = result['segmentation']
        result_noduletype = result['nodule-type']
        result_malignancy = result['malignancy']
        seg_loss = dice_loss(result_segmentation, original_masks)
        type_loss = F.cross_entropy(result_noduletype, noduletype_labels.squeeze().long(), label_smoothing=0.3)
        malig_loss = F.binary_cross_entropy(result_malignancy, malignancy_labels)
        overall_loss = malig_loss + type_loss + 2*seg_loss
        return seg_loss, type_loss, malig_loss, overall_loss