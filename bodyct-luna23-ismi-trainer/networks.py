import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(n_in, n_out, padding=1):
    return [
        nn.Conv3d(n_in, n_out, kernel_size=3, padding=padding, bias=False),
        nn.BatchNorm3d(n_out, affine=True, track_running_stats=False),
        nn.ReLU(inplace=True),
    ]


class ContractionBlock(nn.Module):
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


class ExpansionBlock(nn.Module):
    def __init__(self, n_input_channels, n_filters, dropout=None):
        super(ExpansionBlock, self).__init__()

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


class UNet(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_filters,
        n_output_channels=1,
        dropout=None,
        sigmoid=True,
    ):
        super().__init__()

        # Build contraction path
        self.contraction = nn.ModuleList()
        for i in range(1, 5):
            n_in = n_filters if i > 1 else n_input_channels
            self.contraction.append(
                ContractionBlock(
                    n_in, n_filters, dropout=dropout if i > 1 else None, pooling=i > 1
                )
            )

        # Build expansion path
        self.expansion = nn.ModuleList()
        for i in range(1, 4):
            self.expansion.append(ExpansionBlock(n_filters, n_filters, dropout=dropout))

        output_layer = nn.Conv3d(
            in_channels=n_filters,
            out_channels=n_output_channels,
            kernel_size=1,
        )
        if sigmoid:
            self.segmentation = nn.Sequential(output_layer, nn.Sigmoid())
        else:
            self.segmentation = output_layer

    def forward(self, image):
        y = image

        # Pass image through contraction path
        cf = []
        for contract in self.contraction:
            y = contract(y)
            cf.append(y)

        # Pass features through expansion path
        for expand, features in zip(self.expansion, reversed(cf[:-1])):
            y = expand(y, features)

        # Collect final output
        segmentation = self.segmentation(y)
        features = cf[-1]

        outputs = {"segmentation": segmentation, "features": cf[-1]}

        return outputs


class Flatten(nn.Module):
    def forward(self, y):
        return y.view(y.size(0), -1)


class CNN3D(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels=1,
        task="malignancy",
    ) -> None:
        super().__init__()

        assert task in ["noduletype", "malignancy"]

        self.task = task

        if task == "malignancy":
            activation = nn.Sigmoid()
        else:
            activation = nn.Softmax(dim=1)

        self.classification = nn.Sequential(
            *conv3x3(n_input_channels, 32, padding=0),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(32, 64, padding=0),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(64, 64, padding=0),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(64, 128, padding=0),
            nn.MaxPool3d(kernel_size=2),
            Flatten(),
            nn.Linear(1024, out_features=n_output_channels),
            activation,
        )

    def forward(self, x):
        outputs = self.classification(x)
        outputs = {self.task: outputs}
        return outputs
