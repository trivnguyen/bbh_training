
import torch
import torch.nn as nn

# Define convienient NN block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=padding),
            nn.MaxPool1d(4, 4),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        return self.fc(x)


# Define Classifier
class Classifier(nn.Module):
    def __init__(self, input_shape, corr_dim=0, padding=1):
        super().__init__()

        # classifier attributes
        self.name = 'CNN-LARGE'

        # nn architecture: CONV + FC -> FC
        # 1D CONV layer for time series data
        self.featurizer = nn.Sequential(
            ConvBlock(input_shape[0], 64, padding=padding),
            ConvBlock(64, 128, padding=padding),
            ConvBlock(128, 256, padding=padding),
            ConvBlock(256, 512, padding=padding)

        )
        # FC layer for Pearson correlation series
        fc_dim = self.get_flattened_size(input_shape)
        if corr_dim > 0:
            self.corr_featurizer = nn.Sequential(
                FCBlock(corr_dim, 32),
                FCBlock(32, 32),
            )
            fc_dim += 32

        # FC classifier combines features output
        self.classifier = nn.Sequential(
            FCBlock(fc_dim, 128),
            FCBlock(128, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x, x_corr=None):
        x = self.featurizer(x)
        x = x.view(x.size(0), -1)
        if x_corr is not None:
            x_corr = self.corr_featurizer(x_corr)
            x = torch.cat([x, x_corr], axis=1)
        x = self.classifier(x)
        return x

    def get_flattened_size(self, input_shape):
        x = torch.rand(1, *input_shape)
        out = self.featurizer(x)
        out_dim = 1
        for i in range(len(out.shape)):
            out_dim *= out.shape[i]
        return out_dim
