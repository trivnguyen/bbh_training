
import torch
import torch.nn as nn

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
    def __init__(self, in_channels):
        super().__init__()

        self.name = 'FC-CORR'
        self.fc = nn.Sequential(
            FCBlock(in_channels, 32),
            FCBlock(32, 32),
            FCBlock(32, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)
