import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, dropout=0.2, num_classes=7):
        super(Net, self).__init__()
        # Input Block
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, 2),  # output_size = 24
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(2, 2),  # output_size = 12
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(1024),

            nn.MaxPool2d(2, 2),  # output_size = 6
            nn.Dropout(dropout)
        )

        self.fcn = nn.Sequential(
            nn.Linear(in_features=1024 * 6 * 6, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),

            nn.Linear(in_features=16, out_features=num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcn(x)
        return F.log_softmax(x, dim=-1)
