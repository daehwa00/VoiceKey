import torch
import torch.nn as nn


class Conv1DNet(nn.Module):
    def __init__(self, stride: int = 3):
        super(Conv1DNet, self).__init__()
        self.out_features = 64
        self.conv1 = nn.Conv1d(
            1, self.out_features // 4, kernel_size=25, stride=1, padding=12, dilation=1
        )
        self.conv2 = nn.Conv1d(
            self.out_features // 4,
            self.out_features // 2,
            kernel_size=25,
            stride=1,
            padding=24,
            dilation=2,
        )
        self.conv3 = nn.Conv1d(
            self.out_features // 2,
            self.out_features,
            kernel_size=25,
            stride=1,
            padding=36,
            dilation=3,
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Conv1d(self.out_features // 4, self.out_features, kernel_size=1)
        self.fc2 = nn.Conv1d(self.out_features // 2, self.out_features, kernel_size=1)
        self.out_features = 64

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1_ = self.fc1(out1)

        out2 = self.conv2(out1)
        out2 = self.relu(out2)
        out2_ = self.fc2(out2)

        out3 = self.conv3(out2)
        out3 = self.relu(out3)

        out = torch.cat([out1_, out2_, out3], dim=2)
        out = out[:, :, ::3]  # Downsample to time_seq / 3 * 3
        return out
