import torch.autograd as autograd
import torch.nn as nn

class Conv_QNet(nn.Module):
    def __init__(self, in_features, in_channels, num_actions, width_multiplicator=1):
        super().__init__()

        self.in_features = in_features
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv_out_size = width_multiplicator * 16

        # conv layers
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.conv_out_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.conv_out_size, self.conv_out_size, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.size_linear_unit(), 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def size_linear_unit(self):
        return (
            self.features(autograd.torch.zeros(*self.in_features)).view(1, -1).size(1)
        )

    def forward(self, x):
        x = x.float()
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x