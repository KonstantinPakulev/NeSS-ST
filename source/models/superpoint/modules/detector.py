import torch.nn as nn

from source.utils.model_utils import depth_to_space


class SuperPointDetector(nn.Module):

    def __init__(self):
        super().__init__()
        c4, c5 = 128, 256
        self.relu = nn.ReLU(inplace=True)

        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Retrieve probabilities
        probabilities = semi.softmax(dim=1)
        # Remove dustbin
        probabilities = probabilities[:, :-1, :, :]
        # Reshape to get full resolution score map
        score = depth_to_space(probabilities, 8)

        return score