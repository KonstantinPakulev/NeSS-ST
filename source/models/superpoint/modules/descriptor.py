import torch.nn as nn

from torch.nn.functional import normalize

from source.utils.model_utils import depth_to_space


class SuperPointDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        c4, c5, d1 = 128, 256, 256
        self.relu = nn.ReLU(inplace=True)

        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = normalize(desc)

        return desc