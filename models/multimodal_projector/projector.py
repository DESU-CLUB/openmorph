import torch.nn as nn
import torch.nn.functional as F


class VisionProjector(nn.Module):
    def __init__(self):
        self.ffn1 = nn.Linear(128, 128)
        self.act1 = nn.GELU()
        self.ffn2 = nn.Linear(128, 128)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act1(x)
        x = self.ffn2(x)
        x = self.act2(x)
        return x
