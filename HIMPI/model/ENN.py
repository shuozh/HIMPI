import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# Define the main model
class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.sub = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.cont = nn.Sequential(
            nn.ReLU(),
            nn.Identity()
        )
        
    def forward(self, x):
        sub_out = self.sub(x)
        return self.cont(sub_out) + x

class ECNN(nn.Module):
    def __init__(self):
        super(ECNN, self).__init__()
        self.mModel = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            *[SubNet() for _ in range(5)],
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mModel(x)
class ICNN(nn.Module):
    def __init__(self):
        super(ICNN, self).__init__()
        self.mModel = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            *[SubNet() for _ in range(5)],
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.mModel(x)