import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import cv2

class SRCNN(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)

    
    def forward(self, x):
        x = x.view((x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x.view((x.shape[0], x.shape[2], x.shape[3], x.shape[1]))