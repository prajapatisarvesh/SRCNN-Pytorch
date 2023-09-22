'''
LAST UPDATE: 2023.09.20
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import cv2

'''
Defining SRCNN model
'''
class SRCNN(BaseModel):
    def __init__(self):
        super().__init__()
        ### First conv2d layer, which takes in bicubic interpolated image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        ### Non linear mapping
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        ### Output for SRCNN
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)

    '''
    Forward function for tensors
    '''
    def forward(self, x):
        x = x.view((x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return torch.clip(x.view((x.shape[0], x.shape[2], x.shape[3], x.shape[1])), min=0.0, max=1.0)