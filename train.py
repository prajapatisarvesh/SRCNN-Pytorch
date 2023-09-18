from data_loader import data_loaders
import os
from torch.utils.data import DataLoader
import cv2

if __name__ == '__main__':
    data = data_loaders.Div2kDataLoader('train.csv', os.getcwd())
    