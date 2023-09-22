import torch
import numpy as np
from data_loader import data_loaders
import os
from torch.utils.data import DataLoader
import cv2
from model.model import SRCNN
from model.loss import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = data_loaders.Div2kDataLoader('/valid.csv', os.getcwd(), scale=2)
    testloader = DataLoader(test)
    criterion = torch.nn.MSELoss()
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('checkpoints_mse/checkpoints/model_weight_40.pth'))
    model.eval()
    for i, data in enumerate(testloader):
        lr_image = data['lr_image'].to(device)
        hr_image = data['hr_image'].to(device)
        output = model(lr_image)
        loss = criterion(output, hr_image)
        print("LOSS: ", loss)
        print(output.min(), output.max())
        # print(hr_image.min(), hr_image.max())
        print(lr_image.min(), lr_image.max())
        output = output.view((output.shape[1], output.shape[2], 3))
        output = np.abs(output.to('cpu').detach().numpy())* 255
        output = output.astype(np.uint8)
        print(output.shape)
        test = hr_image.view((hr_image.shape[1], hr_image.shape[2], 3))
        test = np.abs(test.to('cpu').detach().numpy()) * 255
        test = test.astype(np.uint8)
        cv2.imwrite(f'output/pred_{i}.png', output)
        cv2.imwrite(f'output/out_{i}.png', test)