import torch
from data_loader import data_loaders
import os
from torch.utils.data import DataLoader
import cv2
from model.model import SRCNN
from model.loss import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 5
    batch_size = 2
    learning_rate = 0.001
    data = data_loaders.Div2kDataLoader('train.csv', os.getcwd(), scale=2)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    model = SRCNN().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(dataloader)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            lr_image = data['lr_image'].to(device)
            hr_image = data['hr_image'].to(device)
            output = model(lr_image)
            loss = criterion(output, hr_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i + 1) % 50:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
    print("[+] Training Finished!")