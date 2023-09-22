'''
LAST UPDATE: 2023.09.20
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 
Training script for SRCNN

'''
import torch
import numpy as np
from data_loader import data_loaders
import os
from torch.utils.data import DataLoader
import cv2
from model.model import SRCNN
from model.loss import *
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # Summary writer for Tensorboard
    writer = SummaryWriter()
    # Use Cuda if available, todo-> Make arguments parser
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ### Number of Epochs
    num_epochs = 20
    ### Number of Batch Size    
    batch_size = 2
    ### Learning Rate
    learning_rate = 0.001
    ### Data loader for div2k, takes scale as parameter for bicubic interpolation
    data = data_loaders.Div2kDataLoader('train.csv', os.getcwd(), scale=4)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    test = data_loaders.Div2kDataLoader('valid.csv', os.getcwd(), scale=4)
    testloader = DataLoader(test)
    ### Model of SRCNN
    model = SRCNN()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # model.load_state_dict(torch.load('checkpoints/model_weight_rgb.pth'))
    ### For MSE LOSS
    criterion = torch.nn.MSELoss()
    ### Uncomment For Perceptual Loss
    # criterion = PerceptualLoss()

    ### Adam optimizer defined here
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_total_steps = len(dataloader)
    loss_counter = 0
    ### Start training over n epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader):
            lr_image = data['lr_image'].to(device)
            hr_image = data['hr_image'].to(device)
            output = model(lr_image)
            loss = criterion(output, hr_image)
            loss_counter+=1
            writer.add_scalar("current_loss", loss.item(), loss_counter)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 4==0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        ### For now we are saving every weight to see if there was overfitting.
        torch.save(model.state_dict(), f'checkpoints/model_weight_{epoch}.pth')
        epoch_loss /= n_total_steps
        writer.add_scalar("Loss", epoch_loss, epoch)
    print("[+] Training Finished!")
    torch.save(model.state_dict(), 'checkpoints/model_weight_rgb.pth')
