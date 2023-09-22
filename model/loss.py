'''
LAST UPDATE: 2023.09.21
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 

Perceptual Loss that takes pretrained VGG19 model, and we extract specific layer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

class PerceptualLoss(nn.Module):
    '''
    '''
    def __init__(self):
        super().__init__()
        ### Dictionary for required layer
        self.required_layers = {
            'features.8':'relu2_2',
        }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ## Get VGG19 pretrained model
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', weights='VGG19_Weights.DEFAULT')
        ## get output from specific layer
        self.model2 = create_feature_extractor(self.model, return_nodes=self.required_layers)
        self.model2.requires_grad_ = False
        self.model2 = nn.DataParallel(self.model2, device_ids=[0, 1])
        self.model2.to(device)
        ## Hyperparameters for perceptual loss
        self.lambda_0 = [1]
        self.lambda_1 = 0.1
        self.lambda_2 = 1 - self.lambda_1

    '''
    Whenever object is invoked this object is called, and loss is returned based on the 
    hyperparameters defined.
    '''
    def __call__(self, output, target):
        output = output.view((output.shape[0], output.shape[3], output.shape[1], output.shape[2]))
        target = target.view((target.shape[0], target.shape[3], target.shape[1], target.shape[2]))
        feature_mse_list = []
        feature_mse = 0


        # Calculate feature loss
        feature_mse_list = [
            F.mse_loss(self.model2(output)[layer], self.model2(target)[layer])
            for layer in self.required_layers.values()
        ]

        feature_mse = sum(w * loss for w, loss in zip(self.lambda_0, feature_mse_list))

        mse = F.mse_loss(input=output, target=target)
        torch.cuda.empty_cache
        return self.lambda_1 * feature_mse + self.lambda_2 * mse