import torch
import torch.nn as nn

import torchvision.transforms as transforms

class MultiCropToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

class MultiCropNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def forward(self, crops):
        return torch.stack([self.normalize(crop) for crop in crops])


