from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch


class CombineDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        for d in datasets:
            assert len(d) == len(datasets[0])

    def __getitem__(self, idx):
        return tuple([d[idx] for d in self.datasets])

    def __len__(self):
        return len(self.datasets[0])


class ModelOutputDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, model_info):
        self.model_outputs = torch.load(model_info)['outputs'].cpu()

    def __getitem__(self, idx):
        return self.model_outputs[idx]

    def __len__(self):
        return len(self.model_outputs)


class NFLikelihoodDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, model_info):
        self.nf_likelihoods = torch.load(model_info)['likelihoods'].cpu()

    def __getitem__(self, idx):
        return self.nf_likelihoods[idx]

    def __len__(self):
        return len(self.nf_likelihoods)


class ImageListDataset(Dataset):
    def __init__(self, img_list, transform, folder='./'):
        self.folder = folder
        self.transform = transform
        self.path_list = [line.split(' ')[0] for line in open(img_list)]
        self.label_list = [int(line.split(' ')[1].strip('\n')) for line in open(img_list)]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.folder, self.path_list[idx])).convert('RGB'))
        return image, self.label_list[idx]

    @property
    def class_num(self):
        return np.unique(self.label_list).shape[0]