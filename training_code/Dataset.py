from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import json

class PredictorData(Dataset):
    def __init__(self, root_dir, json_name):
        self.root_dir = root_dir
        self.json = None
        with open(root_dir + "/" + json_name, 'r') as f:
            self.json = json.load(f)
        self.feat_list = list(self.json.keys())
        
    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self,index):
        image_filename = self.feat_list[index]
        feature = torch.load(self.root_dir + "features/" + image_filename)
        label = torch.Tensor([1]) if self.json[image_filename] else torch.Tensor([0])
        
        return feature[0], label
