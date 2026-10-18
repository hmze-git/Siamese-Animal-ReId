import torch
from torch.utils.data import Dataset,DataLoader

from torchvision.transforms import v2
import os
import random
import numpy as np
from PIL import Image
class SingleIMGSet(Dataset):


    def __init__(self,rootDatasetDir,training=True):
        super().__init__()
        self.rootDatasetDir=rootDatasetDir
        self.imageFolds=os.listdir(rootDatasetDir)


        if training:
            self.transformTrain=v2.Compose([
                v2.Resize((224,224)),
                v2.ToTensor(),
                v2.ToDtype(torch.float32,scale=True),
                v2.RandAugment(3,3)]
            )

        else:
            self.transformTrain=v2.Compose([

                   v2.Resize((224,224)),
                        v2.ToTensor(),
                        v2.ToDtype(torch.float32,scale=True),
        ])
        self.images = []
        for label, folderName in enumerate(sorted(os.listdir(rootDatasetDir))):
            folderPath = os.path.join(rootDatasetDir, folderName)
            for fileName in os.listdir(folderPath):
                self.images.append((os.path.join(folderPath, fileName), label)) #create the labels for the dogs evaluated
 




    def __getitem__(self, index):

        path,label=self.images[index]
        img=Image.open(path)
        return self.transformTrain(img),label




    def __len__(self):
        return (len(self.images))