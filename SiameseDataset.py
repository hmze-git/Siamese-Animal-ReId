import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from torchvision.transforms import v2
import os
import random
import numpy as np
class SiameseDataaset(Dataset):


    def __init__(self,rootDatasetDir,isTraining=True):
        super().__init__()
        self.rootDatasetDir=rootDatasetDir
        self.imageFolds=os.listdir(rootDatasetDir)

        self.transformTrain=v2.Compose(
            v2.Resize((224,224)),
            v2.ToTensor()
        )
        #self.transformValid=
        self.training=isTraining



    def __getitem__(self, index):
        return super().__getitem__()

        #folder with the anchor and the positive 
        #pick the amchor and positive
        #test.py scri   pt has confirmed no sub folders have less than 2 images so anchor and postive can work with that 
        #assumption
        anchorPosFold=self.imageFolds[index]

        anchorFoldList=np.array(os.listdir(anchorPosFold))

        anchor,postive=np.random.choice(anchorFoldList,size=2,replace=False)









    def __len__(self):
        return (len(self.imageFolds))