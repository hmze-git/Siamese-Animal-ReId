import torch
from torch.utils.data import Dataset,DataLoader

from torchvision.transforms import v2
import os
import random
import numpy as np
from PIL import Image
class SiameseDataaset(Dataset):


    def __init__(self,rootDatasetDir,isTraining=True):
        super().__init__()
        self.rootDatasetDir=rootDatasetDir
        self.imageFolds=os.listdir(rootDatasetDir)

        self.transformTrain=v2.Compose([
            v2.Resize((224,224)),
            v2.ToTensor(),
            v2.ToDtype(torch.float32,scale=True)]
        )
        #self.transformValid=
        self.training=isTraining



    def __getitem__(self, index):

        #folder with the anchor and the positive 
        #pick the amchor and positive
        #test.py scri   pt has confirmed no sub folders have less than 2 images so anchor and postive can work with that 
        #assumption
        anchorPosFold=self.imageFolds[index]
        anchorDir=os.path.join(self.rootDatasetDir,anchorPosFold)



        anchorName,postiveName=random.sample(os.listdir(anchorDir),2)

        anchorPath=os.path.join(anchorDir,anchorName)
        positivePath=os.path.join(anchorDir,postiveName)

        #get the negative
        negativeIndex=random.choice([i for i in range(len(self.imageFolds)) if i!=index]) # build list of i's that are not index then pick one

        negativeDir=os.path.join(self.rootDatasetDir,self.imageFolds[negativeIndex])
        negativePath=os.path.join(negativeDir,random.choice(os.listdir(negativeDir)))


        anchorImg=np.array(Image.open(anchorPath))
        postiveImg=np.array(Image.open(positivePath))
        negativeImg=np.array(Image.open(negativePath))


        anchorTens=self.transformTrain(anchorImg)
        positiveTens=self.transformTrain(postiveImg)
        negativeTens=self.transformTrain(negativeImg)

        return anchorTens,positiveTens,negativeTens





    def __len__(self):
        return (len(self.imageFolds))