
import torch
from torch import nn
from torchvision import models

class VGG16Siamese(nn.Module):


    def __init__(self,embdeddingSize=256):
        super().__init__()

        model=models.vgg16(models.VGG16_Weights.DEFAULT)
        model.classifier[6]=nn.Identity()
        self.VGG=model

        self.featureEncode=nn.Sequential(
            self.VGG,
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.Linear(2048,embdeddingSize)
        )

    def forward(self,anchor,positive,negative):
        outputA=self.featureEncode(anchor)
        outputP=self.featureEncode(positive)
        outputN=self.featureEncode(negative)

        return nn.functional.normalize(outputA,p=2,dim=1),nn.functional.normalize(outputP,p=2,dim=1),nn.functional.normalize(outputN,p=2,dim=1)