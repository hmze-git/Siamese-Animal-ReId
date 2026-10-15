import torch
import torchvision

from SiameseDataset import SiameseDataaset
from SiameseModel import VGG16Siamese
from torch.utils.data import DataLoader,Dataset
from torch.nn import TripletMarginLoss

#---------------------------------HYPERPRAMETERS------------------------------------

LR=1e-4
BATCH=16
EPOCHS=50

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#---------------------------------Loss------------------------------------
trainLossFn=TripletMarginLoss(margin=1,p=2)
validLossFN=TripletMarginLoss(margin=1,p=2)


#------------------------------End-----------------------------------------------------------------



#-----------------------------DATASET STUFF-----------------------------------------------

simTrainSet=SiameseDataaset(r'C:\Users\Hamzah\Desktop\HYP\Dataset\Big Data\DogFaceNet\train',True)
simValSet=SiameseDataaset(r'C:\Users\Hamzah\Desktop\HYP\Dataset\Big Data\DogFaceNet\validation',False)


trainingLoader=DataLoader(simTrainSet,batch_size=BATCH,shuffle=True)
validationLoader=DataLoader(simValSet,batch_size=BATCH,shuffle=True)



#------------------------------End-----------------------------------------------------------------



#---------------------------------Model------------------------------------------------------------

siameseModel=VGG16Siamese(256)




def train()
