import torch
import torchvision
import random
import matplotlib.pyplot as plt
from SiameseDataset import SiameseDataaset
from SingleImageDataset import SingleIMGSet
from SiameseModel import VGG16Siamese
from torch.utils.data import DataLoader,Dataset,Subset
from torch.nn import TripletMarginLoss
from torch.optim import Adam

#---------------------------------HYPERPRAMETERS------------------------------------

LRVGG=1e-5
LRREst=1e-3
BATCH=16
EPOCHS=2

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#---------------------------------Loss------------------------------------
trainLossFn=TripletMarginLoss(margin=0.7,p=2)
validLossFN=TripletMarginLoss(margin=0.7,p=2)



#------------------------------End-----------------------------------------------------------------



#-----------------------------DATASET STUFF-----------------------------------------------\


#NEED METHOD TO CREATE A SUBSET OF RANDOM DOG IMAGES THAT matches the size of val and test at 700ish 





simTrainSet=SiameseDataaset(r'C:\Users\Hamzah\Desktop\HYP\Dataset\Big Data\DogFaceNet\train',True)
simValSet=SiameseDataaset(r'C:\Users\Hamzah\Desktop\HYP\Dataset\Big Data\DogFaceNet\validation',False)

simMetricTrain=SingleIMGSet(r'C:\Users\Hamzah\Desktop\HYP\Dataset\Big Data\DogFaceNet\train')
simMetricVal=SingleIMGSet(r'C:\Users\Hamzah\Desktop\HYP\Dataset\Big Data\DogFaceNet\validation')

simMetricTrain=Subset(simMetricTrain,range(len(simMetricVal)))



traainMetload=DataLoader(simMetricTrain,batch_size=BATCH,shuffle=False)
valMetLoad=DataLoader(simMetricVal,batch_size=BATCH,shuffle=False)


trainingLoader=DataLoader(simTrainSet,batch_size=BATCH,shuffle=True)
validationLoader=DataLoader(simValSet,batch_size=BATCH,shuffle=True)



#------------------------------End-----------------------------------------------------------------



#---------------------------------Model------------------------------------------------------------

siameseModel=VGG16Siamese(256)

siameseModel=siameseModel.to(DEVICE)

BESTMODELPATH='best.tar'


discrim=[
     {
          'params':siameseModel.VGG.parameters(),'lr':LRVGG
     },
     {
           'params':siameseModel.featureEncode[1:].parameters(),'lr':LRREst
     }
]

optimiser=Adam(discrim)

def trainStep(model,epoch,lossFn,optimiser):
    model.train()
    loss=0.0
    for a,p,n in trainingLoader:
        optimiser.zero_grad()
        a=a.to(DEVICE)
        p=p.to(DEVICE)
        n=n.to(DEVICE)


        anchorFeat,positiveFeat,negativeFeat=model(a,p,n)

        l=lossFn(anchorFeat,positiveFeat,negativeFeat)

        l.backward()
        optimiser.step()

        loss+=l.item()
    return loss/len(trainingLoader)


def validStep(model,lossFn):
    model.eval()
    loss=0.0
    with torch.no_grad():
          for a,p,n in validationLoader:
                a=a.to(DEVICE)
                p=p.to(DEVICE)
                n=n.to(DEVICE)
        
        
                anchorFeat,positiveFeat,negativeFeat=model(a,p,n)
        
                l=lossFn(anchorFeat,positiveFeat,negativeFeat)
        
                loss+=l.item()
    return loss/len(validationLoader)

def metricsStep(model,loader):
    model.eval()
    allEmbeddings=[]
    allLabels=[]

    with torch.no_grad():
        for dogs,doglabels in loader:
              dogs=dogs.to(DEVICE)

              featre=model.featureEncode(dogs)
              features=torch.nn.functional.normalize(featre,p=2,dim=1)
              allEmbeddings.append(features.cpu())
              allLabels.append(doglabels)

    embdeedingsALL=torch.cat(allEmbeddings)
    labelsALL=torch.cat(allLabels)

    #compute distances between each vector giving distance between images
    distanceMat=torch.cdist(embdeedingsALL,embdeedingsALL) 

    rank1Sum=0.0
    rank5sum=0.0
    apSum=0.0
    numQueries=0.0

        #each lookuop for a dog is a query and we compare it to the other dogs
    for q in range(len(labelsALL)):

        dist=distanceMat[q].clone()


        #dont care about the first image
        dist[q]=float('inf')

        #Get everything but the last item 
        rankOrderIndices=dist.argsort()[:-1] 

        matches=(labelsALL[rankOrderIndices]==labelsALL[q]).float()


        numMatches=matches.sum().item() # how many matches are theere where it matches the dog being queried

        if numMatches==0:
             continue

        rank1Sum+=matches[0].item() # add 1 or 0 if its a match for dog queried

        # if  at least one image matches the current dog
        if matches[:5].sum().item()>0:
             rank5sum+=1

                #average precision: at every position where a match appears, take (matches so far / position), then average those
        positions=torch.arange(1,len(matches)+1).float()
        precisionAtPosition=matches.cumsum(0)/positions
        apSum+=(precisionAtPosition*matches).sum().item()/numMatches
 
        numQueries+=1


    return rank1Sum/numQueries,rank5sum/numQueries,apSum/numQueries

 
#draws ONE chart and saves it as its own image file
#the chart has a train line and a validation line so the two can be compared
def plotOne(trainArr,valArr,title,savePath):
    epochs=range(1,len(trainArr)+1)
 
    plt.figure(figsize=(8,5))
    plt.plot(epochs,trainArr,label='train')
    plt.plot(epochs,valArr,label='validation')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savePath)
    plt.close()   #frees the figure so memory doesn't grow every epoch
 
 
#takes the arrays collected during training and makes one separate image per metric
def plotMetrics(trainingLossArr,valLossArr,rank1TRArr,rank1VLArr,rank5TRARr,rank5VLARr,mAPTRARr,mAPVLARr):
    plotOne(trainingLossArr,valLossArr,'Loss','loss.png')
    plotOne(rank1TRArr,rank1VLArr,'Rank 1 accuracy','rank1.png')
    plotOne(rank5TRARr,rank5VLARr,'Rank 5 accuracy','rank5.png')
    plotOne(mAPTRARr,mAPVLARr,'mAP','mAP.png')

    
def train(model,trainLoss,validLoss,optimiser):

    trainingLossArr=[]
    valLossArr=[]
    rank1TRArr=[]
    rank1VLArr=[]
    rank5TRARr=[]
    rank5VLARr=[]
    mAPTRARr=[]
    mAPVLARr=[]

    bestValLoss=float('inf')
    for E in range(1,EPOCHS+1):
        tL=trainStep(model,E,trainLoss,optimiser)
        vL=validStep(model,validLoss)

        tMR1,tmR5,tmMAP=metricsStep(model,traainMetload)
        tVR1,tVR5,tVMAP=metricsStep(model,valMetLoad)


        print(f"Epoch:{E} | Training Loss: {tL} | Validation Loss: {vL}")
        print(f"Epoch:{E} | Training Rank1: {tMR1} | Training Rank5: {tmR5} | Training mAP1: {tmMAP}")
        print(f"Epoch:{E} | Validation Rank1: {tVR1} | Validation Rank5: {tVR5} | Validation mAP1: {tVMAP}")
        trainingLossArr.append(tL)
        valLossArr.append(vL)

        rank1TRArr.append(tMR1)
        rank1VLArr.append(tVR1)

        rank5TRARr.append(tmR5)
        rank5VLARr.append(tVR5)

        mAPTRARr.append(tmMAP)
        mAPVLARr.append(tVMAP)

        if vL<bestValLoss:
             bestValLoss=vL

             torch.save(model.state_dict(),BESTMODELPATH)
        
      
        plotMetrics(trainingLossArr,valLossArr,rank1TRArr,rank1VLArr,rank5TRARr,rank5VLARr,mAPTRARr,mAPVLARr)
 



train(siameseModel,trainLossFn,validLossFN,optimiser) 