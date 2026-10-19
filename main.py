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
from PIL import Image
from torchvision.transforms import v2


import torch
from torch.utils.data import Dataset,DataLoader

from torchvision.transforms import v2
import os
import random
import numpy as np
from PIL import Image

class EmbeddDataset(Dataset):


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
        return self.transformTrain(img),label,path




    def __len__(self):
        return (len(self.images))





DATASTOREPATH=r"D:\BD\DogFaceNet\test"

BESTMODELPATH=r"D:\best.tar"
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simMetricVal=EmbeddDataset(DATASTOREPATH)

valMetLoad=DataLoader(simMetricVal,batch_size=16,shuffle=False)
siameseModel=VGG16Siamese(512)



siameseModel.load_state_dict(torch.load(BESTMODELPATH,map_location=DEVICE))

siameseModel=siameseModel.to(DEVICE)
def embeddSet(model,saveDir,fileRootname):
    allEmbeddings=[]
    allLabels=[]
    allPaths=[]

        
    with torch.no_grad():
        for dogs,doglabels,dogPaths in valMetLoad:
            dogs=dogs.to(DEVICE)
            feat=model.featureEncode(dogs)
            feat=torch.nn.functional.normalize(feat,p=2,dim=1)
            allEmbeddings.append(feat.cpu())
            allLabels.append(doglabels)
            allPaths.extend(dogPaths)

    embeddings=torch.cat(allEmbeddings)
    labels=torch.cat(allLabels)
    paths=np.array(allPaths,dtype="U300")

    os.makedirs(saveDir,exist_ok=True)
    np.save(os.path.join(saveDir,f"{fileRootname}_embedd"),embeddings.numpy())
    np.save(os.path.join(saveDir,f"{fileRootname}_labels"),labels.numpy())
    np.save(os.path.join(saveDir,f"{fileRootname}_paths"),paths)
    
def infer(image,model,npyDir,npyRootName):


    embedd=torch.from_numpy(np.load(os.path.join(npyDir,f'{npyRootName}_embedd.npy')))
    labels=np.load(os.path.join(npyDir,f'{npyRootName}_labels.npy'))
    Paths=np.load(os.path.join(npyDir,f'{npyRootName}_paths.npy'))


    petArr=np.array(Image.open(image))

    transForms=v2.Compose([
        
                v2.Resize((224,224)),
                v2.ToTensor(),
                v2.ToDtype(torch.float32,scale=True),
                
    ])

    petTens=transForms(petArr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        featEmbedd=model.featureEncode(petTens)
        featEmbedd=torch.nn.functional.normalize(featEmbedd,p=2,dim=1)


    embedd=embedd.to(DEVICE)


    dist=torch.cdist(featEmbedd,embedd)[0] #only need one of the vectors

    #deafult is ascending order so this is fine
    indices=dist.argsort()[:min(5,len(dist))]

    indices=indices.cpu().numpy()
    return Paths[indices],labels[indices]
embeddSet(siameseModel,r'.\dataNPY','Dogs')


"""
paths,labels=infer(r"D:\BD\DogFaceNet\test\1281\1281.4_42d145868e2eb302.jpg",siameseModel,r'./data','Dogs')



#1 query image + 5 results = 6 panels in one row
fig,axes=plt.subplots(1,6,figsize=(18,4))

#first panel is the uploaded image
axes[0].imshow(Image.open(r"D:\BD\DogFaceNet\test\1281\1281.4_42d145868e2eb302.jpg").convert('RGB'))
axes[0].set_title('Query')
axes[0].axis('off')

#the other panels are the top 5, closest first
for i in range(len(paths)):
    axes[i+1].imshow(Image.open(paths[i]).convert('RGB'))
    axes[i+1].set_title(f'Rank {i+1}\nDog {labels[i]}')
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig('results.png')   #optional, remove if you don't want a file
plt.show()

"""