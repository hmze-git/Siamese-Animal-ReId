import torch
from torchvision.transforms import v2
from SingleImageDataset import SingleIMGSet
from SiameseModel import VGG16Siamese
from torch.utils.data import DataLoader

#---------------------------------SETTINGS------------------------------------

BATCH=16
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(DEVICE)
BESTMODELPATH=r'D:\BD\best.tar'
TESTPATH=r'D:\BD\DogFaceNet\test'

#---------------------------------DATA------------------------------------

testSet=SingleIMGSet(TESTPATH)

#SingleIMGSet applies RandAugment, which we don't want at test time, so swap in a plain transform
testSet.transformTrain=v2.Compose([
    v2.Resize((224,224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True)]
)

testLoader=DataLoader(testSet,batch_size=BATCH,shuffle=False)

#---------------------------------MODEL------------------------------------

model=VGG16Siamese(512)
model.load_state_dict(torch.load(BESTMODELPATH,map_location=DEVICE))
model=model.to(DEVICE)
model.eval()

#---------------------------------EMBED EVERY TEST IMAGE------------------------------------

allEmbeddings=[]
allLabels=[]

with torch.no_grad():
    for dogs,doglabels in testLoader:
        dogs=dogs.to(DEVICE)
        feat=model.featureEncode(dogs)
        feat=torch.nn.functional.normalize(feat,p=2,dim=1)
        allEmbeddings.append(feat.cpu())
        allLabels.append(doglabels)

embeddings=torch.cat(allEmbeddings)
labels=torch.cat(allLabels)

#distance between every pair of images
distanceMat=torch.cdist(embeddings,embeddings)

#---------------------------------METRICS------------------------------------

top1Sum=0.0
top5Sum=0.0
precision1Sum=0.0
precision5Sum=0.0
recall1Sum=0.0
recall5Sum=0.0
numQueries=0

#every image takes a turn as the query
for q in range(len(labels)):

    dist=distanceMat[q].clone()

    #the query is distance 0 from itself, so push it to the end of the ranking and then cut it off
    dist[q]=float('inf')
    ranked=dist.argsort()[:-1]

    #1 where the retrieved image is the same dog as the query, 0 where it isn't
    matches=(labels[ranked]==labels[q]).float()

    #how many other photos of this dog exist
    totalRelevant=matches.sum().item()

    #a dog with only one photo has nothing to find, so skip it
    if totalRelevant==0:
        continue

    correctIn1=matches[:1].sum().item()
    correctIn5=matches[:5].sum().item()

    #top k accuracy: 1 if at least one of the k closest is the right dog
    if correctIn1>0:
        top1Sum+=1
    if correctIn5>0:
        top5Sum+=1

    #precision: correct results / results returned
    precision1Sum+=correctIn1/1
    precision5Sum+=correctIn5/5

    #recall: correct results / all right photos that exist
    recall1Sum+=correctIn1/totalRelevant
    recall5Sum+=correctIn5/totalRelevant

    numQueries+=1

#average over all queries to get one value each
print(f"Queries scored: {numQueries}")
print(f"Top-1 accuracy: {top1Sum/numQueries}")
print(f"Top-5 accuracy: {top5Sum/numQueries}")
print(f"Precision@1:    {precision1Sum/numQueries}")
print(f"Precision@5:    {precision5Sum/numQueries}")
print(f"Recall@1:       {recall1Sum/numQueries}")
print(f"Recall@5:       {recall5Sum/numQueries}")