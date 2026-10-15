import os


dirPath=r'C:\Users\hamza\Downloads\DogFaceNet\DogFaceNet\test'
dirList=os.listdir(dirPath)


count=0
for f in dirList:

    pathTemp=os.path.join(dirPath,f)

    if len(os.listdir(pathTemp))<2:
        count+=1



print(count)