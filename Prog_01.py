import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

ReadImage = Image.open("./images/dont_commit/01.webp")
Xdim,Ydim = ReadImage.size
ReadImage = np.array(ReadImage)
if len(ReadImage.shape)==2:
    ReadImage = ReadImage.reshape(ReadImage.shape[0],ReadImage.shape[1],1)

def DetectBoundaries(ReadImage, verbose=False, Channels=3, invert=False, samplePercentage=0.10):
    ReadImage = ReadImage/255.0
    # sortedFlattened = np.random.choice(ReadImage.flatten(),size=int(ReadImage.shape[0]*ReadImage.shape[1]*samplePercentage), replace=False)
    # Threshold = np.mean(sortedFlattened)
    # ReadImage[ReadImage<=Threshold] = 0.0
    # print(ReadImage.shape)
    Xshape, Yshape, ChannelsPresent = ReadImage.shape
    ReturnedArray = np.zeros((Xshape,Yshape,Channels),dtype=np.float64)
    LUD_Detection = ReadImage[1:,1:,:Channels] - ReadImage[:Xshape-1,:Yshape-1,:Channels]
    ReturnedArray[1:,1:,:Channels] = LUD_Detection
    U_Detection = ReadImage[1:,:,:Channels] - ReadImage[:Xshape-1,:,:Channels]
    ReturnedArray[1:,:,:Channels] = U_Detection
    RUD_Detection = ReadImage[1:,1:,:Channels] - ReadImage[:Xshape-1,:Yshape-1,:Channels]
    ReturnedArray[1:,1:,:Channels] = RUD_Detection
    R_Detection = ReadImage[:Xshape-1,:Yshape-1,:Channels] - ReadImage[1:,1:,:Channels]
    ReturnedArray[:Xshape-1,:Yshape-1,:Channels] = R_Detection
    
    RDD_Detection = ReadImage[:Xshape-1,:Yshape-1,:Channels] - ReadImage[1:,1:,:Channels]
    ReturnedArray[:Xshape-1,:Yshape-1,:Channels] = RDD_Detection
    D_Detection = ReadImage[:Xshape-1,:,:Channels] - ReadImage[1:,:,:Channels]
    ReturnedArray[:Xshape-1,:,:Channels] = D_Detection
    LDD_Detection = ReadImage[:Xshape-1,:Yshape-1,:Channels]- ReadImage[1:,1:,:Channels]
    ReturnedArray[:Xshape-1,:Yshape-1,:Channels] = LDD_Detection
    L_Detection = ReadImage[1:,1:,:Channels] - ReadImage[:Xshape-1,:Yshape-1,:Channels]
    ReturnedArray[1:,1:,:Channels] = L_Detection
    ReturnedArray = ReturnedArray.reshape(Xshape,Yshape,Channels)
    if invert:
        ReturnedArray = np.abs(ReturnedArray - 1.00)
##    Boundaries = ReturnedArray>DetectionThreshold
##    NewImage = Boundaries
##    FlattenedImage = NewImage.flatten()
##    FlattenedImage[FlattenedImage>DetectionThreshold] = 1.0
##    NewImage = FlattenedImage.reshape(NewImage.shape)
##    NewImage = NewImage - np.mean(NewImage[:,:,:Channels]).reshape(1,1,Channels)*MeanThresholdPercentage/MeanSTDThreshold   
    if verbose:
        plt.imshow(ReturnedArray)
        plt.show()
    return ReturnedArray
    
EdgeDetectedImage = DetectBoundaries(ReadImage, True)

