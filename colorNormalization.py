from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import matplotlib
import scipy.misc
import png
import cv2
import imageio
from sklearn.preprocessing import normalize
import random
from os import listdir

#Function to select random images from the propper classes
def getImages(types, p, tts):
    #Determine all the files in the path
    path = 'images_processed_2/lungs/'
    fileNames = listdir(path)
    #Identify unique patient IDs
    pIDs = []
    lengths = []
    for type in types:
        mask = [type in file for file in fileNames]
        pID = [d.split('_')[1] for d, s in zip(fileNames, mask) if s]
        lengths.append(len(set(pID)))
        pIDs.append(pID)
    nTrain = int(int(min(lengths) * p) * tts)
    nTest = int(min(lengths) * p) - nTrain
    #Iterate through image classes and read in images
    trainImages = []
    testImages = []
    trainLabels = []
    testLabels = []
    for i in range(len(types)):
        #Randomly select the max patient IDs
        samples = random.sample(range(lengths[i]), nTrain + nTest)
        trainSamples = samples[:nTrain]
        testSamples = samples[nTrain:]
        #Add images to training set
        for sample in trainSamples:
            #Find the set of images associated with a patient and select one
            mask = ['{}_{}_'.format(types[i], pIDs[i][sample]) in file for file in fileNames]
            index = random.choice([d.split('_')[2] for d, s in zip(fileNames, mask) if s])
            #Read in both the left and right lung
            for lr in range(2):
                fileName = '{}{}_{}_{}_{}.png'.format(path, types[i], pIDs[i][sample], index, lr)
                image = plt.imread(fileName)
                trainImages.append(image[:, :, :3])
                trainLabels.append([types[i], pIDs[i][sample], index, lr, 'None', 'None', 'None'])
        #Add images to resting set
        for sample in testSamples:
            #Find the set of images associated with a patient and select one
            mask = ['{}_{}_'.format(types[i], pIDs[i][sample]) in file for file in fileNames]
            index = random.choice([d.split('_')[2] for d, s in zip(fileNames, mask) if s])
            #Read in both the left and right lung
            for lr in range(2):
                fileName = '{}{}_{}_{}_{}.png'.format(path, types[i], pIDs[i][sample], index, lr)
                image = plt.imread(fileName)
                testImages.append(image[:, :, :3])
                testLabels.append([types[i], pIDs[i][sample], index, lr, 'None', 'None'])
    return trainImages, testImages, trainLabels, testLabels

#Function to normalize the colors of the images
def colorNormalize(imageList):
    normalizedImages = []
    #target = imageList[0]
    #normalizedImages.append(target)
    #targetMean = np.mean(target, (0, 1))
    #targetSTD = np.std(target, (0, 1))
    targetMean = [0.4, 0.4, 0.4]
    targetSTD = [0.2, 0.2, 0.2]
    for i in range(len(imageList)):
        #Determine image mean and STD
        image = imageList[i]
        imageMean = np.mean(image, (0, 1))
        imageSTD = np.std(image, (0, 1))
        #Normalize lab layers
        r = (image[:, :, 0] - imageMean[0]) * targetSTD[0] / imageSTD[0] + targetMean[0]
        g = (image[:, :, 1] - imageMean[1]) * targetSTD[1] / imageSTD[1] + targetMean[1]
        b = (image[:, :, 2] - imageMean[2]) * targetSTD[2] / imageSTD[2] + targetMean[2]
        r[r < 0] = 0
        g[g < 0] = 0
        b[b < 0] = 0
        normalizedImages.append(np.stack((r, g, b), 2))
    return normalizedImages

def cropRotateFlip(imageList, labelList, n):
    imagesToReturn = []
    lablelsToReturn = []
    for i in range(len(imageList)):
        origionalImage = imageList[i]
        for j in range(n):
            label = labelList[i]
            #Crop Image
            cropMax = 512 - 224
            shift = np.random.randint(0, cropMax, 2)
            label[1] = '{}, {}'.format(shift[0], shift[1])
            image = origionalImage[shift[0]: shift[0] + 224, shift[1]: shift[1] + 224, :]
            #Flip Image
            flip = np.random.randint(0,2,2)
            if flip[0] == 1 and flip[1] == 1:
                image = image[::-1, :, :]
                image = image[:, ::-1, :]
                label[2] = 'H and V'
            elif flip[0] == 1:
                image = image[::-1, :, :]
                label[2] = 'H Only'
            elif flip[1] == 1:
                image = image[:, ::-1, :]
                label[2] = 'V Only'
            #Rotate Image
            rotations = np.random.randint(0, 4, 1)
            image = np.rot90(image, k = rotations[0])
            label[3] = '{}'.format(rotations[0] * 90)
            imagesToReturn.append(image)
            lablelsToReturn.append(label)
    return imagesToReturn, lablelsToReturn

def saveSampleImages(imageList, labelList):
    samples = random.sample(range(0, len(imageList)), 25)
    toReturn = []
    count = 0
    for i in samples:
        scipy.misc.toimage(imageList[i], cmin = 0, cmax = 1).save('DisplayImages/{}.png'.format(count))
        toReturn.append(labelList[i])
        count += 1
    return toReturn
