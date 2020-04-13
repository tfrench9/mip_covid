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

#Function to select random images from the propper classes
def getImages(types, n):
    #Iterate through image classes and normalize the colors
    images = []
    labels = []
    for type in types:
        #Open the files and append them to an images list
        samples = random.sample(range(1, 101), n)
        for sample in samples:
            fileName = 'ImageData/{}_{}.png'.format(type, sample)
            #image = Image.open(fileName)
            image = plt.imread(fileName)
            images.append(image)
            labels.append(['{}_{}.png'.format(type, sample), 'None', 'None', 'None'])
    return images, labels

#Function to normalize the colors of the images
def colorNormalize(imageList, labelList, n):
    normalizedImages = []
    target = rgb2lab(imageList[0])
    normalizedImages.append(lab2rgb(target))
    targetMean = np.mean(target, (0, 1))
    targetSTD = np.std(target, (0, 1))
    for i in range(1, len(imageList)):
        #Determine image mean and STD
        image = rgb2lab(imageList[i])
        imageMean = np.mean(image, (0, 1))
        imageSTD = np.std(image, (0, 1))
        #Normalize lab layers
        l = (image[:, :, 0] - imageMean[0]) * targetSTD[0] / imageSTD[0] + targetMean[0]
        a = (image[:, :, 1] - imageMean[1]) * targetSTD[1] / imageSTD[1] + targetMean[1]
        b = (image[:, :, 2] - imageMean[2]) * targetSTD[2] / imageSTD[2] + targetMean[2]
        l[l < 0] = 0
        a[a < 0] = 0
        b[b < 0] = 0
        normalizedImages.append(lab2rgb(np.stack((l, a, b), 2)))
    return normalizedImages, labelList

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
