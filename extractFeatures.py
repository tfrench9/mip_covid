from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from scipy.fftpack import dct
from skimage.feature import canny, greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.misc

#Get RGB, HSV, and LAB color space means and STDs per layer (Indexes 0 - 11)
def getColorFeatures(images):
    toReturn = None
    for image in images:
        hsv = rgb2hsv(image)
        rgbMean = np.mean(image, axis = (0, 1))
        rgbSTD = np.std(image, axis = (0, 1))
        hsvMean = np.mean(hsv, axis = (0, 1))
        hsvSTD = np.std(hsv, axis = (0, 1))
        imageFeatures = np.concatenate((rgbMean.T, rgbSTD.T, hsvMean.T, hsvSTD.T))
        if toReturn is None:
            toReturn = imageFeatures
        else:
            toReturn = np.vstack((toReturn, imageFeatures))
    return toReturn

#Get a bunch of other features
def getOtherFeatures(images):
    row = 0
    toReturn = np.zeros([len(images), 87])
    for image in images:
        gray = rgb2gray(image)
        #Extract Canny Edge Density Mean and STDs (Indexes 18 - 23)
        cannyImage1 = canny(gray, sigma = 1)
        cannyMean1 = np.mean(cannyImage1, axis = (0, 1))
        cannySTD1 = np.std(cannyImage1, axis = (0, 1))
        cannyImage2 = canny(gray, sigma = 2)
        cannyMean2 = np.mean(cannyImage2, axis = (0, 1))
        cannySTD2 = np.std(cannyImage2, axis = (0, 1))
        cannyImage3 = canny(gray, sigma = 3)
        cannyMean3 = np.mean(cannyImage3, axis = (0, 1))
        cannySTD3 = np.std(cannyImage3, axis = (0, 1))
        cannyFeatures = np.array((cannyMean1, cannySTD1, cannyMean2, cannySTD2, cannyMean3, cannySTD3))
        #Extract DCT Frequencies and Magnitudes (Indexes 24 - 63)
        imageDCT = dct(dct(gray, axis = 0),  axis = 1)
        top10Freqs = np.argsort(abs(imageDCT.flatten()))[::-1][1:21]
        dctMags = imageDCT.flatten()[top10Freqs]
        dctFeatures = np.concatenate((top10Freqs, dctMags))
        #Extract GLCM Properties (Indexes 64 -78)
        glcmFeatures = np.zeros(15)
        props = ["contrast", "homogeneity", "energy", "correlation"]
        intGray = np.round(255 * gray).astype(np.uint8)
        glcm1 = greycomatrix(intGray, distances=[1], angles = np.linspace(0, 2 * np.pi, 9)[:-1], levels = 256, symmetric = True, normed = True)
        glcm3 = greycomatrix(intGray, distances=[3], angles = np.linspace(0, 2 * np.pi, 9)[:-1], levels = 256, symmetric = True, normed = True)
        glcm5 = greycomatrix(intGray, distances=[5], angles = np.linspace(0, 2 * np.pi, 9)[:-1], levels = 256, symmetric = True, normed = True)
        glcms = [glcm1, glcm3, glcm5]
        count = 0
        for i in range(len(glcms)):
            for j in range(len(props)):
                glcmFeatures[count] = np.mean(greycoprops(glcms[i], props[j]))
                count += 1
            glcmFeatures[count] = shannon_entropy(glcms[i])
            count += 1
        #Extract Dialation and Erosion Means and STDs (Indexes 79 - 102)
        erodeDialateFeatures = np.zeros(24)
        kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], np.uint8)
        for i in range(3):
            erosion = cv2.erode(gray, kernel, iterations = i + 1)
            dialation = cv2.dilate(gray, kernel, iterations = i + 1)
            erosionEdge = canny(erosion, sigma = 3 - i)
            dialationEdge = canny(dialation, sigma = 3 - i)
            erodeDialateFeatures[(i * 8)] = np.mean(erosion, axis = (0, 1))
            erodeDialateFeatures[(i * 8) + 1] = np.std(erosion, axis = (0, 1))
            erodeDialateFeatures[(i * 8) + 2] = np.mean(dialation, axis = (0, 1))
            erodeDialateFeatures[(i * 8) + 3] = np.std(dialation, axis = (0, 1))
            erodeDialateFeatures[(i * 8) + 4] = np.mean(erosionEdge, axis = (0, 1))
            erodeDialateFeatures[(i * 8) + 5] = np.std(erosionEdge, axis = (0, 1))
            erodeDialateFeatures[(i * 8) + 6] = np.mean(dialationEdge, axis = (0, 1))
            erodeDialateFeatures[(i * 8) + 7] = np.std(dialationEdge, axis = (0, 1))
        #Circle Detection Counter (Indexes 103 - 104)
        circleFeatures = np.zeros(2)
        grayBlurred = cv2.blur(intGray, (3, 3))
        smallCircles = cv2.HoughCircles(grayBlurred, cv2.HOUGH_GRADIENT, 1, 5, param1 = 50, param2 = 10, minRadius = 2, maxRadius = 5)
        bigCircles = cv2.HoughCircles(grayBlurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 6, maxRadius = 30)
        if smallCircles is not None:
            circleFeatures[0] = smallCircles.shape[1]
        if bigCircles is not None:
            circleFeatures[1] = bigCircles.shape[1]
        #Concatonate all these features together
        imageFeatures = np.concatenate((cannyFeatures, dctFeatures, glcmFeatures, erodeDialateFeatures, circleFeatures))
        toReturn[row, :] = imageFeatures
        row += 1
    return toReturn

def performLDA(features, labels):
    X = features
    Y = np.zeros(len(labels))
    for i in range(len(labels)):
        if 'No' in labels[i][0]:
            Y[i] = 0
        elif 'Pn' in labels[i][0]:
            Y[i] = 1
        elif 'CO' in labels[i][0]:
            Y[i] = 2
    lda = LinearDiscriminantAnalysis(n_components = 2)
    projection = lda.fit(X, Y).transform(X)
    # Percentage of variance explained for each components
    return projection, Y, lda.explained_variance_ratio_, lda

def performProjectionLDA(features, labels, lda):
    X = features
    Y = np.zeros(len(labels))
    for i in range(len(labels)):
        if 'No' in labels[i][0]:
            Y[i] = 0
        elif 'Pn' in labels[i][0]:
            Y[i] = 1
        elif 'CO' in labels[i][0]:
            Y[i] = 2
    projection = lda.transform(X)
    return projection, Y

def savePicture(image, name):
    scipy.misc.toimage(image, cmin = 0, cmax = 1).save('DisplayImages/{}.png'.format(name))

def saveCER(image):
    gray = rgb2gray(image)
    kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], np.uint8)
    cannyImage = canny(gray, sigma = 1)
    erosion = cv2.erode(gray, kernel, iterations = 1)
    dialation = cv2.dilate(gray, kernel, iterations = 1)
    erosionEdge = canny(erosion, sigma = 3)
    dialationEdge = canny(dialation, sigma = 3)
    savePicture(cannyImage, 'canny')
    savePicture(erosion, 'erosion')
    savePicture(dialation, 'dialation')
    savePicture(erosionEdge, 'erosionEdge')
    savePicture(dialationEdge, 'dialationEdge')
