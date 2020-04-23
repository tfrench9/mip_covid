from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from scipy.fftpack import dct
from skimage.feature import canny, greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from skimage.filters import sobel, sobel_h, sobel_v, prewitt
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import scipy.misc
from scipy.fftpack import dct, idct
import pywt

#Get RGB, HSV, and LAB color space means and STDs per layer (Indexes 0 - 2)
def getColorFeatures(images):
    toReturn = None
    for image in images:
        grayMean = np.mean(image[:,:,0], axis = (0, 1))
        graySTD = np.std(image[:,:,0], axis = (0, 1))
        imageFeatures = np.array([grayMean, graySTD])
        if toReturn is None:
            toReturn = imageFeatures
        else:
            toReturn = np.vstack((toReturn, imageFeatures))
    return toReturn

def dctReduced(image, xBlocks, yBlocks, hfThresh, nFreqs):

    # Output Arrays
    imageDCT = np.full(image.shape, 0)
    imageHF = np.full(image.shape, 0)
    imageHFReconstruct = np.full(image.shape, 1.0)
    sobelFeatures = []
    waveletFeatures = []
    freqFeatures = []

    for i in range(yBlocks):
        for j in range(xBlocks):

            # Break into Blocks
            block = image[round(i*image.shape[0]/yBlocks):round((i + 1)*image.shape[0]/yBlocks), round(j*image.shape[1]/xBlocks):round((j+1)*image.shape[1]/xBlocks)]

            # Compute DCT
            freqs = dct(dct(block, axis=0),  axis=1)

            # Sort Frequencies and Slice indices
            hold = freqs.flatten()
            highFreqsInds = np.argsort(abs(hold))[::-1][hfThresh:]

            # Safe Frequency Features for output
    #        freqFeatures.append(highFreqsInds[:nFreqs])
    #        freqFeatures.append(hold[highFreqsInds][:nFreqs])

            # Take only HF components
            highFreqs = np.zeros(hold.shape)
            highFreqs[highFreqsInds] = hold[highFreqsInds]
            highFreqs = highFreqs.reshape(np.shape(freqs))

            # Reconstruct
            highFreqsReconstruct = idct(idct(highFreqs, axis=0), axis=1)
            hfForSobel = np.round(256 * highFreqsReconstruct / (highFreqsReconstruct.max() - highFreqsReconstruct.min()) + highFreqsReconstruct.min())

            sobelHFBlock = sobel(hfForSobel)
            sobelFeatures.append(np.mean(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.max(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.median(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.std(sobelHFBlock, axis = (0, 1)))

            sobelHFBlock = sobel_h(hfForSobel)
            sobelFeatures.append(np.mean(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.max(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.median(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.std(sobelHFBlock, axis = (0, 1)))

            sobelHFBlock = sobel_v(hfForSobel)
            sobelFeatures.append(np.mean(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.max(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.median(sobelHFBlock, axis = (0, 1)))
            sobelFeatures.append(np.std(sobelHFBlock, axis = (0, 1)))

            coeffs2 = pywt.dwt2(block, 'bior1.3')
            LL, (LH, HL, HH) = coeffs2
            waveletFeatures.append(np.mean(HL, axis = (0, 1)))
            waveletFeatures.append(np.max(HL, axis = (0, 1)))
            waveletFeatures.append(np.median(HL, axis = (0, 1)))
            waveletFeatures.append(np.std(HL, axis = (0, 1)))

            holdHF = np.round(256*(highFreqs - np.min(highFreqs))/(np.max(highFreqs)-np.min(highFreqs)))
            holdBlur = gaussian_filter(holdHF, sigma = 10)
            holdSobel = sobel(holdBlur)
            freqFeatures.append(np.mean(holdSobel, axis = (0, 1)))
            freqFeatures.append(np.max(holdSobel, axis = (0, 1)))
            freqFeatures.append(np.median(holdSobel, axis = (0, 1)))
            freqFeatures.append(np.std(holdSobel, axis = (0, 1)))

            # Log scale included for display
            imageDCT[round(i*image.shape[0]/yBlocks):round((i+1)*image.shape[0]/yBlocks), round(j*image.shape[1]/xBlocks):round((j+1)*image.shape[1]/xBlocks)] = freqs
            imageHF[round(i*image.shape[0]/yBlocks):round((i+1)*image.shape[0]/yBlocks), round(j*image.shape[1]/xBlocks):round((j+1)*image.shape[1]/xBlocks)] = highFreqs
            imageHFReconstruct[round(i*image.shape[0]/yBlocks):round((i+1)*image.shape[0]/yBlocks), round(j*image.shape[1]/xBlocks):round((j+1)*image.shape[1]/xBlocks)] = highFreqsReconstruct

    # Return outputs
    imageHFReconstruct = np.round(256 * imageHFReconstruct / (imageHFReconstruct.max() - imageHFReconstruct.min()) + imageHFReconstruct.min())
    return imageHFReconstruct, sobelFeatures, waveletFeatures, freqFeatures

#Get a bunch of other features
def getOtherFeatures(images):
    row = 0
    toReturn = np.zeros([len(images), 207])
    for image in images:
        gray = rgb2gray(image)

        # Extract Frequency and HF Sobel Features
        imageHFReconstruct, sobelFeatures, waveletFeatures, freqFeatures = dctReduced(gray, 2, 4, 400, 50)

        # Extract Canny Edge Density Mean and STDs (Indexes 18 - 23)
        cannyImage1 = canny(gray, sigma = 1)
        cannyMean1 = np.mean(cannyImage1, axis = (0, 1))
        cannySTD1 = np.std(cannyImage1, axis = (0, 1))

        cannyImage2 = canny(gray, sigma = 2)
        cannyMean2 = np.mean(cannyImage1, axis = (0, 1))
        cannySTD2 = np.std(cannyImage1, axis = (0, 1))

        cannyImage3 = canny(gray, sigma = 3)
        cannyMean3 = np.mean(cannyImage3, axis = (0, 1))
        cannySTD3 = np.std(cannyImage3, axis = (0, 1))
        cannyFeatures = np.array((cannyMean1, cannySTD1, cannyMean2, cannySTD2, cannyMean3, cannySTD3))

        #Extract DCT Frequencies and Magnitudes (Indexes 24 - 63)
        # topFreqs = np.array(freqFeatures[::2]).flatten()
        # dctMags = np.array(freqFeatures[1::2]).flatten()
        # dctFeatures = np.concatenate((topFreqs, dctMags))

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
        imageFeatures = np.concatenate((sobelFeatures, waveletFeatures, freqFeatures, cannyFeatures, glcmFeatures, erodeDialateFeatures, circleFeatures))
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

def performQDA(features, labels):
    X = features
    Y = np.zeros(len(labels))
    for i in range(len(labels)):
        if 'No' in labels[i][0]:
            Y[i] = 0
        elif 'Pn' in labels[i][0]:
            Y[i] = 1
        elif 'CO' in labels[i][0]:
            Y[i] = 2
    qda = QuadraticDiscriminantAnalysis(n_components = 2)
    projection = qda.fit(X, Y).transform(X)
    # Percentage of variance explained for each components
    return projection, Y, qda.explained_variance_ratio_, lda

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
