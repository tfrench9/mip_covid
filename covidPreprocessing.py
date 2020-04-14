import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from scipy.ndimage import gaussian_filter

def addSaltBlurandCrop(imageList, labelList, n):
    imagesToReturn = []
    lablelsToReturn = []

    for i in range(len(imageList)):
        originalImage = imageList[i]
        print(originalImage.ndim)
        print(originalImage.shape)
        if originalImage.ndim == 2:
            originalImage = np.stack((originalImage, originalImage, originalImage), 2)

        if originalImage.shape[2]==4:
            originalImage = originalImage[:, :, :3]

        print(originalImage.shape)
        for j in range(n):
            label = labelList[i]



            cropMax = 50
            shift = np.random.randint(0, cropMax, 2)
            label[1] = '{}, {}'.format(shift[0], shift[1])

            image = originalImage[ shift[1]:shift[1] + 250, shift[0]:shift[0] + 450, :]
            s = image.shape

            nSaltPepper = np.random.randint(0, 500)
            for p in range(nSaltPepper):
                a = np.random.randint(0, s[0])
                b = np.random.randint(0, s[1])
                image[a, b, :] = np.random.uniform(low=0.0, high=1.0, size=3)

            label[2] = '{} SnP'.format(nSaltPepper)

            gBlur = np.random.randint(0, 4)
            image = gaussian_filter(image, sigma=gBlur)
            label[3] = '{}'.format(gBlur)

            imagesToReturn.append(image)
            lablelsToReturn.append(label)

    return imagesToReturn
