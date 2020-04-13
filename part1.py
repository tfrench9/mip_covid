from skimage.filters import sobel, prewitt
from skimage import transform
from skimage import feature
from skimage.color import rgb2gray, rgb2hsv
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

features = []
image_gray = rgb2gray(image)

# Use SKImage toolbox to compute Sobel, Prewitt and Canny Edge Detection

image_dct = dct(dct(image_gray, axis=0),  axis=1)
top_10_freqs = np.argsort(abs(image_dct))[::-1][1:21]
dct_mags = image_dct[top_10_freqs]

int_image_gray = np.round(255*image_gray).astype(np.uint8)
glcm = greycomatrix(int_image_gray[::4, ::4], distances=[1], angles=np.linspace(0, 2*np.pi, 9)[:-1], levels=256, symmetric=True, normed=True)

properties = ["contrast", "homogeneity", "energy", "correlation"]


kernel = np.ones((7, 7), np.uint8)

img_erosion = cv2.erode(image_gray, kernel, iterations=1)
img_dilation = cv2.dilate(image_gray, kernel, iterations=1)
img_erosion2 = cv2.erode(image_gray, kernel, iterations=2)
img_dilation2 = cv2.dilate(image_gray, kernel, iterations=2)
img_erosion3 = cv2.erode(image_gray, kernel, iterations=3)
img_dilation3 = cv2.dilate(image_gray, kernel, iterations=3)

img_dilation_edge = feature.canny(img_dilation, sigma=3)
img_erosion_edge = feature.canny(img_erosion, sigma=3)
img_dilation_edge2 = feature.canny(img_dilation2, sigma=2)
img_erosion_edge2 = feature.canny(img_erosion2, sigma=2)
img_dilation_edge3 = feature.canny(img_dilation3, sigma=1)
img_erosion_edge3 = feature.canny(img_erosion3, sigma=1)

# circles
gray_blurred = cv2.blur(int_image_gray, (3, 3))
detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 2, maxRadius = 5)
detected_circles_big = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 6, maxRadius = 30)


features.append(np.mean(img_canny, axis=(0,1)))
features.append(np.std(img_canny, axis=(0,1)))
features.append(np.max(sampled_edges))
features.append(np.min(sampled_edges))
features.append(top_10_freqs)
features.append(dct_mags)

for p in properties:
    features.append(greycoprops(glcm, p))

features.append(np.mean(img_dilation_edge))
features.append(np.std(img_dilation_edge))
features.append(np.mean(img_erosion_edge))
features.append(np.std(img_erosion_edge))

features.append(np.mean(img_dilation_edge2))
features.append(np.std(img_dilation_edge2))
features.append(np.mean(img_erosion_edge2))
features.append(np.std(img_erosion_edge2))

features.append(np.mean(img_dilation_edge3))
features.append(np.std(img_dilation_edge3))
features.append(np.mean(img_erosion_edge3))
features.append(np.std(img_erosion_edge3))

try:
    features.append(detected_circles.shape[1])
except:
    features.append(0)
try:
    features.append(detected_circles_big.shape[1])
except:
    features.append(0)

flat = np.concatenate((np.array(feature).flatten() for feature in features)).flatten()
print(flat)
