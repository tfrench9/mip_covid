import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy.misc


if not os.path.exists('images_processed_2/lungs'):
    os.makedirs('images_processed_2/lungs')

for filename in glob.glob('images_processed_2/*.png'):
    image = plt.imread(filename)
    if (image.ndim == 2):
        image = np.stack((image, image, image), 2)

    lung1 = image[150:950, 125:475, :]
    lung2 = image[150:950, 525:875, :]
    lung2 = lung2[:, ::-1, :]

    im = Image.fromarray((255*lung1).astype(np.uint8))
    im.save("images_processed_2/lungs/" + filename[19:-4] + "_0" + ".png")
    im.close()
    im = Image.fromarray((255*lung2).astype(np.uint8))
    im.save("images_processed_2/lungs/" + filename[19:-4] + "_1" + ".png")
    print("done with" + filename)
    im.close()
