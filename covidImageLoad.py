import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt

imagesC = []
imagesNC = []
i = 0
for filename in glob.glob('images_raw/CT_COVID/*.png'):
    print(filename)
    image = Image.open(filename)
    imageNew = image.resize((500, 300))
    #imageNew = resizeimage.resize_crop(image, [500, 300])
    imageNew.save('images_processed/c_{}.png'.format(i))
    i += 1
    image.close()

i = 0
for filename in glob.glob('images_raw/CT_nonCOVID/*.png'):
    print(filename)
    image = Image.open(filename)
    imageNew = image.resize((500, 300))
    #imageNew = resizeimage.resize_crop(image, [500, 300])
    imageNew.save('images_processed/n_{}.png'.format(i))
    i += 1
    image.close()
