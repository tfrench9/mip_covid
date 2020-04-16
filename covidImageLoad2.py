import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import pandas as pd
import os


meta = pd.read_csv("metadata.csv")[['patientid', 'finding', 'filename', 'view']]

i = []
types = []

for row in meta.iterrows():
    pid = int(row[1][0])
    t = row[1][1]
    if not (t in types):
        types.append(t)
        i.append(0)
    filename = row[1][2]
    ti = types.index(t)
    view = row[1][3]

    if not os.path.exists('images_processed_2/{}'.format(view)):
        os.makedirs('images_processed_2/{}'.format(view))

    try:
        image = Image.open("images_newset/" + filename)
        imageNew = image.resize((1000, 1250))
        if (types[ti][:2] == 'E.'):
            imageNew.save('images_processed_2/{}/{}_{}_{}.png'.format(view, "EC", pid, i[ti]))
        else:
            imageNew.save('images_processed_2/{}/{}_{}_{}.png'.format(view, types[ti][:2], pid, i[ti]))
        i[ti] += 1
        print('images_processed_2/{}/{}_{}_{}.png'.format(view, types[ti][:2], pid, i[ti]))
        image.close()
    except:
        print("could not open: images_newset/" + filename)
