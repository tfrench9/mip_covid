import colorNormalization as cn
import matplotlib.pyplot as plt

images = cn.getNormedImageArray(3)
croppedImages = cn.crop(images)
print(images[0])
for i in range(9):
    plt.subplot('33{}'.format(i + 1))
    plt.imshow(images[i])
    plt.axis('off')

plt.show()

for i in range(9):
    plt.subplot('33{}'.format(i + 1))
    plt.imshow(croppedImages[i])
    plt.axis('off')

plt.show()
