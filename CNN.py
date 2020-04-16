from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import random
import os

images = os.listdir('AllSamples')
random.shuffle(images)
#images.sort()
#images = images[0:500]
data = np.zeros([len(images), 224, 224])
labels = np.zeros(len(images))
for i in range(len(images)):
    arr = plt.imread('AllSamples/{}'.format(images[i]))
    gray = rgb2gray(arr)
    data[i, :, :] = gray
    if 'Necrosis' in images[i]:
        labels[i] = 0
    elif 'Tumor' in images[i]:
        labels[i] = 1
    elif 'Stroma' in images[i]:
        labels[i] = 2

percentTrain = .8
trainData = data[:int(percentTrain * data.shape[0])]
testData = data[int(percentTrain * data.shape[0]):]
trainLabels = labels[:int(percentTrain * labels.shape[0])]
testLabels = labels[int(percentTrain * labels.shape[0]):]

print('Training data shape : ', trainData.shape, trainLabels.shape)
print('Testing data shape : ', testData.shape, testLabels.shape)

classes = np.unique(trainLabels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize = [5,5])

trainData = trainData.reshape(-1, 224, 224, 1)
testData = testData.reshape(-1, 224, 224, 1)
print(trainData.shape, testData.shape)

trainLabelsOH = to_categorical(trainLabels)
testLabelsOH = to_categorical(testLabels)

# Display the change for category label using one-hot encoding
print('Original label:', trainLabels[0])
print('After conversion to one-hot:', trainLabelsOH[0])

from sklearn.model_selection import train_test_split
trainData, validData, trainLabels, validLabels = train_test_split(trainData, trainLabelsOH, test_size = 0.1)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 3

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation = 'linear', input_shape= (224,224,1), padding='same'))
fashion_model.add(LeakyReLU(alpha = 0.1))
fashion_model.add(MaxPooling2D((2, 2), padding = 'same'))
fashion_model.add(Dropout(0.1))
fashion_model.add(Conv2D(64, (5, 5), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.1))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.1))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

fashion_model.summary()

fashion_train = fashion_model.fit(trainData, trainLabels, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(validData, validLabels))

test_eval = fashion_model.evaluate(testData, testLabelsOH, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

fashion_model.save("fashion_model_dropout.h5py")

accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

com = '''
predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))'''
