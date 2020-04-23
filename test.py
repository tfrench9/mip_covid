from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn
import pandas as pd
import numpy as np
import random

train = pd.read_csv('trainFeatures.csv')
test = pd.read_csv('testFeatures.csv')
cols = [str(x) for x in range(78)]

X = train[cols].to_numpy()
y = []
for label in list(train["Class"]):
    if label == 'No':
        y.append(0)
    if label == 'Pn':
        y.append(1)
    if label == 'CO':
        y.append(2)

Xtest = test[cols].to_numpy()
ytest = []
for label in list(test["Class"]):
    if label == 'No':
        ytest.append(0)
    if label == 'Pn':
        ytest.append(1)
    if label == 'CO':
        ytest.append(2)

com = '''
newX = np.zeros((3 * 208, X.shape[1]))
newY = np.zeros(3 * 208)
noSamples = random.sample(range(1243), 208)
for i in range(len(noSamples)):
    newX[i, :] = X[noSamples[i], :]
    newY[i] = y[noSamples[i]]
pnSamples = random.sample(range(1244, 1691), 208)
for i in range(len(pnSamples)):
    newX[i + 208, :] = X[pnSamples[i], :]
    newY[i + 208] = y[pnSamples[i]]
for i in range(208):
    newX[i + 416, :] = X[1691 + i, :]
    newY[i + 416] = y[1691 + i]

print(newX)
print(newY)
#lda = LinearDiscriminantAnalysis(n_components = 2, store_covariance = True)
#projection = lda.fit(X, y).transform(X)

#X = X.reshape(-1, X.shape[0], X.shape[1], 1)
#print(newX.shape)
#y = np.array(y)
ytest = np.array(ytest)
#print(y.shape)

#Normalize training and testing data
#X = tf.keras.utils.normalize(X, axis = 1)
#Xtest = tf.keras.utils.normalize(Xtest, axis = 1)


#Use Feed Forward model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

#Add 2 hidden layers with 128 neurons and a rectified linear and sigmoid activation
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))

#Add decision layer with 10 neurons for 10 classes
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

#Specify parameters for model training
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Train the model iterating thorugh the training set 3 times
model.fit(newX, newY, epochs = 1000)


pred = model.predict_classes(Xtest).flatten()
valLoss, valAcc = model.evaluate(Xtest, ytest)
print("Loss: " + str(valLoss))
print("Accuracy: " + str(valAcc * 100))


print(ytest)
#pred = [int(round(xx)) for xx in pred]
print(pred)
'''
#qda = QuadraticDiscriminantAnalysis()
#projection = qda.fit(X, y)
#pred = qda.predict(Xtest)

#sn.heatmap(lda.covariance_, annot=True, fmt='g')
#plt.show()
#print(len(pred))
#diff = pred - ytest
#print((list(diff).count(0) / len(diff)) * 100)

df = pd.DataFrame(zip(test["Class"], test["PID"], test["L/R Lung"], list(ytest), list(pred)), columns = ['Class', 'PID', 'Lung', 'Actual', 'Predicted'])
df.to_csv("results.csv")

color = []
for c in y:
    if c == 0:
        color.append('b')
    if c == 1:
        color.append('r')
    if c == 2:
        color.append('g')

plt.scatter(projection[:, 0], projection[:, 1], c = color)
plt.show()
