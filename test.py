from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn
import pandas as pd
import numpy as np
import random

train = pd.read_csv('trainFeatures3.csv')
test = pd.read_csv('testFeatures3.csv')
print(test.shape)
cols = [str(x) for x in range(test.shape[1] - 4)]
cols.remove('10')
cols.remove('18')
cols.remove('22')
cols.remove('30')
cols.remove('34')
cols.remove('42')
cols.remove('46')
cols.remove('54')
cols.remove('58')
cols.remove('66')
cols.remove('70')
cols.remove('78')
cols.remove('82')
cols.remove('90')
cols.remove('94')


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


#qda = QuadraticDiscriminantAnalysis()
#projection = qda.fit(X, y)
#pred = qda.predict(Xtest)
#print(ytest)
#print(pred)
#cm = confusion_matrix(ytest, pred)
#print(cm)

print('LDA:')
lda = LinearDiscriminantAnalysis()
projection = lda.fit(X, y)
pred = lda.predict(Xtest)
print(ytest)
print(pred)
cm = confusion_matrix(ytest, pred)
print(cm)
print('')
#
# print('QDA:')
# qda = QuadraticDiscriminantAnalysis()
# projection = qda.fit(X, y)
# pred = qda.predict(Xtest)
# print(ytest)
# print(pred)
# cm = confusion_matrix(ytest, pred)
# print(cm)

print(len(ytest))
mat = np.zeros((int(len(ytest) / 2), 4))
mat[:, 0] = list(test["PID"])[::2]
mat[:, 1] = ytest[::2]
mat[:, 2] = pred[::2]
mat[:, 3] = pred[1::2]
df = pd.DataFrame(mat, columns = ['PID', 'Numeric Class', 'Left Lung Prediction', 'Right Lung Prediction'])
df.to_csv('ResultsSummary.csv', index = False)

both_lungs = np.zeros((3, 3))
one_lung = np.zeros((3, 3))
andre_method = np.zeros((3, 3))
for rowi in range(len(mat[:, 0])):
    row = mat[rowi, :]
    curr_class = int(row[1])
    one_pred = max(pred)
    andre_method[curr_class, one_pred] += 1
    pred = row[2:].astype(int)

    if pred[0]==pred[1]:
        both_lungs[curr_class, pred[0]] += 1
        one_lung[curr_class, pred[0]] += 1

    elif curr_class in pred:
        good = np.where(pred==curr_class)[0]
        bad = (good + 1) % 2
        if len(good) == 1:
            one_lung[curr_class, pred[good]] += 1
            both_lungs[curr_class, pred[bad]] += 1
    else:
        both_lungs[curr_class, np.max(pred)] += 1
        one_lung[curr_class, np.max(pred)] += 1


def ss(arr):
    sensitivity = np.sum(arr[2, :])/(np.sum(arr[2, :]) + np.sum(arr[2, :2]))
    specificity = np.sum(arr[:2, :], (0,1))/(np.sum(arr[:2, :], (0,1)) + np.sum(arr[:2, 2]))
    return sensitivity, specificity



print(both_lungs)
print(one_lung)
print(andre_method)
print(ss(both_lungs))
print(ss(one_lung))
print(ss(andre_method))



#sn.heatmap(lda.covariance_, annot=True, fmt='g')
#plt.show()
#print(len(pred))
#diff = pred - ytest
#print((list(diff).count(0) / len(diff)) * 100)

#df = pd.DataFrame(zip(test["Class"], test["PID"], test["L/R Lung"], list(ytest), list(pred)), columns = ['Class', 'PID', 'Lung', 'Actual', 'Predicted'])
#df.to_csv("results.csv")
com = '''
color = []
for c in y:
    if c == 0:
        color.append('b')
    if c == 1:
        color.append('r')
    if c == 2:
        color.append('g')

plt.scatter(projection[:, 0], projection[:, 1], c = color)
plt.show()'''
