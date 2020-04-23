from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

train = pd.read_csv('trainFeatures.csv')
test = pd.read_csv('testFeatures.csv')
cols = [str(x) for x in range(78)]

print(test)

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

#lda = LinearDiscriminantAnalysis(n_components = 2, store_covariance = True)
#projection = lda.fit(X, y).transform(X)


qda = QuadraticDiscriminantAnalysis()
projection = qda.fit(X, y)
pred = qda.predict(Xtest)

#sn.heatmap(lda.covariance_, annot=True, fmt='g')
#plt.show()
print(len(pred))
diff = pred - ytest
print((list(diff).count(0) / len(diff)) * 100)

df = pd.DataFrame(zip(test["PID"], test["L/R Lung"], list(ytest), list(pred)), columns = ['PID', 'Lung', 'Actual', 'Predicted'])
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
