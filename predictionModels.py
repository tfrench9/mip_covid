from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from scipy import stats

def naiveBays(X, y, XTest, yTest):
    #Create the Gaussian Naive Bayes Model
    gnb = GaussianNB().fit(X, y)
    yPred = gnb.predict(XTest)
    #Create Results Matrix
    results = np.zeros((3,3))
    for i in range(len(yTest)):
        results[int(yTest[i]), int(yPred[i])] = results[int(yTest[i]), int(yPred[i])] + 1
    print(results)
    return results

def kMeans(X, y, XTest, yTest, k):
    #Create the K-Means Model
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(X)
    #Associate Clusters with Classes
    yVec = np.zeros(k)
    for i in range(k):
        yVec[i] = stats.mode(y[kmeans.labels_ == i])[0]
    yPred = kmeans.predict(XTest)
    #Create Results Matrix
    results = np.zeros((3,3))
    for i in range(len(yTest)):
        results[int(yTest[i]), int(yVec[yPred[i]])] = results[int(yTest[i]), int(yVec[yPred[i]])] + 1
    print(results)
    return results

def kNN(X, y, XTest, yTest, k):
    #Create the K Nearest Neighbors Model
    knn = KNeighborsClassifier(n_neighbors = k).fit(X, y)
    yPred = knn.predict(XTest)
    #Create Results Matrix
    results = np.zeros((3,3))
    for i in range(len(yTest)):
        results[int(yTest[i]), int(yPred[i])] = results[int(yTest[i]), int(yPred[i])] + 1
    print(results)
    return results

def svm(X, y, XTest, yTest, kernel, C):
    #Create the K Nearest Neighbors Model
    svm = SVC(C = C, kernel = kernel, gamma = 'auto').fit(X, y)
    yPred = svm.predict(XTest)
    #Create Results Matrix
    results = np.zeros((3,3))
    for i in range(len(yTest)):
        results[int(yTest[i]), int(yPred[i])] = results[int(yTest[i]), int(yPred[i])] + 1
    print(results)
    return results

def formatResults(type, a):
    oa = (a[0, 0] + a[1, 1] + a[2, 2]) / np.sum(a)
    n = a.diagonal()[0] / np.sum(a, axis = 1)[0]
    ns = a[0, 1] / np.sum(a, axis = 1)[0]
    nt = a[0, 2] / np.sum(a, axis = 1)[0]
    s = a.diagonal()[1] / np.sum(a, axis = 1)[1]
    sn = a[1, 0] / np.sum(a, axis = 1)[1]
    st = a[1, 2] / np.sum(a, axis = 1)[1]
    t = a.diagonal()[2] / np.sum(a, axis = 1)[2]
    tn = a[2, 0] / np.sum(a, axis = 1)[2]
    ts = a[2, 1] / np.sum(a, axis = 1)[2]
    return [type, '{}%'.format(round(oa * 100), 2), '{}%'.format(round(n * 100), 2), '{}%'.format(round(ns * 100), 2), '{}%'.format(round(nt * 100), 2), '{}%'.format(round(s * 100), 2), '{}%'.format(round(sn * 100), 2), '{}%'.format(round(st * 100), 2), '{}%'.format(round(t * 100), 2), '{}%'.format(round(tn * 100), 2), '{}%'.format(round(ts * 100))]
