

"""
Compare the clustering methods in scikits.learn to see which ones are fastest
and most accurate 
"""
import time 
import numpy 
import sklearn.cluster as cluster
from apgl.data.Standardiser import Standardiser
import scipy.cluster.vq as vq

numExamples = 10000
numFeatures = 500

X = numpy.random.rand(numExamples, numFeatures)
X = Standardiser().standardiseArray(X)

k = 10
numRuns = 10
maxIter = 100
tol = 10**-4

intialCentroids = X[0:k, :]

#Quite fast
print("Running scikits learn k means")
clusterer = cluster.KMeans(k=k, n_init=numRuns, tol=tol, init=intialCentroids, max_iter=maxIter)
start = time.clock()
clusterer.fit(X)
totalTime = time.clock() - start
print(totalTime)

startArray = X[0:k, :]

#Really fast - good alternative but check cluster accuracy
print("Running mini batch k means")
clusterer = cluster.MiniBatchKMeans(k=k, max_iter=maxIter, tol=tol, init=intialCentroids)
start = time.clock()
clusterer.fit(X)
totalTime = time.clock() - start
print(totalTime)

clusters1 = clusterer.labels_

#Run k means clustering a number of times
print("Running vq k means")
start = time.clock()
centroids, distortion = vq.kmeans(X, intialCentroids, iter=numRuns, thresh=tol)
totalTime = time.clock() - start
print(totalTime)

#Run k means just once 
print("Running vq k means2")
start = time.clock()
centroids, distortion = vq.kmeans2(X, intialCentroids, iter=maxIter, thresh=tol)
totalTime = time.clock() - start
print(totalTime)


clusters, distortion = vq.vq(X, centroids)

#Very slow
#clusterer = cluster.Ward(n_clusters=k)
#start = time.clock()
#clusterer.fit(X)
#totalTime = time.clock() - start
#print(totalTime)

#Conclusion: k means is fast even on 10000 examples 