import sys 
import numpy 
import logging 
import sklearn.cluster
import matplotlib.pyplot as plt 
from exp.sandbox.ClusterBound import ClusterBound
from apgl.data.Standardiser import Standardiser
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

numExamples = 100 
numFeatures = 3
std = 0.1 

V = numpy.random.rand(numExamples, numFeatures)
V[0:20 ,:] = numpy.random.randn(20, numFeatures)*std 
V[0:20 ,0:3] += numpy.array([1, 0.2, -1]) 

V[20:70 ,:] = numpy.random.randn(50, numFeatures)*std  
V[20:70, 0:3] += numpy.array([-0.5, 1, -1])

V[70: ,:] = numpy.random.randn(30, numFeatures)*std  
V[70:, 0:3] += numpy.array([-0.3, 0.4, -0.1])

U = V - numpy.mean(V, 0)
U = Standardiser().normaliseArray(U.T).T

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U[0:20, 0], U[0:20, 1], U[0:20, 2], c="red")
ax.scatter(U[20:70, 0], U[20:70, 1], U[20:70, 2], c="blue")
ax.scatter(U[70:, 0], U[70:, 1], U[70:, 2], c="green")

UU = U.dot(U.T)
#s, X = numpy.linalg.eig(UU)
X, a, Y = numpy.linalg.svd(U)

#Now compute true cluster error 
k = 3
kmeans = sklearn.cluster.KMeans(k)
kmeans.fit(U)
error = 0

for i in range(numExamples): 
    error += numpy.linalg.norm(U[i, :] - kmeans.cluster_centers_[kmeans.labels_[i], :])**2

print(error)

deltas = numpy.arange(0, 100, 1)
realDeltas = numpy.zeros(deltas.shape[0])
continuousBounds = numpy.zeros(deltas.shape[0])
continuous = numpy.zeros(deltas.shape[0])
upperBounds = numpy.zeros(deltas.shape[0])
realError = numpy.zeros(deltas.shape[0])

randScore = numpy.zeros(deltas.shape[0])
realLabels = numpy.zeros(deltas.shape[0])
realLabels[0:20] = 0 
realLabels[20:70] = 1 
realLabels[70:] = 2 

#This is a matrix of standardised original examples 
Un = Standardiser().normaliseArray(U.T).T

for i in range(deltas.shape[0]): 
    continuousBounds[i], bestSigma = ClusterBound.computeKClusterBound(U, deltas[i], k, a)
        
    #Simulate brownian motion, sort of 
    E = numpy.random.randn(numExamples, numFeatures)
    E = Standardiser().normaliseArray(E.T).T
    E = E*numpy.sqrt(deltas[i])/numpy.linalg.norm(E)
    U2 = U + E
    U2 = Standardiser().normaliseArray(U2.T).T
    U2 = U2 - numpy.mean(U2, 0)
    
    if i == deltas.shape[0]-1: 
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(U2[0:20, 0], U2[0:20, 1], U2[0:20, 2], c="red")
        ax.scatter(U2[20:70, 0], U2[20:70, 1], U2[20:70, 2], c="blue")
        ax.scatter(U2[70:, 0], U2[70:, 1], U2[70:, 2], c="green")    
    
    realDeltas[i] = ((U - U2)**2).sum()    
    
    UU2 = U2.dot(U2.T)
    s, X = numpy.linalg.eig(UU2)
    s = numpy.flipud(numpy.sort(s))
    
    continuous[i] = numpy.trace(UU2) - s[0:k-1].sum()
    upperBounds[i] = numpy.trace(UU2)
    
    kmeans = sklearn.cluster.KMeans(k)
    kmeans.fit(U2)
  
    for j in range(numExamples): 
        realError[i] += numpy.linalg.norm(U2[j, :] - kmeans.cluster_centers_[kmeans.labels_[j], :])**2
        
    randScore[i]= metrics.adjusted_rand_score(realLabels, kmeans.labels_) 
        
inds = numpy.argsort(realDeltas)
realDeltas = realDeltas[inds]        
continuousBounds = continuousBounds[inds]
continuous = continuous[inds]
realError = realError[inds]  

print("realDelta[-1]= " + str(realDeltas[-1]))
print("norm(U2)**2 = " + str(numpy.linalg.norm(U)**2))

fig = plt.figure(2)
ax = fig.add_subplot(111)    
ax.plot(realDeltas, continuousBounds, label="Worst continuous") 
ax.plot(realDeltas, continuous, label="Continuous solution") 
ax.plot(realDeltas, realError, label="k-means solution")
#ax.plot(realDeltas, upperBounds, label="Upper bound")
ax.set_ylim(0, ax.get_ylim()[1])
plt.xlabel("delta")
plt.ylabel("J_k")
plt.legend(loc="upper left")  


fig = plt.figure(3)
ax = fig.add_subplot(111)  
ax.plot(realDeltas, randScore, label="Worst continuous") 
plt.xlabel("delta")
plt.ylabel("Rand score")

plt.show()


