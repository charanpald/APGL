import sys 
import numpy 
import logging 
import sklearn.cluster
import matplotlib.pyplot as plt 
from exp.sandbox.ClusterBound import ClusterBound

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

numExamples = 100 
numFeatures = 3
V = numpy.random.rand(numExamples, numFeatures)
V[0:20 ,:] = numpy.random.randn(20, numFeatures) 
V[0:20 ,0:3] += numpy.array([1, 0.2, -1]) 
#V[0:20 ,0:5] += numpy.array([1, 0.2, -1, 0.5, -0.4])
V[20:70 ,:] = numpy.random.randn(50, numFeatures) 
V[20:70, 0:3] += numpy.array([1, 1, -1])
#V[20:70, 0:5] += numpy.array([1, 1, -1, 0.5, -0.4])
V[70: ,:] = numpy.random.randn(30, numFeatures) 
V[70:, 0:3] += numpy.array([-0.3, 0.4, -0.1])
#V[70:, 0:5] += numpy.array([-0.3, 0.4, -0.1, 0.5, 0.2])
U = V - numpy.mean(V, 0)

UU = U.dot(U.T)
s, X = numpy.linalg.eig(UU)

#Now compute true cluster error 
k = 3
kmeans = sklearn.cluster.KMeans(k)
kmeans.fit(U)
error = 0

for i in range(numExamples): 
    error += numpy.linalg.norm(U[i, :] - kmeans.cluster_centers_[kmeans.labels_[i], :])**2

print(error)
print("norm(U)**2 = " + str(numpy.linalg.norm(U)**2))


deltas = numpy.arange(0, 800, 5)
realDeltas = numpy.zeros(deltas.shape[0])
continuousBounds = numpy.zeros(deltas.shape[0])
continuous = numpy.zeros(deltas.shape[0])
upperBounds = numpy.zeros(deltas.shape[0])
realError = numpy.zeros(deltas.shape[0])

for i in range(deltas.shape[0]): 
    continuousBounds[i], bestSigma = ClusterBound.computeKClusterBound(U, deltas[i], k)
        
    #Now add random matrix to U 
    E = numpy.random.randn(numExamples, numFeatures)
    E = E*numpy.sqrt(deltas[i])/numpy.linalg.norm(E)
    U2 = U + E
    U2 = U2 - numpy.mean(U2, 0)
    
    realDeltas[i] = ((U - U2)**2).sum()    
    
    UU2 = U2.dot(U2.T)
    s, X = numpy.linalg.eig(UU2)
    s = numpy.flipud(numpy.sort(s))
    
    continuous[i] = numpy.trace(UU2) - s[0:k-1].sum()
    upperBounds[i] = numpy.trace(UU2)
    
    kmeans = sklearn.cluster.KMeans(2)
    kmeans.fit(U2)
  
    for j in range(numExamples): 
        realError[i] += numpy.linalg.norm(U2[j, :] - kmeans.cluster_centers_[kmeans.labels_[j], :])**2
        
inds = numpy.argsort(realDeltas)
realDeltas = realDeltas[inds]        
continuousBounds = continuousBounds[inds]
continuous = continuous[inds]
realError = realError[inds]  


print("norm(U)**2 = " + str(numpy.linalg.norm(U)**2))

fig = plt.figure()
ax = fig.add_subplot(111)    
ax.plot(realDeltas, continuousBounds, label="Worst continuous") 
ax.plot(realDeltas, continuous, label="Continuous solution") 
ax.plot(realDeltas, realError, label="k-means solution")
#ax.set_ylim(0, numpy.max(continuousBounds)+10)
plt.xlabel("delta")
plt.ylabel("J_k")
plt.legend(loc="upper left")  
plt.show()


