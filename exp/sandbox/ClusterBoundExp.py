
#Test some work on the cluster bound 
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
numFeatures = 2

V = numpy.random.rand(numExamples, numFeatures)

V[0:80, :] += 2
U = V - numpy.mean(V)

UU = U.dot(U.T)
s, X = numpy.linalg.eig(UU)


#Lower and upper bounds on the cluster error 
print(numpy.trace(UU) - numpy.max(s), numpy.trace(UU))
print(numpy.linalg.norm(U)**2)
 
#Now compute true cluster error 
kmeans = sklearn.cluster.KMeans(2)
kmeans.fit(U)
error = 0

for i in range(numExamples): 
    #print(U[i, :])
    #print(kmeans.cluster_centers_[kmeans.labels_[i], :])
    error += numpy.linalg.norm(U[i, :] - kmeans.cluster_centers_[kmeans.labels_[i], :])**2

print(error)


deltas = numpy.arange(0, 100, 0.1)
worstLowerBounds = numpy.zeros(deltas.shape[0])
lowerBounds = numpy.zeros(deltas.shape[0])
upperBounds = numpy.zeros(deltas.shape[0])
realError = numpy.zeros(deltas.shape[0])

for i in range(deltas.shape[0]): 
    worstLowerBounds[i], bestSigma = ClusterBound.compute2ClusterBound(U, deltas[i])
    
    #Now add random matrix to U 
    E = numpy.random.randn(numExamples, numFeatures)
    E = E*numpy.sqrt(deltas[i])/numpy.linalg.norm(E)
    U2 = U + E
    
    #print(numpy.linalg.norm(U2 -U)**2, deltas[i])
    
    UU2 = U2.dot(U2.T)
    s, X = numpy.linalg.eig(UU2)
    
    lowerBounds[i] = numpy.trace(UU2) - numpy.max(s)
    upperBounds[i] = numpy.trace(UU2)
    
    kmeans = sklearn.cluster.KMeans(2)
    kmeans.fit(U2)
  
    for j in range(numExamples): 
        realError[i] += numpy.linalg.norm(U2[j, :] - kmeans.cluster_centers_[kmeans.labels_[j], :])**2
        
    
plt.plot(deltas, worstLowerBounds, label="Worst Continuous") 
#plt.plot(deltas, upperBounds, label="Upper") 
plt.plot(deltas, lowerBounds, label="Continuous Solution") 
plt.plot(deltas, realError, label="k-means")
plt.xlabel("delta")
plt.ylabel("J_k")
plt.legend(loc="upper left") 
plt.show()

#print("objective = " + str(computeBound(U, delta)))
