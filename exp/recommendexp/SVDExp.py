

"""
In this short experiment we compare PROPACK, RSVD, updating RSVD and the initial 
solution on the synthetic dataset. We want to compare error in approximation 
and also the times of the methods.  
"""
import numpy 
import scipy.sparse 
import sppy.linalg 
from exp.util.SparseUtils import SparseUtils 
from exp.recommendexp.SyntheticDataset1 import SyntheticDataset1
from exp.sandbox.RandomisedSVD import RandomisedSVD
import time 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

k = 100
p = 20
q = 3
generator = SyntheticDataset1(startM=5000, endM=6000, startN=1000, endN=1200, pnz=0.10, noise=0.01, nonUniform=False)

numMethods = 4
numMatrices = 10
times = numpy.zeros((numMatrices, numMethods))
errors = numpy.zeros((numMatrices, numMethods))
trainIterator = generator.getTrainIteratorFunc()()
trainIterator = list(trainIterator)
trainIterator = trainIterator[0:numMatrices]

for i, X in enumerate(trainIterator): 
    print(i)
      
    #First normal RSVD 
    startTime = time.time()
    U2, s2, V2 = sppy.linalg.core.rsvd(X, k, q=q)
    times[i, 0] = time.time() - startTime 
    
    errors[i, 0] = numpy.linalg.norm(X - (U2*s2).dot(V2.T)) 

    #Now RSVD + update 
    if i == 0: 
        startTime = time.time()
        U3, s3, V3 = sppy.linalg.core.rsvd(X, k,q=q)
        times[i, 1] = time.time() - startTime 
        lastX = X 
    else: 
        E = X - lastX
        E.eliminate_zeros()
        print(X.nnz, E.nnz)
        startTime = time.time()
        U3, s3, V3 = RandomisedSVD.updateSvd(X, U3, s3, V3, E, k, p)
        times[i, 1] = time.time() - startTime 
        
        lastX = X  
        
    errors[i, 1] = numpy.linalg.norm(X - (U3*s3).dot(V3.T)) 
    
    #Accurate method 
    startTime = time.time()
    U4, s4, V4 = SparseUtils.svdPropack(X, k)    
    times[i, 2] = time.time() - startTime 
    
    errors[i, 2] = numpy.linalg.norm(X - (U4*s4).dot(V4.T)) 
    
    #Final method - just use the same SVD
    if i == 0: 
        startTime = time.time()
        U5, s5, V5 = SparseUtils.svdPropack(X, k)    
        times[i, 3] = time.time() - startTime 
    
    errors[i, 3] = numpy.linalg.norm(X - (U5*s5).dot(V5.T)) 
    
    
cumtimes = numpy.cumsum(times, 0)
print(cumtimes)
print(errors)

plt.figure(0)
plt.plot(numpy.arange(cumtimes.shape[0]), cumtimes[:, 0], label="RSVD")
plt.plot(numpy.arange(cumtimes.shape[0]), cumtimes[:, 1], label="RSVD+")
plt.plot(numpy.arange(cumtimes.shape[0]), cumtimes[:, 2], label="SVD")
plt.plot(numpy.arange(cumtimes.shape[0]), cumtimes[:, 3], label="Initial")
plt.legend()

plt.figure(1)
plt.plot(numpy.arange(errors.shape[0]), errors[:, 0], label="RSVD")
plt.plot(numpy.arange(errors.shape[0]), errors[:, 1], label="RSVD+")
plt.plot(numpy.arange(errors.shape[0]), errors[:, 2], label="SVD")
plt.plot(numpy.arange(errors.shape[0]), errors[:, 3], label="Initial")
plt.legend()

plt.show()