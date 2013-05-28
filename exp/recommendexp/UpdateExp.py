

"""
Compare the different update methods on a set of sparse matrices 
"""
import numpy 
import scipy.sparse 
import time 
import matplotlib.pyplot as plt 
from exp.util.SparseUtils import SparseUtils 
from exp.recommendexp.SyntheticDataset1 import SyntheticDataset1
from exp.sandbox.RandomisedSVD import RandomisedSVD

numpy.set_printoptions(suppress=True, precision=3)

startM = 5000
endM = 6000 
startN = 7000 
endN = 8000
pnz=0.001
trainIterator = SyntheticDataset1(startM=startM, endM=endM, startN=startN, endN=endN, pnz=pnz).getTrainIteratorFunc()() 

k = 100 
ps = [0] 
qs = [1, 2]

errors = []
times = []

for i, X in enumerate(trainIterator):
    print(i)
    if i == 10: 
        break 
    
    tempTimes = []
    tempErrors = []
    
    startTime = time.time()
    U, s, V = SparseUtils.svdPropack(X, k)
    tempTimes.append(time.time()-startTime)
    tempErrors.append(numpy.linalg.norm(numpy.array(X.todense()) - (U*s).dot(V.T))/numpy.linalg.norm(X.todense()))
    
    for p in ps: 
        for q in qs: 
            startTime = time.time()
            U2, s2, V2 = RandomisedSVD.svd(X, k, p, q)
            tempTimes.append(time.time()-startTime)
            tempErrors.append(numpy.linalg.norm(numpy.array(X.todense()) - (U2*s2).dot(V2.T))/numpy.linalg.norm(X.todense()) )
            
            startTime = time.time()
            if i == 0: 
                U3, s3, V3 = RandomisedSVD.svd(X, k, p, q)
            else: 
                U3, s3, V3 = RandomisedSVD.svd(X, k, p, q, omega=lastV)    
            tempTimes.append(time.time()-startTime)
            tempErrors.append(numpy.linalg.norm(numpy.array(X.todense()) - (U3*s3).dot(V3.T))/numpy.linalg.norm(X.todense()) )

    lastU = U2 
    lastS = s2 
    lastV = V2 

    times.append(tempTimes)
    errors.append(tempErrors)
    
errors = numpy.array(errors)
times = numpy.array(times)

plotStyles = ['k-', 'k--', 'k-.', 'r--', 'r-', 'g-', 'b-', 'b--', 'b-.', 'g--', 'g--', 'g-.', 'r-', 'r--', 'r-.']

plt.figure(0)
plt.plot(numpy.arange(errors.shape[0]), errors[:, 0], plotStyles[0], label="propack")

plt.figure(1)
plt.plot(numpy.arange(errors.shape[0]), times[:, 0], plotStyles[0], label="propack")

i = 1

for p in ps: 
    for q in qs: 
        plt.figure(0)
        plt.plot(numpy.arange(errors.shape[0]), errors[:, i], plotStyles[i], label="rsvd p="+str(p) + " q="+str(q))
        plt.plot(numpy.arange(errors.shape[0]), errors[:, i+1], plotStyles[i+1], label="rsvd update p="+str(p) + " q="+str(q))
        plt.ylabel("Error")
        plt.legend()
    
        plt.figure(1)
        plt.plot(numpy.arange(times.shape[0]), times[:, i], plotStyles[i], label="rsvd p="+str(p) + " q="+str(q))
        plt.plot(numpy.arange(times.shape[0]), times[:, i+1], plotStyles[i+1], label="rsvd update p="+str(p) + " q="+str(q))
        plt.ylabel("Time (s)")
        plt.legend() 
        
        i += 2

plt.show()