
"""
Investigate why the Nystrom method does so badly on our synthetic dataset given 
in LaplacianExp2 
"""

import sys 
import logging
import numpy
import scipy 
import itertools 
import copy
import matplotlib.pyplot as plt 
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults
from apgl.graph.GraphUtils import GraphUtils
from apgl.util.Util import Util 
from exp.clusterexp.BoundGraphIterator import BoundGraphIterator 
from exp.sandbox.Nystrom import Nystrom 
from exp.sandbox.EigenUpdater import EigenUpdater 
from apgl.graph import SparseGraph 
from exp.clusterexp.FowlkesExp import createDataset 

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=200, precision=5)

def computeBound(A, omega, Q, omega2, Q2, k):
    """
    Compute the perturbation bound on L using exact eigenvalues/vectors omega and 
    Q and approximate ones omega2, Q. 
    """
    A = A.todense()
    M = Q2.T.dot(A).dot(Q2)
    R = A.dot(Q2) - Q2.dot(M)
    
    normR = numpy.linalg.norm(R)    
    lmbda, U = numpy.linalg.eigh(M)
    L2 = omega[k:]

    delta = float("inf")
    
    for i in lmbda: 
        for j in L2: 
            if abs(i-j) < delta: 
                delta = abs(i-j)      
    
    #print(lmbda, L2)
    print("normR=" + str(normR), "delta="+ str(delta), "bound="+str(normR/delta))
    
    return normR/delta


k = 4
numGraphs = 100 
numClusterVertices = 250
iterator = BoundGraphIterator(changeEdges=50, numGraphs=numGraphs, numClusterVertices=numClusterVertices, numClusters=k, p=0.1)

numMethods = 2


#Plot bound as Nystrom cols change 


W = iterator.next() 
nystromNs = numpy.arange(200, 1000, 50) 

#Same plots with Fowlkes dataset 
#There is no eigengap in this case so bound does poorly 
#W = scipy.sparse.csr_matrix(createDataset())
#nystromNs = numpy.arange(20, 151, 10) 
#k = 2

errors = numpy.zeros((len(nystromNs), numMethods))  

L = GraphUtils.shiftLaplacian(W)
L2 = GraphUtils.normalisedLaplacianSym(W)

#Find connected components 
graph = SparseGraph(GeneralVertexList(W.shape[0]))
graph.setWeightMatrix(W)
components = graph.findConnectedComponents()
print(len(components)) 


#Compute exact eigenvalues 
omega, Q = numpy.linalg.eigh(L.todense())
inds = numpy.flipud(numpy.argsort(omega)) 
omega, Q = omega[inds], Q[:, inds]
omegak, Qk = omega[0:k], Q[:, 0:k]    

print(omega)

omegaHat, Qhat = numpy.linalg.eigh(L2.todense())
inds = numpy.argsort(omegaHat)
omegaHat, Qhat = omegaHat[inds], Qhat[:, inds]
omegaHatk, Qhatk = omegaHat[0:k], Qhat[:, 0:k] 

print(omegaHat)

for i, nystromN in enumerate(nystromNs):
    #omega2, Q2 = numpy.linalg.eigh(L.todense())
    omega2, Q2 = Nystrom.eigpsd(L, int(nystromN))
    inds = numpy.flipud(numpy.argsort(omega2))
    omega2, Q2 = omega2[inds], Q2[:, inds]
    omega2k, Q2k = omega2[0:k], Q2[:, 0:k]

    errors[i, 0] = computeBound(L, omega, Q, omega2k, Q2k, k)
    
    omega2, Q2 = numpy.linalg.eigh(L2.todense())
    #omega2, Q2 = Nystrom.eigpsd(L2, int(nystromN))
    inds = numpy.argsort(omega2)
    omega2, Q2 = omega2[inds], Q2[:, inds]
    omega2k, Q2k = omega2[0:k], Q2[:, 0:k]
    #print(omega2)

    errors[i, 1] = computeBound(L2, omegaHat, Qhat, omega2k, Q2k, k)

print(errors)

plt.figure(0)
plt.plot(numpy.arange(omega.shape[0]), omega, label="Shift Laplacian")
plt.plot(numpy.arange(omegaHat.shape[0]), omegaHat, label="Normalised Laplacian")
plt.legend()

plt.figure(1)    
#plt.plot(nystromNs, errors[:, 0], label="Shift Laplacian")
plt.plot(nystromNs, errors[:, 1], label="Normalised Laplacian")
plt.xlabel("Columns")
plt.ylabel("||sin(theta)||")
plt.legend()
plt.show()

#TODO: Look at 2nd eigenvector and angle with real one
#
    