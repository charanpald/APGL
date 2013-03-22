"""
Compare eigen-updating and the Nystrom method to the exact eigen-decomposition 
using the bound on the canonical angles of two subspaces.  
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
from exp.sandbox.RandomisedSVD import RandomisedSVD

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=200, precision=3)
       
def eigenUpdate(L1, L2, omega, Q, k): 
    """
    Find the eigen-update between two matrices L1 (with eigenvalues omega, and 
    eigenvectors Q), and L2.  
    """
    deltaL = L2 - L1 
    deltaL.prune()
    inds = numpy.unique(deltaL.nonzero()[0]) 
    
    if len(inds) > 0:
        Y1 = deltaL[:, inds]
        Y1 = numpy.array(Y1.todense())
        Y1[inds, :] = Y1[inds, :]/2
        
        logging.debug("rank(deltaL)=" + str(Y1.shape[1]))
        
        Y2 = numpy.zeros((L1.shape[0], inds.shape[0]))
        Y2[(inds, numpy.arange(inds.shape[0]))] = 1
        
        omega, Q = EigenUpdater.eigenAdd2(omega, Q, Y1, Y2, min(k, L1.shape[0]))
    
    return omega, Q


def computeBound(A, omega, Q, omega2, Q2, k):
    """
    Compute the perturbation bound on L using exact eigenvalues/vectors omega and 
    Q and approximate ones omega2, Q2. 
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
nystromNs = [300, 600, 900]
randSVDVecs = [300, 600, 900]
numClusterVertices = 250
numMethods = len(nystromNs) + len(randSVDVecs) + 2 
errors = numpy.zeros((numGraphs, numMethods)) 

numRepetitions = 20 
#numRepetitions = 5

saveResults = True
resultsDir = PathDefaults.getOutputDir() + "cluster/"
fileName = resultsDir + "ErrorBoundNystrom.npy"

if saveResults: 
    for r in range(numRepetitions): 
        i = 0 
        iterator = BoundGraphIterator(changeEdges=50, numGraphs=numGraphs, numClusterVertices=numClusterVertices, numClusters=k, p=0.1)
        
        for W in iterator: 
            print("i="+str(i))
            L = GraphUtils.shiftLaplacian(W)
          
            if i == 0: 
                lastL = L
                lastOmega, lastQ = numpy.linalg.eigh(L.todense())
                inds = numpy.flipud(numpy.argsort(lastOmega))
                lastOmega, lastQ = lastOmega[inds], lastQ[:, inds]
                initialOmega, initialQ = lastOmega[0:k], lastQ[:, 0:k]
                #Fix for weird error in EigenAdd2 later on 
                lastQ = numpy.array(lastQ)
            
            #Compute exact eigenvalues 
            omega, Q = numpy.linalg.eigh(L.todense())
            inds = numpy.flipud(numpy.argsort(omega))
            omega, Q = omega[inds], Q[:, inds]
            omegak, Qk = omega[0:k], Q[:, 0:k]
               
            #Nystrom method 
            print("Running Nystrom")
            for j, nystromN in enumerate(nystromNs):  
                omega2, Q2 = Nystrom.eigpsd(L, nystromN)
                inds = numpy.flipud(numpy.argsort(omega2))
                omega2, Q2 = omega2[inds], Q2[:, inds]
                omega2k, Q2k = omega2[0:k], Q2[:, 0:k]
                
                errors[i, j] += computeBound(L, omega, Q, omega2k, Q2k, k)
        
            
            #Incremental updates 
            print("Running Eigen-update")
            omega3, Q3 = eigenUpdate(lastL, L, lastOmega, lastQ, k)
            inds = numpy.flipud(numpy.argsort(omega3)) 
            omega3, Q3 = omega3[inds], Q3[:, inds]
            omega3k, Q3k = omega3[0:k], Q3[:, 0:k]

            #Randomised SVD method 
            print("Running Random SVD")
            for j, r in enumerate(randSVDVecs):  
                Q4, omega4, R4 = RandomisedSVD.svd(L, r)
                inds = numpy.flipud(numpy.argsort(omega4))
                omega4, Q4 = omega4[inds], Q4[:, inds]
                omega4k, Q4k = omega4[0:k], Q4[:, 0:k]
                
                errors[i, j+len(nystromNs)] += computeBound(L, omega, Q, omega4k, Q4k, k)
            
            #Use previous results for update, not 1st ones 
            lastL = L 
            lastOmega = omega3 
            lastQ = Q3
            
            errors[i, len(nystromNs)+len(randSVDVecs)] += computeBound(L, omega, Q, omega3k, Q3k, k)
            
            #Compare vs initial solution     
            errors[i, len(nystromNs)+len(randSVDVecs)+1] += computeBound(L, omega, Q, initialOmega, initialQ, k)
            
            i += 1 
    
    errors /= numRepetitions 
    print(errors)
    
    numpy.save(fileName, errors)
    logging.debug("Saved results as " + fileName)
else: 
    errors = numpy.load(fileName)   
    print(errors)
    plotStyles1 = ['k-', 'k--', 'k-.', 'b-', 'b--', 'b-.', 'g-', 'g--', 'g-.', 'r-', 'r--', 'r-.']    
    
    plt.figure(0)
    #plt.plot(numpy.arange(errors.shape[0]), errors[:, 0], plotStyles1[0], label="Nystrom m=300")
    plt.plot(numpy.arange(errors.shape[0]), errors[:, 1], plotStyles1[0], label="Nystrom m=600")
    plt.plot(numpy.arange(errors.shape[0]), errors[:, 2], plotStyles1[1], label="Nystrom m=900")
    plt.plot(numpy.arange(errors.shape[0]), errors[:, 5], plotStyles1[2], label="RandSVD r=600")
    plt.plot(numpy.arange(errors.shape[0]), errors[:, 6], plotStyles1[3], label="RandSVD r=900") 
    plt.plot(numpy.arange(errors.shape[0]), errors[:, 3], plotStyles1[4], label="Eigen-update") 
    #plt.plot(numpy.arange(errors.shape[0]), errors[:, 4], label="Initial sol.")
    plt.legend(loc="upper left")
    plt.xlabel("Graph no.")
    plt.ylabel("||sin(theta)||")
    plt.grid(True)
    
    plt.show()
    #Result are terrible 
