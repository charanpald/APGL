"""
Test some recommendation with the movielen data 
"""

from apgl.util.PathDefaults import PathDefaults 
import nimfa
import numpy 
import matplotlib.pyplot as plt 
from os.path import dirname, abspath, sep
from warnings import warn
import scipy.io 
import logging 
import sys
from apgl.util.Sampling import Sampling 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(linewidth=1000, threshold=10000000)

def preprocess(V):
    """
    Normalize data so that each row has max value 1. 
    
    :param V: The data matrix. 
    :type V: `scipy.sparse.csr_matrix`
    """
    logging.debug("Preprocessing data matrix")
    maxs = [numpy.max(V[i, :].todense()) for i in range(V.shape[0])]
    
    now = 0
    for row in range(V.shape[0]):
        upto = V.indptr[row+1]
        while now < upto:
            V.data[now] /= maxs[row]
            now += 1
    logging.debug("Finished.")  
    return V, maxs

def readCoauthors(): 
    """
    Take a list of coauthors and read in the complete graph into a sparse 
    matrix R such that R_ij = k means author i has worked with j, k times. Then 
    do matrix factorisation on the resulting methods. 
    """
    matrixFileName = PathDefaults.getOutputDir() + "erasm/R"
    
    numExamples = 100 
    numFolds = 3    
    ranks = numpy.arange(10, 14, 3)
    
    trainErrors = numpy.zeros((ranks.shape[0], numFolds))
    testErrors = numpy.zeros((ranks.shape[0], numFolds))
    
    R = scipy.io.mmread(matrixFileName)
    logging.debug("Loaded matrix " + str(R.shape) + " with " + str(R.getnnz()) + " non zeros")
    R = R.tocsr()
    R = R[0:numExamples ,:]
    R, maxS = preprocess(R)

    #Take out some ratings to form a training set
    rowInds, colInds = R.nonzero()
    randInds = numpy.random.permutation(rowInds.shape[0])
    indexList = Sampling.crossValidation(numFolds, rowInds.shape[0])
    
    for j, (trnIdx, tstIdx) in enumerate(indexList): 
        logging.debug("Fold index " + str(j))
        trainInds = randInds[trnIdx]
        testInds = randInds[tstIdx]
        
        trainR = scipy.sparse.csr_matrix(R.shape)
        for ind in trainInds: 
            trainR[rowInds[ind], colInds[ind]] = R[rowInds[ind], colInds[ind]]
            
        testR = scipy.sparse.csr_matrix(R.shape) 
        for ind in testInds: 
            testR[rowInds[ind], colInds[ind]] = R[rowInds[ind], colInds[ind]]  
        
        for i, rank in enumerate(ranks):
            logging.debug("Rank=" + str(rank))
            model = nimfa.mf(trainR, method = "lsnmf", max_iter = 50, rank = rank, update = 'Euclidean', objective = 'div')
            logging.debug("Performing factorisation") 
            fit = nimfa.mf_run(model)
            logging.debug("Finished")
            sparse_w, sparse_h = fit.fit.sparseness()
        
            W = fit.basis()
            H = fit.coef()
            
            predR = W.dot(H)
            
            #Compute train error  
            for ind in trainInds: 
                trainErrors[i, j] += (trainR[rowInds[ind], colInds[ind]] - predR[rowInds[ind], colInds[ind]])**2
          
            trainErrors[i, j] /= trainInds.shape[0]  
          
            #Compute test error 
            for ind in testInds: 
                testErrors[i, j] += (testR[rowInds[ind], colInds[ind]] - predR[rowInds[ind], colInds[ind]])**2
                
            testErrors[i, j] /= testInds.shape[0]  
            logging.debug(testErrors[i, j])
    
    meanTrainErrors = trainErrors.mean(1)
    meanTestErrors = testErrors.mean(1)
        
    logging.debug("Train errors = " + str(meanTrainErrors))
    logging.debug("Test errors = " + str(meanTestErrors))
    
    plt.plot(ranks, meanTrainErrors)
    plt.plot(ranks, meanTestErrors)
    plt.show()
       
readCoauthors()

#TODO: 
#Try a selection of factorisation methods 
#Parallelism 