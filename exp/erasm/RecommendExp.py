"""
Test some recommendation with the Mendeley coauthor data 
"""

from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Sampling import Sampling 
import nimfa
import numpy 
import matplotlib.pyplot as plt 
from os.path import dirname, abspath, sep
import scipy.io 
import logging 
import sys
import multiprocessing 

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


def factorise(args):
    maxIter = 30
    R, trainInds, testInds, rank, method = args 
    
    logging.debug("Computing factorisation with rank " + str(rank))
    rowInds, colInds = R.nonzero()
    
    trainR = scipy.sparse.csr_matrix(R.shape)
    for ind in trainInds: 
        trainR[rowInds[ind], colInds[ind]] = R[rowInds[ind], colInds[ind]]
        
    testR = scipy.sparse.csr_matrix(R.shape) 
    for ind in testInds: 
        testR[rowInds[ind], colInds[ind]] = R[rowInds[ind], colInds[ind]] 
    
    model = nimfa.mf(trainR, method=method, max_iter=maxIter, rank=rank)
    fit = nimfa.mf_run(model)
    W = fit.basis()
    H = fit.coef()
    
    predR = W.dot(H)
    
    #Compute train error 
    rowInds, colInds = trainR.nonzero()
    trainError = 0 

    for i in range(rowInds.shape[0]): 
        trainError += (trainR[rowInds[i], colInds[i]] - predR[rowInds[i], colInds[i]])**2 
        
    trainError /= rowInds.shape[0]
    
    #Compute test error
    rowInds, colInds = testR.nonzero()
    testError = 0 

    for i in range(rowInds.shape[0]): 
        testError += (testR[rowInds[i], colInds[i]] - predR[rowInds[i], colInds[i]])**2 
        
    testError /= rowInds.shape[0]
    
    return trainError, testError 
    

def recommend(method): 
    """
    Take a list of coauthors and read in the complete graph into a sparse 
    matrix R such that R_ij = k means author i has worked with j, k times. Then 
    do matrix factorisation on the resulting methods. 
    """
    outputDir = PathDefaults.getOutputDir() + "erasm/" 
    matrixFileName = outputDir + "R"
    
    
    numExamples = 1000 
    numFolds = 3    
    ranks = numpy.arange(10, 25, 3)
    
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
    
    paramList = [] 
    for j, (trnIdx, tstIdx) in enumerate(indexList): 
        trainInds = randInds[trnIdx]
        testInds = randInds[tstIdx]
        
        for i, rank in enumerate(ranks):
            paramList.append((R, trainInds, testInds, rank, method))
        
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(factorise, paramList)
    
    k = 0 
    for j in range(len(indexList)): 
        for i in range(len(ranks)):
            trainErrors[i, j], testErrors[i, j] = results[k]
            k += 1 
            
    meanTrainErrors = trainErrors.mean(1)
    meanTestErrors = testErrors.mean(1)
        
    logging.debug("Train errors = " + str(meanTrainErrors))
    logging.debug("Test errors = " + str(meanTestErrors))
    
    errorFileName = outputDir + "results_" + method
    numpy.savez(errorFileName, meanTrainErrors, meanTestErrors)    
    
    """
    plt.plot(ranks, meanTrainErrors, label="Train Error")
    plt.plot(ranks, meanTestErrors, label="Test Error")
    plt.xlabel("Rank")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
    """

methods = ["icm", "lfnmf", "lsnmf", "nmf", "nsnmf" "pmf", "psnmf", "snmf"]      

for method in methods: 
    recommend(method)

#TODO: 
#Try matrix completion 
#Profile 
#Different error methods 
#Save results 