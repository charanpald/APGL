"""
Test some recommendation with the Mendeley coauthor data 
"""

from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Sampling import Sampling 
from apgl.util.SparseUtils import SparseUtils 
from exp.sandbox.recommendation.AbstractMatrixCompleter import computeTestError
from exp.sandbox.recommendation.NimfaFactorise import NimfaFactorise 
from exp.sandbox.recommendation.SoftImpute import SoftImpute 
import numpy 
import matplotlib.pyplot as plt 
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

def recommend(learner): 
    """
    Take a list of coauthors and read in the complete graph into a sparse 
    matrix X such that X_ij = k means author i has worked with j, k times. Then 
    do matrix factorisation on the resulting methods. 
    """
    outputDir = PathDefaults.getOutputDir() + "erasm/" 
    matrixFileName = outputDir + "Toy"
    
    numExamples = 50 
    numFolds = 5    
      
    X = scipy.io.mmread(matrixFileName)
    X = scipy.sparse.csr_matrix(X)
    logging.debug("Loaded matrix " + str(X.shape) + " with " + str(X.getnnz()) + " non zeros")
    X = X.tocsr()
    X = X[0:numExamples ,:]
    X, maxS = preprocess(X)

    #Take out some ratings to form a training set
    rowInds, colInds = X.nonzero()
    randInds = numpy.random.permutation(rowInds.shape[0])
    indexList = Sampling.crossValidation(numFolds, rowInds.shape[0])
    
    paramList = [] 
    for j, (trnIdx, tstIdx) in enumerate(indexList): 
        trainInds = randInds[trnIdx]
        testInds = randInds[tstIdx]
        
        trainX = SparseUtils.selectMatrix(X, rowInds[trainInds], colInds[trainInds]).tocsr()
        testX = SparseUtils.selectMatrix(X, rowInds[testInds], colInds[testInds]).tocsr()
        
        paramList.append((trainX, testX, learner))
        
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(computeTestError, paramList)
    #results = map(computeTestError, paramList)
    
    testErrors = numpy.array(results)
    meanTestErrors = testErrors.mean()
    logging.debug("Test errors = " + str(meanTestErrors))
    
    errorFileName = outputDir + "results_" + learner.name()
    numpy.savez(errorFileName, meanTestErrors)   
    logging.debug("Saved results as " + errorFileName)
    
nimfaFactorise = NimfaFactorise("lsnmf")
lmbdas = numpy.array([0.1])
softImpute = SoftImpute(lmbdas)

learners = [softImpute, nimfaFactorise]      

for learner in learners: 
    recommend(learner)
    #plotResults(method)

#plt.show()

#TODO: 
#Profile 
#Different error methods 
#Save results 