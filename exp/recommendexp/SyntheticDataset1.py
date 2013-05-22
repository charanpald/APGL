
"""
Start with a simple toy dataset with time-varying characteristics 
"""

import numpy 
import logging
import scipy.sparse 
from exp.util.SparseUtils import SparseUtils 
from exp.util.SparseUtilsCython import SparseUtilsCython

class SyntheticDataset1(object): 
    def __init__(self): 
        """
        This function returns a list of 20 train/test matrices for incremental 
        collaborative filtering. Each item in the list is (trainX, testX).
        """    
        numpy.random.seed(21)    
        startM = 5000 
        startN = 10000 
        endM = 6000
        endN = 12000
        r = 150 
        
        noise = 0.1
        U, s, V = SparseUtils.generateLowRank((endM, endN), r)
        
        startNumInds = 9000
        endNumInds = 12000
        inds = numpy.random.randint(0, startM*startN-1, endNumInds)
        inds = numpy.unique(inds)
        numpy.random.shuffle(inds)
        endNumInds = inds.shape[0]
        
        rowInds, colInds = numpy.unravel_index(inds, (endM, endN))
        vals = SparseUtilsCython.partialReconstructVals(rowInds, colInds, U, s, V)
        vals +=  numpy.random.randn(vals.shape[0])*noise 
        
        trainSplit = 2.0/3 
        isTrainInd = numpy.array(numpy.random.rand(inds.shape[0]) <= trainSplit, numpy.bool)
        XMaskTrain = scipy.sparse.csc_matrix((isTrainInd, (rowInds, colInds)), dtype=numpy.bool, shape=(endM, endN)) 
        XMaskTest = scipy.sparse.csc_matrix((numpy.logical_not(isTrainInd), (rowInds, colInds)), dtype=numpy.bool, shape=(endM, endN))

        #In the first phase, the matrices stay the same size but there are more nonzero 
        #entries   
        numMatrices = 10 
        stepList = numpy.linspace(startNumInds, endNumInds, numMatrices) 
        trainXList = []
        testXList = []    
        
        for i in range(numMatrices):  
            currentVals = vals[0:stepList[i]]
            currentRowInds = rowInds[0:stepList[i]]
            currentColInds = colInds[0:stepList[i]]
            
            X = scipy.sparse.csc_matrix((currentVals, (currentRowInds, currentColInds)), dtype=numpy.float, shape=(endM, endN))
            
            logging.debug("Centering rows and cols of X with shape " + str(X.shape))
            tempRowInds, tempColInds = X.nonzero()
            X, muRows = SparseUtils.centerRows(X)
            X, muCols = SparseUtils.centerCols(X, inds=(tempRowInds, tempColInds))   
    
            trainX = X.multiply(XMaskTrain)[0:startM, 0:startN]
            testX = X.multiply(XMaskTest)[0:startM, 0:startN]
            
            trainXList.append(trainX)
            testXList.append(testX)
            
        #Now we increase the size of matrix 
        numMatrices = 10 
        mStepList = numpy.linspace(startM, endM, numMatrices)
        nStepList = numpy.linspace(startN, endN, numMatrices)
    
        X = scipy.sparse.csc_matrix((vals, (rowInds, colInds)), dtype=numpy.float, shape=(endM, endN))
        
        logging.debug("Centering rows and cols of X with shape " + str(X.shape))
        rowInds, colInds = X.nonzero()
        X, muRows = SparseUtils.centerRows(X)
        X, muCols = SparseUtils.centerCols(X, inds=(rowInds, colInds))       
    
        for i in range(numMatrices): 
            trainX = X.multiply(XMaskTrain)[0:mStepList[i], :][:, 0:nStepList[i]]
            testX = X.multiply(XMaskTest)[0:mStepList[i], :][:, 0:nStepList[i]]
            
            trainXList.append(trainX)
            testXList.append(testX)
            
        self.trainXList = trainXList
        self.testXList = testXList 
        
    def getTrainIteratorFunc(self):
        def trainIteratorFunc(): 
            return iter(self.trainXList)
        
        return trainIteratorFunc
        
    def getTestIteratorFunc(self):
        def testIteratorFunc(): 
            return iter(self.testXList)
        
        return testIteratorFunc
