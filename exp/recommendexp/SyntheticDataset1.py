
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
        pass 
    
    def generateMatrices(self):
        """
        This function returns a list of 20 train/test matrices for incremental 
        collaborative filtering. Each item in the list is (trainX, testX).
        """    
        numpy.random.seed(21)    
        startM = 500 
        startN = 1000 
        endM = 600
        endN = 1200
        r = 50 
        
        noise = 0.2
        U, s, V = SparseUtils.generateLowRank((endM, endN), r)
        
        startNumInds = 90000
        endNumInds = 120000
        inds = numpy.random.randint(0, endM*endN-1, endNumInds)
        inds = numpy.unique(inds)
        numpy.random.shuffle(inds)
        endNumInds = inds.shape[0]
        
        rowInds, colInds = numpy.unravel_index(inds, (endM, endN))
        vals = SparseUtilsCython.partialReconstructVals(rowInds, colInds, U, s, V)
        vals /= vals.std()
        vals +=  numpy.random.randn(vals.shape[0])*noise
        
        trainSplit = 2.0/3 
        isTrainInd = numpy.array(numpy.random.rand(inds.shape[0]) <= trainSplit, numpy.bool)
        
        assert (trainSplit - isTrainInd.sum()/float(isTrainInd.shape[0]))
        
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
            
            trainX = X.multiply(XMaskTrain)[0:startM, 0:startN]
            trainX.eliminate_zeros()
            trainX.prune() 
            
            testX = X.multiply(XMaskTest)[0:startM, 0:startN]
            testX.eliminate_zeros()
            testX.prune() 
            
            trainXList.append(trainX)
            testXList.append(testX)
            
        #Now we increase the size of matrix 
        numMatrices = 10 
        mStepList = numpy.linspace(startM, endM, numMatrices)
        nStepList = numpy.linspace(startN, endN, numMatrices)
    
        X = scipy.sparse.csc_matrix((vals, (rowInds, colInds)), dtype=numpy.float, shape=(endM, endN))
    
        for i in range(numMatrices): 
            trainX = X.multiply(XMaskTrain)[0:mStepList[i], :][:, 0:nStepList[i]]
            trainX.eliminate_zeros()
            trainX.prune() 
            
            testX = X.multiply(XMaskTest)[0:mStepList[i], :][:, 0:nStepList[i]]
            testX.eliminate_zeros()
            testX.prune() 
            
            trainXList.append(trainX)
            testXList.append(testX)
                    
        return trainXList, testXList
        
    def getTrainIteratorFunc(self):
        def trainIteratorFunc(): 
            trainXList, testXList = self.generateMatrices()       
            return iter(trainXList)
        
        return trainIteratorFunc
        
    def getTestIteratorFunc(self):
        def testIteratorFunc(): 
            trainXList, testXList = self.generateMatrices()   
            return iter(testXList)
        
        return testIteratorFunc
