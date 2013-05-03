
"""
Start with a simple toy dataset with time-varying characteristics 
"""

import numpy 
from exp.util.SparseUtils import SparseUtils 

class SyntheticDataset1(object): 
    def __init__(self): 
        """
        This function returns a list of 20 train/test matrices for incremental 
        collaborative filtering. Each item in the list is (trainX, testX).
        
        Could add noise to reconstruction 
        """    
        numpy.random.seed(21)    
        startM = 5000 
        startN = 10000 
        endM = 6000
        endN = 12000
        r = 150 
        
        U, s, V = SparseUtils.generateLowRank((endM, endN), r)
        
        startNumInds = 9000
        endNumInds = 12000
        inds = numpy.random.randint(0, startM*startN-1, endNumInds)
        inds = numpy.unique(inds)
        numpy.random.shuffle(inds)
        endNumInds = inds.shape[0]
        
        trainSplit = 2.0/3 
        trainInds = inds[0:inds.shape[0]*trainSplit]
        testInds = inds[inds.shape[0]*trainSplit:]
        
        trainXList = []
        testXList = []    
        
        #In the first phase, the matrices stay the same size but there are more nonzero 
        #entries   
        numMatrices = 10 
        stepList = numpy.linspace(startNumInds*trainSplit, endNumInds*trainSplit, numMatrices)
        
        for i in range(numMatrices): 
            trainX = SparseUtils.reconstructLowRank(U, s, V, trainInds[0:stepList[i]])
            trainX = trainX[0:startM, :][:, 0:startN]
            
            testX =  SparseUtils.reconstructLowRank(U, s, V, testInds)  
            testX = testX[0:startM, :][:, 0:startN]
            trainXList.append(trainX)
            testXList.append(testX)
            
        #Now we increase the size of matrix 
        numMatrices = 10 
        mStepList = numpy.linspace(startM, endM, numMatrices)
        nStepList = numpy.linspace(startN, endN, numMatrices)
    
        for i in range(numMatrices): 
            trainX = SparseUtils.reconstructLowRank(U, s, V, trainInds)
            trainX = trainX[0:mStepList[i], :][:, 0:nStepList[i]]
            
            testX =  SparseUtils.reconstructLowRank(U, s, V, testInds)  
            testX = testX[0:mStepList[i], :][:, 0:nStepList[i]]
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
    
