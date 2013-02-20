from apgl.util.Parameter import Parameter 
import numpy 
import multiprocessing
import itertools 

class AbstractMatrixCompleter(object): 
    def __init__(self): 
        pass 
    
    def parallelModelSelect(self, X, idx, paramDict):
        """
        Perform parallel model selection using any learner. 
        Using the best set of parameters train using the whole dataset.

        :param X: The matrix to complete 
        :type X: :class:`scipy.sparse.csr_matrix`

        :param idx: A list of train/test splits
        
        :param paramDict: A dictionary index by the method name and with value as an array of values
        :type X: :class:`dict`
        """
        folds = len(idx)

        gridSize = [] 
        gridInds = [] 
        for key in paramDict.keys(): 
            gridSize.append(paramDict[key].shape[0])
            gridInds.append(numpy.arange(paramDict[key].shape[0])) 
            
        meanErrors = numpy.zeros(tuple(gridSize))
        m = 0
        paramList = []
        
        for trainInds, testInds in idx:
            trainX, trainY = X[trainInds, :], y[trainInds]
            testX, testY = X[testInds, :], y[testInds]
            
            indexIter = itertools.product(*gridInds)
            
            for inds in indexIter: 
                learner = self.copy()     
                currentInd = 0             
            
                for key, val in paramDict.items():
                    method = getattr(learner, key)
                    method(val[inds[currentInd]])
                    currentInd += 1                    
                
                paramList.append((trainX, trainY, testX, testY, learner))
            
            m += 1 
            
        pool = multiprocessing.Pool(processes=self.processes, maxtasksperchild=100)
        resultsIterator = pool.imap(computeTestError, paramList, self.chunkSize)
        
        for trainInds, testInds in idx:
            indexIter = itertools.product(*gridInds)
            for inds in indexIter: 
                error = resultsIterator.next()
                meanErrors[inds] += error/float(folds)

        pool.terminate()

        learner = self.getBestLearner(meanErrors, paramDict, X, y, idx)

        return learner, meanErrors
        