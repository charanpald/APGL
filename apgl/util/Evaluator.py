

import numpy
from apgl.util.Parameter import Parameter 
#TODO: Test this file 

class Evaluator(object):
    """
    A class to evaluate machine learning performance.
    """
    def __init__(self):
        pass

    @staticmethod 
    def evaluateBinary1DLabels(testY, predY):
        numEvaluations = 6
        evaluations = numpy.zeros(numEvaluations)

        evaluations[0] = Evaluator.binaryError(testY, predY)
        #evaluations[1] = mlpy.sens(testY, predY)
        #evaluations[2] = mlpy.spec(testY, predY)
        evaluations[3] = Evaluator.binaryErrorP(testY, predY)
        evaluations[4] = Evaluator.binaryErrorN(testY, predY)
        evaluations[5] = Evaluator.balancedError(testY, predY)

        return evaluations
		
    @staticmethod
    def balancedError(testY, predY):
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")

        return 0.5*(Evaluator.binaryErrorP(testY, predY)+Evaluator.binaryErrorN(testY, predY))

    @staticmethod
    def weightedRootMeanSqError(testY, predY):
        """
        Weighted root mean square error.
        """
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")

        alpha = 1.0 
        w = numpy.exp(alpha * testY)

        return numpy.linalg.norm((testY - predY)*numpy.sqrt(w))/numpy.sqrt(testY.shape[0])

    @staticmethod
    def rootMeanSqError(testY, predY):
        """
        This is the error given by sqrt{1//y.shape sum_i (py - y)^2} 
        """
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")

        return numpy.linalg.norm(testY - predY)/numpy.sqrt(testY.size)
    
    @staticmethod
    def meanAbsError(testY, predY): 
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")
            
        return numpy.abs(testY - predY).mean()
    
    @staticmethod
    def meanSqError(testY, predY):
        """
        This is the error given by the mean squared difference between examamples 
        """
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")

        return ((testY - predY)**2).mean()

    @staticmethod
    def evaluateWindowError(D, windowSize, pairIndices):
        """
        The input is a matrix D of distances between examples such that
        D_ij = d(x_i, x_j). The aim is to match each example to the corresponding
        pair based on ranking in order of their distance. An error is
        counted if the given item in the pair is not in the window. 
        """
        if D.shape[0]!=D.shape[1]: 
            raise ValueError("D must be a square and symmetric matrix")

        numExamples = D.shape[0]
        numPairs = numExamples/2

        error = 0 

        for i in pairIndices[:, 0]:
            windowInds = numpy.argsort(D[i, :])[0:windowSize]
            error = error + (windowInds != pairIndices[i, 1]).all()

        return float(error)/numPairs

    @staticmethod
    def binaryError(testY, predY):
        """
        Work out the error on a set of -1/+1 labels
        """
        Parameter.checkClass(testY, numpy.ndarray)
        Parameter.checkClass(predY, numpy.ndarray)
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")

        error = numpy.sum(testY != predY)/float(predY.shape[0]) 
        return error

    @staticmethod
    def binaryBootstrapError(testY, predTestY, trainY, predTrainY, weight):
        """
        Evaluate an error in conjunction with a bootstrap method by computing
        w*testErr + (1-w)*trainErr
        """
        Parameter.checkFloat(weight, 0.0, 1.0)

        return weight*Evaluator.binaryError(testY, predTestY) + (1-weight)*Evaluator.binaryError(trainY, predTrainY)

    @staticmethod
    def binaryErrorP(testY, predY):
        """
        Work out the error on a set of -1/+1 labels
        """
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")

        posInds = (testY == 1)

        if testY[posInds].shape[0] != 0:
            error = numpy.sum(numpy.abs(testY[posInds] - predY[posInds]))/(2.0*testY[posInds].shape[0])
        else:
            error = 0.0
            
        return error

    @staticmethod
    def binaryErrorN(testY, predY):
        """
        Work out the error on a set of -1/+1 labels
        """
        if testY.shape[0] != predY.shape[0]:
            raise ValueError("Labels vector much be same dimensions as predicted labels")

        negInds = (testY == -1)

        if testY[negInds].shape[0] != 0:
            error = numpy.sum(numpy.abs(testY[negInds] - predY[negInds]))/(2.0*testY[negInds].shape[0])
        else:
            error = 0.0

        return error

    @staticmethod
    def auc2(trueY, predY):
        return Evaluator.auc(predY, trueY)

    @staticmethod
    def auc(predY, trueY):
        """
        Can be used in conjunction with evaluateCV using the scores, and true
        labels. Note the order of parameters. 
        """
        try:
            import sklearn.metrics 
        except ImportError:
            raise

        Parameter.checkClass(predY, numpy.ndarray)
        Parameter.checkClass(trueY, numpy.ndarray)
        if predY.ndim != 1:
            raise ValueError("Expecting predY to be 1D")
        if trueY.ndim != 1:
            raise ValueError("Expecting trueY to be 1D")
        if numpy.unique(trueY).shape[0] > 2:
            raise ValueError("Found more than two label types in trueY")

        if numpy.unique(trueY).shape[0] == 1:
            return 0.5

        fpr, tpr, threshold = sklearn.metrics.roc_curve(trueY.ravel(), predY.ravel())
        return sklearn.metrics.metrics.auc(fpr, tpr)

    @staticmethod
    def roc(testY, predY):
        try:
            import sklearn.metrics 
        except ImportError:
            raise

        if numpy.unique(testY).shape[0] == 1:
            fpr = numpy.array([])
            tpr = numpy.array([])
        else:
            fpr, tpr, threshold = sklearn.metrics.roc_curve(testY.ravel(), predY.ravel())

        #Insert 0,0 at the start of fpr and tpr
        if fpr[0] != 0.0 or tpr[0] != 0.0:
            fpr = numpy.insert(fpr, 0, 0)
            tpr = numpy.insert(tpr, 0, 0)
        return (fpr, tpr)

    @staticmethod
    def localAuc(testY, predY, u):
        """
        Compute the local AUC measure for a given ROC curve. The parameter u is
        the proportion of best instances to use u = P(s(X) > t).

        """
        Parameter.checkFloat(u, 0.0, 1.0)
        fpr, tpr = Evaluator.roc(testY, predY)

        minExampleIndex = numpy.floor((predY.shape[0]-1)*u)
        minExampleScore = numpy.flipud(numpy.sort(predY))[minExampleIndex]
        intersectInd = numpy.searchsorted(numpy.sort(numpy.unique(predY)), minExampleScore)
        intersectInd = numpy.unique(predY).shape[0] - intersectInd

        alpha = fpr[intersectInd]
        beta = tpr[intersectInd]

        localAuc = numpy.sum(0.5*numpy.diff(fpr[0:intersectInd])*(tpr[0:max(intersectInd-1, 0)] + tpr[1:intersectInd]))
        localAuc += beta*(1-alpha)

        return localAuc

    @staticmethod 
    def precisionFromIndLists(testList, predList): 
        """
        Measure the precision of a predicted list given the true list. The precision is 
        |relevant items \cup retrieved items| / |retrieved items|. The items of the 
        lists are indices. 
        """
        import sklearn.metrics 
        n  = max(numpy.max(testList), numpy.max(predList))+1
        
        predY = -1*numpy.ones(n)
        predY[predList] = 1
        
        testY = -1*numpy.ones(n)
        testY[testList] = 1
        
        return sklearn.metrics.precision_score(testY, predY)
        
    @staticmethod 
    def averagePrecisionFromLists(testList, predList, k=100):

        """
        Computes the average precision at k. Borrowed from https://github.com/benhamner/Metrics. 
        
        This function computes the average prescision at k between two lists of
        items.
        
        Parameters
        ----------
        testList : list
        A list of elements that are to be predicted (order doesn't matter)
        predList : list
        A list of predicted elements (order does matter)
        k : int, optional
        The maximum number of predicted elements
        
        Returns
        -------
        score : double
        The average precision at k over the input lists
        
        """
        if len(predList)>k:
            predList = predList[:k]
    
        score = 0.0
        num_hits = 0.0
    
        for i,p in enumerate(predList):
            if p in testList and p not in predList[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
    
        if not testList:
            return 1.0
    
        return score / min(len(testList), k)
      
    @staticmethod 
    def ndcg(testY, predY, n): 
        """
        Compute the Normalised Discounted Cumulative Gain at N. The relevance 
        of the items in testY is 1 otherwise the item has relevance 0. 
        
        :param testY: A partial list of indices 
        
        :param predY: A list of predicted indices
        """       
        raise ValueError("Method not implemented completely")
        
        testY = testY[0:n]
        predY = predY[0:n]
        
        m = max(numpy.max(testY), numpy.max(predY))+1
        
        rel = numpy.zeros(m)
        rel[predY] = 1
        dcg = rel[0] + rel[1:]/numpy.log2(numpy.arange(2, m+1, dtype=numpy.float))
        dcg = dcg.sum()
        
        rel = numpy.zeros(m)
        rel[testY] = 1
        dcg2 = rel[0] + rel[1:]/numpy.log2(numpy.arange(2, m+1, dtype=numpy.float))
        dcg2 = dcg2.sum()
        
        return dcg/dcg2