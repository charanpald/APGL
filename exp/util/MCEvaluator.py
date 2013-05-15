import numpy 
import numpy.testing as nptst 

class MCEvaluator(object):
    """
    A class to evaluate machine learning performance for the matrix completion
    problem.
    """
    def __init__(self):
        pass
    
    @staticmethod 
    def meanSqError(testX, predX): 
        """
        Find the mean squared error between two sparse matrices testX and predX. 
        Note that the matrices must have nonzero elements in the same places. 
        """
        #Note that some predictions might be zero 
        assert numpy.in1d(predX.nonzero()[0], testX.nonzero()[0]).all() 
        assert numpy.in1d(predX.nonzero()[1], testX.nonzero()[1]).all() 
        
        diff = testX - predX     
        error = numpy.sum(diff.data**2)/testX.data.shape[0]
        return error

        
    @staticmethod 
    def rootMeanSqError(testX, predX): 
        """
        Find the root mean squared error between two sparse matrices testX and predX. 
        """
        
        return numpy.sqrt(MCEvaluator.meanSqError(testX, predX)) 