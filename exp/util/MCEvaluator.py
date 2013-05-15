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
        
        nptst.assert_array_equal(testX.nonzero()[0], predX.nonzero()[0])
        nptst.assert_array_equal(testX.nonzero()[1], predX.nonzero()[1])
        
        diff = testX - predX     
        
        if diff.data.shape[0] != 0: 
            return numpy.mean(diff.data**2) 
        else: 
            return 0 
        
    @staticmethod 
    def rootMeanSqError(testX, predX): 
        """
        Find the root mean squared error between two sparse matrices testX and predX. 
        """
        
        return numpy.sqrt(MCEvaluator.meanSqError(testX, predX)) 