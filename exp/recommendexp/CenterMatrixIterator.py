import gc
import logging 
from exp.util.SparseUtils import SparseUtils 


class CenterMatrixIterator(object): 
    """
    Takes an iterator of matrices and generates a new centered set of matrices. 
    Note that the original matrices are changed in-place. 
    """
    def __init__(self, iterator): 
        self.iterator = iterator  
        self.i = 0
    
    def next(self): 
        X = next(self.iterator)
        logging.debug("Centering train matrix of size: " + str(X.shape))
        tempRowInds, tempColInds = X.nonzero()    
        X, self.muRows = SparseUtils.centerRows(X)
        X.eliminate_zeros()
        X.prune() 
        gc.collect()
        
        self.i += 1 
        
        return X 

    def __iter__(self): 
        return self
     
    def centerMatrix(self, X):
        """
        Center a test matrix given we have already centered the training one. 
        """
        logging.debug("Centering test matrix of size: " + str(X.shape))
        tempRowInds, tempColInds = X.nonzero()    
        X, muRows = SparseUtils.centerRows(X, self.muRows)
        X.eliminate_zeros()
        X.prune() 
        
        return X     
        
    def uncenter(self, X): 
        """
        Uncenter a training or test matrix. 
        """
        #logging.debug("Uncentering matrix of size: " + str(X.shape))
        return SparseUtils.uncenterRows(X, self.muRows)
        