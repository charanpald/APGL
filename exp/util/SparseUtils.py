import numpy 
import scipy.sparse 
from exp.util.SparseUtilsCython import SparseUtilsCython

class SparseUtils(object): 
    def __init__(self): 
        pass 
    
    @staticmethod 
    def generateSparseLowRank(shape, r, k, verbose=False): 
        """
        A method to efficiently generate large sparse matrices of low rank. We 
        use rank r and k indices are sampled from the full matrix to form a 
        sparse one. Returns a scipy csc_matrix. 
        """
        U, s, V = SparseUtils.generateLowRank(shape, r)
        X = SparseUtils.reconstructLowRank(U, s, V, k)        
        
        if verbose: 
            return X, U, s, V
        else: 
            return X 
        
    @staticmethod 
    def generateLowRank(shape, r, sampleVals=500): 
        """
        Return the singular values/vectors of a random matrix with rank r. The 
        resulting matrix has entries within the range [-1, +1].
        """
        (n, m) = shape 

        A = numpy.random.rand(n, r)
        U, R = numpy.linalg.qr(A)
        
        B = numpy.random.rand(m, r)
        V, R = numpy.linalg.qr(B)
        
        #Generate singular values 
        indsU = numpy.unique(numpy.random.randint(0, n, sampleVals)) 
        indsV = numpy.unique(numpy.random.randint(0, m, sampleVals)) 
        X = U[indsU, :].dot(V[indsV, :].T)
        scaling = 1/max(abs(X.min()), X.max())
  
        #s = numpy.ones(r)*scaling
        s = numpy.random.rand(r)*scaling
        
        return U, s, V
        
    @staticmethod 
    def reconstructLowRank(U, s, V, k): 
        """
        Take the SVD of a low rank matrix and partially compute it with at most 
        k values. If k is an array of values [0, U.shape[0]*V.shape[0]] then these 
        indices are used for reconstruction. 
        """
        (n, m) = (U.shape[0], V.shape[0])  
        
        if type(k) == numpy.ndarray: 
            inds = k 
        else: 
            inds = numpy.random.randint(0, n*m, k)
        
        inds = numpy.unique(inds)
        rowInds = inds/m 
        colInds = inds%m  
        
        X = SparseUtilsCython.partialReconstruct2((rowInds, colInds), U, s, V)
        
        return X 