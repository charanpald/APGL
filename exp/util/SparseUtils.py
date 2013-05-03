import numpy 
import scipy.sparse 
import scipy.sparse.linalg 
from sparsesvd import sparsesvd
from exp.util.SparseUtilsCython import SparseUtilsCython
from apgl.util.Util import Util 
from pypropack import svdp

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
        (m, n) = (U.shape[0], V.shape[0])  
        
        if type(k) == numpy.ndarray: 
            inds = k 
            inds = numpy.unique(inds)
            rowInds, colInds = numpy.unravel_index(inds, (m, n))
        elif type(k) == tuple: 
            rowInds, colInds = k 
            rowInds = numpy.array(rowInds, numpy.int)
            colInds = numpy.array(colInds, numpy.int)
        else: 
            inds = numpy.random.randint(0, n*m, k)
            inds = numpy.unique(inds)
            rowInds, colInds = numpy.unravel_index(inds, (m, n))

        X = SparseUtilsCython.partialReconstruct2((rowInds, colInds), U, s, V)
        
        return X 
        
    @staticmethod
    def svdSparseLowRank(X, U, s, V, k=10): 
        """
        Find the partial SVD of a matrix A = X + U s V.T in which X is sparse and B = 
        U s V.T is a low rank matrix. We use PROPACK to find the first k 
        eigenvectors/values. 
        """
        
        def matvec(v): 
            return X.dot(v) + (U*s).dot(V.T.dot(v)) 
        
        def rmatvec(v): 
            return X.T.dot(v) + (V*s).dot(U.T.dot(v))
        
        L = scipy.sparse.linalg.LinearOperator(X.shape, matvec, rmatvec) 
        U, s, V = svdp(L, k, kmax=30*k)
        
        return U, s, V.T 
        

    @staticmethod 
    def svdSoft(X, lmbda, k): 
        """
        Take the soft-thresholded SVD of sparse matrix X under threshold lmbda for the first 
        k singular values and vectors. Returns the left and right singular 
        vectors and the singular values. 
        """
        if not scipy.sparse.issparse(X): 
            raise ValueError("X must be a sparse matrix")
        
        U, s, V = sparsesvd(X, k) 
        U = U.T
        inds = numpy.flipud(numpy.argsort(s))
        U, s, V = Util.indSvd(U, s, V, inds) 
        
        #Soft threshold 
        s = s - lmbda
        s = numpy.clip(s, 0, numpy.max(s))

        return U, s, V 
      
    @staticmethod 
    def svdSoft2(X, U, s, V, lmbda, k): 
        """
        Compute the SVD of a matrix X + UsV^T where X is sparse and U s V^T is 
        low rank. The first k singular values are computed and then soft thresholded 
        with threshold lmbda. 
        """
        U2, s2, V2 = SparseUtils.svdSparseLowRank(X, U, s, V, k)
        
        #Soft threshold 
        s2 = s2 - lmbda
        s2 = numpy.clip(s2, 0, numpy.max(s2))
        
        return U2, s2, V2 