import numpy 
import scipy.sparse 
import gc 
from sppy.linalg.GeneralLinearOperator import GeneralLinearOperator
from apgl.util.Parameter import Parameter 
import sppy.linalg 

class RandomisedSVD(object): 
    """
    Compute the randomised SVD using the algorithm on page 9 of Halko et al., 
    Finding Structure with randomness: stochastic algorithms for constructing 
    approximate matrix decompositions, 2009. 
    """
    def ___init__(self): 
        pass
    
    @staticmethod
    def svd(A, k, p=10, q=2, omega=None): 
        """
        Compute the SVD of a sparse or dense matrix A, finding the first k 
        singular vectors/values, using exponent q. Returns the left and right singular 
        vectors, and the singular values. The resulting matrix can be approximated 
        using A ~ U s V.T. 
        
        :param A: A matrix or GeneralLinearOperator 
        
        :param k: The number of singular values and random projections
        
        :param p: The oversampling parameter 
        
        :param q: The exponent for the projections.
        
        :param omega: An initial matrix to perform random projections onto with at least k columns 
        """
        return sppy.linalg.rsvd(A, k, p, q, omega)
      
    @staticmethod 
    def updateSvd(A, U, s, V, E, k, p): 
        """
        Given a matrix A whose approximate SVD is U s V.T, compute the SVD 
        of the new matrix A + E, using previous info. A and E are sparse 
        matrices. The rank of the approximation is p, and k is an oversampling
        parameter. 
        """
        Parameter.checkInt(k, 1, float("inf"))
        Parameter.checkInt(p, 0, float("inf"))     
                    
        if isinstance(E, GeneralLinearOperator): 
            M = E 
        else: 
            M = GeneralLinearOperator.asLinearOperator(E) 
            
        N = GeneralLinearOperator.asLinearOperator(A + E)
        
        n = A.shape[1]
        omega = numpy.random.randn(n, p)
        
        Y = U*s + M.matmat(V)
        Y = numpy.c_[Y, N.matmat(omega)]
        
        Q, R = numpy.linalg.qr(Y)
        del omega 
            
        del Y 
        del R 
        gc.collect() 
        
        B = N.rmatmat(Q).T 
        U, s, V = numpy.linalg.svd(B, full_matrices=False)
        del B 
        V = V.T
        U = Q.dot(U)
    
        U = U[:, 0:k]
        s = s[0:k]
        V = V[:, 0:k]        
        
        return U, s, V 
        
        