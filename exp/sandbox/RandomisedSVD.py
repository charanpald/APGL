import numpy 
from apgl.util.Parameter import Parameter 

class RandomisedSVD(object): 
    """
    Compute the randomised SVD using the algorithm on page 9 of Halko et al., 
    Finding Structure with randomness: stochastic algorithms for constructing 
    approximate matrix decompositions, 2009. 
    """
    def ___init__(self): 
        pass
    
    @staticmethod
    def svd(A, k, q=2): 
        """
        Compute the SVD of a sparse or dense matrix A, finding the first 2k 
        singular vectors/values, using exponent q. Returns the left and right singular 
        vectors, and the singular values. The resulting matrix can be approximated 
        using A ~ U s V.T. 
        """
        Parameter.checkInt(k, 1, float("inf"))
        Parameter.checkInt(q, 1, float("inf"))        
        
        n = A.shape[0]
        omega = numpy.random.randn(n, 2*k)
        Y = A.dot(omega)
        
        for i in range(q): 
            Y = A.T.dot(Y)
            Y = A.dot(Y)
        
        Q, R = numpy.linalg.qr(Y)
        B = A.T.dot(Q).T   
        U, s, V = numpy.linalg.svd(B, full_matrices=False)
        V = V.T
        U = Q.dot(U)
        
        return U, s, V 