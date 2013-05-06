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
        Compute the SVD of a sparse or dense matrix A, finding the first k 
        singular vectors/values, using exponent q. Returns the left and right singular 
        vectors, and the singular values. The resulting matrix can be approximated 
        using A ~ U s V.T. 
        """
        Parameter.checkInt(k, 1, float("inf"))
        Parameter.checkInt(q, 1, float("inf"))        
        
        n = A.shape[0]
        omega = numpy.random.randn(n, k)
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
        
    def svd2(A, lmbda): 
        """
        Compute the randomised SVD with error estimate up to a tolerance of 
        lmbda, i.e. ||A - QQ^TA||_2 <= lmbda. 
        """ 
        Parameter.checkFloat(lmbda, 0.0, float("inf"))
        r = 10 

        m, n = A.shape        
        Omega = numpy.random.randn(n, k)
        Y = A.dot(Omega)
        
        Q = numpy.zeros((m, 0))
        
        i = 0 

        while numpy.max((Y**2).sum(0)) > eps/(10*numpy.sqrt(2/numpy.pi)): 
            Y[:, i] = Y[:, i] - Q.dot(Q.T).dot(Y[:, i])
            q = Y[:, i]
            Q = numpy.c_[Q, q]
            
            omega = numpy.random.randn(n) 
        
        