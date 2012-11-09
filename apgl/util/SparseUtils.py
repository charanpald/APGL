


"""
Extra methods for scipy sparse matrices 
"""

import numpy
import scipy.sparse

class SparseUtils(object):
    @staticmethod
    def equals(A, B):
        """
        Test if matrices A and B are identical.
        """
        if A.data.shape[0] != B.data.shape[0]:
            return False 
        if (A.data != B.data).any():
            return False
        if hasattr(A, 'indices')  and hasattr(B, 'indices')  and (A.indices != B.indices).any():
            return False
        if hasattr(A, 'rows')  and hasattr(B, 'rows')  and (A.rows != B.rows).any():
            return False

        return True

    @staticmethod
    def diag(X):
        """
        Find the diagonal of a sparse matrix and return as a numpy array. 
        """
        d = numpy.zeros(X.shape[0])

        for i in range(X.shape[0]):
            d[i] = X[i, i]

        return d

    @staticmethod
    def norm(X):
        """
        Find the frobenius norm of a sparse matrix X
        """
        elements = X.data
        return numpy.sqrt((elements**2).sum()) 
        
    @staticmethod
    def resize(X, shape): 
        """
        Resize a sparse matrix to the given shape, padding with zero if required. 
        """
        Y = scipy.sparse.csr_matrix((shape))
        rows, cols = X.nonzero()
        
        for ind in range(rows.shape[0]):
            i = rows[ind]
            j = cols[ind]
            if i < Y.shape[0] and j < Y.shape[1]: 
                Y[i, j] = X[i, j]
        
        return Y 
        
        