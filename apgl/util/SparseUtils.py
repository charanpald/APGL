


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
        if (A.indices != B.indices).any():
            return False

        return True

    @staticmethod
    def diag(X):
        """
        Find the diagonal of a lil_matrix
        """
        d = scipy.sparse.lil_matrix((X.shape[0], 1))

        for i in range(0, X.shape[0]):
            d[i, 0] = X[i, i]

        return d

    @staticmethod
    def norm(X):
        """
        Find the frobenius norm of a sparse matrix X
        """
        elements = X.data
        return numpy.sqrt((elements**2).sum()) 
        