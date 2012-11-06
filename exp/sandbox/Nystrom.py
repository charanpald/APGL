import numpy
import logging
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
from apgl.util.Util import Util 

class Nystrom(object):
    """
    A class to find approximations based on the Nystrom method.
    """

    def __init__(self):
        pass

    @staticmethod
    def eig(X, n):
        """
        Find the eigenvalues and eigenvectors of an indefinite symmetric matrix X.

        :param X: The matrix to find the eigenvalues of.
        :type X: :class:`ndarray`

        :param n: If n is an int, then it is the number of columns to sample otherwise n is an array of column indices. 
        """
        logging.warn("This method can result in large errors with indefinite matrices")

        if type(n) == int:
            inds = numpy.sort(numpy.random.permutation(X.shape[0])[0:n])
        else:
            inds = n
        invInds = numpy.setdiff1d(numpy.arange(X.shape[0]), inds)

        A = X[inds, :][:, inds]
        B = X[inds, :][:, invInds]
        
        Am12 = numpy.linalg.inv(scipy.linalg.sqrtm(A))
        #Am12 = Util.matrixPowerh(A, -0.5)
        S = A + Am12.dot(B).dot(B.T).dot(Am12)

        lmbda, U = numpy.linalg.eig(S)
        Ubar = numpy.r_[U, B.T.dot(U).dot(numpy.diag(1/lmbda))]
        Z = Ubar.dot(numpy.diag(lmbda**0.5))
        sigma, F = numpy.linalg.eig(Z.T.dot(Z))
        V = Z.dot(F).dot(numpy.diag(sigma**-0.5))

        return sigma, V

    @staticmethod
    def eigpsd(X, n):
        """
        Find the eigenvalues and eigenvectors of a positive semi-definite symmetric matrix.
        The input matrix X can be a numpy array or a scipy sparse matrix. In the case that
        n==X.shape[0] we convert to an ndarray. 

        :param X: The matrix to find the eigenvalues of.
        :type X: :class:`ndarray`

        :param n: If n is an int, then it is the number of columns to sample otherwise n is an array of column indices.

        :return lmbda: The set of eigenvalues 
        :return V: The matrix of eigenvectors as a ndarray
        """
        if type(n) == int:
            n = min(n, X.shape[0])
            inds = numpy.sort(numpy.random.permutation(X.shape[0])[0:n])
        elif type(n) == numpy.ndarray:
            inds = n 
        else: 
            raise ValueError("Invalid n value: " + str(n))
            
        invInds = numpy.setdiff1d(numpy.arange(X.shape[0]), inds)

        if numpy.sort(inds).shape[0] == X.shape[0] and (numpy.sort(inds) == numpy.arange(X.shape[0])).all():
            if scipy.sparse.issparse(X):
                X = numpy.array(X.todense())
            lmbda, V = Util.safeEigh(X)
            return lmbda, V

        tmp = X[inds, :] 
        A = tmp[:, inds]
        B = tmp[:, invInds]

        if scipy.sparse.issparse(X): 
            A = numpy.array(A.todense())
            BB = numpy.array((B*B.T).todense())
        else:
            BB = B.dot(B.T)
        
        #Following line is very slow 
        #Am12 = scipy.linalg.sqrtm(numpy.linalg.pinv(A)) 
        Am12 = Util.matrixPowerh(A, -0.5)
        S = A + Am12.dot(BB).dot(Am12)
        S = (S.T + S)/2

        lmbda, U = Util.safeEigh(S)

        tol = 10**-10
        lmbdaN = lmbda.copy()
        lmbdaN[numpy.abs(lmbda) < tol] = 0
        lmbdaN[numpy.abs(lmbda) > tol] = lmbdaN[numpy.abs(lmbda) > tol]**-0.5
        
        V = X[:, inds].dot(Am12.dot(U)*lmbdaN)
        
        return lmbda, V

    @staticmethod 
    def matrixApprox(X, n):
        """
        Compute the matrix approximation using the Nystrom method.

        :param X: The matrix to approximate.
        :type X: :class:`ndarray`

        :param n: If n is an int, then it is the number of columns to sample otherwise n is an array of column indices.
        """
        if type(n) == int:
            inds = numpy.sort(numpy.random.permutation(X.shape[0])[0:n])
        else:
            inds = n

        A = X[inds, :][:, inds]
        B = X[:, inds]

        if scipy.sparse.issparse(X):
            A = numpy.array(A.todense())
            Ainv = scipy.sparse.csr_matrix(numpy.linalg.pinv(A))
            XHat = B.dot(Ainv).dot(B.T)
        else:
            XHat = B.dot(numpy.linalg.pinv(A)).dot(B.T)
        return XHat 
