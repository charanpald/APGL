import os 
import sys 
import numpy 
import scipy.sparse 
import scipy.sparse.linalg 
from scipy.sparse.linalg import LinearOperator  
import logging 
from exp.util.SparseUtilsCython import SparseUtilsCython
from apgl.util.Util import Util 
from exp.util.LinOperatorUtils import LinOperatorUtils 
from pypropack import svdp

class SparseUtils(object): 
    def __init__(self): 
        pass 
    
    @staticmethod 
    def generateSparseLowRank(shape, r, k, noise=0.0, verbose=False): 
        """
        A method to efficiently generate large sparse matrices of low rank. We 
        use rank r and k indices are sampled from the full matrix to form a 
        sparse one. One can also perturb the observed values with a Gaussian 
        noise component. Returns a scipy csc_matrix. 
        """
        U, s, V = SparseUtils.generateLowRank(shape, r)
        X = SparseUtils.reconstructLowRank(U, s, V, k)

        X.data += numpy.random.randn(X.data.shape[0])*noise        
        
        if verbose: 
            return X, U, s, V
        else: 
            return X 
    
    @staticmethod 
    def generateSparseLowRank2(shape, r, k, snr=1.0, verbose=False): 
        """
        Generate large sparse low rank matrices according to the model given 
        in Mazumder et al., 2010. We have Z = UV.T + E where all three matrices 
        are gaussian. The signal to noise ratio is given by snr. 
        """
        (m, n) = shape
        U = numpy.random.randn(m, r)
        V = numpy.random.randn(n, r)
        s = numpy.ones(r)
        
        noise = r/snr**2
        X = SparseUtils.reconstructLowRank(U, s, V, k)
        X.data += numpy.random.randn(X.data.shape[0])*noise    
        
        if verbose: 
            return X, U, V
        else: 
            return X 
        
    @staticmethod 
    def generateLowRank(shape, r, sampleVals=500): 
        """
        Return the singular values/vectors of a random matrix with rank r. The 
        resulting matrix has entries within the range [-1, +1].
        """
        (n, m) = shape 
        r = numpy.min([n, m, r])

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
    def svdSparseLowRank(X, U, s, V, k=None, kmax=None, usePropack=True): 
        """
        Find the partial SVD of a matrix A = X + U s V.T in which X is sparse and B = 
        U s V.T is a low rank matrix. We use PROPACK to find the singular  
        vectors/values. 
        
        :param X: The input matrix in csc_matrix format. 
        
        :param U: The left singular vectors 
        
        :parma s: The singulal values 
        
        :param V: The right singular vectors 
        
        :param k: The number of singular values/vectors or None for all 
        
        :param kmax: The number of Krylov vectors or None to use SparseUtils.kmaxMultiplier*k
        """

        L = LinOperatorUtils.sparseLowRankOp(X, U, s, V)
        
        if usePropack: 
            U, s, V = SparseUtils.svdPropack(L, k, kmax=kmax)
        else: 
            U, s, V = SparseUtils.svdArpack(L, k, kmax=kmax)
   
        logging.debug("Number of SVs: " + str(s.shape[0]) + " and min SV: " + str(numpy.min(s)))
        
        return U, s, V 
        

    @staticmethod
    def svdPropack(X, k, kmax=None):
        """
        Perform the SVD of a sparse matrix X using PROPACK for the largest k 
        singular values. 
        
        :param X: The input matrix as scipy.sparse.csc_matrix or a LinearOperator
        
        :param k: The number of singular vectors/values for None for all 
        
        :param kmax: The maximal number of iterations / maximal dimension of Krylov subspace.
        """
        if k==None: 
            k = min(X.shape[0], X.shape[1])
        if kmax==None: 
            kmax = SparseUtils.kmaxMultiplier*k
        
        if scipy.sparse.isspmatrix(X): 
            L = scipy.sparse.linalg.aslinearoperator(X) 
        else: 
            L = X
        
        U, s, VT, info, sigma_bound = svdp(L, k, kmax=kmax, full_output=True)

        if info > 0: 
            logging.debug("An invariant subspace of dimension " + str(info) + " was found.")
        elif info==-1: 
            logging.warning(str(k) + " singular triplets did not converge within " + str(kmax) + " iterations")
              
        return U, s, VT.T 
        
    @staticmethod
    def svdArpack(X, k, kmax=None):
        """
        Perform the SVD of a sparse matrix X using ARPACK for the largest k 
        singular values. Note that the input matrix should be of float dtype.  
        
        :param X: The input matrix as scipy.sparse.csc_matrix or a LinearOperator
        
        :param k: The number of singular vectors/values for None for all 
        
        :param kmax: The maximal number of iterations / maximal dimension of Krylov subspace.
        """
        if k==None: 
            k = min(X.shape[0], X.shape[1])
        if kmax==None: 
            kmax = SparseUtils.kmaxMultiplier*k        
        
        if scipy.sparse.isspmatrix(X): 
            L = scipy.sparse.linalg.aslinearoperator(X) 
        else: 
            L = X        
        
        m, n = L.shape

        def matvec_AH_A(x):
            Ax = L.matvec(x)
            return L.rmatvec(Ax)

        AH_A = LinearOperator(matvec=matvec_AH_A, shape=(n, n), dtype=L.dtype)

        eigvals, eigvec = scipy.sparse.linalg.eigsh(AH_A, k=k, ncv=kmax)
        s2 = scipy.sqrt(eigvals)
        V2 = eigvec
        U2 = L.matmat(V2)/s2

        inds = numpy.flipud(numpy.argsort(s2))
        U2, s2, V2 = Util.indSvd(U2, s2, V2.T, inds)
        
        return U2, s2, V2


    @staticmethod 
    def svdSoft(X, lmbda, kmax=None): 
        """
        Find the partial SVD of the sparse or dense matrix X, for which singular 
        values are >= lmbda. Soft threshold the resulting singular values 
        so that s <- max(s - lambda, 0)
        """
        if scipy.sparse.issparse(X): 
            k = min(X.shape[0], X.shape[1])
            L = scipy.sparse.linalg.aslinearoperator(X)  
            
            U, s, V = SparseUtils.svdPropack(L, k, kmax=kmax)
            V = V.T
        else: 
            U, s, V = numpy.linalg.svd(X)
            
        inds = numpy.flipud(numpy.argsort(s))
        inds = inds[s[inds] >= lmbda]
        U, s, V = Util.indSvd(U, s, V, inds) 
        
        #Soft threshold 
        if s.shape[0] != 0: 
            s = s - lmbda
            s = numpy.clip(s, 0, numpy.max(s))

        return U, s, V 
      
    @staticmethod 
    def svdSoft2(X, U, s, V, lmbda): 
        """
        Compute the SVD of a matrix X + UsV^T where X is sparse and U s V^T is 
        low rank. The singular values >= lmbda are computed and then soft thresholded 
        with threshold lmbda. 
        """
        U2, s2, V2 = SparseUtils.svdSparseLowRank(X, U, s, V)
        
        #Soft threshold 
        s2 = s2 - lmbda
        s2 = numpy.clip(s2, 0, numpy.max(s2))
        
        return U2, s2, V2 
        
    kmaxMultiplier = 15 