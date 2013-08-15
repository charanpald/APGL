'''
Created on 31 Jul 2009

@author: charanpal
'''
from __future__ import print_function

import sys 
import os
import numpy
from contextlib import contextmanager
import numpy.random as rand
import logging
import scipy.linalg
import scipy.sparse as sparse
import scipy.special
import pickle 
from apgl.util.Parameter import Parameter


class Util(object):
    '''
    A class with some general useful function that don't fit in anywhere else. Not very OO unfortunately. 
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    @staticmethod   
    def histogram(v):
        """
        Compute a histogram based on all unique elements in vector v 
        """ 
        if v.ndim != 1: 
            raise ValueError("Input must be a dimension 1 vector")
        
        uniqElements = numpy.unique(v)
        numElements = uniqElements.shape[0]
        hist = numpy.zeros(numElements)
        
        for i in range(0, numElements): 
            hist[i] = sum(v == uniqElements[i])

        return (hist, uniqElements)

    @staticmethod
    def mode(v):
        """
        Returns the mode of a 1D vectors, and the 1st more frequent element if more than 1
        """
        if v.ndim != 1:
            raise ValueError("Input must be a dimension 1 vector")

        uniqElements = numpy.unique(v)
        freqs = numpy.zeros(uniqElements.shape[0])

        for i in range(uniqElements.shape[0]):
            freqs[i] = numpy.sum(v == uniqElements[i])

        return uniqElements[numpy.argmax(freqs)]
    
    @staticmethod  
    def sampleWithoutReplacement(sampleSize, totalSize):
        """ 
        Create a list of integers from 0 to totalSize, and take a random sample of size sampleSize. The 
        sample ordered. 
        """
        perm = rand.permutation(totalSize)
        perm = perm[0:sampleSize]
        perm = numpy.sort(perm)
        
        return perm 
    
    @staticmethod
    def randNormalInt(mean, sd, min, max):
        """
        Returns a normally distributed integer within a range (inclusive of min, max) 
        """
        i = round(rand.normal(mean, sd)); 
        
        while i<min or i>max: 
            i = round(random.normal(mean, sd)); 
            
        return i

    @staticmethod 
    def computeMeanVar(X):
        mu = numpy.mean(X, 0)
        X2 = X - mu
        sigma = numpy.dot(X2.T, X2)/X.shape[0]

        return (mu, sigma)

    @staticmethod
    def iterationStr(i, step, maxIter, preStr="Iteration: "):
        outputStr = ""
        if maxIter == 1:
            outputStr = preStr + str(i) + " (1.0)"
        elif i % step == 0:
            #frm = inspect.stack()[1]
            #mod = inspect.getmodule(frm[0])
            #logging.info(mod.__name__ +  ": " + str(i) + " (" + str(float(i)/maxIter) + ")")
            outputStr = preStr + str(i) + " (" + str("%.3f" % (float(i)/(maxIter-1))) + ")"
        elif i == maxIter-1:
            outputStr = preStr + str(i) + " (" + str("%.3f" % (float(i)/(maxIter-1))) + ")"
        else:
            raise ValueError("Got invalid input: " + str((i, step, maxIter)))
        return outputStr 

    @staticmethod 
    def printIteration(i, step, maxIter, preStr="Iteration: "):
        if i % step == 0 or i==maxIter-1:
            logging.debug(Util.iterationStr(i, step, maxIter, preStr))

    @staticmethod 
    def printConciseIteration(i, step, maxIter, preStr="Iteration: "):
        if i==0:
            print(Util.iterationStr(i, step, maxIter, preStr), end=""),
        elif i!=maxIter-1:
            print(Util.iterationStr(i, step, maxIter, " "), end="")
        else:
            print(Util.iterationStr(i, step, maxIter, " "))



    @staticmethod
    def abstract():
        """
        This is a method to be put in abstract methods so that they are identified
        as such when called. 
        """
        import inspect
        caller = inspect.getouterframes(inspect.currentframe())[1][3]
        raise NotImplementedError("Method " + caller + ' must be implemented in subclass')

    @staticmethod
    def rank(A, tol=1e-8):
        """
        Kindly borrowed from the following forum thread: 
        http://mail.scipy.org/pipermail/numpy-discussion/2008-February/031218.html
        """
        s = numpy.linalg.svd(A, compute_uv=False)
        return numpy.sum(numpy.where(s>tol, 1, 0))

    @staticmethod
    def randomChoice(V, n=1):
        """
        Make a random choice from a vector V of values which are unnormalised
        probabilities. Return the corresponding index. For example if v = [1, 2, 4]
        then the probability of the indices repectively are [1/7, 2/7, 4/7]. The
        parameter n is the number of random choices to make. If V is a matrix,
        then the rows are taken as probabilities, and a choice is made for each
        row. 
        """
        Parameter.checkClass(V, numpy.ndarray)

        if V.shape[0]==0:
            return -1 

        if V.ndim == 1:
            cumV = numpy.cumsum(V)
            p = numpy.random.rand(n)*cumV[-1]
            return numpy.searchsorted(cumV, p)
        elif V.ndim == 2:
            cumV = numpy.cumsum(V, 1)
            P = numpy.random.rand(V.shape[0], n)*numpy.array([cumV[:, -1]]).T

            inds = numpy.zeros(P.shape, numpy.int)
            for i in range(P.shape[0]):
                inds[i, :] = numpy.searchsorted(cumV[i, :], P[i, :])

            return inds
        else:
            raise ValueError("Invalid number of dimensions")

    @staticmethod
    def fitPowerLaw(x, xmin):
        """
        Take a sample of data points which are drawn from a power law probability
        distribution (p(x) = (x/xmin)**-alpha) and return the exponent. This works
        best for continuous data.
        """
        x = x[x >= xmin]
        n = x.shape[0]

        lnSum = n / numpy.sum(numpy.log(x/xmin))
        #gamma = 1 + lnSum
        gamma = lnSum

        return gamma 

    @staticmethod
    def fitDiscretePowerLaw(x, xmins = None):
        """
        Take a sample of discrete data points which are drawn from a power law probability
        distribution (p(x) = x-alpha / zeta(alpha, xmin)) and return the exponent.
        If xmins is supplied then it searches through the set of xmins rather than
        using all possible xmins. Most of the time it helps to keep xmins low. 

        Returns the goodness of fit, best alpha and xmin. If there is only 1 unique 
        value of x then -1, -1 min(x) is returned.

        """
        
        xmax = numpy.max(x)
        if xmins == None:
            xmin = numpy.max(numpy.array([numpy.min(x), 1]))
            xmins = numpy.arange(xmin, xmax)

        #Note that x must have at least 2 unique elements 
        if xmins.shape[0] == 0:
            return -1, -1, numpy.min(x)

        alphas = numpy.arange(1.5, 3.5, 0.01)
        ksAlpha = numpy.zeros((xmins.shape[0], 2))

        for j in range(xmins.shape[0]):
            xmin = xmins[j]
            z = x[x >= xmin]
            n = z.shape[0]

            sumLogx = numpy.sum(numpy.log(z))
            likelyhoods = numpy.zeros(alphas.shape[0])
            
            for i in range(alphas.shape[0]):
                likelyhoods[i] = -n*numpy.log(scipy.special.zeta(alphas[i], xmin)) -alphas[i]*sumLogx
        
            k = numpy.argmax(likelyhoods)

            #Compute KS statistic
            cdf = numpy.cumsum(numpy.bincount(z)[xmin:xmax]/float(n))
            fit = numpy.arange(xmin, xmax)**-alphas[k] /scipy.special.zeta(alphas[k], xmin)
            fit = numpy.cumsum(fit)
            ksAlpha[j, 0] = numpy.max(numpy.abs(cdf - fit))
            ksAlpha[j, 1] = alphas[k]

        i = numpy.argmin(ksAlpha[:, 0])

        return ksAlpha[i, 0], ksAlpha[i, 1], xmins[i]


    @staticmethod
    def entropy(v):
        """
        Compute the information entropy of a vector of random vector observations
        using the log to the base 2.
        """

        items = numpy.unique(v)
        infEnt = 0

        for i in items:
            prob = numpy.sum(v==i)/float(v.shape[0])
            infEnt -= prob * numpy.log2(prob)

        return infEnt

    @staticmethod
    def expandIntArray(v):
        """
        Take a vector of integers and expand it into a vector with counts of the
        corresponding integers. For example, with v = [1, 3, 2, 4], the expanded
        vector is [0, 1, 1, 1, 2, 2, 3, 3, 3, 3]. 
        """
        Parameter.checkClass(v, numpy.ndarray)
        if v.dtype != numpy.int:
            raise ValueError("Can only expand arrays of integers")
        Parameter.checkList(v, Parameter.checkInt, [0, float('inf')])
        
        w = numpy.zeros(numpy.sum(v), numpy.int)
        currentInd = 0
        
        for i in range(v.shape[0]):
            w[currentInd:currentInd+v[i]] = i
            currentInd += v[i]

        return w


    @staticmethod
    def random2Choice(V, n=1):
        """
        Make a random binary choice from a vector V of values which are unnormalised
        probabilities. Return the corresponding index. For example if v = [1, 2]
        then the probability of the indices repectively are [1/3, 2/3]. The
        parameter n is the number of random choices to make. If V is a matrix,
        then the rows are taken as probabilities, and a choice is made for each
        row.
        """
        Parameter.checkClass(V, numpy.ndarray)

        if V.ndim == 1 and V.shape[0] != 2:
            raise ValueError("Function only works on binary probabilities")
        if V.ndim == 2 and V.shape[1] != 2:
            raise ValueError("Function only works on binary probabilities")

        if V.ndim == 1:
            cumV = numpy.cumsum(V)
            p = numpy.random.rand(n)*cumV[-1]
            cumV2 = numpy.ones(n)*cumV[0] - p
            return numpy.array(cumV2 <= 0, numpy.int)
        elif V.ndim == 2:
            cumV = numpy.cumsum(V, 1)
            P = numpy.random.rand(V.shape[0], n)*numpy.array([cumV[:, -1]]).T
            cumV2 = numpy.outer(cumV[:, 0], numpy.ones(n)) - P
            return numpy.array(cumV2 <= 0, numpy.int)
        else:
            raise ValueError("Invalid number of dimensions")

    @staticmethod
    def loadPickle(filename):
        """
        Loads a pickled file with the given filename. 
        """
        file = open(filename, 'rb')
        obj = pickle.load(file)
        file.close()
        #logging.debug("Loaded " + filename + " with object " + str(type(obj)))

        return obj

    @staticmethod
    def savePickle(obj, filename, overwrite=True, debug=False):
        if os.path.isfile(filename) and not overwrite:
            raise IOError("File exists: " + filename)

        file = open(filename, 'wb')
        pickle.dump(obj, file)
        file.close()
        
        if debug: 
            logging.debug("Saved " + filename + " object type " + str(type(obj)))

    @staticmethod
    def incompleteCholesky(X, k):
        """
        Compute the incomplete cholesky decomposition of positive semi-define 
        square matrix X. Use an approximation of k rows.
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError("X must be a square matrix")

        ell = X.shape[0]
        R = numpy.zeros((k, ell))
        d = numpy.diag(X)
        
        aInd = numpy.argmax(d)
        a = d[aInd]

        nu = numpy.zeros(k)

        for j in range(k):
            nu[j] = numpy.sqrt(a)

            for i in range(ell):
                R[j, i] = (X[aInd, i] - R[:, i].T.dot(R[:, aInd]))/nu[j]
                d[i] = d[i] - R[j, i]**2

            aInd = numpy.argmax(d)
            a = d[aInd]

        return R

    @staticmethod
    def incompleteCholesky2(X, k):
        """
        Compute the incomplete cholesky decomposition of positive semi-define
        square matrix X. Use an approximation of k rows.
        """
        ell = X.shape[0]
        A = numpy.zeros((ell, k))
        Xj = X
        Xaj =  numpy.zeros((ell, k))

        for j in range(k):
            d = numpy.diag(Xj)
            ind = numpy.argmax(d)

            A[ind, j] = 1/numpy.sqrt(Xj[ind, ind])
            Xaj[:, j] = Xj.dot(A[:, j])

            Xj = Xj - numpy.outer(Xaj[:, j], Xaj[:, j])/numpy.dot(A[:, j].T, Xaj[:, j])

        return Xaj.T


    @staticmethod
    def indEig(s, U, inds):
        """
        Take the output of numpy.linalg.eig and return the eigenvalue and vectors
        sorted in order indexed by ind. 
        """
        U = U[:, inds]
        s = s[inds]
        return s, U

    @staticmethod
    def indSvd(P, s, Q, inds):
        """
        Take the output of numpy.linalg.svd and return the eigenvalue and vectors
        sorted in order indexed by ind.
        """
        if inds.shape[0] != 0:
            P = P[:, inds]
            s = s[inds]
            Q = Q.conj().T
            Q = Q[:, inds]
        else:
            P = numpy.zeros((P.shape[0], 0))
            s = numpy.zeros(0)
            Q = Q.conj().T
            Q = numpy.zeros((Q.shape[0], 0))

        return P, s, Q

    @staticmethod 
    def svd(A, eps=10**-8, tol=10**-8):
        """
        Wrapper for 'svd_from_eigh' to work on the smallest dimention of A
        """
        if A.shape[0] > A.shape[1]:
            return Util.svd_from_eigh(A, eps)
        else:
            P, s, Qh = Util.svd_from_eigh(A.conj().T, eps, tol)
            return Qh.conj().T, s.conj(), P.conj().T

    @staticmethod 
    def svd_from_eigh(A, eps=10**-8, tol=10**-8):
        """
        Find the SVD of an ill conditioned matrix A. This uses numpy.linalg.eig
        but conditions the matrix so is not as precise as numpy.linalg.svd, but
        can be useful if svd does not coverge. Uses the eigenvectors of A^T*A and
        return singular vectors corresponding to nonzero singular values.

        Note: This is slightly different to linalg.svd which returns zero singular
        values. 
        """
        AA = A.conj().T.dot(A)
        lmbda, Q = scipy.linalg.eigh(AA + eps*numpy.eye(A.shape[1]))
        lmbda = lmbda-eps

        inds = numpy.arange(lmbda.shape[0])[lmbda>tol]
        lmbda, Q = Util.indEig(lmbda, Q, inds)

        sigma = lmbda**0.5
        P = A.dot(Q) / sigma
        Qh = Q.conj().T

        if __debug__:
            if not scipy.allclose(A, (P*sigma).dot(Qh), atol=tol):
                logging.warn(" SVD obtained from EVD is too poor")
            Parameter.checkArray(P, softCheck=True, arrayInfo="P in svd_from_eigh()")
            if not Parameter.checkOrthogonal(P, tol=tol, softCheck=True, arrayInfo="P in svd_from_eigh()", investigate=True):
                print("corresponding sigma: ", sigma)
            Parameter.checkArray(sigma, softCheck=True, arrayInfo="sigma in svd_from_eigh()")
            Parameter.checkArray(Qh, softCheck=True, arrayInfo="Qh in svd_from_eigh()")
            if not Parameter.checkOrthogonal(Qh.conj().T, tol=tol, softCheck=True, arrayInfo="Qh.H in svd_from_eigh()"):
                print("corresponding sigma: ", sigma)


        return P, sigma, Qh

    @staticmethod
    def safeSvd(A, eps=10**-8, tol=10**-8):
        """
        Compute the SVD of a matrix using scipy.linalg.svd, and if convergence fails
        revert to Util.svd.
        """
        # check input matrix
        if __debug__:
            if not Parameter.checkArray(A, softCheck = True):
                logging.info("... in Util.safeSvd")

        try:
            # run scipy.linalg.svd
            try:
                P, sigma, Qh = scipy.linalg.svd(A, full_matrices=False)
            except scipy.linalg.LinAlgError as e:
                logging.warn(str(e))
                raise Exception('SVD decomposition has to be computed from EVD decomposition')
                
            # --- only when the SVD decomposition comes from scipy.linalg.svd ---
            # clean output singular values (sometimes scipy.linalg.svd returns NaN or negative singular values, let's remove them)
            inds = numpy.arange(sigma.shape[0])[sigma > tol]
            if inds.shape[0] < sigma.shape[0]:
                P, sigma, Q = Util.indSvd(P, sigma, Qh, inds)
                Qh = Q.conj().T
                # an expensive check but we really need it
                # rem: A*s = A.dot(diag(s)) ; A*s[:,new] = diag(s).dot(A)
                if not scipy.allclose(A, (P*sigma).dot(Qh)):
                    logging.warn(" After cleaning singular values from scipy.linalg.svd, the SVD decomposition is too far from the original matrix")
#                    numpy.savez("matrix_leading_to_bad_SVD.npz", A)
                    raise Exception('SVD decomposition has to be computed from EVD decomposition')
                    
            # check scipy.linalg.svd output matrices (expensive)
            if __debug__:
                badAnswerFromScipySvd = False
                if not Parameter.checkArray(P, softCheck=True, arrayInfo="P in Util.safeSvd()"):
                    badAnswerFromScipySvd = True
                if not Parameter.checkArray(sigma, softCheck = True, arrayInfo="sigma in Util.safeSvd()"):
                    badAnswerFromScipySvd = True
                if not Parameter.checkArray(Qh, softCheck = True, arrayInfo="Qh in Util.safeSvd()"):
                    badAnswerFromScipySvd = True
                if badAnswerFromScipySvd:
                    logging.warn(" After cleaning singular values from scipy.linalg.svd, the SVD decomposition still contains 'NaN', 'inf' or complex values")
                    raise Exception('SVD decomposition has to be computed from EVD decomposition')

        except Exception as inst:
            if inst.args != ('SVD decomposition has to be computed from EVD decomposition',):
                raise
            logging.warn(" Using EVD method to compute the SVD.")
            P, sigma, Qh = Util.svd(A, eps, tol)

            # check Util.svd output matrices (expensive)
            if __debug__:
                badAnswerFromUtilSvd = False
                if not Parameter.checkArray(P, softCheck = True):
                    logging.info("... in P in Util.safeSvd")
                    badAnswerFromUtilSvd = True
#                        print nan_rows in P: numpy.isnan(P).sum(0).nonzero()
                if not Parameter.checkArray(sigma, softCheck = True):
                    logging.info("... in sigma in Util.safeSvd")
                    badAnswerFromUtilSvd = True
#                        print numpy.isnan(sigma).nonzero()
                if not Parameter.checkArray(Qh, softCheck = True):
                    logging.info("... in Q in Util.safeSvd")
                    badAnswerFromUtilSvd = True
#                        blop = numpy.isnan(Qh).sum(1)
#                        print blop.nonzero()
#                        print blop[blop.nonzero()]
                if badAnswerFromUtilSvd:
                    logging.warn(" SVD decomposition obtained from EVD decomposition contains 'NaN', 'inf' or real values")

        from apgl.util.ProfileUtils import ProfileUtils
        if ProfileUtils.memory() > 10**9:
            ProfileUtils.memDisplay(locals())

        return P, sigma, Qh

    @staticmethod
    def safeEigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, turbo=True, eigvals=None, type=1):
        """
        Compute the EigenDecomposition of a hermitian matrix using scipy.linalg.eigh,
        and if convergence fails revert to scipy.linalg.eig.
        """
        try:
            return scipy.linalg.eigh(a, b=b, lower=lower, eigvals_only=eigvals_only, overwrite_a=overwrite_a, overwrite_b=overwrite_b, turbo=turbo, eigvals=eigvals) #, type=type) I do not know how to manage it
        except:
            if __debug__:
                logging.warning(" scipy.linalg.eigh raised an error, scipy.linalg.eig() is used instead")
            lmbda, q = scipy.linalg.eig(a, b=b, overwrite_a=overwrite_a, overwrite_b=overwrite_b)
            if eigvals == None:
                eigvals = (0, len(lmbda))
            if eigvals_only:
                return lmbda[eigvals[0]:eigvals[1]]
            else :
                return lmbda[eigvals[0]:eigvals[1]], q[eigvals[0]:eigvals[1]]
                

    @staticmethod
    def powerLawProbs(alpha, zeroVal=0.5, maxInt=100):
        """
        Generate a vector of power law probabilities such that p(x) = C x^-alpha for some
        C and 0 < x <= maxInt. The value of zeroVal^-alpha is the probability to assign
        to x==0. 
        """

        p = numpy.arange(0, maxInt, dtype=numpy.float)
        p[0] = zeroVal
        p = p ** -alpha
        p /= p.sum()
        return p


    @staticmethod 
    def matrixPower(A, n):
        """
        Compute the matrix power of A using the exponent n. The computation simply
        evaluated the eigendecomposition of A and then powers the eigenvalue
        matrix accordingly.
        
        Warning: if at least one eigen-value is negative, n should be an integer.
        """
        Parameter.checkClass(A, numpy.ndarray)
        tol = 10**-10

        lmbda, V = scipy.linalg.eig(A)
        lmbda[numpy.abs(lmbda) <= tol] = 0
        lmbda[numpy.abs(lmbda) > tol] = lmbda[numpy.abs(lmbda) > tol]**n
        
        if n >= 0: 
            return (V*lmbda).dot(numpy.linalg.inv(V))
        else: 
            A = scipy.linalg.pinv(A)
            n = numpy.abs(n)
            lmbda, V = scipy.linalg.eig(A)
            lmbda[numpy.abs(lmbda) > tol] = lmbda[numpy.abs(lmbda) > tol]**n
            return (V*lmbda).dot(numpy.linalg.inv(V))  

            
    @staticmethod 
    def matrixPowerh(A, n):
        """
        Compute the matrix power of A using the exponent n. The computation simply
        evaluated the eigendecomposition of A and then powers the eigenvalue
        matrix accordingly.
        
        This version assumes that A is hermitian.
        Warning: if at least one eigen-value is negative, n should be an integer.
        """
        Parameter.checkClass(A, numpy.ndarray)
        tol = 10**-10

        lmbda, V = scipy.linalg.eigh(A)
        lmbda[numpy.abs(lmbda) < tol] = 0
        lmbda[numpy.abs(lmbda) > tol] = lmbda[numpy.abs(lmbda) > tol]**n
        # next line uses the fact that eigh claims returning an orthonormal basis (even if 
        #one sub-space is of dimension >=2) (to be precise, it claims using dsyevd which claims returning an orthonormal matrix)
        return (V*lmbda).dot(V.T) 

    @staticmethod 
    def extendArray(A, newShape, val=0): 
        """
        Take a 2D matrix A and extend the shape to newShape adding zeros to the 
        right and bottom of it. One can optionally pass in scalar or array val 
        and this will be broadcast into the new array. 
        """
        
        tempA = numpy.zeros(newShape)
        tempA[:, :] = val
        tempA[0:A.shape[0], 0:A.shape[1]] = A 
        return tempA 
        
    @staticmethod 
    def distanceMatrix(U, V): 
        """
        Compute a distance matrix between n x d matrix U and m x d matrix V, such 
        that D_ij = ||u_i - v_i||. 
        """
        if U.shape[1] != V.shape[1]: 
            raise ValueError("Arrays must have the same number of columns")
        
        normU = numpy.sum(U**2, 1)
        normV = numpy.sum(V**2, 1)
                
        D = numpy.outer(normU, numpy.ones(V.shape[0])) - 2*U.dot(V.T) + numpy.outer(numpy.ones(U.shape[0]), normV) 
        #Fix for slightly negative numbers 
        D[D<0] = 0
        
        try: 
            D **= 0.5
        except FloatingPointError: 
            numpy.set_printoptions(suppress=True, linewidth=200, threshold=2000)
            print(D.shape)
            print(D)
            raise 
        
        return D 

    @staticmethod
    def cumMin(v): 
        """
        Find the minimum element of a 1d array v for each subarray, starting 
        with the 1st elemnt. 
        """
        u = numpy.zeros(v.shape[0])
        
        for i in range(v.shape[0]):
            u[i] = numpy.min(v[0:i+1])
            
        return u 
        
    @staticmethod         
    def argsort(seq):
        """
        Find the indices of a sequence after being sorted. Code taken from 
        http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        """
        return sorted(range(len(seq)), key = seq.__getitem__)
    
    @staticmethod     
    @contextmanager
    def suppressStdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
    @staticmethod     
    @contextmanager
    def suppressStderr():
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
            
            
    @staticmethod
    def powerEigs(A, eps=0.001): 
        """
        Compute the largest eigenvector of A using power iteration. Returns 
        the eigenvector and corresponding eigenvalue. 
        """
        v = numpy.random.rand(A.shape[1])
        oldV = v 
        error = eps+1
        
        while error > eps: 
            v = A.dot(v)
            v = v/numpy.sqrt((v**2).sum())
            
            error = numpy.linalg.norm(oldV - v)
            oldV = v 
            
        return v.T.dot(A).dot(v), v  
        