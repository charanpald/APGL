import gc 
import numpy
import logging
import itertools
import scipy.sparse.linalg
import exp.util.SparseUtils as ExpSU
import numpy.testing as nptst 
from sppy import csarray 
from exp.sandbox.RandomisedSVD import RandomisedSVD
from exp.util.MCEvaluator import MCEvaluator
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from exp.sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter
from exp.util.SparseUtilsCython import SparseUtilsCython
from exp.sandbox.SVDUpdate import SVDUpdate
from exp.util.LinOperatorUtils import LinOperatorUtils
from exp.util.SparseUtils import SparseUtils

def learnPredict(args): 
    """
    A function to train on a training set and test on a test set, for a number 
    of values of rho. 
    """
    learner, trainX, testX, rhos = args 
    logging.debug("k=" + str(learner.getK()))
    
    testInds = testX.nonzero()
    trainXIter = []
    testIndList = []    
    
    for rho in rhos: 
        trainXIter.append(trainX)
        testIndList.append(testInds)
    
    trainXIter = iter(trainXIter)

    ZIter = learner.learnModel(trainXIter, iter(rhos))
    predXIter = learner.predict(ZIter, testIndList)
    
    errors = numpy.zeros(rhos.shape[0])
    for j, predX in enumerate(predXIter): 
        errors[j] = MCEvaluator.rootMeanSqError(testX, predX)
        logging.debug("Error = " + str(errors[j]))
        del predX 
        gc.collect()
        
    return errors 

class IterativeSoftImpute(AbstractMatrixCompleter):
    """
    Given a set of matrices X_1, ..., X_T find the completed matrices.
    """
    def __init__(self, rho=0.1, eps=0.01, k=None, svdAlg="propack", updateAlg="initial", r=10, logStep=10, kmax=None, postProcess=False, p=50, q=2):
        """
        Initialise imputing algorithm with given parameters. The rho is a value
        for use with the soft thresholded SVD. Eps is the convergence threshold and
        k is the rank of the SVD.

        :param rho: The regularisation parameter for soft-impute in [0, 1] (lambda = rho * maxSv)

        :param eps: The convergence threshold

        :param k: The number of SVs to compute

        :param svdAlg: The algorithm to use for computing a low rank + sparse matrix

        :param updateAlg: The algorithm to use for updating an SVD for a new matrix

        :param r: The number of random projections to use for randomised SVD
        
        :param p: The oversampling used for the randomised SVD
        
        :param q: The exponent used for the randomised SVD 
        """
        super(AbstractMatrixCompleter, self).__init__()

        self.rho = rho
        self.eps = eps
        self.k = k
        self.svdAlg = svdAlg
        self.updateAlg = updateAlg
        self.r = r
        self.p = p
        self.q = q
        if k != None:
            self.kmax = k*5
        else:
            self.kmax = None
        self.logStep = logStep
        self.postProcess = postProcess 
        self.postProcessSamples = 10**6
        self.maxIterations = 30
        self.weighted = False 
        self.implicit = False 

    def learnModel(self, XIterator, rhos=None):
        """
        Learn the matrix completion using an iterator which outputs
        a sequence of sparse matrices X. The output of this method is also
        an iterator which outputs a sequence of completed matrices in factorised 
        form. 
        
        :param XIterator: An iterator which emits scipy.sparse.csc_matrix objects 
        
        :param rhos: An optional array of rhos for model selection using warm restarts 
        """

        class ZIterator(object):
            def __init__(self, XIterator, iterativeSoftImpute):
                self.tol = 10**-6
                self.j = 0
                self.XIterator = XIterator
                self.iterativeSoftImpute = iterativeSoftImpute
                self.rhos = rhos 

            def __iter__(self):
                return self
            
            def next(self):
                X = self.XIterator.next()
                logging.debug("Learning on matrix with shape: " + str(X.shape) + " and " + str(X.nnz) + " non-zeros")    
                
                if self.iterativeSoftImpute.weighted: 
                    #Compute row and col probabilities 
                    u, v = SparseUtils.nonzeroRowCols(X)

                    U = scipy.sparse.eye(X.shape[0], format="csr")
                    U.data = 1/u 
                    
                    V = scipy.sparse.eye(X.shape[1], format="csr")                    
                    V.data = 1/v 
                    
                    X = U.dot(X).dot(V)                    
                    
                if self.rhos != None: 
                    self.iterativeSoftImpute.setRho(self.rhos.next())

                if not scipy.sparse.isspmatrix_csc(X):
                    raise ValueError("X must be a csc_matrix")
                    
                #Figure out what lambda should be 
                #PROPACK has problems with convergence 
                Y = scipy.sparse.csc_matrix(X, dtype=numpy.float)
                U, s, V = SparseUtils.svdArpack(Y, 1, kmax=20)
                del Y
                #U, s, V = SparseUtils.svdPropack(X, 1, kmax=20)
                lmbda = s[0]*self.iterativeSoftImpute.rho
                logging.debug("Largest singular value : " + str(s[0]) + " and lambda: " + str(lmbda))

                (n, m) = X.shape

                if self.j == 0:
                    self.oldU = numpy.zeros((n, 1))
                    self.oldS = numpy.zeros(1)
                    self.oldV = numpy.zeros((m, 1))
                else:
                    oldN = self.oldU.shape[0]
                    oldM = self.oldV.shape[0]

                    if self.iterativeSoftImpute.updateAlg == "initial":
                        if n > oldN:
                            self.oldU = Util.extendArray(self.oldU, (n, self.oldU.shape[1]))
                        elif n < oldN:
                            self.oldU = self.oldU[0:n, :]

                        if m > oldM:
                            self.oldV = Util.extendArray(self.oldV, (m, self.oldV.shape[1]))
                        elif m < oldN:
                            self.oldV = self.oldV[0:m, :]
                    elif self.iterativeSoftImpute.updateAlg == "svdUpdate":
                        pass
                    elif self.iterativeSoftImpute.updateAlg == "zero":
                        self.oldU = numpy.zeros((n, 1))
                        self.oldS = numpy.zeros(1)
                        self.oldV = numpy.zeros((m, 1))
                    else:
                        raise ValueError("Unknown SVD update algorithm: " + self.updateAlg)

                rowInds, colInds = X.nonzero()

                gamma = self.iterativeSoftImpute.eps + 1
                i = 0

                while gamma > self.iterativeSoftImpute.eps:
                    if i == self.iterativeSoftImpute.maxIterations: 
                        logging.debug("Maximum number of iterations reached")
                        break 
                    
                    ZOmega = SparseUtilsCython.partialReconstructPQ((rowInds, colInds), self.oldU*self.oldS, self.oldV)
                    Y = X - ZOmega
                    #Y = Y.tocsc()
                    #del ZOmega
                    Y = csarray.fromScipySparse(Y, storagetype="row")
                    gc.collect()

                    if self.iterativeSoftImpute.svdAlg=="propack":
                        newU, newS, newV = ExpSU.SparseUtils.svdSparseLowRank(Y, self.oldU, self.oldS, self.oldV, k=self.iterativeSoftImpute.k, kmax=self.iterativeSoftImpute.kmax)
                    elif self.iterativeSoftImpute.svdAlg=="arpack":
                        newU, newS, newV = ExpSU.SparseUtils.svdSparseLowRank(Y, self.oldU, self.oldS, self.oldV, k=self.iterativeSoftImpute.k, kmax=self.iterativeSoftImpute.kmax, usePropack=False)
                    elif self.iterativeSoftImpute.svdAlg=="svdUpdate":
                        newU, newS, newV = SVDUpdate.addSparseProjected(self.oldU, self.oldS, self.oldV, Y, self.iterativeSoftImpute.k)
                    elif self.iterativeSoftImpute.svdAlg=="rsvd":
                        L = LinOperatorUtils.sparseLowRankOp(Y, self.oldU, self.oldS, self.oldV, parallel=True)
                        newU, newS, newV = RandomisedSVD.svd(L, self.iterativeSoftImpute.k, p=self.iterativeSoftImpute.p, q=self.iterativeSoftImpute.q)
                    elif self.iterativeSoftImpute.svdAlg=="rsvdUpdate": 
                        L = LinOperatorUtils.sparseLowRankOp(Y, self.oldU, self.oldS, self.oldV)
                        if i == 0: 
                            newU, newS, newV = RandomisedSVD.svd(L, self.iterativeSoftImpute.k, p=self.iterativeSoftImpute.p, q=self.iterativeSoftImpute.q)
                        else: 
                            newU, newS, newV = RandomisedSVD.svd(L, self.iterativeSoftImpute.k, p=0, q=1, omega=self.oldV)
                    else:
                        raise ValueError("Unknown SVD algorithm: " + self.iterativeSoftImpute.svdAlg)

                    #Soft threshold
                    newS = newS - lmbda
                    newS = numpy.clip(newS, 0, numpy.max(newS))

                    normOldZ = (self.oldS**2).sum()
                    normNewZmOldZ = (self.oldS**2).sum() + (newS**2).sum() - 2*numpy.trace((self.oldV.T.dot(newV*newS)).dot(newU.T.dot(self.oldU*self.oldS)))

                    #We can get newZ == oldZ in which case we break
                    if normNewZmOldZ < self.tol:
                        gamma = 0
                    elif abs(normOldZ) < self.tol:
                        gamma = self.iterativeSoftImpute.eps + 1
                    else:
                        gamma = normNewZmOldZ/normOldZ

                    self.oldU = newU.copy()
                    self.oldS = newS.copy()
                    self.oldV = newV.copy()

                    logging.debug("Iteration " + str(i) + " gamma="+str(gamma))
                    i += 1

                if self.iterativeSoftImpute.postProcess: 
                    #Add the mean vectors 
                    previousS = newS
                    newU = numpy.c_[newU, numpy.array(X.mean(1)).ravel()]
                    newV = numpy.c_[newV, numpy.array(X.mean(0)).ravel()]
                    newS = self.iterativeSoftImpute.unshrink(X, newU, newV)  
                    
                    #print("Difference in s after postprocessing: " + str(numpy.linalg.norm(previousS - newS[0:-1]))) 
                    logging.debug("Difference in s after postprocessing: " + str(numpy.linalg.norm(previousS - newS[0:-1]))) 

                logging.debug("Number of iterations for rho="+str(self.iterativeSoftImpute.rho) + ": " + str(i))

                self.j += 1
                return (newU, newS, newV)

        return ZIterator(XIterator, self)

    def predict(self, ZIter, indList):
        """
        Make a set of predictions for a given iterator of completed matrices and
        an index list.
        """
        class ZTestIter(object):
            def __init__(self, iterativeSoftImpute):
                self.i = 0
                self.iterativeSoftImpute = iterativeSoftImpute

            def __iter__(self):
                return self

            def next(self):    
                Xhat = self.iterativeSoftImpute.predictOne(ZIter.next(), indList[self.i])  
                self.i += 1
                return Xhat 

        return ZTestIter(self)

    def predictOne(self, Z, inds): 
        U, s, V = Z
        
        if type(inds) == tuple: 
            logging.debug("Predicting on matrix with shape: " + str((U.shape[0], V.shape[0])) + " and " + str(inds[0].shape[0]) + " non-zeros")  
        Xhat = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds)
    
        return Xhat

    def modelSelect(self, X, rhos, ks, cvInds):
        """
        Pick a value of rho based on a single matrix X. We do cross validation
        within, and return the best value of lambda (according to the mean
        squared error). The rhos must be in decreasing order and we use 
        warm restarts. 
        """
        print("nbytes = " + str(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes))
        if (numpy.flipud(numpy.sort(rhos)) != rhos).all(): 
            raise ValueError("rhos must be in descending order")    

        X = X.tocoo()
        gc.collect()
        errors = numpy.zeros((rhos.shape[0], ks.shape[0], len(cvInds)))

        for i, (trainInds, testInds) in enumerate(cvInds):
            Util.printIteration(i, 1, len(cvInds), "Fold: ")

            trainX = SparseUtils.submatrix(X, trainInds)
            testX = SparseUtils.submatrix(X, testInds)

            assert trainX.nnz == trainInds.shape[0]
            assert testX.nnz == testInds.shape[0]
            nptst.assert_array_almost_equal((testX+trainX).data, X.data)

            paramList = []
        
            for m, k in enumerate(ks): 
                learner = self.copy()
                learner.setK(k)
                paramList.append((learner, trainX, testX, rhos)) 
                
            #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()/2, maxtasksperchild=10)
            #results = pool.imap(learnPredict, paramList)

            results = itertools.imap(learnPredict, paramList)
            
            for m, rhoErrors in enumerate(results): 
                errors[:, m, i] = rhoErrors
                
            #pool.terminate()

        meanErrors = errors.mean(2)
        stdErrors = errors.std(2)

        return meanErrors, stdErrors

    def unshrink(self, X, U, V): 
        """
        Perform post-processing on a factorisation of a matrix X use factor 
        vectors U and V. 
        """
        logging.debug("Post processing singular values")
               
        #Fix for versions of numpy < 1.7 
        inds = numpy.unique(numpy.random.randint(0, X.data.shape[0], numpy.min([self.postProcessSamples, X.data.shape[0]]))) 
        a = X.data[inds]
            
        B = numpy.zeros((a.shape[0], U.shape[1])) 
            
        rowInds, colInds = X.nonzero() 
        rowInds = numpy.array(rowInds[inds], numpy.int32)
        colInds = numpy.array(colInds[inds], numpy.int32)  
        
        #Populate B 
        for i in range(U.shape[1]): 
            B[:, i] = SparseUtilsCython.partialOuterProduct(rowInds, colInds, U[:, i], V[:, i])
        
        s = numpy.linalg.pinv(B.T.dot(B)).dot(B.T).dot(a)
        
        return s 

    def setK(self, k):
        #Parameter.checkInt(k, 1, float('inf'))

        self.k = k

    def getK(self):
        return self.k

    def setRho(self, rho):
        Parameter.checkFloat(rho, 0.0, 1.0)

        self.rho = rho

    def getRho(self):
        return self.rho

    def getMetricMethod(self):
        return MCEvaluator.meanSqError

    def copy(self):
        """
        Return a new copied version of this object.
        """
        iterativeSoftImpute = IterativeSoftImpute(rho=self.rho, eps=self.eps, k=self.k, svdAlg=self.svdAlg, updateAlg=self.updateAlg, r=self.r, logStep=self.logStep, kmax=self.kmax, postProcess=self.postProcess)

        return iterativeSoftImpute

    def name(self):
        return "IterativeSoftImpute"