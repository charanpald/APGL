
#from exp.sandbox.recommendation.SGDNorm2Reg import SGDNorm2Reg
from exp.sandbox.recommendation.SGDNorm2RegCython import SGDNorm2Reg
import logging
import gc
import scipy
from apgl.util.Sampling import Sampling 
from apgl.util import Util
from exp.util.SparseUtils import SparseUtils 
import numpy.testing as nptst 
import itertools
from exp.util.MCEvaluator import MCEvaluator

"""
An iterative version of matrix factoisation using frobenius norm penalisation. 
"""

class IterativeSGDNorm2Reg(object): 
    def __init__(self, k, lmbda, eps=0.000001, tmax=100000, gamma=1): 
        self.baseLearner = SGDNorm2Reg(k, lmbda, eps, tmax, gamma)
        
    def learnModel(self, XIterator): 
        
        class ZIterator(object):
            def __init__(self, XIterator, baseLearner):
                self.XIterator = XIterator 
                self.baseLearner = baseLearner 
                self.ZListSGD = None

            def __iter__(self):
                return self

            def next(self):
                X = self.XIterator.next()
                
                #Return the matrices P, Q as the learnt model
                if self.ZListSGD == None:
                    # assumption : training matrix centered by row and column
                    self.ZListSGD = self.baseLearner.learnModel(X, storeAll=False)
                else:
                    #In the case the matrix size changes, we alter P and Q to fit the new data     
                    P, Q = self.ZListSGD[0]
                                        
                    if X.shape[0] > P.shape[0]:
                        P = Util.extendArray(P, (X.shape[0], P.shape[1]))
                    elif X.shape[0] < P.shape[0]:
                        P = P[0:X.shape[0], :]

                    if X.shape[1] > Q.shape[0]:
                        Q = Util.extendArray(Q, (X.shape[1], Q.shape[1]))
                    elif X.shape[1] < Q.shape[0]:
                        Q = Q[0:X.shape[1], :]
                        
                    self.ZListSGD = [(P, Q)]
                    
                    try:
                        self.ZListSGD = self.baseLearner.learnModel(X, Z=self.ZListSGD, storeAll=False)
                    except FloatingPointError:
                        logging.warning("FloatingPointError encountered, reinitialise the matrix decomposition")
                        self.ZListSGD = self.baseLearner.learnModel(X, storeAll=False)
                    except ValueError:
                        logging.warning("ValueError encountered, reinitialise the matrix decomposition")
                        self.ZListSGD = self.baseLearner.learnModel(X, storeAll=False)
                    except SGDNorm2Reg.ArithmeticError:
                        logging.warning("ArithmeticError encountered, reinitialise the matrix decomposition")
                        self.ZListSGD = self.baseLearner.learnModel(X, storeAll=False)
                return self.ZListSGD
                
        return ZIterator(XIterator, self.baseLearner)
        

    def predict(self, ZIter, indList):
        """
        Make a set of predictions for a given iterator of completed matrices and
        an index list.
        """
        class ZTestIter(object):
            def __init__(self, baseLearner):
                self.i = 0
                self.baseLearner = baseLearner

            def __iter__(self):
                return self

            def next(self):
                Z = next(ZIter) 
                Xhat = self.baseLearner.predict(Z, indList[self.i])  
                self.i += 1
                return Xhat 

        return ZTestIter(self.baseLearner)
        
    def predictOne(self, Z, indList): 
        return self.baseLearner.predict(Z, indList)  
        
    def learnPredict(self, trainX, testX, k, lmbda, gamma, maxNTry=1):
        """
        A function to train on a training set and test on a test set.
        Use a copy of the base learner (allow to run several parameter sets in
        parallel) 
        """
        logging.debug("k = " + str(k) + "    lmbda = " + str(lmbda) + "    gamma = " +str(gamma))
        learner = self.baseLearner.copy()
        learner.k = k
        learner.lmbda = lmbda
        learner.gamma = gamma
        
        testInds = testX.nonzero()
    
        # train (try several time if floating point error is raised
        haveRes = False
        nTry = 0
        while not haveRes and nTry<maxNTry:
            nTry += 1
            try:
                ZIter = learner.learnModel(trainX, storeAll = False)
                haveRes = True
            except (FloatingPointError, ValueError, SGDNorm2Reg.ArithmeticError):
                pass

        if haveRes:
            logging.debug("result obtained in " + str(nTry) + " try(ies)")
            predX = learner.predict(ZIter, testX.nonzero())
            error = MCEvaluator.rootMeanSqError(testX, predX)
        else:
            logging.debug("enable to make SGD converge")
            error = float("inf")
            
        return error

    def modelSelect(self, X, ks, lmbdas, gammas, nFolds, maxNTry=5):
        """
        Choose parameters based on a single matrix X. We do cross validation
        within, and set parameters according to the mean squared error.
        Return nothing.
        """
        logging.debug("Performing model selection")

        # usefull
        X = X.tocoo()
        gc.collect()
        nK = len(ks) 
        nLmbda = len(lmbdas) 
        nGamma = len(gammas) 
        nLG = nLmbda * nGamma
        errors = scipy.zeros((nK, nLmbda, nGamma, nFolds))
       
        # generate cross validation sets
        cvInds = Sampling.randCrossValidation(nFolds, X.nnz)
        
        # compute error for each fold / setting
        for icv, (trainInds, testInds) in enumerate(cvInds):
            Util.printIteration(icv, 1, nFolds, "Fold: ")

            trainX = SparseUtils.submatrix(X, trainInds)
            testX = SparseUtils.submatrix(X, testInds)

            assert trainX.nnz == trainInds.shape[0]
            assert testX.nnz == testInds.shape[0]
            nptst.assert_array_almost_equal((testX+trainX).data, X.data)

            paramList = []
        
            for ik, k in enumerate(ks):
                for ilmbda, lmbda in enumerate(lmbdas):
                    for igamma, gamma in enumerate(gammas):
                        paramList.append((trainX, testX, k, lmbda, gamma, maxNTry)) 
            
            # ! Remark !
            # we can parallelize the run of parameters easely.
            # parallelize the run of cv-folds is not done as it is much more
            # memory-consuming 
            
            # parallel version (copied from IteraticeSoftImpute, but not tested) 
            #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()/2, maxtasksperchild=10)
            #results = pool.imap(self.learnPredict, paramList)
            #pool.terminate()

            # non-parallel version 
            results = scipy.array(list(itertools.starmap(self.learnPredict, paramList)))

            errors[:, :, :, icv] = scipy.array(results).reshape((nK, nLmbda, nGamma))
        
        # compute cross validation error for each setting
        errors[errors == float("inf")] = errors[errors != float("inf")].max()
        meanErrors = errors.mean(3)
        stdErrors = errors.std(3)
        logging.debug("Mean errors given (k, lambda, gamma):")
        logging.debug(meanErrors)
        logging.debug("... with standard deviation:")
        logging.debug(stdErrors)

        # keep the best
        iMin = meanErrors.argmin()
        kMin = ks[int(scipy.floor(iMin/(nLG)))]
        lmbdaMin = lmbdas[int(scipy.floor((iMin%nLG)/nGamma))]
        gammaMin = gammas[int(scipy.floor(iMin%nGamma))]
        logging.debug("argmin: (k, lambda, gamma) = (" + str(kMin) + ", " + str(lmbdaMin) + ", " + str(gammaMin) + ")")
        logging.debug("min = " + str(meanErrors[int(scipy.floor(iMin/(nLG))), int(scipy.floor((iMin%nLG)/nGamma)), int(scipy.floor(iMin%nGamma))]))
        
        self.baseLearner.k = kMin
        self.baseLearner.lmbda = lmbdaMin
        self.baseLearner.gamma = gammaMin
        
        return

    def getLambda(self): 
        return self.baseLearner.lmbda 