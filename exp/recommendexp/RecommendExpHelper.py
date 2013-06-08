
"""
Some common functions used for the recommendation experiments 
"""
import gc 
import logging
import numpy
import argparse
import scipy.sparse
import time 
from copy import copy
from apgl.util.PathDefaults import PathDefaults
from apgl.util import Util
from exp.util.MCEvaluator import MCEvaluator 
from exp.sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute 
from exp.sandbox.recommendation.IterativeSGDNorm2Reg import IterativeSGDNorm2Reg 
from exp.util.SparseUtils import SparseUtils 
from apgl.util.Sampling import Sampling 
from apgl.util.FileLock import FileLock 
from exp.recommendexp.CenterMatrixIterator import CenterMatrixIterator

class RecommendExpHelper(object):
    defaultAlgoArgs = argparse.Namespace()
    defaultAlgoArgs.runSoftImpute = False
    defaultAlgoArgs.runSgdMf = False
    defaultAlgoArgs.rhos = numpy.linspace(0.5, 0.0, 10)     
    defaultAlgoArgs.folds = 3
    defaultAlgoArgs.ks = numpy.array(2**numpy.arange(3, 7, 0.5), numpy.int)
    defaultAlgoArgs.kmax = None 
    defaultAlgoArgs.svdAlgs = ["propack", "rsvd", "rsvdUpdate"]
    defaultAlgoArgs.modelSelect = False
    defaultAlgoArgs.postProcess = False 
    defaultAlgoArgs.trainError = False 
    defaultAlgoArgs.lmbdas = [0.001, 0.002, 0.005, 0.01]
    defaultAlgoArgs.gammas = [0.5, 1, 2]
    
    def __init__(self, trainXIteratorFunc, testXIteratorFunc, cmdLine=None, defaultAlgoArgs = None, dirName=""):
        """ priority for default args
         - best priority: command-line value
         - middle priority: set-by-function value
         - lower priority: class value
        """
        # Parameters to choose which methods to run
        # Obtained merging default parameters from the class with those from the user
        self.algoArgs = RecommendExpHelper.newAlgoParams(defaultAlgoArgs)
        
        #Function to return iterators to the training and test matrices  
        self.trainXIteratorFunc = trainXIteratorFunc
        self.testXIteratorFunc = testXIteratorFunc
        
        #How often to print output 
        self.logStep = 10
        
        #The max number of observations to use for model selection
        self.sampleSize = 5*10**6

        # basic resultsDir
        self.resultsDir = PathDefaults.getOutputDir() + "recommend/" + dirName + "/"

        # update algoParams from command line
        self.readAlgoParams(cmdLine)

    @staticmethod
    # update parameters with those from the user
    def updateParams(params, update=None):
        if update:
            for key, val in vars(update).items():
                params.__setattr__(key, val) 

    @staticmethod
    # merge default algoParameters from the class with those from the user
    def newAlgoParams(algoArgs=None):
        algoArgs_ = copy(RecommendExpHelper.defaultAlgoArgs)
        RecommendExpHelper.updateParams(algoArgs_, algoArgs)
        return(algoArgs_)
    
    @staticmethod
    def newAlgoParser(defaultAlgoArgs=None, add_help=False):
        # default algorithm args
        defaultAlgoArgs = RecommendExpHelper.newAlgoParams(defaultAlgoArgs)
        
        # define parser
        algoParser = argparse.ArgumentParser(description="", add_help=add_help)
        for method in ["runSoftImpute", "runSgdMf"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        algoParser.add_argument("--rhos", type=float, nargs="+", help="Regularisation parameter (default: %(default)s)", default=defaultAlgoArgs.rhos)
        algoParser.add_argument("--ks", type=int, nargs="+", help="Max number of singular values/vectors (default: %(default)s)", default=defaultAlgoArgs.ks)
        algoParser.add_argument("--kmax", type=int, help="Max number of Krylov/Lanczos vectors for PROPACK/ARPACK (default: %(default)s)", default=defaultAlgoArgs.kmax)
        algoParser.add_argument("--svdAlgs", type=str, nargs="+", help="Algorithmss to compute SVD for each iteration of soft impute (default: %(default)s)", default=defaultAlgoArgs.svdAlgs)
        algoParser.add_argument("--modelSelect", action="store_true", help="Whether to do model selection on the 1st iteration (default: %(default)s)", default=defaultAlgoArgs.modelSelect)
        algoParser.add_argument("--postProcess", action="store_true", help="Whether to do post processing for soft impute (default: %(default)s)", default=defaultAlgoArgs.postProcess)
        algoParser.add_argument("--trainError", action="store_true", help="Whether to compute the error on the training matrices (default: %(default)s)", default=defaultAlgoArgs.trainError)
        algoParser.add_argument("--lmbdas", type=int, nargs="+", help="Weight of norm2 regularisation (default: %(default)s)", default=defaultAlgoArgs.lmbdas)
        algoParser.add_argument("--gammas", type=int, nargs="+", help="Weight of SGD update (default: %(default)s)", default=defaultAlgoArgs.gammas)
        return(algoParser)
    
    # update current algoArgs with values from user and then from command line
    def readAlgoParams(self, cmdLine=None, defaultAlgoArgs=None):
        # update current algoArgs with values from the user
        self.__class__.updateParams(defaultAlgoArgs)
        
        # define parser, current values of algoArgs are used as default
        algoParser = self.__class__.newAlgoParser(self.algoArgs, True)

        # parse
        algoParser.parse_args(cmdLine, namespace=self.algoArgs)
            
    def printAlgoArgs(self):
        logging.info("Algo params")
        keys = list(vars(self.algoArgs).keys())
        keys.sort()
        for key in keys:
            logging.info("    " + str(key) + ": " + str(self.algoArgs.__getattribute__(key)))
            
    def getTrainIterator(self): 
        """
        Return the training iterator wrapped in an iterator which centers the rows. 
        Note that the original iterator must generate *new* matrices on repeated 
        calls since the original ones are modified by centering. 
        """
        return CenterMatrixIterator(self.trainXIteratorFunc())            
          
    def recordResults(self, ZIter, learner, fileName):
        """
        Save results for a particular recommendation 
        """
        trainIterator = self.getTrainIterator()
        testIterator = self.testXIteratorFunc()
        measures = []
        metadata = []
        logging.debug("Computing recommendation errors")
        
        while True: 
            try: 
                start = time.time()
                Z = next(ZIter) 
                learnTime = time.time()-start 
            except StopIteration: 
                break 
            
            trainX = next(trainIterator)
            if not self.algoArgs.trainError: 
                del trainX 
                gc.collect()
            
            testX = next(testIterator)
            predTestX = learner.predictOne(Z, testX.nonzero())
            predTestX.eliminate_zeros()
            predTestX = trainIterator.uncenter(predTestX)
            currentMeasures = [MCEvaluator.rootMeanSqError(testX, predTestX), MCEvaluator.meanAbsError(testX, predTestX)]
            
            if self.algoArgs.trainError:
                assert trainX.shape == testX.shape
                predTrainX = learner.predictOne(Z, trainX.nonzero())  
                predTrainX.eliminate_zeros()
                predTrainX = trainIterator.uncenter(predTrainX)
                trainX.eliminate_zeros()
                trainX = trainIterator.uncenter(trainX)
                currentMeasures.append(MCEvaluator.rootMeanSqError(trainX, predTrainX))
                del trainX 
                gc.collect()
            
            logging.debug("Error measures: " + str(currentMeasures))
            logging.debug("Standard deviation of test set " + str(testX.data.std()))
            measures.append(currentMeasures)
            
            #Store some metadata about the learning process 
            if type(learner) == IterativeSoftImpute: 
                metadata.append([Z[0].shape[1], learner.getRho(), learnTime])
            elif type(learner) == IterativeSGDNorm2Reg: 
                metadata.append([Z[0][0].shape[1], learner.getLambda(), learnTime])

        measures = numpy.array(measures)
        metadata = numpy.array(metadata)
        
        logging.debug(measures)
        numpy.savez(fileName, measures, metadata)
        logging.debug("Saved file as " + fileName)


    def runExperiment(self):
        """
        Run the selected clustering experiments and save results
        """
        if self.algoArgs.runSoftImpute:
            logging.debug("Running soft impute")
            
            for svdAlg in self.algoArgs.svdAlgs: 
                resultsFileName = self.resultsDir + "ResultsSoftImpute_alg=" + svdAlg +  ".npz"
                fileLock = FileLock(resultsFileName)  
                
                if not fileLock.isLocked() and not fileLock.fileExists(): 
                    fileLock.lock()
                    
                    try: 
                        learner = IterativeSoftImpute(svdAlg=svdAlg, logStep=self.logStep, kmax=self.algoArgs.kmax, postProcess=self.algoArgs.postProcess)
                        
                        if self.algoArgs.modelSelect: 
                            trainIterator = self.getTrainIterator()
                            #Let's find the optimal lambda using the first matrix 
                            X = trainIterator.next() 
                            
                            logging.debug("Performing model selection, taking subsample of entries of size " + str(self.sampleSize))
                            X = SparseUtils.subsample(X, self.sampleSize)
                            
                            cvInds = Sampling.randCrossValidation(self.algoArgs.folds, X.nnz)
                            meanErrors, stdErrors = learner.modelSelect(X, self.algoArgs.rhos, self.algoArgs.ks, cvInds)
                            
                            logging.debug("Mean errors = " + str(meanErrors))
                            logging.debug("Std errors = " + str(stdErrors))
                            rho = self.algoArgs.rhos[numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)[0]]
                            k = self.algoArgs.ks[numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)[1]]
                        else: 
                            rho = self.algoArgs.rhos[0]
                            k = self.algoArgs.ks[0]
                            
                        learner.setK(k)  
                        learner.setRho(rho)   
                        logging.debug("Training with k = " + str(k) + " and rho = " + str(rho))                    
                        trainIterator = self.getTrainIterator()
                        ZIter = learner.learnModel(trainIterator)
                        
                        self.recordResults(ZIter, learner, resultsFileName)
                    finally: 
                        fileLock.unlock()
                else: 
                    logging.debug("File is locked or already computed: " + resultsFileName)
                
                
        if self.algoArgs.runSgdMf:
            logging.debug("Running SGD MF")
            
            resultsFileName = self.resultsDir + "ResultsSgdMf.npz"
            fileLock = FileLock(resultsFileName)  
            
            if not fileLock.isLocked() and not fileLock.fileExists(): 
                fileLock.lock()
                
                try: 
                    learner = IterativeSGDNorm2Reg(k=self.algoArgs.ks[0], lmbda=self.algoArgs.lmbdas[0], gamma=self.algoArgs.gammas[0])               

                    if self.algoArgs.modelSelect:
                        # Let's find optimal parameters using the first matrix 
                        learner.modelSelect(self.getTrainIterator().next(), self.algoArgs.ks, self.algoArgs.lmbdas, self.algoArgs.gammas, self.algoArgs.folds)
                        trainIterator = self.getTrainIterator()

                    trainIterator = self.getTrainIterator()
                    ZIter = learner.learnModel(trainIterator)
                    
                    self.recordResults(ZIter, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)            
            
        logging.info("All done: see you around!")
