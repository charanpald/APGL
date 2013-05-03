
"""
Some common functions used for the recommendation experiments 
"""
import logging
import numpy
import argparse
from copy import copy
from apgl.util.PathDefaults import PathDefaults
from apgl.util import Util
from apgl.util.MCEvaluator import MCEvaluator 
from exp.sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute 
from exp.util.SparseUtils import SparseUtils 

class RecommendExpHelper(object):
    defaultAlgoArgs = argparse.Namespace()
    defaultAlgoArgs.runSoftImpute = False
    defaultAlgoArgs.lmbdas = [0.1, 0.2, 0.5, 1.0]        
    
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
        for method in ["runSoftImpute"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        algoParser.add_argument("--lmbdas", type=float, help="Regularisation parameter (default: %(default)s)", default=defaultAlgoArgs.lmbdas)

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
    
    def getIterator(self):
        return self.getIteratorFunc()

    def recordResults(self, ZIter, fileName):
        """
        Save results for a particular recommendation 
        """
        testIterator = self.testXIteratorFunc()
        measures = []
        logging.debug("Computing recommendation errors")

        for Z in ZIter:
            #Util.printIteration(i, self.logStep, len(clusterList))
            testX = next(testIterator)
            
            U, s, V = Z 
            predX = SparseUtils.reconstructLowRank(U, s, V, testX.nonzero())
            currentMeasures = [MCEvaluator.meanSqError(testX, predX)] 
            print(currentMeasures)
            measures.append(currentMeasures) 

        measures = numpy.array(measures)
        
        numpy.savez(fileName, measures)
        logging.debug("Saved file as " + fileName)

    def runExperiment(self):
        """
        Run the selected clustering experiments and save results
        """
        if self.algoArgs.runSoftImpute:
            logging.debug("Running exact method")
            learner = IterativeSoftImpute(self.algoArgs.lmbdas[0], svdAlg="propack", logStep=self.logStep)
            trainIterator = self.trainXIteratorFunc()
            ZIter = learner.learnModel(trainIterator)
            

            resultsFileName = self.resultsDir + "ResultsSoftImpute_lmbda=" + str(self.algoArgs.lmbdas[0]) + ".npz"
            self.recordResults(ZIter, resultsFileName)

        logging.info("All done: see you around!")
