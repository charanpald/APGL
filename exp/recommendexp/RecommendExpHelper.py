
"""
Some common functions used for the recommendation experiments 
"""
import logging
import numpy
import argparse
from copy import copy
from apgl.util.PathDefaults import PathDefaults
from apgl.util import Util
from exp.sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute 

class RecommendExpHelper(object):
    def __init__(self, iteratorFunc, cmdLine=None, defaultAlgoArgs = None, dirName=""):
        """ priority for default args
         - best priority: command-line value
         - middle priority: set-by-function value
         - lower priority: class value
        """
        self.defaultAlgoArgs = argparse.Namespace()
        self.defaultAlgoArgs.runSoftImpute = False
        self.defaultAlgoArgs.lmbdas = [0.1, 0.2, 0.5, 1.0]        
        
        # Parameters to choose which methods to run
        # Obtained merging default parameters from the class with those from the user
        self.algoArgs = RecommendExpHelper.newAlgoParams(defaultAlgoArgs)
        
        # Variables related to the dataset
        self.getIteratorFunc = iteratorFunc
        
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
        algoParser.add_argument("--lmbdas", type=float, help="Regularisation parameter (default: %(default)s)", default=defaultAlgoArgs.k1)

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

    def recordResults(self, recommendList, timeList, fileName):
        """
        Save results for a particular recommendation 
        """
        iterator = self.getIterator()
        measures = []
        graphInfo =  []
        logging.debug("Computing cluster measures")

        for i in range(len(clusterList)):
            Util.printIteration(i, self.logStep, len(clusterList))
            W = next(iterator)
            #G = networkx.Graph(W)
            #Store modularity, k-way normalised cut, and cluster size 
            currentMeasures = [GraphUtils.modularity(W, clusterList[i]), GraphUtils.kwayNormalisedCut(W, clusterList[i]), len(numpy.unique(clusterList[i]))] 
            measures.append(currentMeasures) 

            # graph size
            currentGraphInfo = [W.shape[0]]
            graphInfo.append(currentGraphInfo)
            # nb connected components
            #graphInfo[i, 1] = networkx.number_connected_components(G)
        
        measures = numpy.array(measures)
        graphInfo = numpy.array(graphInfo)
        
        numpy.savez(fileName, measures, timeList, graphInfo)
        logging.debug("Saved file as " + fileName)

    def runExperiment(self):
        """
        Run the selected clustering experiments and save results
        """
        if self.algoArgs.runSoftImpute:
            logging.debug("Running exact method")
            learner = IterativeSoftImpute(self.algoArgs.lmbdas, svdAlg="propack", logStep=self.logStep)
            Xiterator = self.getIterator()
            ZList = learner.learnModel(iterator)
            
            for lmbda in self.algoArgs.lmbdas: 
                resultsFileName = self.resultsDir + "ResultsSoftImpute_lmbda=" + str(lmbda) + ".npz"
                self.recordResults(ZList, resultsFileName)

        logging.info("All done: see you around!")
