"""
Some common functions used for the clustering experiments 
"""
import logging
import numpy
from copy import copy
from apgl.graph.GraphUtils import GraphUtils 
from apgl.util.PathDefaults import PathDefaults
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.NingSpectralClustering import NingSpectralClustering
from exp.sandbox.IterativeModularityClustering import IterativeModularityClustering
from exp.sandbox.GraphIterators import toDenseGraphListIterator
import networkx
import argparse
from apgl.util import Util

class ClusterExpHelper(object):
    # priority for default args
    # - best priority: command-line value
    # - middle priority: set-by-function value
    # - lower priority: class value
    defaultAlgoArgs = argparse.Namespace()
    defaultAlgoArgs.runIASC = False
    defaultAlgoArgs.runExact = False
    defaultAlgoArgs.runNystrom = False
    defaultAlgoArgs.runEfficientNystrom = False
    defaultAlgoArgs.runNing = False
    defaultAlgoArgs.runModularity = False
    defaultAlgoArgs.runRandomisedSvd = False

    defaultAlgoArgs.k1 = 10
    defaultAlgoArgs.k2s = [10, 20] 
    defaultAlgoArgs.k3s = [500, 1000]
    
    defaultAlgoArgs.T = 10
    
    defaultAlgoArgs.computeBound = False

    @staticmethod
    # update parameters with those from the user
    def updateParams(params, update=None):
        if update:
            for key, val in vars(update).items():
                params.__setattr__(key, val) 
    
    @staticmethod
    # merge default algoParameters from the class with those from the user
    def newAlgoParams(algoArgs=None):
        algoArgs_ = copy(ClusterExpHelper.defaultAlgoArgs)
        ClusterExpHelper.updateParams(algoArgs_, algoArgs)
        return(algoArgs_)
    
    @staticmethod
    def newAlgoParser(defaultAlgoArgs=None, add_help=False):
        # default algorithm args
        defaultAlgoArgs = ClusterExpHelper.newAlgoParams(defaultAlgoArgs)
        
        # define parser
        algoParser = argparse.ArgumentParser(description="", add_help=add_help)
        for method in ["runIASC", "runExact", "runModularity", "runNystrom", "runEfficientNystrom", "runNing", "runRandomisedSvd"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        algoParser.add_argument("--k1", type=int, help="Number of clusters to construct at each iteration (default: %(default)s)", default=defaultAlgoArgs.k1)
        algoParser.add_argument("--k2s", nargs="+", type=int, help="Rank of the approximated laplacian matrix (default: %(default)s)", default=defaultAlgoArgs.k2s)
        algoParser.add_argument("--k3s", nargs="+", type=int, help="Number of rows/columns used by the Nystrom approach (default: %(default)s)", default=defaultAlgoArgs.k3s)
        algoParser.add_argument("--T", type=int, help="The exact decomposition is recomputed any T-th iteration (default: %(default)s)", default=defaultAlgoArgs.T)

        return(algoParser)
    
    def __init__(self, iteratorFunc, cmdLine=None, defaultAlgoArgs = None, dirName=""):
        # Parameters to choose which methods to run
        # Obtained merging default parameters from the class with those from the user
        self.algoArgs = ClusterExpHelper.newAlgoParams(defaultAlgoArgs)
        
        # Variables related to the dataset
        self.getIteratorFunc = iteratorFunc
        
        #How often to print output 
        self.logStep = 10

        # basic resultsDir
        self.resultsDir = PathDefaults.getOutputDir() + "cluster/" + dirName + "/"

        # update algoParams from command line
        self.readAlgoParams(cmdLine)

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

    def recordResults(self, clusterList, timeList, fileName):
        """
        Save results for a particular clustering
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
        
        if self.algoArgs.runIASC:
            logging.debug("Running approximate method")
            
            for k2 in self.algoArgs.k2s: 
                logging.debug("k2=" + str(k2))
                clusterer = IterativeSpectralClustering(self.algoArgs.k1, k2=k2, T=self.algoArgs.T, alg="IASC", logStep=self.logStep)
                clusterer.nb_iter_kmeans = 20
                clusterer.computeBound = self.algoArgs.computeBound
                iterator = self.getIterator()
                clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)
    
                resultsFileName = self.resultsDir + "ResultsIASC_k1=" + str(self.algoArgs.k1) + "_k2=" + str(k2) + "_T=" + str(self.algoArgs.T) + ".npz"
                self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runExact:
            logging.debug("Running exact method")
            clusterer = IterativeSpectralClustering(self.algoArgs.k1, alg="exact", logStep=self.logStep)
            clusterer.nb_iter_kmeans = 20
            iterator = self.getIterator()
            clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)

            resultsFileName = self.resultsDir + "ResultsExact_k1=" + str(self.algoArgs.k1) + ".npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runNystrom:
            logging.debug("Running Nystrom method")
            
            for k3 in self.algoArgs.k3s: 
                logging.debug("k3=" + str(k3))
                clusterer = IterativeSpectralClustering(self.algoArgs.k1, k3=k3, alg="nystrom", logStep=self.logStep)
                clusterer.nb_iter_kmeans = 20
                clusterer.computeBound = self.algoArgs.computeBound
                iterator = self.getIterator()
                clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)
    
                resultsFileName = self.resultsDir + "ResultsNystrom_k1="+ str(self.algoArgs.k1) + "_k3=" + str(k3) + ".npz"
                self.recordResults(clusterList, timeList, resultsFileName)
                
        if self.algoArgs.runRandomisedSvd:
            logging.debug("Running randomised SVD method")
            
            for k2 in self.algoArgs.k2s: 
                logging.debug("k2=" + str(k2))
                clusterer = IterativeSpectralClustering(self.algoArgs.k1, k2=k2, alg="randomisedSvd", logStep=self.logStep)
                clusterer.nb_iter_kmeans = 20
                clusterer.computeBound = self.algoArgs.computeBound
                iterator = self.getIterator()
                clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)
    
                resultsFileName = self.resultsDir + "ResultsRandomisedSvd_k1="+ str(self.algoArgs.k1) + "_k2=" + str(k2) + ".npz"
                self.recordResults(clusterList, timeList, resultsFileName)
                
        if self.algoArgs.runEfficientNystrom:
            logging.debug("Running efficient Nystrom method")
            
            for k3 in self.algoArgs.k3s: 
                logging.debug("k3=" + str(k3))
                clusterer = IterativeSpectralClustering(self.algoArgs.k1, k3=k3, alg="efficientNystrom", logStep=self.logStep)
                clusterer.nb_iter_kmeans = 20
                clusterer.computeBound = self.algoArgs.computeBound
                iterator = self.getIterator()
                clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)
    
                resultsFileName = self.resultsDir + "ResultsEfficientNystrom_k1="+ str(self.algoArgs.k1) + "_k3=" + str(k3) + ".npz"
                self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runModularity: 
            logging.info("Running modularity clustering")
            clusterer = IterativeModularityClustering(self.algoArgs.k1)
            iterator = self.getIterator()

            clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)

            resultsFileName = self.resultsDir + "ResultsModularity_k1=" + str(self.algoArgs.k1) + ".npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runNing:
            logging.info("Running Nings method")
            iterator = self.getIterator()
            clusterer = NingSpectralClustering(self.algoArgs.k1, T=self.algoArgs.T)
            clusterList, timeList, boundList = clusterer.cluster(iterator, verbose=True)

            resultsFileName = self.resultsDir + "ResultsNing_k1=" + str(self.algoArgs.k1) + "_T=" + str(self.algoArgs.T) + ".npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        logging.info("All done: see you around!")
