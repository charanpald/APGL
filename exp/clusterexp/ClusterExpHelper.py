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

class ClusterExpHelper(object):
    # priority for default args
    # - best priority: command-line value
    # - middle priority: set-by-function value
    # - lower priority: class value
    defaultAlgoArgs = argparse.Namespace()
    defaultAlgoArgs.runIASC = False
    defaultAlgoArgs.runExact = False
    defaultAlgoArgs.runNystrom = False
    defaultAlgoArgs.runNing = False
    defaultAlgoArgs.runModularity = False

    defaultAlgoArgs.k1 = 10
    defaultAlgoArgs.k2s = [10, 20] 
    defaultAlgoArgs.k3s = [500, 1000]
    
    defaultAlgoArgs.T = 10
    
    defaultAlgoArgs.computeBound = False

    @staticmethod
    def newDefaultAlgoArgs(defaultAlgoArgs=None):
        defaultAlgoArgs_ = copy(ClusterExpHelper.defaultAlgoArgs)
        if defaultAlgoArgs:
            for key, val in vars(defaultAlgoArgs).items():
                defaultAlgoArgs_.__setattr__(key, val) 
        
        return(defaultAlgoArgs_)
    
    @staticmethod
    def newAlgoParser(defaultAlgoArgs=None, add_help=False):
        # default algorithm args
        defaultAlgoArgs = ClusterExpHelper.newDefaultAlgoArgs(defaultAlgoArgs)
        
        # define parser
        algoParser = argparse.ArgumentParser(description="", add_help=add_help)
        for method in ["runIASC", "runExact", "runModularity", "runNystrom", "runNing"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        
        return(algoParser)
    
    def __init__(self, iteratorFunc, numGraphs, cmdLine=None, defaultAlgoArgs = None, dirName=""):
        # Default values for variables to choose which methods to run
        self.algoArgs = copy(self.__class__.defaultAlgoArgs)
        
        # Variables related to the dataset
        self.getIteratorFunc = iteratorFunc
        self.numGraphs = numGraphs

        # basic resultsDir
        self.resultsDir = PathDefaults.getOutputDir() + "cluster/" + dirName + "/"

        # read params from command line
        self.readAlgoParams(cmdLine, defaultAlgoArgs)

    def readAlgoParams(self, cmdLine=None, defaultAlgoArgs=None):
        # update current algorithm args
        self.algoArgs = self.__class__.newDefaultAlgoArgs(self.algoArgs)
        
        # define parser
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
        numMeasures = 3
        measures = numpy.zeros((self.numGraphs, numMeasures))
        numGraphInfo = 2
        graphInfo =  numpy.zeros((self.numGraphs, numGraphInfo))
        logging.debug("Computing cluster measures")

        for i in range(self.numGraphs):
            logging.debug("Iteration " + str(i) + " of " + str(self.numGraphs))
            W = next(iterator)
            G = networkx.Graph(W)
            measures[i, 0] = GraphUtils.modularity(W, clusterList[i])
            measures[i, 1] = GraphUtils.kwayNormalisedCut(W, clusterList[i])
            # nb clust
            measures[i, 2] = len(numpy.unique(clusterList[i]))
            # graph size
            graphInfo[i, 0] = W.shape[0]
            # nb connected components
            graphInfo[i, 1] = networkx.number_connected_components(G)
           
        numpy.savez(fileName, measures, timeList, graphInfo)
        logging.debug("Saved file as " + fileName)

    def runExperiment(self):
        """
        Run the selected clustering experiments and save results
        """
        TLogging = max(self.numGraphs // 100, 1)
        
        if self.algoArgs.runIASC:
            logging.debug("Running approximate method")
            
            for k2 in self.algoArgs.k2s: 
                logging.debug("k2=" + str(k2))
                clusterer = IterativeSpectralClustering(self.algoArgs.k1, k2=k2, T=self.algoArgs.T, alg="IASC")
                clusterer.nb_iter_kmeans = 20
                clusterer.computeBound = self.algoArgs.computeBound
                iterator = self.getIterator()
                clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True, TLogging=TLogging)
    
                resultsFileName = self.resultsDir + "ResultsIASC_k1=" + str(self.algoArgs.k1) + "_k2=" + str(k2) + "_T=" + str(self.algoArgs.T) + ".npz"
                self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runExact:
            logging.debug("Running exact method")
            clusterer = IterativeSpectralClustering(self.algoArgs.k1, alg="exact")
            clusterer.nb_iter_kmeans = 20
            iterator = self.getIterator()
            clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True, TLogging=TLogging)

            resultsFileName = self.resultsDir + "ResultsExact_k1=" + str(self.algoArgs.k1) + ".npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runNystrom:
            logging.debug("Running nystrom method")
            
            for k3 in self.algoArgs.k3s: 
                logging.debug("k3=" + str(k3))
                clusterer = IterativeSpectralClustering(self.algoArgs.k1, k3=k3, alg="nystrom")
                clusterer.nb_iter_kmeans = 20
                clusterer.computeBound = self.algoArgs.computeBound
                iterator = self.getIterator()
                clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True, TLogging=TLogging)
    
                resultsFileName = self.resultsDir + "ResultsNystrom_k1="+ str(self.algoArgs.k1) + "_k3=" + str(k3) + ".npz"
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
