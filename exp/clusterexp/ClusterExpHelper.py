"""
Some common functions used for the clustering experiments 
"""
import logging
import numpy
from apgl.graph.GraphUtils import GraphUtils 
from apgl.util.PathDefaults import PathDefaults
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.NingSpectralClustering import NingSpectralClustering
from exp.sandbox.IterativeModularityClustering import IterativeModularityClustering
from exp.sandbox.GraphIterators import toDenseGraphListIterator
import networkx
import argparse


class ClusterExpHelper(object):
    def __init__(self, iteratorFunc, numGraphs, cmdLine=None, algoArgs = None):
        # Default values for variables to choose which methods to run
        self.algoArgs = argparse.Namespace()
        self.algoArgs.runIASC = False
        self.algoArgs.runExact = False
        self.algoArgs.runNystrom = False
        self.algoArgs.runNing = False
        self.algoArgs.runModularity = False

        self.algoArgs.k1 = 10
        self.algoArgs.k2 = 10 
        self.algoArgs.k3 = 500
        
        self.algoArgs.T = 10
        
        # Variables related to the dataset
        self.getIteratorFunc = iteratorFunc
        self.numGraphs = numGraphs

        # basic resultsDir
        self.resultsDir = PathDefaults.getOutputDir() + "cluster/"

        # read params from command line
        self.readAlgoParams(cmdLine, algoArgs)

    def readAlgoParams(self, cmdLine=None, args = None):
        # default values
        if not args:
            args = argparse.Namespace()
        for key, val in vars(args).items():
            self.algoArgs.__setattr__(key, val) 

        # define parser
        algoParser = argparse.ArgumentParser(description="")
        for method in ["runIASC", "runExact", "runModularity", "runNystrom", "runNing"]:
            algoParser.add_argument("--" + method, action="store_true", default=self.algoArgs.__getattribute__(method))
        algoParser.add_argument("--k1", type=int, help="Number of clusers", default=self.algoArgs.k1)
        algoParser.add_argument("--k2", type=int, help="Rank of the approximation", default=self.algoArgs.k2)
        algoParser.add_argument("--k3", type=int, help="Number of row/cols used by to find the approximate eigenvalues with Nystrom approach", default=self.algoArgs.k3)
        algoParser.add_argument("--T", type=int, help="The exact decomposition is recomputed any T-ith iteration", default=self.algoArgs.T)

        # parse
        algoParser.parse_args(cmdLine, namespace=args)
        for key, val in vars(args).items():
            self.algoArgs.__setattr__(key, val)
            
    def extendResultsDir(self, middle = None):
        if middle == None:
            middle = ""
        self.resultsDir += middle + "__k1_" + str(self.algoArgs.k1) + "__k2_" + str(self.algoArgs.k2) + "__k3_" + str(self.algoArgs.k3) + "__T_" + str(self.algoArgs.T) + "/"

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

        for i in range(self.numGraphs):
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
            
            
            
        file = open(fileName, 'wb')
        numpy.savez(file, measures, timeList, graphInfo)
        logging.info("Saved file as " + fileName)

    def runExperiment(self):
        """
        Run the selected clustering experiments and save results
        """
        TLogging = max(self.numGraphs // 100, 1)
        
        if self.algoArgs.runIASC or self.algoArgs.runExact:
            clusterer = IterativeSpectralClustering(self.algoArgs.k1, self.algoArgs.k2)
            clusterer.nb_iter_kmeans = 20

        if self.algoArgs.runIASC:
            logging.info("Running approximate method")
            iterator = self.getIterator()
            clusterList, timeList = clusterer.clusterFromIterator(iterator, timeIter=True, T=self.algoArgs.T, TLogging=TLogging)

            resultsFileName = self.resultsDir + "ResultsIASC.npz"

            self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runExact:
            logging.info("Running exact method")
            iterator = self.getIterator()
            clusterList, timeList = clusterer.clusterFromIterator(iterator, False, timeIter=True, TLogging=TLogging)

            resultsFileName = self.resultsDir + "ResultsExact.npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runNystrom:
            logging.info("Running nystrom method without updates")
            clusterer = IterativeSpectralClustering(self.algoArgs.k1, self.algoArgs.k2, k3=self.algoArgs.k3, nystromEigs=True)
            clusterer.nb_iter_kmeans = 20
            iterator = self.getIterator()
            clusterList, timeList = clusterer.clusterFromIterator(iterator, False, timeIter=True, T=self.algoArgs.T, TLogging=TLogging)

            resultsFileName = self.resultsDir + "ResultsNystrom.npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runModularity: 
            logging.info("Running modularity clustering")
            clusterer = IterativeModularityClustering(self.algoArgs.k1)
            iterator = self.getIterator()

            clusterList, timeList = clusterer.clusterFromIterator(iterator, timeIter=True, T=self.algoArgs.T)

            resultsFileName = self.resultsDir + "ResultsModularity.npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        if self.algoArgs.runNing:
            logging.info("Running Nings method")
            iterator = self.getIterator()
            clusterer = NingSpectralClustering(self.algoArgs.k1)
            clusterList, timeList = clusterer.cluster(iterator, timeIter=True, T=self.algoArgs.T)

            resultsFileName = self.resultsDir + "ResultsNing.npz"
            self.recordResults(clusterList, timeList, resultsFileName)

        logging.info("All done: see you around!")
