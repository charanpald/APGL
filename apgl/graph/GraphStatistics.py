"""
A class to compute a series of graph statistics over a sequence of subgraphs
"""
import gc 
import numpy
import logging 
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter 
from apgl.graph.GraphUtils import GraphUtils
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph

class GraphStatistics(object):
    def __init__(self):
        self.numVerticesIndex = 0
        self.numEdgesIndex = 1
        self.numDirEdgesIndex = 2
        self.maxComponentSizeIndex = 3
        self.numComponentsIndex = 4
        self.meanComponentSizeIndex = 5
        self.meanDegreeIndex = 6
        self.diameterIndex = 7
        self.effectiveDiameterIndex = 8
        self.densityIndex = 9
        self.powerLawIndex = 10
        self.geodesicDistanceIndex = 11
        self.harmonicGeoDistanceIndex = 12
        self.geodesicDistMaxCompIndex = 13
        self.meanTreeSizeIndex = 14
        self.meanTreeDepthIndex = 15
        self.numNonSingletonComponentsIndex = 16
        self.maxComponentEdgesIndex = 17
        self.numTriOrMoreComponentsIndex = 18
        self.secondComponentSizeIndex = 19
        self.maxTreeSizeIndex = 20
        self.maxTreeDepthIndex = 21
        self.secondTreeSizeIndex = 22
        self.secondTreeDepthIndex = 23
        self.maxCompMeanDegreeIndex = 24

        self.numTreesIndex = 25
        self.numNonSingletonTreesIndex = 26 

        self.numStats = 27
        self.q = 0.9
        self.printStep = 5
        self.vectorPrintStep = 1

        self.useFloydWarshall = False 

    def getNumStats(self):
        return self.numStats 

    def strScalarStatsArray(self, statsArray):
        """
        Get a string representation of the scalar statastics array.
        """
        s = "Num vertices: " + str(statsArray[self.numVerticesIndex]) + "\n"
        s += "Num edges: " + str(statsArray[self.numEdgesIndex]) + "\n"
        s += "Num directed edges: " + str(statsArray[self.numDirEdgesIndex]) + "\n"
        s += "Max component size: " + str(statsArray[self.maxComponentSizeIndex]) + "\n"
        s += "Max component edges: " + str(statsArray[self.maxComponentEdgesIndex]) + "\n"
        s += "Max component mean degree: " + str(statsArray[self.maxCompMeanDegreeIndex]) + "\n"
        s += "Max component geodesic disance: " + str(statsArray[self.geodesicDistMaxCompIndex]) + "\n"
        s += "2nd largest component size: " + str(statsArray[self.secondComponentSizeIndex]) + "\n"
        s += "Num components: " + str(statsArray[self.numComponentsIndex]) + "\n"
        s += "Mean component size: " + str(statsArray[self.meanComponentSizeIndex]) + "\n"
        s += "Mean degree: " + str(statsArray[self.meanDegreeIndex]) + "\n"
        s += "Diameter: " + str(statsArray[self.diameterIndex]) + "\n"
        s += "Effective diameter: " + str(statsArray[self.effectiveDiameterIndex]) + "\n"
        s += "Density: " + str(statsArray[self.densityIndex]) + "\n"
        s += "Power law exponent: " + str(statsArray[self.powerLawIndex]) + "\n"
        s += "Geodesic distance: " + str(statsArray[self.geodesicDistanceIndex]) + "\n"
        s += "Harmonic geodesic distance: " + str(statsArray[self.harmonicGeoDistanceIndex]) + "\n"
        s += "Mean tree size: " + str(statsArray[self.meanTreeSizeIndex]) + "\n"
        s += "Mean tree depth: " + str(statsArray[self.meanTreeDepthIndex]) + "\n"
        s += "Num non-singleton components: " + str(statsArray[self.numNonSingletonComponentsIndex]) + "\n"
        s += "Num tri-or-more components: " + str(statsArray[self.numTriOrMoreComponentsIndex]) + "\n"

        return s 


    def scalarStatistics(self, graph, slowStats=True, treeStats=False):
        """
        Find a series of statistics for the given input graph which can be represented
        as scalar values. Return results as a vector.
        """
        #This method is a bit of a mess 
        Parameter.checkClass(graph, AbstractMatrixGraph)
        Parameter.checkBoolean(slowStats)
        Parameter.checkBoolean(treeStats)
        
        statsArray = numpy.ones(self.numStats)*-1
        statsArray[self.numVerticesIndex] = graph.getNumVertices()
        statsArray[self.numEdgesIndex] = graph.getNumEdges()
        statsArray[self.numDirEdgesIndex] = graph.getNumDirEdges()
        statsArray[self.densityIndex] = graph.density()

        if graph.isUndirected():
            logging.debug("Finding connected components")
            subComponents = graph.findConnectedComponents()
            logging.debug("Done")
            statsArray[self.numComponentsIndex] = len(subComponents)
            
            nonSingletonSubComponents = [c for c in subComponents if len(c) > 1]
            statsArray[self.numNonSingletonComponentsIndex] = len(nonSingletonSubComponents)

            triOrMoreSubComponents = [c for c in subComponents if len(c) > 2]
            statsArray[self.numTriOrMoreComponentsIndex] = len(triOrMoreSubComponents)
            
            logging.debug("Studying max component")
            if len(subComponents) != 0:
                maxCompGraph = graph.subgraph(list(subComponents[0]))
                statsArray[self.maxComponentSizeIndex] = len(subComponents[0])

                if len(subComponents) >= 2:
                    statsArray[self.secondComponentSizeIndex] = len(subComponents[1])

                statsArray[self.maxComponentEdgesIndex] = maxCompGraph.getNumEdges()
                statsArray[self.meanComponentSizeIndex] = sum([len(x) for x in subComponents])/float(statsArray[self.numComponentsIndex])
                statsArray[self.maxCompMeanDegreeIndex] = numpy.mean(maxCompGraph.outDegreeSequence())
            else:
                statsArray[self.maxComponentSizeIndex] = 0
                statsArray[self.maxComponentEdgesIndex] = 0 
                statsArray[self.meanComponentSizeIndex] = 0
                statsArray[self.geodesicDistMaxCompIndex] = 0

        if graph.getNumVertices() != 0:
            statsArray[self.meanDegreeIndex] = numpy.mean(graph.outDegreeSequence())
        else:
            statsArray[self.meanDegreeIndex] = 0
            
        if slowStats:
            if self.useFloydWarshall:
                logging.debug("Running Floyd-Warshall")
                P = graph.floydWarshall(False)
            else:
                logging.debug("Running Dijkstra's algorithm")
                P = graph.findAllDistances(False)

            statsArray[self.diameterIndex] = graph.diameter(P=P)
            statsArray[self.effectiveDiameterIndex] = graph.effectiveDiameter(self.q, P=P)
            statsArray[self.powerLawIndex] = graph.fitPowerLaw()[0]
            statsArray[self.geodesicDistanceIndex] = graph.geodesicDistance(P=P)
            statsArray[self.harmonicGeoDistanceIndex] = graph.harmonicGeodesicDistance(P=P)

            if graph.isUndirected() and len(subComponents) != 0:
                statsArray[self.geodesicDistMaxCompIndex] = graph.geodesicDistance(P=P, vertexInds=list(subComponents[0]))

        if treeStats:
            logging.debug("Computing statistics on trees")
            trees = graph.findTrees()
            statsArray[self.numTreesIndex] = len(trees)

            nonSingletonTrees = [c for c in trees if len(c) > 1]
            statsArray[self.numNonSingletonTreesIndex] = len(nonSingletonTrees)

            statsArray[self.meanTreeSizeIndex] = numpy.mean([len(x) for x in trees])
            treeDepths = [GraphUtils.treeDepth((graph.subgraph(list(x)))) for x in trees]
            statsArray[self.meanTreeDepthIndex] = numpy.mean(treeDepths)

            if len(trees) != 0:
                maxTreeGraph = graph.subgraph(trees[0])
                statsArray[self.maxTreeSizeIndex] = len(trees[0])
                statsArray[self.maxTreeDepthIndex] = GraphUtils.treeDepth(maxTreeGraph)

                if len(trees) >= 2:
                    secondTreeGraph = graph.subgraph(trees[1])
                    statsArray[self.secondTreeSizeIndex] = len(trees[1])
                    statsArray[self.secondTreeDepthIndex] = GraphUtils.treeDepth(secondTreeGraph)

        return statsArray

    def vectorStatistics(self, graph, treeStats=False, eigenStats=True):
        """
        Find a series of statistics for the given input graph which can be represented 
        as vector values.
        """
        Parameter.checkClass(graph, AbstractMatrixGraph)
        Parameter.checkBoolean(treeStats)
        statsDict = {}

        statsDict["inDegreeDist"] = graph.inDegreeDistribution()
        statsDict["outDegreeDist"] = graph.degreeDistribution()
        logging.debug("Computing hop counts")
        P = graph.findAllDistances(False)
        statsDict["hopCount"] = graph.hopCount(P)
        logging.debug("Computing triangle count")
        if graph.getNumVertices() != 0:
            statsDict["triangleDist"] = numpy.bincount(graph.triangleSequence())
        else:
            statsDict["triangleDist"] = numpy.array([])
        
        #Get the distribution of component sizes 
        logging.debug("Finding distribution of component sizes")
        
        if graph.isUndirected(): 
            components = graph.findConnectedComponents()
            if len(components) != 0: 
                statsDict["componentsDist"] = numpy.bincount(numpy.array([len(c) for c in components], numpy.int))

        #Make sure weight matrix is symmetric
        
        if graph.getNumVertices()!=0 and eigenStats:
            logging.debug("Computing eigenvalues/vectors")
            W = graph.getWeightMatrix()
            W = (W + W.T)/2
            eigenDistribution, V = numpy.linalg.eig(W)
            i = numpy.argmax(eigenDistribution)
            statsDict["maxEigVector"] = V[:, i]
            statsDict["eigenDist"] = numpy.flipud(numpy.sort(eigenDistribution[eigenDistribution>0]))
            gc.collect() 
        else:
            statsDict["maxEigVector"] = numpy.array([])
            statsDict["eigenDist"] = numpy.array([])

        if treeStats:
            logging.debug("Computing statistics on trees")
            trees = graph.findTrees()
            statsDict["treeSizesDist"] = numpy.bincount([len(x) for x in trees])
            treeDepths = [GraphUtils.treeDepth((graph.subgraph(x))) for x in trees]
            statsDict["treeDepthsDist"] = numpy.bincount(treeDepths)

        return statsDict

    def sequenceScalarStats(self, graph, subgraphIndices, slowStats=True, treeStats=False):
        """
        Pass in a graph and list of subgraph indices and returns a series of statistics. Each row
        corresponds to the statistics on the subgraph. 
        """
        Parameter.checkClass(graph, AbstractMatrixGraph)
        for inds in subgraphIndices:
            Parameter.checkList(inds, Parameter.checkInt, [0, graph.getNumVertices()])
        Parameter.checkBoolean(slowStats)
        Parameter.checkBoolean(treeStats)

        numGraphs = len(subgraphIndices)
        statsMatrix = numpy.zeros((numGraphs, self.numStats))

        for i in range(numGraphs):
            Util.printIteration(i, self.printStep, numGraphs)
            logging.debug("Subgraph size: " + str(len(subgraphIndices[i])))
            subgraph = graph.subgraph(subgraphIndices[i])
            statsMatrix[i, :] = self.scalarStatistics(subgraph, slowStats, treeStats)

        return statsMatrix

    def meanSeqScalarStats(self, graphList, slowStats=True, treeStats=False):
        """
        Pass in a list of tuples (graph, subgraphIndices) and returns a series of statistics. Each row
        corresponds to the statistics on the subgraph. All graphs must be the same size and computed 
        from the same distribution, and the number of subgraphs must be the same.
        """
        Parameter.checkBoolean(slowStats)
        Parameter.checkBoolean(treeStats)
        if len(graphList)==0:
            return -1 

        numGraphs = len(graphList)
        numSubgraphs = len(graphList[0][1])
        statsMatrix = numpy.zeros((numSubgraphs, self.numStats, numGraphs))

        for i in range(len(graphList)):
            (graph, subgraphIndices) = graphList[i]
            statsMatrix[:, :, i] = self.sequenceScalarStats(graph, subgraphIndices, slowStats, treeStats)

        return numpy.mean(statsMatrix, 2), numpy.std(statsMatrix, 2)

    def sequenceVectorStats(self, graph, subgraphIndices, treeStats=False, eigenStats=True):
        """
        Pass in a list of graphs are returns a series of statistics. Each list
        element is a dict of vector statistics. 
        """
        Parameter.checkClass(graph, AbstractMatrixGraph)
        for inds in subgraphIndices:
            Parameter.checkList(inds, Parameter.checkInt, [0, graph.getNumVertices()])
        Parameter.checkBoolean(treeStats)

        numGraphs = len(subgraphIndices)
        statsDictList = []

        for i in range(numGraphs):
            Util.printIteration(i, self.vectorPrintStep, numGraphs)
            subgraph = graph.subgraph(subgraphIndices[i])
            statsDictList.append(self.vectorStatistics(subgraph, treeStats, eigenStats))

        return statsDictList

    def sequenceClustering(self, graph, subgraphIndices, clusterFunc, maxComponent=True):
        """
        Take a graph and a sequence of indices corresponding to subgraphs and
        compute some clusters indices for each one. 
        """
        numGraphs = len(subgraphIndices)
        clusterList = []

        for i in range(numGraphs):
            Util.printIteration(i, self.vectorPrintStep, numGraphs)
            subgraph = graph.subgraph(subgraphIndices[i])

            if maxComponent:
                subComponents = subgraph.findConnectedComponents()
                subgraph = subgraph.subgraph(subComponents[-1])

            clusterList.append(clusterFunc(subgraph))

        return clusterList
        