"""
A class to compute a series of graph statistics over a sequence of subgraphs
for the HIV data. 
"""

import numpy
import logging
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.graph.GraphUtils import GraphUtils
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.viroscopy.HIVGraphReader import CsvConverters

class HIVGraphStatistics(object):
    def __init__(self, fInds):
        self.msmGeodesicIndex = 0
        self.menSubgraphGeodesicIndex = 1
        self.mostConnectedGeodesicIndex = 2
        self.ctSubgraphGeodesicIndex = 3

        self.numStats = 4
        self.printStep = 5
        self.topConnect = 0.1

        self.fInds = fInds

    def scalarStatistics(self, graph):

        Parameter.checkClass(graph, AbstractMatrixGraph)
        statsArray = numpy.ones(self.numStats)*-1

        #Find geodesic distance between MSMs
        logging.debug("Running Floyd-Warshall")
        P = graph.floydWarshall(False)
        V = graph.getVertexList().getVertices(list(range(graph.getNumVertices())))
        
        bisexual = CsvConverters.orientConv('HB')
        msmIndices = list(numpy.nonzero(V[:, self.fInds["orient"]]==bisexual)[0])
        if len(msmIndices) != 0:
            statsArray[self.msmGeodesicIndex] = graph.harmonicGeodesicDistance(P, msmIndices)

        male = CsvConverters.genderConv('M')
        menIndices = list(numpy.nonzero(V[:, self.fInds["gender"]]==male)[0])
        if len(menIndices) != 0: 
            menGraph = graph.subgraph(menIndices)
            statsArray[self.menSubgraphGeodesicIndex] = menGraph.harmonicGeodesicDistance()

        contactTrIndices = list(numpy.nonzero(V[:, self.fInds["contactTrace"]]==1)[0])
        if len(contactTrIndices) != 0:
            ctGraph = graph.subgraph(contactTrIndices)
            statsArray[self.ctSubgraphGeodesicIndex] = ctGraph.harmonicGeodesicDistance()

        degreeSequence = graph.outDegreeSequence()
        sortedInds = numpy.argsort(degreeSequence)
        numInds = int(float(graph.getNumVertices())*self.topConnect)
        topConnectInds = sortedInds[-numInds:]

        statsArray[self.mostConnectedGeodesicIndex] = graph.harmonicGeodesicDistance(P, topConnectInds)

        return statsArray 

    def sequenceScalarStats(self, graph, subgraphIndices):
        """
        Pass in a list of graphs are returns a series of statistics. Each row
        corresponds to the statistics on the subgraph.
        """

        numGraphs = len(subgraphIndices)
        statsMatrix = numpy.zeros((numGraphs, self.numStats))

        for i in range(numGraphs):
            Util.printIteration(i, self.printStep, numGraphs)
            logging.debug("Subgraph size: " + str(len(subgraphIndices[i])))
            subgraph = graph.subgraph(subgraphIndices[i])
            statsMatrix[i, :] = self.scalarStatistics(subgraph)

        return statsMatrix